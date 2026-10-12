"""CPU tests for Part 2 (tiny random Mixtral; no download): uv run pytest tests/"""

import copy

import torch
from transformers import MixtralConfig, MixtralForCausalLM

from lab1.common import load_config
from lab1.data import doc_key, pack, sentence_batches, split_of
from lab1.part2.balancing import BiasBalancer, aux_loss, load_stats, stats_from_counts, z_loss
from lab1.part2.pretrain import train
from lab1.part2.router_stats import evaluate, jaccard_drift, js_divergence, noise_floor, patch_routers

E, K = 8, 2


def tiny_model():
    torch.manual_seed(0)
    cfg = MixtralConfig(vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_local_experts=E,
                        num_experts_per_tok=K, max_position_embeddings=64, router_aux_loss_coef=0.0)
    return MixtralForCausalLM(cfg)


def test_aux_loss_is_one_at_balance_and_larger_when_collapsed():
    uniform = torch.zeros(E * 10, E)
    idx = torch.arange(E * 10 * K).remainder(E).view(-1, K)
    assert torch.isclose(aux_loss(uniform, K, E, idx), torch.tensor(1.0))
    collapsed = torch.zeros(100, E)
    collapsed[:, :K] = 10.0
    assert aux_loss(collapsed, K, E) > 3.0


def test_aux_loss_uses_step_counts_when_given():
    collapsed = torch.zeros(100, E)
    collapsed[:, :K] = 10.0
    balanced_step = torch.full((E,), 50)
    assert torch.isclose(aux_loss(collapsed, K, E, counts=balanced_step), torch.tensor(1.0))
    assert aux_loss(collapsed, K, E) > 3.0


def test_z_loss():
    assert torch.isclose(z_loss(torch.zeros(5, E)), torch.log(torch.tensor(float(E))) ** 2)


def test_maxvio_and_entropy():
    s = stats_from_counts(torch.full((E,), 10))
    assert s["maxvio"].item() == 0.0 and abs(s["entropy"].item() - 1.0) < 1e-9
    s = load_stats(torch.zeros(10, K, dtype=torch.long) + torch.arange(K), E)
    assert abs(s["maxvio"].item() - (E / K - 1)) < 1e-9


def test_bias_changes_selection_not_gates():
    model = tiny_model().eval()
    balancers = [BiasBalancer(E, gamma=1.0) for _ in range(2)]
    routers = patch_routers(model, balancers)
    x = torch.randn(20, 32)
    r = routers[0]
    _, w0, i0 = r(x)
    balancers[0].bias[i0[0, 0]] = -100.0
    logits, w1, i1 = r(x)
    assert not torch.equal(i0, i1)
    probs = torch.softmax(logits.float(), -1).gather(-1, i1)
    assert torch.allclose(w1, probs / probs.sum(-1, keepdim=True))


def test_bias_update_sign_rule():
    b = BiasBalancer(4, gamma=0.1)
    b.update(torch.tensor([10, 0, 5, 5]))
    assert torch.allclose(b.bias, torch.tensor([-0.1, 0.1, 0.0, 0.0]))


def test_patched_router_equals_original():
    model = tiny_model().eval()
    ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
        patch_routers(model)
        assert torch.equal(ref, model(input_ids=ids).logits)


def test_noise_floor_shrinks_with_tokens():
    small = noise_floor(500, 64, 8, 24, repeats=50)
    large = noise_floor(20_000, 64, 8, 24, repeats=50)
    assert large["maxvio_mean"] < small["maxvio_mean"] < small["maxvio_max_over_layers_mean"]


def test_drift_and_js():
    a = torch.rand(100, 2, E).topk(K, dim=-1).indices.to(torch.uint8).numpy()
    assert torch.allclose(jaccard_drift(a, a, E), torch.ones(2))
    c = torch.randint(1, 10, (3, E))
    assert torch.allclose(js_divergence(c, c), torch.zeros(3), atol=1e-6)


def test_split_is_deterministic_and_text_based():
    k1, k2 = doc_key("Київ — столиця України." * 10), doc_key("Київ — столиця України." * 10)
    assert k1 == k2 and split_of(k1, 30, 10) == split_of(k2, 30, 10)
    splits = [split_of(doc_key(str(i)), 30, 10) for i in range(20_000)]
    assert 0.02 < splits.count("test") / 20_000 < 0.04 and 0.005 < splits.count("val") / 20_000 < 0.015


def test_pack_and_padding_mask():
    assert pack([[1, 5, 6, 2], [1, 7, 2]], 3).tolist() == [[1, 5, 6], [2, 1, 7]]
    ids, mask = next(sentence_batches([[1, 3], [1, 3, 4, 5]], 2, pad_id=2))
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]


def test_eval_ignores_padding():
    model = tiny_model()
    routers = patch_routers(model)
    out = evaluate(model, routers, sentence_batches([[1, 3], [1, 3, 4, 5]], 2, pad_id=2), keep_topk=True)
    assert out["n_pred_tokens"] == 1 + 3 and out["n_routed_tokens"] == 6 and out["topk"].shape == (6, 2, K)


def smoke_cfg(strategy_overrides=None):
    cfg = copy.deepcopy(load_config())
    cfg["part2"]["train"].update(token_budget=4 * 2 * 2 * 16, micro_batch=2, grad_accum=2, warmup_steps=1,
                                 lr=1e-2, bias_update_rate=0.1)
    cfg["part2"]["eval"]["block_batch"] = 4
    return cfg


def run_smoke(strategy, tmp_path):
    tmp_path.mkdir()
    model = tiny_model()
    data = {"train": torch.randint(3, 64, (40, 16), dtype=torch.int32), "val": torch.randint(3, 64, (4, 16))}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train(model, data, smoke_cfg(), strategy, 42, tmp_path, save=True, validate=True)
    ckpt = torch.load(tmp_path / "final.pt", weights_only=False)
    changed = {n for n, p in model.named_parameters() if not torch.equal(p.detach().float(), before[n].float())}
    return changed, ckpt


def test_smoke_strategies(tmp_path):
    changed, ckpt = run_smoke("none", tmp_path / "a")
    assert any(".mlp.gate." in n for n in changed) and any(".self_attn." in n for n in changed)
    assert not any(".mlp.experts." in n or "embed" in n for n in changed) and ckpt["bias"] is None

    changed, ckpt = run_smoke("frozen_router", tmp_path / "e")
    assert not any(".mlp.gate." in n for n in changed) and any(".self_attn." in n for n in changed)

    _, ckpt = run_smoke("bias", tmp_path / "d")
    assert ckpt["bias"].abs().sum() > 0 and ckpt["bias"].shape == (2, E)

    _, ckpt = run_smoke("aux_1e-2", tmp_path / "c")
    assert (tmp_path / "c" / "train_log.jsonl").read_text().count("\n") == 4
