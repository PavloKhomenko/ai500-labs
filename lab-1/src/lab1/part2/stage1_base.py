"""Stage 1: everything that needs no training.

    python -m lab1.part2.stage1_base

1. data splits (+ contamination checks)     6. base model: PPL + routing on UA test / FLORES uk / en
2. tokenizer checks + fertility uk vs en    7. random-router noise floor
3. HF aux-loss sanity check                 8. gradient check (aux loss reaches every router)
4. model card (total / active params)       9. throughput test -> token budget and time per run
5. router patch equivalence                10. gamma check for the bias run
Outputs: <output_dir>/part2/{data,base,figures}/, W&B run "stage1-base".
"""

import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

from ..common import load_config, out_dir
from ..data import prepare_part2_data
from . import wandb_log as wl
from .balancing import aux_loss
from .pretrain import load_model, prepare_for_training, train
from .router_stats import describe_layers, eval_sets, evaluate_all, noise_floor, patch_routers


def check_data(data, report, tok):
    print(json.dumps(report, indent=2, ensure_ascii=False))
    rows = [{"split": s, **{k: v for k, v in r.items() if k != "sources"}} for s, r in report["splits"].items()]
    rows += [{"split": f"flores_{lang}", "docs": report[f"flores_{lang}"]["sentences"],
              "tokens": report[f"flores_{lang}"]["tokens"]} for lang in ["uk", "en"]]
    if any(report["flores_sentences_in_train"].values()):
        print(f"WARNING: FLORES sentences found in train: {report['flores_sentences_in_train']}")
    # Tokenization: we add BOS / EOS ourselves; there must never be two BOS in a row.
    assert (tok.bos_token_id, tok.eos_token_id) == (1, 2), "unexpected special-token ids"
    for split in ["train", "val", "test"]:
        flat = data[split].flatten()
        assert not ((flat[:-1] == tok.bos_token_id) & (flat[1:] == tok.bos_token_id)).any(), f"double BOS in {split}"
    for lang in ["uk", "en"]:
        assert all(s[0] == tok.bos_token_id and s.count(tok.bos_token_id) == 1 for s in data["flores"][lang])
    return pd.DataFrame(rows)


def fertility(data, flores_texts):
    rows = []
    for lang in ["uk", "en"]:
        n_tok = sum(len(s) - 1 for s in data["flores"][lang])  # minus BOS
        n_words = sum(len(t.split()) for t in flores_texts[lang])
        n_chars = sum(len(t) for t in flores_texts[lang])
        rows.append({"lang": lang, "tokens": n_tok, "tokens_per_word": n_tok / n_words,
                     "tokens_per_char": n_tok / n_chars, "tokens_per_sentence": n_tok / len(flores_texts[lang])})
    return pd.DataFrame(rows)


def hf_aux_check(name, ids, device):
    """The assignment's check: HF aux loss is non-zero and changes with the coefficient,
    which must be passed to from_pretrained (changing the config afterwards does nothing)."""
    def run(model):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            return model(input_ids=ids, labels=ids)

    m = load_model(name, device, aux_coef=0.01, output_router_logits=True)
    o1 = run(m)
    assert o1.aux_loss is not None and o1.aux_loss.item() > 0, "HF aux loss missing"
    k, n_exp = m.config.num_experts_per_tok, m.config.num_local_experts
    ours = torch.stack([aux_loss(lg, k, n_exp) for lg in o1.router_logits]).mean().item()
    m.config.router_aux_loss_coef = 0.02
    o1b = run(m)
    del m
    torch.cuda.empty_cache()
    m = load_model(name, device, aux_coef=0.02, output_router_logits=True)
    o2 = run(m)
    del m
    torch.cuda.empty_cache()
    assert o2.loss.item() != o1.loss.item(), "loss does not change with router_aux_loss_coef"
    res = {
        "hf_aux": o1.aux_loss.item(),
        "ours_paper_form_mean_over_layers": ours,
        "hf_over_ours": o1.aux_loss.item() / ours,
        "loss_coef_0.01": o1.loss.item(),
        "loss_coef_0.01_then_config_set_0.02": o1b.loss.item(),
        "loss_coef_0.02": o2.loss.item(),
        "loss_diff_vs_0.01_x_aux": (o2.loss - o1.loss).item() / (0.01 * o1.aux_loss.item()),
    }
    print("HF aux check:", json.dumps(res, indent=2))
    return res


def model_card(model):
    c = model.config
    count = lambda key: sum(p.numel() for n, p in model.named_parameters() if key in n)  # noqa: E731
    total = sum(p.numel() for p in model.parameters())
    expert, emb = count(".mlp.experts."), count("embed_tokens")
    active = total - expert + expert * c.num_experts_per_tok / c.num_local_experts
    card = {
        "layers": c.num_hidden_layers, "hidden": c.hidden_size, "routed_experts": c.num_local_experts,
        "shared_experts": 0, "top_k": c.num_experts_per_tok, "expert_intermediate": c.intermediate_size,
        "gating": "softmax over all experts -> top-k -> renormalize the k weights",
        "pretraining_balancing": "aux loss 0.01 (global batch) + z-loss 0.001 (EuroMoE report)",
        "router_jitter": c.router_jitter_noise,
        "total_params_B": total / 1e9, "expert_params_B": expert / 1e9,
        "attention_params_M": count(".self_attn.") / 1e6, "router_params_M": count(".mlp.gate.") / 1e6,
        "active_params_B": active / 1e9,
        "active_params_B_untied_lm_head": (active + emb) / 1e9,
        "tie_word_embeddings": c.tie_word_embeddings,
    }
    print("model card:", json.dumps(card, indent=2))
    return card


def router_equivalence(model, ids):
    """The patched router (bias off) must reproduce the original model exactly."""
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=ids.is_cuda):
        ref = model(input_ids=ids).logits.float()
        routers = patch_routers(model)
        new = model(input_ids=ids).logits.float()
    diff = (ref - new).abs().max().item()
    assert torch.isfinite(ref).all(), "non-finite logits in fp16"
    assert diff == 0.0, f"patched router changes the output (max diff {diff})"
    print(f"router patch equivalence: max abs diff {diff}")
    return routers


def grad_check(model, routers, ids):
    """Our aux loss under gradient checkpointing (use_reentrant=False) reaches every router."""
    prepare_for_training(model, train_router=True)
    model.train()
    with torch.autocast("cuda", dtype=torch.float16, enabled=ids.is_cuda):
        model(input_ids=ids)
    k, n_exp = routers[0].top_k, routers[0].num_experts
    sum(aux_loss(r.last_logits, k, n_exp, r.last_idx) for r in routers).backward()
    norms = [r.weight.grad.norm().item() if r.weight.grad is not None else 0.0 for r in routers]
    model.zero_grad(set_to_none=True)
    assert all(n > 0 for n in norms), f"router without gradient: {norms}"
    print(f"grad check: router grad norm min {min(norms):.3e} max {max(norms):.3e}")
    return norms


def main():
    cfg = load_config()
    p2 = cfg["part2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir, fig_dir = out_dir(cfg, "part2", "base"), out_dir(cfg, "part2", "figures")
    data_dir = out_dir(cfg, "part2", "data")
    wl.init_run(cfg, "stage1-base", job_type="analysis", config={
        **p2, "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"})
    summary = {}

    print("\n== 1-2. data + tokenizer ==")
    tok = AutoTokenizer.from_pretrained(p2["model"])
    data, report = prepare_part2_data(cfg, tok, data_dir)
    wl.log_table("data/splits", check_data(data, report, tok), base_dir / "splits")
    flores_texts = json.loads((data_dir / "flores_texts.json").read_text())
    fert = fertility(data, flores_texts)
    print(fert.to_string(index=False))
    wl.log_table("data/fertility", fert, base_dir / "fertility")
    summary["split_report"], summary["fertility"] = report, fert.to_dict("records")

    print("\n== 3. HF aux-loss sanity check ==")
    probe = data["test"][:2].long().to(device)
    summary["hf_aux_check"] = hf_aux_check(p2["model"], probe, device)

    print("\n== 4-5. model card + router patch equivalence ==")
    model = load_model(p2["model"], device)
    summary["model_card"] = model_card(model)
    wl.log_table("model/card", pd.DataFrame([summary["model_card"]]), base_dir / "model_card")
    routers = router_equivalence(model, probe)
    n_layers, n_exp, k = len(routers), routers[0].num_experts, routers[0].top_k

    print("\n== 6. base model evaluation ==")
    t0 = time.time()
    metrics, raw = evaluate_all(model, routers, eval_sets(data, p2["eval"], tok.eos_token_id), base_dir)
    summary["eval_minutes"] = (time.time() - t0) / 60
    summary["base"] = {s: {k_: v for k_, v in m.items() if k_ != "per_layer"} for s, m in metrics.items()}
    analysis = describe_layers(raw["ua_test"]["counts"])
    (base_dir / "layer_analysis.json").write_text(json.dumps(analysis, indent=2))
    print("early / middle / late:", json.dumps(analysis["groups"], indent=2))
    wl.log_table("base/layer_groups", pd.DataFrame([{"group": g, **v} for g, v in analysis["groups"].items()]),
                 base_dir / "layer_groups")
    wl.log_table("base/layers", pd.DataFrame(analysis["layers"]).astype(str), base_dir / "layers")
    for name in ["ua_test", "flores_uk", "flores_en"]:
        wl.log_figure(f"heatmap/base_{name}", wl.heatmap_fig(raw[name]["counts"], f"base, {name}"),
                      fig_dir / f"heatmap_base_{name}.png")
    lang = pd.DataFrame([{
        "lang": l_, "ppl": metrics[f"flores_{l_}"]["ppl"], "nll_per_sentence": metrics[f"flores_{l_}"]["nll_per_seq_mean"],
        "maxvio_mean": metrics[f"flores_{l_}"]["maxvio_mean"], "entropy_mean": metrics[f"flores_{l_}"]["entropy_mean"],
    } for l_ in ["uk", "en"]])
    print(lang.to_string(index=False), f"\nJS(uk, en) mean over layers: {metrics['js_uk_en']['mean']:.4f}")
    wl.log_table("base/uk_vs_en", lang, base_dir / "uk_vs_en")
    wl.log_figure("base/js_uk_en", wl.layer_lines_fig({"base": metrics["js_uk_en"]["per_layer"]}, "JS divergence",
                  "FLORES: uk vs en expert load, base"), fig_dir / "js_uk_en_base.png")

    print("\n== 7. noise floor ==")
    tc = p2["train"]
    batch_tokens = tc["micro_batch"] * tc["grad_accum"] * p2["data"]["seq_len"]
    sizes = {"train_batch": batch_tokens, **{s: metrics[s]["n_routed_tokens"] for s in ["ua_test", "flores_uk", "flores_en"]}}
    floor = {s: noise_floor(n, n_exp, k, n_layers, p2["eval"]["noise_floor_repeats"], device.type) for s, n in sizes.items()}
    (base_dir / "noise_floor.json").write_text(json.dumps(floor, indent=2))
    wl.log_table("base/noise_floor", pd.DataFrame([{"set": s, **v} for s, v in floor.items()]), base_dir / "noise_floor")
    print(pd.DataFrame(floor).T.to_string())
    summary["noise_floor"] = floor
    wl.log_figure("base/maxvio_per_layer", wl.layer_lines_fig(
        {s: metrics[s]["per_layer"]["maxvio"] for s in ["ua_test", "flores_uk", "flores_en"]}, "global MaxVio",
        "base model", hline=floor["ua_test"]["maxvio_p95"], hline_label="random router p95 (UA test)"),
        fig_dir / "maxvio_per_layer_base.png")

    print("\n== 8. gradient check ==")
    summary["router_grad_norms"] = grad_check(model, routers, data["train"][:1].long().to(device))

    print("\n== 9. throughput test (changes the weights; runs last) ==")
    throughput = {}
    for mb, accum in [(2, 4), (4, 2)]:
        c = copy.deepcopy(cfg)
        c["part2"]["train"].update(micro_batch=mb, grad_accum=accum)
        try:
            throughput[f"{mb}x{accum}"] = train(model, data, c, "none", cfg["seed"], base_dir, max_steps=10,
                                                save=False, validate=False)
        except torch.cuda.OutOfMemoryError:
            throughput[f"{mb}x{accum}"] = "OOM"
        torch.cuda.empty_cache()
    print(json.dumps(throughput, indent=2))
    ok = [v for v in throughput.values() if v != "OOM"]
    best = max(v["tok_per_s"] for v in ok)
    budgets = {f"{b // 10**6}M": round(b / best / 60, 1) for b in [5 * 10**6, 8 * 10**6, 10 * 10**6]}
    print(f"best {best:,.0f} tok/s -> minutes per run (training only): {budgets}; eval per model "
          f"{summary['eval_minutes']:.1f} min")
    summary["throughput"], summary["minutes_per_run"] = throughput, budgets

    print("\n== 10. gamma check for the bias run ==")
    n_steps = tc["token_budget"] // batch_tokens
    max_move = tc["bias_update_rate"] * n_steps
    margins = np.array(metrics["ua_test"]["per_layer"]["margin_median"])
    gamma = {"n_steps": n_steps, "gamma": tc["bias_update_rate"], "max_bias_move": max_move,
             "median_top8_top9_margin": float(np.median(margins)),
             "layers_with_margin_below_max_move": int((margins < max_move).sum())}
    print(json.dumps(gamma, indent=2))
    if gamma["median_top8_top9_margin"] > max_move:
        print("-> the bias cannot reach typical margins within the budget; consider gamma = 1e-2 (decide BEFORE training)")
    summary["gamma_check"] = gamma

    (base_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    wl.log_summary({"base_ua_ppl": metrics["ua_test"]["ppl"], "base_ua_maxvio_mean": metrics["ua_test"]["maxvio_mean"],
                    "tok_per_s": best})
    wl.log_artifact(base_dir, "stage1-base", "analysis")
    wl.finish()
    print(f"\nStage 1 done. Outputs in {base_dir}. Write report/part2_predictions.yaml before Stage 2.")


if __name__ == "__main__":
    main()
