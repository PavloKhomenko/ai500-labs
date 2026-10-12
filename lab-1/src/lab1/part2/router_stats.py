"""Per-layer f_i, MaxVio, load entropy, logsumexp, random-router noise floor, routing drift."""

import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import MixtralTopKRouter

from ..data import block_batches, sentence_batches
from .balancing import stats_from_counts


class InstrumentedRouter(MixtralTopKRouter):
    balancer = None

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        if self.balancer is None:
            top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        else:
            router_indices = self.balancer.select(router_logits, self.top_k)
            top_value = router_probs.gather(-1, router_indices)
        router_scores = top_value / top_value.sum(dim=-1, keepdim=True)
        self.last_logits, self.last_idx = router_logits, router_indices
        return router_logits, router_scores, router_indices


def patch_routers(model, balancers=None):
    routers = []
    for i, layer in enumerate(model.model.layers):
        router = layer.mlp.gate
        router.__class__ = InstrumentedRouter
        router.balancer = balancers[i] if balancers else None
        routers.append(router)
    return routers


@torch.no_grad()
def evaluate(model, routers, batches, keep_topk=False):
    model.eval()
    device = next(model.parameters()).device
    n_layers, n_experts = len(routers), routers[0].num_experts
    counts = torch.zeros(n_layers, n_experts, dtype=torch.long, device=device)
    lse_sum = torch.zeros(n_layers, dtype=torch.float64, device=device)
    margins, topk, seq_nll = [[] for _ in routers], [[] for _ in routers], []
    nll_sum, n_pred = 0.0, 0

    for ids, mask in batches:
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=mask, output_router_logits=False).logits
        nll = torch.stack([F.cross_entropy(logits[b, :-1].float(), ids[b, 1:], reduction="none") for b in range(len(ids))])
        nll = nll * mask[:, 1:]
        seq_nll += nll.sum(dim=1).tolist()
        nll_sum += nll.sum().item()
        n_pred += mask[:, 1:].sum().item()

        keep = mask.reshape(-1).bool()
        for layer, router in enumerate(routers):
            idx = router.last_idx[keep]
            lg = router.last_logits[keep].float()
            counts[layer] += torch.bincount(idx.reshape(-1), minlength=n_experts)
            lse_sum[layer] += torch.logsumexp(lg, dim=-1).double().sum()
            top = lg.topk(router.top_k + 1, dim=-1).values
            margins[layer].append((top[:, -2] - top[:, -1]).half().cpu())
            if keep_topk:
                topk[layer].append(idx.to(torch.uint8).cpu())

    n_routed = counts[0].sum().item() // routers[0].top_k
    out = {
        "nll_sum": nll_sum,
        "n_pred_tokens": n_pred,
        "ppl": float(np.exp(nll_sum / n_pred)),
        "n_routed_tokens": n_routed,
        "counts": counts.cpu(),
        "lse_mean": (lse_sum / n_routed).float().cpu(),
        "margin_median": torch.stack([torch.cat(m).float().median() for m in margins]),
        "seq_nll": seq_nll,
    }
    if keep_topk:
        out["topk"] = torch.stack([torch.cat(t) for t in topk], dim=1)
    return out


def eval_sets(data, eval_cfg, pad_id):
    return {
        "ua_test": lambda: block_batches(data["test"], eval_cfg["block_batch"]),
        "flores_uk": lambda: sentence_batches(data["flores"]["uk"], eval_cfg["sentence_batch"], pad_id),
        "flores_en": lambda: sentence_batches(data["flores"]["en"], eval_cfg["sentence_batch"], pad_id),
    }


def evaluate_all(model, routers, sets, path):
    metrics, raw = {}, {}
    for name, make_batches in sets.items():
        raw[name] = evaluate(model, routers, make_batches(), keep_topk=True)
        metrics[name] = save_eval(raw[name], path / name)
        print(f"  {name}: ppl {metrics[name]['ppl']:.3f}  maxvio mean {metrics[name]['maxvio_mean']:.3f} "
              f"max {metrics[name]['maxvio_max']:.3f}  entropy {metrics[name]['entropy_mean']:.4f}")
    js = js_divergence(raw["flores_uk"]["counts"], raw["flores_en"]["counts"])
    metrics["js_uk_en"] = {"mean": js.mean().item(), "per_layer": js.tolist()}
    (path / "metrics_all.json").write_text(json.dumps(metrics, indent=2))
    return metrics, raw


def summarize(result):
    s = stats_from_counts(result["counts"])
    return {
        "ppl": result["ppl"],
        "nll_sum": result["nll_sum"],
        "n_pred_tokens": result["n_pred_tokens"],
        "n_routed_tokens": result["n_routed_tokens"],
        "maxvio_mean": s["maxvio"].mean().item(),
        "maxvio_max": s["maxvio"].max().item(),
        "entropy_mean": s["entropy"].mean().item(),
        "lse_mean": result["lse_mean"].mean().item(),
        "nll_per_seq_mean": float(np.mean(result["seq_nll"])),
        "per_layer": {
            "maxvio": s["maxvio"].tolist(),
            "entropy": s["entropy"].tolist(),
            "lse": result["lse_mean"].tolist(),
            "margin_median": result["margin_median"].tolist(),
        },
    }


def save_eval(result, path):
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "counts.npy", result["counts"].numpy())
    np.save(path / "seq_nll.npy", np.array(result["seq_nll"]))
    if "topk" in result:
        np.save(path / "topk_uint8.npy", result["topk"].numpy())
    metrics = summarize(result)
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def noise_floor(n_tokens, n_experts, k, n_layers, repeats=1000, device="cpu", chunk=50_000):
    gen = torch.Generator(device=device).manual_seed(0)
    maxvio = []
    for _ in range(repeats):
        counts = torch.zeros(n_experts, dtype=torch.long, device=device)
        for start in range(0, n_tokens, chunk):
            t = min(chunk, n_tokens - start)
            idx = torch.rand(t, n_experts, generator=gen, device=device).topk(k, dim=-1).indices
            counts += torch.bincount(idx.reshape(-1), minlength=n_experts)
        maxvio.append(stats_from_counts(counts)["maxvio"].item())
    maxvio = torch.tensor(maxvio)
    pick = torch.randint(0, repeats, (repeats, n_layers), generator=torch.Generator().manual_seed(1))
    layer_max = maxvio[pick].max(dim=1).values
    return {
        "n_tokens": n_tokens,
        "maxvio_mean": maxvio.mean().item(),
        "maxvio_p95": maxvio.quantile(0.95).item(),
        "maxvio_max_over_layers_mean": layer_max.mean().item(),
        "maxvio_max_over_layers_p95": layer_max.quantile(0.95).item(),
    }


def jaccard_drift(topk_a, topk_b, n_experts, chunk=20_000):
    assert topk_a.shape == topk_b.shape, "drift needs the same tokens"
    k = topk_a.shape[-1]
    total = torch.zeros(topk_a.shape[1], dtype=torch.float64)
    for i in range(0, len(topk_a), chunk):
        a = F.one_hot(torch.as_tensor(topk_a[i : i + chunk]).long(), n_experts).sum(-2)
        b = F.one_hot(torch.as_tensor(topk_b[i : i + chunk]).long(), n_experts).sum(-2)
        inter = (a * b).sum(-1).double()
        total += (inter / (2 * k - inter)).sum(0)
    return (total / len(topk_a)).float()


def js_divergence(counts_p, counts_q):
    p = counts_p.double() / counts_p.sum(-1, keepdim=True)
    q = counts_q.double() / counts_q.sum(-1, keepdim=True)
    m = (p + q) / 2

    def kl(a, b):
        return (a * (a.clamp_min(1e-12) / b.clamp_min(1e-12)).log2()).sum(-1)

    return ((kl(p, m) + kl(q, m)) / 2).float()


def describe_layers(counts, n_top=3):
    s = stats_from_counts(counts)
    n_layers, n_experts = counts.shape
    f = s["f"] * n_experts
    layers = []
    for layer in range(n_layers):
        order = f[layer].argsort(descending=True)
        layers.append({
            "layer": layer,
            "maxvio": s["maxvio"][layer].item(),
            "entropy": s["entropy"][layer].item(),
            "n_overloaded": int((f[layer] > 2).sum()),
            "n_rare": int((f[layer] < 0.5).sum()),
            "top_experts": [(int(e), round(f[layer, e].item(), 2)) for e in order[:n_top]],
            "bottom_experts": [(int(e), round(f[layer, e].item(), 3)) for e in order[-n_top:]],
        })
    thirds = np.array_split(np.arange(n_layers), 3)
    groups = {
        name: {
            "layers": f"{idx[0]}-{idx[-1]}",
            "maxvio": s["maxvio"][idx].mean().item(),
            "entropy": s["entropy"][idx].mean().item(),
            "n_overloaded": float(np.mean([layers[i]["n_overloaded"] for i in idx])),
            "n_rare": float(np.mean([layers[i]["n_rare"] for i in idx])),
        }
        for name, idx in zip(["early", "middle", "late"], thirds)
        if len(idx)
    }
    return {"layers": layers, "groups": groups}
