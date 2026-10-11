"""Shared with Part 3 (A), keep this interface stable:

aux_loss(router_logits, top_k, n_experts) -> Tensor
class BiasBalancer: select(scores, k) -> topk_idx; update(expert_counts)
load_stats(topk_idx, n_experts) -> dict

All functions work on ONE layer: router_logits [T, n_experts], topk_idx [T, k].
"""

import math

import torch
import torch.nn.functional as F


def aux_loss(router_logits, top_k, n_experts, topk_idx=None, counts=None):
    """Switch / GShard load-balancing loss in paper form: N * sum_i f_i * P_i.

    f_i = share of routed slots on expert i (sums to 1), P_i = mean router probability.
    Equals 1.0 at perfect balance (HF's load_balancing_loss_func equals top_k there).
    f_i is a count and has no gradient; the gradient flows through P_i.
    topk_idx: the experts actually selected (defaults to the top-k of the probabilities).
    counts: expert counts over a larger batch (e.g. the whole optimizer step so far); if given,
        f_i comes from them instead of from this batch (global-batch balancing, Qiu et al. 2025).
    """
    probs = F.softmax(router_logits.float(), dim=-1)
    if counts is None:
        if topk_idx is None:
            topk_idx = probs.topk(top_k, dim=-1).indices
        counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts)
    f = counts.float() / counts.sum()
    P = probs.mean(dim=0)
    return n_experts * (f * P).sum()


def z_loss(router_logits):
    """ST-MoE router z-loss: mean over tokens of logsumexp(logits)^2."""
    return torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()


class BiasBalancer:
    """Auxiliary-loss-free balancing (Wang et al., 2024) for one layer.

    The bias is a float32 tensor outside the optimizer. It is added to the scores
    for top-k SELECTION only; the caller takes gate weights from the unbiased scores.
    """

    def __init__(self, n_experts, gamma, device="cpu"):
        self.gamma = gamma
        self.bias = torch.zeros(n_experts, dtype=torch.float32, device=device)

    def select(self, scores, k):
        return (scores.float() + self.bias).topk(k, dim=-1).indices

    @torch.no_grad()
    def update(self, expert_counts):
        """Sign rule after each optimizer step: underloaded experts go up, overloaded go down."""
        load = expert_counts.float().to(self.bias.device)
        self.bias += self.gamma * torch.sign(load.mean() - load)


def stats_from_counts(counts):
    """Balance metrics from expert counts [..., n_experts] (one row per layer)."""
    counts = counts.double()
    n_experts = counts.shape[-1]
    f = counts / counts.sum(dim=-1, keepdim=True)
    mean = counts.mean(dim=-1)
    maxvio = (counts.max(dim=-1).values - mean) / mean
    entropy = -(f * f.clamp_min(1e-12).log()).sum(dim=-1) / math.log(n_experts)
    return {"f": f, "maxvio": maxvio, "entropy": entropy}


def load_stats(topk_idx, n_experts):
    """Per-layer load metrics from the selected experts [T, k]."""
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts)
    return {"counts": counts, **stats_from_counts(counts)}
