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
    probs = F.softmax(router_logits.float(), dim=-1)
    if counts is None:
        if topk_idx is None:
            topk_idx = probs.topk(top_k, dim=-1).indices
        counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts)
    f = counts.float() / counts.sum()
    P = probs.mean(dim=0)
    return n_experts * (f * P).sum()


def z_loss(router_logits):
    return torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()


class BiasBalancer:
    def __init__(self, n_experts, gamma, device="cpu"):
        self.gamma = gamma
        self.bias = torch.zeros(n_experts, dtype=torch.float32, device=device)

    def select(self, scores, k):
        return (scores.float() + self.bias).topk(k, dim=-1).indices

    @torch.no_grad()
    def update(self, expert_counts):
        load = expert_counts.float().to(self.bias.device)
        self.bias += self.gamma * torch.sign(load.mean() - load)


def stats_from_counts(counts):
    counts = counts.double()
    n_experts = counts.shape[-1]
    f = counts / counts.sum(dim=-1, keepdim=True)
    mean = counts.mean(dim=-1)
    maxvio = (counts.max(dim=-1).values - mean) / mean
    entropy = -(f * f.clamp_min(1e-12).log()).sum(dim=-1) / math.log(n_experts)
    return {"f": f, "maxvio": maxvio, "entropy": entropy}


def load_stats(topk_idx, n_experts):
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts)
    return {"counts": counts, **stats_from_counts(counts)}
