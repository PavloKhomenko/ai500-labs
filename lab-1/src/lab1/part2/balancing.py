"""Shared with Part 3 (A), keep this interface stable:

aux_loss(router_logits, top_k, n_experts) -> Tensor
class BiasBalancer: select(scores, k) -> topk_idx; update(expert_counts)
load_stats(topk_idx, n_experts) -> dict
"""
