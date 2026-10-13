import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from ..part2.balancing import BiasBalancer, aux_loss, stats_from_counts

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


class ExpertLoRA(nn.Module):
    def __init__(self, n_experts, d_in, d_out, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_experts, rank, d_in))
        self.B = nn.Parameter(torch.zeros(n_experts, d_out, rank))
        for a in self.A.data:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x, e):
        return (x @ self.A[e].to(x.dtype).T) @ self.B[e].to(x.dtype).T * self.scale


class MixLoRAMLP(nn.Module):
    def __init__(self, mlp, n_experts, top_k, rank, alpha, balancer=None):
        super().__init__()
        d, h = mlp.hidden_size, mlp.intermediate_size
        self.mlp, self.n_experts, self.top_k, self.balancer = mlp, n_experts, top_k, balancer
        self.router = nn.Linear(d, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.gate = ExpertLoRA(n_experts, d, h, rank, alpha)
        self.up = ExpertLoRA(n_experts, d, h, rank, alpha)
        self.down = ExpertLoRA(n_experts, h, d, rank, alpha)
        self.last_logits = self.last_idx = None

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        with torch.autocast(x.device.type, enabled=False):
            logits = F.linear(x.float(), self.router.weight)
        probs = logits.softmax(dim=-1)
        if self.balancer is None:
            idx = probs.topk(self.top_k, dim=-1).indices
        else:
            idx = self.balancer.select(logits, self.top_k)  # bias picks experts, weights stay unbiased
        w = probs.gather(-1, idx)
        w = w / w.sum(dim=-1, keepdim=True)
        self.last_logits, self.last_idx = logits, idx

        g, u = self.mlp.gate_proj(x), self.mlp.up_proj(x)
        mixed, delta = torch.zeros_like(g), torch.zeros_like(x)
        # ponytail: python loop over E experts, a grouped GEMM kernel if time per step matters
        for e in range(self.n_experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = x[tok]
            he = self.mlp.act_fn(g[tok] + self.gate(xe, e)) * (u[tok] + self.up(xe, e))
            we = w[tok, slot, None].to(he.dtype)
            mixed.index_add_(0, tok, we * he)
            delta.index_add_(0, tok, we * self.down(he, e))
        return (self.mlp.down_proj(mixed) + delta).view(shape)


class Adapter:
    def __init__(self, model, rank, alpha, targets, n_experts=0, top_k=0, balance="none", gamma=0.0):
        layers = model.model.layers
        self.cost = {name: sum(m.in_features + m.out_features for n, m in model.named_modules() if n.endswith(name))
                     for name in ATTN + MLP}
        self.rank, self.n_experts, self.top_k, self.balance = rank, n_experts, top_k, balance
        self.n_layers, self.hidden = len(layers), model.config.hidden_size
        self.model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets,
                                                      lora_dropout=0.0, bias="none"))
        self.balancers, self.moes = None, []
        if n_experts:
            device = next(model.parameters()).device
            if balance == "bias":
                self.balancers = [BiasBalancer(n_experts, gamma, device) for _ in layers]
            for i, layer in enumerate(layers):
                balancer = self.balancers[i] if self.balancers else None
                layer.mlp = MixLoRAMLP(layer.mlp, n_experts, top_k, rank, alpha, balancer).to(device)
                self.moes.append(layer.mlp)
        for p in self.trainable():
            p.data = p.data.float()

    @classmethod
    def lora(cls, model, rank, targets=ATTN + MLP):
        return cls(model, rank, 2 * rank, targets)

    @classmethod
    def mixlora(cls, model, p3, balance):
        return cls(model, p3["lora_rank"], p3["lora_alpha"], ATTN, p3["n_experts"], p3["top_k"], balance,
                   p3["bias_update_rate"])

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def n_trainable(self):
        return sum(p.numel() for p in self.trainable())

    def expected_trainable(self):
        attn = self.rank * sum(self.cost[n] for n in ATTN)
        if not self.moes:
            return attn + self.rank * sum(self.cost[n] for n in self.model.peft_config["default"].target_modules
                                          if n in MLP)
        return attn + self.n_experts * self.rank * sum(self.cost[n] for n in MLP) + self.n_layers * self.hidden * self.n_experts

    def n_active(self):
        if not self.moes:
            return self.n_trainable()
        per_expert = self.rank * sum(self.cost[n] for n in MLP)
        return self.n_trainable() - (self.n_experts - self.top_k) * per_expert

    def matched_rank(self, n_params):
        return max(1, round(n_params / sum(self.cost.values())))

    def param_table(self):
        return {"trainable": self.n_trainable(), "active": self.n_active(),
                "lora_rank_matched_trainable": self.matched_rank(self.n_trainable()),
                "lora_rank_matched_active": self.matched_rank(self.n_active())}

    def step_counts(self, mask):
        return torch.stack([torch.bincount(m.last_idx[mask].reshape(-1), minlength=self.n_experts) for m in self.moes])

    def aux_loss(self, mask):
        return sum(aux_loss(m.last_logits[mask], self.top_k, self.n_experts, topk_idx=m.last_idx[mask])
                   for m in self.moes)

    def update_bias(self, counts):
        for b, c in zip(self.balancers, counts):
            b.update(c)

    @staticmethod
    def load_stats(counts):
        st = stats_from_counts(counts)
        return {"counts": counts.tolist(), "f": st["f"].tolist(), "maxvio": st["maxvio"].tolist(),
                "entropy": st["entropy"].tolist()}

    def state_dict(self):
        state = {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}
        for i, b in enumerate(self.balancers or []):
            state[f"balancer_bias.{i}"] = b.bias.cpu()
        return state

    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.no_grad()
    def load(self, path):
        state = torch.load(path)
        for i, b in enumerate(self.balancers or []):
            b.bias.copy_(state.pop(f"balancer_bias.{i}"))
        expected = {n for n, p in self.model.named_parameters() if p.requires_grad}
        assert set(state) == expected, (set(state) ^ expected)
        params = dict(self.model.named_parameters())
        for n, t in state.items():
            params[n].copy_(t)

    def check_param_count(self):
        assert self.n_trainable() == self.expected_trainable(), (self.n_trainable(), self.expected_trainable())
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        assert all("lora_" in n or ".mlp.router." in n or ".mlp.gate." in n or ".mlp.up." in n or ".mlp.down." in n
                   for n in names), "a base weight is trainable"
        print(f"trainable {self.n_trainable():,} == expected, active per token {self.n_active():,}")

    @torch.no_grad()
    def check_identity(self, ids, ref_logits, tol=1e-3):
        logits = self.model(input_ids=ids).logits.float()
        diff = (logits - ref_logits.float()).abs().max().item()
        assert diff < tol, diff
        if self.moes:
            assert all(m.last_idx.shape == (ids.numel(), self.top_k) for m in self.moes)
        print(f"identity: max|diff| = {diff:.2e}")

    @torch.no_grad()
    def check_roundtrip(self, ids, path):
        for p in self.trainable():
            p.normal_(std=0.02)
        for b in self.balancers or []:
            b.bias.normal_()
        ref = self.model(input_ids=ids).logits.float()
        self.save(path)
        for p in self.trainable():
            p.zero_()
        for b in self.balancers or []:
            b.bias.zero_()
        self.load(path)
        diff = (self.model(input_ids=ids).logits.float() - ref).abs().max().item()
        assert diff == 0, diff
        print("save/load round-trip: exact")
