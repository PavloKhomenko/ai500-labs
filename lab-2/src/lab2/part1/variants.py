import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.mamba.modeling_mamba import MambaMixer, apply_mask_to_padding_states

try:
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn

    KERNELS = True
except ImportError:
    causal_conv1d_fn = mamba_inner_fn = selective_scan_fn = None
    KERNELS = False

VARIANTS = {  # static_dt, static_B, static_C
    "V1": (True, True, True),
    "V2": (False, True, True),
    "V3": (True, False, True),
    "V4": (False, False, False),
}


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """Torch reference (CPU checks only). u, delta, z: (b, d, L); A: (d, N); B, C: (d, N) static or (b, N, L)."""
    delta = delta.float() + (delta_bias[..., None].float() if delta_bias is not None else 0)
    if delta_softplus:
        delta = F.softplus(delta)
    dA = torch.exp(delta[..., None] * A[None, :, None, :])  # (b, d, L, N)
    Bt = B[None, :, None, :] if B.dim() == 2 else B.transpose(1, 2)[:, None]  # broadcast to (b, d, L, N)
    Ct = C[None, :, None, :] if C.dim() == 2 else C.transpose(1, 2)[:, None]
    dBu = delta[..., None] * Bt.float() * u.float()[..., None]
    state = torch.zeros_like(dBu[:, :, 0])
    ys = []
    for t in range(u.shape[-1]):
        state = dA[:, :, t] * state + dBu[:, :, t]
        ys.append((state * Ct[:, :, min(t, Ct.shape[2] - 1)].float()).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u.float() * D[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(u.dtype)


class SelectiveMixer(MambaMixer):
    @classmethod
    def from_mixer(cls, mixer, variant, calib=None):
        mixer.__class__ = cls  # in place: keeps every pretrained weight and hook
        mixer.variant = variant
        mixer.static_dt, mixer.static_B, mixer.static_C = VARIANTS[variant]
        r, n, d = mixer.time_step_rank, mixer.ssm_state_size, mixer.intermediate_size
        # live x_proj rows: Δ rows always (zero-weighted when Δ is static), B rows only when B is selective
        mixer.live_rows = r + 2 * n if variant == "V4" else (r + n if not mixer.static_B else r)
        dev = mixer.A_log.device
        with torch.no_grad():
            if mixer.static_dt:
                mixer.dt_proj.weight.zero_()
                mixer.dt_proj.weight.requires_grad_(False)
                mixer.dt_proj.bias.copy_(calib["beta"].to(dev))
            if mixer.static_B:
                mixer.B_static = nn.Parameter(calib["B"].to(dev, torch.float32)[None].expand(d, n).clone())
            if mixer.static_C:
                mixer.C_static = nn.Parameter(calib["C"].to(dev, torch.float32)[None].expand(d, n).clone())
        return mixer

    def ssm_params(self, x):
        r, n = self.time_step_rank, self.ssm_state_size
        x_dbl = F.linear(x.transpose(1, 2), self.x_proj.weight[: self.live_rows])  # (b, L, live_rows)
        dt = self.dt_proj.weight @ x_dbl[..., :r].transpose(1, 2)
        B = self.B_static if self.static_B else x_dbl[..., r : r + n].transpose(1, 2).contiguous()
        C = self.C_static if self.static_C else x_dbl[..., r + n : r + 2 * n].transpose(1, 2).contiguous()
        return dt, B, C

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        assert cache_params is None, "variants run with use_cache=False only"
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        xz = self.in_proj(hidden_states).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        out_bias = self.out_proj.bias.float() if self.use_bias else None

        if self.training and mamba_inner_fn is not None:
            return mamba_inner_fn(
                xz, self.conv1d.weight, self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight[: self.live_rows], self.dt_proj.weight, self.out_proj.weight, out_bias, A,
                self.B_static.float() if self.static_B else None,
                self.C_static.float() if self.static_C else None,
                self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )

        x, z = xz.chunk(2, dim=1)
        w, b = self.conv1d.weight.squeeze(1), self.conv1d.bias if self.use_conv_bias else None
        if causal_conv1d_fn is not None:
            x = causal_conv1d_fn(x, w, b, activation=self.activation)
        else:
            x = self.act(F.conv1d(x, w.unsqueeze(1), b, padding=w.shape[-1] - 1, groups=x.shape[1])[..., : x.shape[-1]])
        x = apply_mask_to_padding_states(x.transpose(1, 2), attention_mask).transpose(1, 2)
        dt, B, C = self.ssm_params(x)
        scan = selective_scan_fn if selective_scan_fn is not None else selective_scan_ref
        y = scan(x, dt, A, B, C, self.D.float(), z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        return self.out_proj(y.transpose(1, 2).to(hidden_states.dtype))


def apply_variant(model, variant, calib=None):
    for i, layer in enumerate(model.backbone.layers):
        layer_calib = None if calib is None else {k: calib[k][i] for k in ("beta", "B", "C")}
        SelectiveMixer.from_mixer(layer.mixer, variant, layer_calib)
    model.config.use_cache = False
    return model


def no_decay(name):
    return name.endswith(".bias") or "norm" in name or name.endswith(("A_log", ".D", "B_static", "C_static"))


class Calibrator:
    def __init__(self, model):
        self.model = model.eval()
        self.layers = [l.mixer for l in model.backbone.layers]
        m = self.layers[0]
        self.r, self.n, self.d = m.time_step_rank, m.ssm_state_size, m.intermediate_size

    @torch.no_grad()
    def run(self, rows, batch_size=8):
        L, dev = len(self.layers), next(self.model.parameters()).device
        acc = {"dt": torch.zeros(L, self.d, dtype=torch.float64), "B": torch.zeros(L, self.n, dtype=torch.float64),
               "C": torch.zeros(L, self.n, dtype=torch.float64), "B_norm": torch.zeros(L, dtype=torch.float64),
               "C_norm": torch.zeros(L, dtype=torch.float64)}
        count = 0

        def hook(i):
            def fn(module, inp, h):
                mixer = self.layers[i]
                h = h.double().flatten(0, 1)  # (tokens, r + 2N)
                dt = h[:, : self.r] @ mixer.dt_proj.weight.double().T + mixer.dt_proj.bias.double()
                B, C = h[:, self.r : self.r + self.n], h[:, self.r + self.n :]
                acc["dt"][i] += dt.sum(0).cpu()
                acc["B"][i] += B.sum(0).cpu()
                acc["C"][i] += C.sum(0).cpu()
                acc["B_norm"][i] += B.norm(dim=-1).sum().cpu()
                acc["C_norm"][i] += C.norm(dim=-1).sum().cpu()
            return fn

        handles = [m.x_proj.register_forward_hook(hook(i)) for i, m in enumerate(self.layers)]
        try:
            for s in range(0, len(rows), batch_size):
                ids = rows[s : s + batch_size].to(dev, torch.long)
                self.model(input_ids=ids, use_cache=False)
                count += ids.numel()
        finally:
            for h in handles:
                h.remove()
        mean = {k: (v / count).float() for k, v in acc.items()}
        return {
            "beta": mean["dt"], "B": mean["B"], "C": mean["C"], "tokens": count,
            "ratio_B": mean["B"].norm(dim=-1) / mean["B_norm"], "ratio_C": mean["C"].norm(dim=-1) / mean["C_norm"],
        }

    @staticmethod
    def save(calib, path):
        torch.save(calib, path)

    @staticmethod
    def load(path):
        return torch.load(path)


class VariantChecks:
    def __init__(self, build, stock, batch, tol=1e-3, grad_rtol=1e-3):
        self.build, self.stock, self.batch, self.tol, self.grad_rtol = build, stock, batch, tol, grad_rtol

    @torch.no_grad()
    def static_invariance(self, variant):
        model = self.build(variant).eval()
        out = {}
        for i, layer in enumerate(model.backbone.layers):
            m = layer.mixer
            x1, x2 = (torch.randn(2, m.intermediate_size, 64, device=m.A_log.device) for _ in range(2))
            (dt1, B1, C1), (dt2, B2, C2) = m.ssm_params(x1), m.ssm_params(x2)
            for name, a, b, static in (("dt", dt1, dt2, m.static_dt), ("B", B1, B2, m.static_B), ("C", C1, C2, m.static_C)):
                diff = (a - b).abs().max().item()
                assert (diff == 0) == static, f"{variant} layer {i}: {name} static={static} but diff={diff}"
                out[f"{name}_diff_max"] = max(out.get(f"{name}_diff_max", 0.0), diff)
        return out

    @torch.no_grad()
    def v4_matches_stock(self):
        model = self.build("V4").eval()
        diff = (model(input_ids=self.batch, use_cache=False).logits
                - self.stock.eval()(input_ids=self.batch, use_cache=False).logits).abs().max().item()
        assert diff < self.tol, f"V4 vs stock max |Δlogits| = {diff}"
        return {"v4_vs_stock_max_abs": diff}

    @torch.no_grad()
    def train_eval_agree(self, variant):
        model = self.build(variant)
        train_logits = model.train()(input_ids=self.batch, use_cache=False).logits
        eval_logits = model.eval()(input_ids=self.batch, use_cache=False).logits
        diff = (train_logits - eval_logits).abs().max().item()
        rel = diff / eval_logits.abs().max().item()
        assert rel < 1e-4, f"{variant}: train vs eval max |Δlogits| = {diff} (rel {rel:.2e})"
        return {"train_vs_eval_max_abs": diff, "train_vs_eval_rel": rel}

    def grads_reproducible(self, variant):
        model = self.build(variant).train()
        grads = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.batch.is_cuda):
                loss = model(input_ids=self.batch, labels=self.batch, use_cache=False).loss
            loss.backward()
            grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                          if p.grad is not None and ("conv1d" in n or "x_proj" in n)})
        worst = 0.0
        for name, g in grads[0].items():
            assert torch.isfinite(g).all() and torch.isfinite(grads[1][name]).all(), f"{variant}: non-finite {name}"
            worst = max(worst, ((g - grads[1][name]).norm() / g.norm().clamp_min(1e-12)).item())
        assert worst < self.grad_rtol, f"{variant}: grads differ between identical passes (rel {worst:.2e})"
        return {"grad_rel_diff_max": worst}

    def run_checks(self):
        rows = [{"check": "V4 vs stock", "variant": "V4", **self.v4_matches_stock()}]
        for v in VARIANTS:
            rows.append({"check": "train vs eval", "variant": v, **self.train_eval_agree(v)})
        for v in ("V1", "V2", "V3"):
            rows.append({"check": "static invariance", "variant": v, **self.static_invariance(v)})
            rows.append({"check": "grad reproducibility", "variant": v, **self.grads_reproducible(v)})
        return rows
