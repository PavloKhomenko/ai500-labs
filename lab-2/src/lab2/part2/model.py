import importlib
from collections import Counter

import torch
import transformers.models.granitemoehybrid.modeling_granitemoehybrid as gmh
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

KERNELS = ["mamba2_split_conv1d_scan_combined", "mamba2_chunk_scan", "causal_conv1d_fn"]


def load_base(cfg, dtype=torch.float16, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=dtype).to(device)
    assert model.dtype == dtype, model.dtype
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def add_lora(model, cfg, name):
    targets = cfg["targets"][name]
    assert "out_proj" not in targets
    lora = cfg["lora"]
    conf = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                      target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, conf)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    return model


def load_adapter(cfg, path, device="cuda"):
    return PeftModel.from_pretrained(load_base(cfg, device=device), path).eval()


def param_counts(model):
    return {"params_total": sum(p.numel() for p in model.parameters()),
            "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def kernel_packages():
    out = {}
    for name in ["mamba_ssm", "causal_conv1d", "triton"]:
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception as e:
            out[name] = f"missing: {type(e).__name__}"
    return out


class KernelCounter:
    def __init__(self):
        self.counts = Counter()

    def wrap(self, name, fn):
        def counted(*args, **kwargs):
            out = fn(*args, **kwargs)
            self.counts[name if out is not None else f"{name}_fallback"] += 1
            return out
        return counted

    def __enter__(self):
        self.originals = {n: getattr(gmh, n) for n in KERNELS}
        for n, fn in self.originals.items():
            setattr(gmh, n, self.wrap(n, fn))
        return self

    def __exit__(self, *exc):
        for n, fn in self.originals.items():
            setattr(gmh, n, fn)

    def fused(self):
        return self.counts["mamba2_split_conv1d_scan_combined"] > 0 and not self.counts[f"{KERNELS[0]}_fallback"]


def lora_grad_problems(model, all_nonzero):
    problems = []
    for name, p in model.named_parameters():
        if "lora_" not in name or not p.requires_grad:
            continue
        if p.grad is None:
            problems.append(f"{name}: none")
        elif not torch.isfinite(p.grad).all():
            problems.append(f"{name}: non-finite")
        elif (all_nonzero or "lora_B" in name) and p.grad.abs().max() == 0:
            problems.append(f"{name}: zero")
    return problems
