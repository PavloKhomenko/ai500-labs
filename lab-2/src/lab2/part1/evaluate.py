import math
import time
from collections import Counter
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import transformers.models.mamba.modeling_mamba as stock

from . import variants as ours

# (module, global name, counter key); a key starting with "fallback" means the slow torch path
_TARGETS = [
    (ours, "mamba_inner_fn", "fused"),
    (ours, "selective_scan_fn", "scan"),
    (ours, "causal_conv1d_fn", "conv"),
    (ours, "selective_scan_ref", "fallback_ref_scan"),
    (stock, "mamba_inner_fn", "stock_fused"),
    (stock, "mamba_selective_scan", "stock_scan"),
    (stock, "causal_conv1d_fn", "stock_conv"),
]


class KernelCounter:
    def __init__(self):
        self.counts = Counter()

    @contextmanager
    def __call__(self):
        saved = []
        for module, name, key in _TARGETS:
            fn = getattr(module, name)
            if fn is None:
                continue
            saved.append((module, name, fn))
            setattr(module, name, self._wrap(fn, key))
        try:
            yield self
        finally:
            for module, name, fn in saved:
                setattr(module, name, fn)

    def _wrap(self, fn, key):
        def counted(*args, **kwargs):
            out = fn(*args, **kwargs)
            self.counts[key] += 1
            if key.startswith("stock") and (not ours.KERNELS or out is None):
                self.counts["fallback_" + key] += 1
            return out
        return counted

    def assert_path(self, mode):
        """mode: 'fused' (training) or 'scan' (evaluation). Raises on any fallback or a missing kernel call."""
        fallback = {k: v for k, v in self.counts.items() if k.startswith("fallback")}
        assert ours.KERNELS and not fallback, f"slow path used: {dict(self.counts)}"
        keys = ("fused", "stock_fused") if mode == "fused" else ("scan", "stock_scan")
        assert sum(self.counts[k] for k in keys) > 0, f"no {mode} kernel call: {dict(self.counts)}"
        return dict(self.counts)


class LMEvaluator:
    """fp32, eval mode, use_cache=False, one document per forward (no padding).
    ponytail: batch size 1; sort-by-length batching if eval time ever matters."""

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def totals(self, model, sample):
        model.eval()
        nll, tokens = 0.0, 0
        for ids in sample["ids"]:
            ids = ids.to(self.device, torch.long)[None]
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
            nll += F.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
            tokens += ids.shape[1] - 1
        return nll, tokens, int(sample["bytes"].sum())

    def evaluate(self, model, sample, kernels=True):
        counter, t0 = KernelCounter(), time.time()
        with counter():
            nll, tokens, nbytes = self.totals(model, sample)
        out = {"loss": nll / tokens, "ppl": math.exp(nll / tokens), "bpb": nll / math.log(2) / nbytes,
               "tokens": tokens, "bytes": nbytes, "docs": len(sample["ids"]), "seconds": time.time() - t0}
        out["kernel_counts"] = counter.assert_path("scan") if kernels else dict(counter.counts)
        return out
