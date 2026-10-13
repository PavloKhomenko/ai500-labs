import pandas as pd
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..part2 import wandb_log as wl


class KVCacheProbe:
    def __init__(self, model):
        self.model, self.c = model.eval(), model.config
        self.bytes_per_el = next(model.parameters()).element_size()

    def predicted(self, seq_len, batch=1):
        c = self.c
        return 2 * c.num_hidden_layers * c.num_key_value_heads * c.head_dim * seq_len * batch * self.bytes_per_el

    @torch.inference_mode()
    def measure(self, seq_len):
        ids = torch.randint(3, self.c.vocab_size, (1, seq_len), device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out = self.model.model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        del out
        torch.cuda.synchronize()
        res = {"cache_tensors": sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values)),
               "allocated_increase": torch.cuda.memory_allocated() - before,
               "peak_increase": torch.cuda.max_memory_allocated() - before}
        del cache
        return res

    def table(self, lengths, path=None):
        df = pd.DataFrame([{"seq_len": n, "predicted": self.predicted(n), **self.measure(n)} for n in lengths])
        df["allocated_over_predicted"] = df["allocated_increase"] / df["predicted"]
        for col in ["predicted", "cache_tensors", "allocated_increase", "peak_increase"]:
            df[col] = df[col] / 2**20
        df = df.rename(columns=lambda col: col if col in ("seq_len", "allocated_over_predicted") else f"{col}_MiB")
        wl.log_table("kv_cache", df, path)
        return df
