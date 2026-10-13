import math

import pandas as pd
import torch
import copy
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

METHODS = ["none", "linear", "ntk", "dynamic_ntk", "yarn"]


def base_inv_freq(base, dim):
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))


class RoPEScaling:
    def __init__(self, base, dim, l_train, scale):
        self.base, self.dim, self.l_train, self.scale = base, dim, l_train, scale

    @classmethod
    def from_config(cls, config, scale):
        params = config.rope_parameters
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * (params.get("partial_rotary_factor") or 1.0))
        return cls(params["rope_theta"], dim, config.max_position_embeddings, scale)

    def yarn_gamma(self, beta_fast=32, beta_slow=1):
        index = lambda r: self.dim * math.log(self.l_train / (2 * math.pi * r)) / (2 * math.log(self.base))
        low, high = max(math.floor(index(beta_fast)), 0), min(math.ceil(index(beta_slow)), self.dim - 1)
        ramp = ((torch.arange(self.dim // 2, dtype=torch.float) - low) / max(high - low, 1e-3)).clamp(0, 1)
        return 1 - ramp

    def inv_freq(self, method, seq_len=None):
        b, d, s, L = self.base, self.dim, self.scale, self.l_train
        theta = base_inv_freq(b, d)
        if method == "none" or s == 1:
            return theta, 1.0
        if method == "linear":
            return theta / s, 1.0
        if method == "ntk":
            return base_inv_freq(b * s ** (d / (d - 2)), d), 1.0
        if method == "dynamic_ntk":
            if seq_len is None or seq_len <= L:
                return theta, 1.0
            return base_inv_freq(b * (s * seq_len / L - (s - 1)) ** (d / (d - 2)), d), 1.0
        if method == "yarn":
            gamma = self.yarn_gamma()
            return (1 - gamma) * theta / s + gamma * theta, 0.1 * math.log(s) + 1.0
        raise ValueError(f"unknown method {method}")

    @torch.no_grad()
    def apply(self, model, method, seq_len=None, verbose=True):
        rot = model.model.rotary_emb
        assert rot.rope_type == "default", "load the model without rope_scaling: we set inv_freq ourselves"
        inv_freq, scaling = self.inv_freq(method, seq_len)
        rot.inv_freq.copy_(inv_freq.to(rot.inv_freq.device))
        rot.attention_scaling = scaling
        if verbose:
            f = rot.inv_freq
            print(f"rope={method} s={self.scale} seq_len={seq_len}: inv_freq[:3]={f[:3].tolist()} "
                  f"inv_freq[-3:]={f[-3:].tolist()} attention_scaling={rot.attention_scaling:.4f}")

    def compare_with_hf(self, config):
        rows = []
        for method, hf_type, seq_len in [("linear", "linear", None), ("yarn", "yarn", None),
                                         *[("dynamic_ntk", "dynamic", k * self.l_train) for k in (1, 2, 4)]]:
            c = copy.deepcopy(config)
            c.rope_parameters = {"rope_type": hf_type, "rope_theta": self.base, "factor": float(self.scale),
                                 "original_max_position_embeddings": self.l_train}
            hf_freq, hf_scaling = ROPE_INIT_FUNCTIONS[hf_type](c, "cpu", seq_len=seq_len)
            freq, scaling = self.inv_freq(method, seq_len)
            rows.append({"method": method, "seq_len": seq_len, "max_abs_diff": (freq - hf_freq).abs().max().item(),
                         "scaling_ours": scaling, "scaling_hf": hf_scaling})
        theta = base_inv_freq(self.base, self.dim)
        ratio = self.inv_freq("ntk")[0] / theta  # NTK has no HF type: check c_0 = 1, c_{d/2-1} = 1/s instead
        rows.append({"method": "ntk", "max_abs_diff": max(abs(ratio[0].item() - 1), abs(ratio[-1].item() - 1 / self.scale))})
        df = pd.DataFrame(rows)
        assert (df["max_abs_diff"] < 1e-5).all(), df
        return df

    def describe(self):
        wavelength = 2 * math.pi / base_inv_freq(self.base, self.dim)
        long = wavelength > self.l_train
        print(f"base={self.base} rotary_dim={self.dim} L_train={self.l_train} s={self.scale}")
        print(f"pairs with lambda_i > L_train: {long.sum().item()} / {self.dim // 2} (first i={long.nonzero()[0].item()}), "
              f"lambda_i > s*L_train: {(wavelength > self.scale * self.l_train).sum().item()}")
        for m in METHODS:
            f, a = self.inv_freq(m, seq_len=self.scale * self.l_train)
            print(f"{m:12s} scaling={a:.4f} inv_freq[0,16,32,48,63]={[f'{f[i].item():.3e}' for i in (0, 16, 32, 48, 63)]}")
