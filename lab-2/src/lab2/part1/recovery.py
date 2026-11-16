import math
import time

import torch
from transformers import set_seed

from .. import wandb_log as wl
from .evaluate import KernelCounter
from .variants import no_decay


def lr_factor(step, warmup, total, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def param_counts(model):
    return {"params_total": sum(p.numel() for p in model.parameters()),
            "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def row_order(n_rows, seed):
    return torch.randperm(n_rows, generator=torch.Generator().manual_seed(seed))


class RecoveryTrainer:
    def __init__(self, cfg, evaluator, val_sample):
        self.tc, self.seq_len = cfg["part1"]["train"], cfg["part1"]["seq_len"]
        self.log_ckpt = cfg["wandb"]["log_checkpoints"]
        self.evaluator, self.val = evaluator, val_sample

    def optimizer(self, model):
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        groups = [{"params": [p for n, p in params if not no_decay(n)], "weight_decay": self.tc["weight_decay"]},
                  {"params": [p for n, p in params if no_decay(n)], "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=self.tc["lr"], betas=tuple(self.tc["betas"]), fused=True)

    def val_loss(self, model):
        nll, tokens, _ = self.evaluator.totals(model, self.val)
        model.train()
        return nll / tokens

    def train(self, model, rows, seed, run_dir, max_steps=None):
        tc, device = self.tc, next(model.parameters()).device
        mb, accum = tc["micro_batch"], tc["grad_accum"]
        tokens_per_step = mb * accum * self.seq_len
        total = tc["budget_tokens"] // tokens_per_step
        steps = total if max_steps is None else min(max_steps, total)
        warmup = max(1, int(tc["warmup_frac"] * total))
        order = row_order(len(rows), seed)
        assert len(order) >= total * mb * accum, "train stream shorter than the token budget"

        set_seed(seed)
        opt, scaler = self.optimizer(model), torch.amp.GradScaler("cuda")
        wl.define_token_axis("val/loss", "train/*")
        history = [{"tokens": 0, "val_loss": self.val_loss(model)}]
        wl.log({"tokens": 0, "val/loss": history[0]["val_loss"]})

        counter, t_start = KernelCounter(), time.time()
        model.train()
        with counter():
            for step in range(steps):
                t0 = time.time()
                for g in opt.param_groups:
                    g["lr"] = tc["lr"] * lr_factor(step, warmup, total, tc["min_lr_ratio"])
                loss_sum = 0.0
                for k in range(accum):
                    idx = order[(step * accum + k) * mb : (step * accum + k + 1) * mb]
                    ids = rows[idx].to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = model(input_ids=ids, labels=ids, use_cache=False).loss / accum
                    scaler.scale(loss).backward()
                    loss_sum += loss.item()
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if step == 0:
                    counter.assert_path("fused")  # fail fast on a silent fallback
                tokens = (step + 1) * tokens_per_step
                wl.log({"tokens": tokens, "train/loss": loss_sum, "train/lr": opt.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm, "train/scale": scaler.get_scale(),
                        "train/step_seconds": time.time() - t0})
                if (step + 1) % tc["val_every"] == 0 or step + 1 == steps:
                    history.append({"tokens": tokens, "val_loss": self.val_loss(model)})
                    wl.log({"tokens": tokens, "val/loss": history[-1]["val_loss"]})

        for layer in model.backbone.layers:
            if getattr(layer.mixer, "static_dt", False):
                assert not layer.mixer.dt_proj.weight.any(), "static Δ weight changed during training"
        wall = time.time() - t_start
        summary = {"steps": steps, "tokens": steps * tokens_per_step, "wall_seconds": wall,
                   "step_seconds": wall / max(1, steps), "kernel_counts": counter.assert_path("fused"),
                   **param_counts(model)}
        torch.save(model.state_dict(), run_dir / "final.pt")
        if self.log_ckpt:
            wl.log_artifact(run_dir / "final.pt", run_dir.name, "model")
        wl.log_summary(summary)
        return history, summary
