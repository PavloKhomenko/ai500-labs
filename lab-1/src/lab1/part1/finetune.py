import json
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import set_seed

from ..part2 import wandb_log as wl
from ..part2.pretrain import lr_factor
from .rope import RoPEScaling


def is_extra_trainable(name):
    return "embed_tokens" in name or "norm" in name


class LongLoRATrainer:
    def __init__(self, model, p1, seed=42):
        self.ft, self.chunk, self.seed = p1["finetune"], p1["lm_head_chunk"], seed
        RoPEScaling.from_config(model.config, p1["scale"]).apply(model, self.ft["method"])
        self.model = self.add_lora(model, self.ft)
        self.base = self.model.get_base_model()
        self.base.config.use_cache = False
        self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base.enable_input_require_grads()
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.lora = [p for n, p in named if not is_extra_trainable(n)]
        self.extra = [p for n, p in named if is_extra_trainable(n)]
        print(f"trainable: LoRA {sum(p.numel() for p in self.lora) / 1e6:.2f}M, "
              f"embeddings + norms {sum(p.numel() for p in self.extra) / 1e6:.2f}M")

    @staticmethod
    def add_lora(model, ft):
        model = get_peft_model(model, LoraConfig(r=ft["lora_rank"], lora_alpha=ft["lora_alpha"],
                                                 target_modules=ft["lora_targets"], lora_dropout=0.0, bias="none"))
        for name, p in model.named_parameters():
            p.requires_grad = p.requires_grad or is_extra_trainable(name)
            if p.requires_grad:
                p.data = p.data.float()
        return model

    @classmethod
    def load_finetuned(cls, model, path, p1):
        RoPEScaling.from_config(model.config, p1["scale"]).apply(model, p1["finetune"]["method"])
        model = cls.add_lora(model, p1["finetune"])
        result = model.load_state_dict(torch.load(path), strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        return model.merge_and_unload().half().eval()

    def lm_loss(self, ids):
        h = self.base.model(input_ids=ids[:, :-1]).last_hidden_state[0]
        labels = ids[0, 1:]
        chunk_loss = lambda h_, y_: F.cross_entropy(self.base.lm_head(h_).float(), y_, reduction="sum")
        total = sum(checkpoint(chunk_loss, h[i : i + self.chunk], labels[i : i + self.chunk], use_reentrant=False)
                    for i in range(0, len(labels), self.chunk))
        return total / len(labels)

    def train(self, blocks, log_path=None, max_steps=None):
        ft, device = self.ft, next(self.model.parameters()).device
        n_steps = min(max_steps or ft["steps"], ft["steps"])
        set_seed(self.seed)
        order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(self.seed))
        opt = torch.optim.AdamW([{"params": self.lora, "lr": ft["lr"]}, {"params": self.extra, "lr": ft["lr_emb_norm"]}],
                                weight_decay=ft["weight_decay"])
        base_lrs = [g["lr"] for g in opt.param_groups]
        scaler = torch.amp.GradScaler("cuda")
        params = self.lora + self.extra
        self.model.train()
        log_file = open(log_path, "a") if log_path else None
        for step in range(n_steps):
            t0 = time.time()
            f = lr_factor(step, ft["warmup_steps"], ft["steps"], ft["min_lr_ratio"])
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            loss_sum = 0.0
            for micro in range(ft["grad_accum"]):
                ids = blocks[order[(step * ft["grad_accum"] + micro) % len(blocks)]][None].long().to(device)
                # the kernel choice must also hold for the checkpoint recompute inside backward()
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = self.lm_loss(ids) / ft["grad_accum"]
                    scaler.scale(loss).backward()
                loss_sum += loss.item()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, ft["grad_clip"]).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            rec = {"step": step, "loss": loss_sum, "grad_norm": grad_norm, "lr": opt.param_groups[0]["lr"],
                   "sec_per_step": time.time() - t0, "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            wl.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=step)
            if log_file:
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
            if step % 10 == 0 or step == n_steps - 1:
                print(json.dumps(rec))
        if log_file:
            log_file.close()
        self.model.eval()

    def save(self, path):
        torch.save({n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}, path)
