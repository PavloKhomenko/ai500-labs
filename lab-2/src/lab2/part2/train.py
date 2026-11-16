import argparse
import csv
import json
import math
import os
import shutil
import statistics
import time
from functools import partial

import accelerate
import peft
import torch
import transformers
import wandb
import yaml
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from ..common import load_config, out_dir
from .data import collate, decode_mask, load_tokenizer, prepare
from .evaluate import stop_rate, val_metrics
from .model import KernelCounter, add_lora, kernel_packages, load_base, lora_grad_problems, param_counts

LOG_FIELDS = ["step", "tokens", "train_loss", "lr", "grad_norm", "scale", "skipped", "sec_per_step", "tok_per_sec",
              "peak_mem_gb", "val_loss", "val_ppl", "val_acc", "eot_top1"]


class SFTTrainer(Trainer):
    def __init__(self, *args, batches, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = batches

    def get_train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_sampler=self.batches, collate_fn=self.data_collator,
                            num_workers=self.args.dataloader_num_workers)
        return self.accelerator.prepare(loader)


class SFTCallback(TrainerCallback):
    def __init__(self, data, tok, cfg, path, kernels, timing):
        self.data, self.tok, self.t, self.kernels, self.timing = data, tok, cfg["train"], kernels, timing
        accum, items = self.t["grad_accum"], data["train"]
        labeled = [sum(sum(l != -100 for l in items[i]["labels"]) for i in b) for b in data["train_batches"]]
        padded = [len(b) * max(len(items[i]["input_ids"]) for i in b) for b in data["train_batches"]]
        self.step_labeled = [sum(labeled[i:i + accum]) for i in range(0, len(labeled), accum)]
        self.step_padded = [sum(padded[i:i + accum]) for i in range(0, len(padded), accum)]
        self.rows, self.grad_checks, self.tokens, self.pending = [], [], 0, None
        self.summary = {}
        self.csv = None if timing else open(path / "logs.csv", "w", newline="")
        self.writer = None if timing else csv.DictWriter(self.csv, LOG_FIELDS, extrasaction="ignore")
        if self.writer:
            self.writer.writeheader()

    def evaluate(self, model):
        return val_metrics(model, self.data["val"], self.data["val_batches"], self.tok.pad_token_id)

    def record(self, row):
        self.rows.append(row)
        wandb.log(row, step=row["step"])
        if self.writer:
            self.writer.writerow(row)
            self.csv.flush()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self.timing:
            self.summary["initial_val"] = self.evaluate(model)
            rate, examples = stop_rate(model, self.tok, self.data["val"], self.t["stop_prompts"], self.t["max_new_tokens"])
            self.summary["initial_stop_rate"], self.summary["initial_examples"] = rate, examples
            self.record({"step": 0, "tokens": 0, **self.summary["initial_val"], "stop_rate": rate})

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.perf_counter()

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if len(self.grad_checks) >= 2:
            return
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads):
            return
        b_moved = any(p.abs().max() > 0 for n, p in model.named_parameters() if "lora_B" in n)
        if len(self.grad_checks) == 1 and not b_moved:
            return
        problems = lora_grad_problems(model, all_nonzero=b_moved)
        self.grad_checks.append({"step": state.global_step + 1, "n_problems": len(problems), "problems": problems[:20]})
        if problems:
            raise RuntimeError(f"LoRA gradient check failed: {problems[:5]}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        dt, step = time.perf_counter() - self.t0, state.global_step
        acc = self.trainer.accelerator
        if step == 1 and torch.cuda.is_available() and not self.kernels.fused():
            raise RuntimeError(f"fused Mamba-2 kernel not used: {dict(self.kernels.counts)}")
        self.tokens += self.step_labeled[step - 1]
        self.pending = {
            "step": step, "tokens": self.tokens, "sec_per_step": dt, "tok_per_sec": self.step_padded[step - 1] / dt,
            "scale": acc.scaler.get_scale() if acc.scaler is not None else None,
            "skipped": int(acc.optimizer_step_was_skipped),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        }
        if step == 3 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if not self.timing and step % self.t["val_every"] == 0:
            self.pending |= self.evaluate(model)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pending is None or "loss" not in logs:
            return
        self.record(self.pending | {"train_loss": logs["loss"], "lr": logs.get("learning_rate"),
                                    "grad_norm": logs.get("grad_norm")})
        self.pending = None

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.timing:
            return
        last = self.rows[-1]
        if "val_loss" not in last:
            last = last | self.evaluate(model)
            self.record({"step": last["step"], **{k: last[k] for k in ["val_loss", "val_ppl", "val_acc", "eot_top1"]}})
        rate, examples = stop_rate(model, self.tok, self.data["val"], self.t["stop_prompts"], self.t["max_new_tokens"])
        self.summary |= {"final_val": {k: last[k] for k in ["val_loss", "val_ppl", "val_acc", "eot_top1"]},
                         "final_stop_rate": rate, "final_examples": examples}
        wandb.log({"stop_rate": rate}, step=last["step"])
        self.csv.close()

    def timing_stats(self):
        rows = [r for r in self.rows if "sec_per_step" in r][3:]
        sec = statistics.median(r["sec_per_step"] for r in rows)
        return {"sec_per_step": sec, "tok_per_sec": statistics.median(r["tok_per_sec"] for r in rows),
                "peak_mem_gb": max(r["peak_mem_gb"] for r in rows)}


def versions():
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "peft": peft.__version__, "accelerate": accelerate.__version__, **kernel_packages()}


def wandb_mode():
    return os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "disabled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", choices=["a", "b"], default="a")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--prepare_only", action="store_true")
    args = ap.parse_args()

    full = load_config()
    cfg, seed, t = full["part2"], full["seed"], full["part2"]["train"]
    root = out_dir(full, "part2")
    tok = load_tokenizer(cfg)
    data = prepare(cfg, tok, root / "data", seed)
    n_steps = math.ceil(len(data["train_batches"]) / t["grad_accum"]) * t["epochs"]
    data_info = {"stats": data["stats"], "n_train": len(data["train"]), "n_val": len(data["val"]),
                 "train_tokens": sum(len(x["input_ids"]) for x in data["train"]),
                 "train_labeled_tokens": sum(sum(l != -100 for l in x["labels"]) for x in data["train"]),
                 "train_micro_batches": len(data["train_batches"]), "optimizer_steps": n_steps}
    print(json.dumps(data_info, indent=2))
    if args.prepare_only:
        return

    timing = args.max_steps > 0
    name = f"sft_{args.cfg}"
    path = out_dir(full, "part2", name)
    set_seed(seed)
    base = load_base(cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    if t["grad_checkpointing"]:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    model = add_lora(base, cfg, args.cfg)
    model.print_trainable_parameters()

    run_config = {"part2": cfg, "target": args.cfg, "seed": seed, "versions": versions(), "data": data_info}
    wandb.init(project=cfg["wandb_project"], entity=full["wandb"]["entity"], name=name + ("_timing" if timing else ""),
               group=name, job_type="timing" if timing else "sft", config=run_config, mode=wandb_mode())

    targs = TrainingArguments(
        output_dir=str(root / f"{name}_trainer"), gradient_accumulation_steps=t["grad_accum"], learning_rate=t["lr"],
        lr_scheduler_type="cosine", warmup_steps=t["warmup_ratio"], weight_decay=0.0, num_train_epochs=t["epochs"],
        max_steps=args.max_steps, max_grad_norm=1.0, fp16=torch.cuda.is_available(), logging_steps=1,
        save_strategy="no", eval_strategy="no", report_to="none", seed=seed, data_seed=seed,
        remove_unused_columns=False, dataloader_num_workers=0,
    )
    with KernelCounter() as kernels:
        callback = SFTCallback(data, tok, cfg, path, kernels, timing)
        trainer = SFTTrainer(model=model, args=targs, train_dataset=data["train"], batches=data["train_batches"],
                             data_collator=partial(collate, pad_id=tok.pad_token_id), callbacks=[callback])
        callback.trainer = trainer
        start = time.time()
        trainer.train()
        train_time = time.time() - start

    stats = callback.timing_stats()
    summary = {"name": name, "target_modules": cfg["targets"][args.cfg], **param_counts(model), **stats,
               "kernel_counts": dict(kernels.counts), "fused_kernel_path": kernels.fused(), "versions": versions(),
               "train_time_h": train_time / 3600, "optimizer_steps": n_steps, "data": data_info}
    if timing:
        summary["projected_train_h"] = stats["sec_per_step"] * n_steps / 3600
        print(json.dumps(summary, indent=2))
        wandb.run.summary.update(summary)
        wandb.finish()
        return

    example = next(x for x in data["train"] if x["messages"][0]["role"] != "system" and len(x["messages"]) >= 4)
    summary |= {"decoded_label_mask": decode_mask(tok, example), "grad_checks": callback.grad_checks, **callback.summary}
    model.save_pretrained(path)
    tok.save_pretrained(path)
    shutil.copy(root / "data" / "batches.json", path / "batches.json")
    (path / "train_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    (path / "summary.json").write_text(json.dumps(summary, indent=2))
    wandb.run.summary.update({k: v for k, v in summary.items() if not isinstance(v, (list, dict))})
    artifact = wandb.Artifact(name, type="model")
    artifact.add_dir(str(path))
    wandb.log_artifact(artifact)
    wandb.finish()
    print(json.dumps({k: summary[k] for k in ["params_trainable", "sec_per_step", "peak_mem_gb", "final_val",
                                              "initial_stop_rate", "final_stop_rate"]}, indent=2))


if __name__ == "__main__":
    main()
