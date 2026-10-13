import json
import random
import time

import pandas as pd
import sacrebleu
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import set_seed

from ..data import normalize
from ..part2 import wandb_log as wl
from ..part2.pretrain import lr_factor


class SpivavtorData:
    def __init__(self, tok, p3, seed=42):
        self.tok, dc = tok, p3["data"]
        ds = load_dataset(p3["task"])
        pick = lambda split: [r for r in ds[split] if r["task"] == p3["task_filter"]]
        rng = random.Random(seed)
        train_pool, test_pool = pick("train"), pick("validation")
        rng.shuffle(train_pool)
        rng.shuffle(test_pool)
        seen = set()
        self.test = self.take(test_pool, dc["n_test"], seen, dc["max_len"])
        self.val = self.take(train_pool, dc["n_val"], seen, dc["max_len"])
        self.train = self.take(train_pool, dc["n_train"], seen, dc["max_len"])
        print({k: len(getattr(self, k)) for k in ("train", "val", "test")})

    def encode(self, src, tgt):
        prompt = self.tok.apply_chat_template([{"role": "user", "content": src}], add_generation_prompt=True,
                                              tokenize=False)
        p = self.tok(prompt, add_special_tokens=False).input_ids
        a = self.tok(tgt + self.tok.eos_token, add_special_tokens=False).input_ids
        return {"src": src, "tgt": tgt, "prompt_ids": p, "ids": p + a, "labels": [-100] * len(p) + a}

    def take(self, pool, n, seen, max_len):
        out = []
        while pool and len(out) < n:
            r = pool.pop()
            key = normalize(r["src"])
            if key in seen:
                continue
            seen.add(key)
            ex = self.encode(r["src"], r["tgt"])
            if len(ex["ids"]) <= max_len:
                out.append(ex)
        return out

    def collate(self, batch, left=False, key="ids"):
        pad = self.tok.pad_token_id
        n = max(len(ex[key]) for ex in batch)
        ids, labels, mask = [], [], []
        for ex in batch:
            seq, lab, k = ex[key], ex["labels"][: len(ex[key])], n - len(ex[key])
            ids.append([pad] * k + seq if left else seq + [pad] * k)
            labels.append([-100] * k + lab if left else lab + [-100] * k)
            mask.append([0] * k + [1] * len(seq) if left else [1] * len(seq) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}

    def check_masking(self):
        ex, eos = self.train[0], self.tok.eos_token_id
        n = len(ex["prompt_ids"])
        assert all(l == -100 for l in ex["labels"][:n]) and ex["labels"][n:] == ex["ids"][n:], "prompt not masked"
        assert ex["labels"][-1] == eos, "answer must end with eos"
        b = self.collate(self.train[:4])
        assert ((b["labels"] != -100) <= b["attention_mask"].bool()).all()
        print("masking ok:", repr(self.tok.decode([t for t in ex["labels"] if t != -100])))

    def check_disjoint(self):
        keys = {k: {normalize(ex["src"]) for ex in getattr(self, k)} for k in ("train", "val", "test")}
        overlaps = {"train&val": keys["train"] & keys["val"], "train&test": keys["train"] & keys["test"],
                    "val&test": keys["val"] & keys["test"]}
        assert not any(overlaps.values()), {k: len(v) for k, v in overlaps.items()}
        print("splits disjoint:", {k: len(v) for k, v in keys.items()})


class SFTTrainer:
    def __init__(self, adapter, data, p3, seed=42):
        self.adapter, self.data, self.p3, self.seed = adapter, data, p3, seed
        self.base = adapter.model.get_base_model()
        self.device = next(self.base.parameters()).device
        self.base.config.use_cache = False
        self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base.enable_input_require_grads()

    def forward(self, batch):
        """Token-mean loss on answer tokens only; lm_head runs on those positions, no full logits."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            h = self.base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            labels = batch["labels"][:, 1:]
            keep = labels != -100
            logits = self.base.lm_head(h[:, :-1][keep]).float()
        loss = F.cross_entropy(logits, labels[keep], reduction="sum")
        return loss, keep.sum(), batch["attention_mask"].reshape(-1).bool()

    def train(self, run_dir, max_steps=None):
        tc, ad = self.p3["train"], self.adapter
        n_steps = min(max_steps or tc["steps"], tc["steps"])
        set_seed(self.seed)
        order = torch.randperm(len(self.data.train), generator=torch.Generator().manual_seed(self.seed)).tolist()
        opt = torch.optim.AdamW(ad.trainable(), lr=tc["lr"], weight_decay=tc["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda st: lr_factor(st, tc["warmup_steps"], tc["steps"], tc["min_lr_ratio"]))
        scaler = torch.amp.GradScaler("cuda")
        mb, accum = tc["micro_batch"], tc["grad_accum"]
        torch.cuda.reset_peak_memory_stats()
        log_file, times = open(run_dir / "train_log.jsonl", "w"), []
        self.adapter.model.train()
        for step in range(n_steps):
            t0 = time.time()
            rec = {"loss": 0.0, "aux_per_layer": 0.0}
            counts = torch.zeros(ad.n_layers, ad.n_experts, dtype=torch.long, device=self.device) if ad.moes else None
            for micro in range(accum):
                start = (step * accum + micro) * mb
                batch = self.data.collate([self.data.train[order[j % len(order)]] for j in range(start, start + mb)])
                loss_sum, n_tok, mask = self.forward(batch)
                loss = loss_sum / n_tok
                rec["loss"] += loss.item() / accum
                if ad.moes:
                    counts += ad.step_counts(mask)
                    with torch.set_grad_enabled(ad.balance == "aux"):
                        aux = ad.aux_loss(mask)
                    rec["aux_per_layer"] += aux.item() / (ad.n_layers * accum)
                    if ad.balance == "aux":
                        loss = loss + self.p3["aux_coef"] * aux
                scaler.scale(loss / accum).backward()
            scaler.unscale_(opt)
            rec["grad_norm"] = torch.nn.utils.clip_grad_norm_(ad.trainable(), tc["grad_clip"]).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if ad.balancers:
                ad.update_bias(counts)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            rec.update(step=step + 1, lr=sched.get_last_lr()[0], sec_per_step=times[-1],
                       max_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
            if ad.moes:
                maxvio = ad.load_stats(counts)["maxvio"]
                rec["maxvio_mean"] = sum(maxvio) / len(maxvio)
                wl.log_per_layer("train/maxvio", maxvio, step=step + 1)
            wl.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=step + 1)
            log_file.write(json.dumps(rec) + "\n")
            if step % 10 == 0 or step == n_steps - 1:
                print(json.dumps({k: float(f"{v:.4g}") if isinstance(v, float) else v for k, v in rec.items()}))
        log_file.close()
        self.adapter.model.eval()
        warm = times[min(5, len(times) - 1):]  # first steps include compilation / allocator warm-up
        return {"steps": n_steps, "sec_per_step": sum(warm) / len(warm),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}

    @torch.no_grad()
    def val_loss(self):
        ad, bs = self.adapter, self.p3["eval"]["batch"]
        self.adapter.model.eval()
        total, n = 0.0, 0
        counts = torch.zeros(ad.n_layers, ad.n_experts, dtype=torch.long, device=self.device) if ad.moes else None
        for i in range(0, len(self.data.val), bs):
            loss_sum, n_tok, mask = self.forward(self.data.collate(self.data.val[i : i + bs]))
            total, n = total + loss_sum.item(), n + n_tok.item()
            if ad.moes:
                counts += ad.step_counts(mask)
        out = {"val_loss": total / n}
        if ad.moes:
            out["load"] = ad.load_stats(counts.cpu())
        return out

    @torch.no_grad()
    def generate(self, examples=None):
        examples = examples or self.data.test
        tok, ec = self.data.tok, self.p3["eval"]
        self.adapter.model.eval()
        preds = []
        for i in range(0, len(examples), ec["batch"]):
            chunk = examples[i : i + ec["batch"]]
            b = self.data.collate(chunk, left=True, key="prompt_ids")
            n_src = max(len(tok(ex["src"], add_special_tokens=False).input_ids) for ex in chunk)
            with torch.autocast("cuda", dtype=torch.float16):
                out = self.adapter.model.generate(
                    input_ids=b["input_ids"].to(self.device), attention_mask=b["attention_mask"].to(self.device),
                    max_new_tokens=int(ec["max_new_tokens_ratio"] * n_src) + 8, do_sample=False, use_cache=True,
                    pad_token_id=tok.pad_token_id)
            preds += tok.batch_decode(out[:, b["input_ids"].shape[1] :], skip_special_tokens=True)
        refs = [ex["tgt"] for ex in examples]
        preds = [p.strip() for p in preds]
        return {"chrf": sacrebleu.corpus_chrf(preds, [refs]).score,
                "exact_match": sum(normalize(p) == normalize(r) for p, r in zip(preds, refs)) / len(refs),
                "predictions": [{"src": ex["src"], "ref": r, "pred": p} for ex, r, p in zip(examples, refs, preds)]}

    def evaluate(self, run_dir, extra=None):
        gen = self.generate()
        preds = gen.pop("predictions")
        metrics = {**self.val_loss(), **gen, "trainable": self.adapter.n_trainable(),
                   "active": self.adapter.n_active(), **(extra or {})}
        (run_dir / "predictions.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds))
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        wl.log_summary({k: v for k, v in metrics.items() if k != "load"})
        if "load" in metrics:
            wl.log_per_layer("val/maxvio", metrics["load"]["maxvio"])
        print({k: v for k, v in metrics.items() if k != "load"})
        return metrics


class Part3Report:
    def __init__(self, part3_dir, runs):
        self.dir, self.runs = part3_dir, runs
        self.metrics = {r: json.loads((part3_dir / r / "metrics.json").read_text()) for r in runs}

    def table(self):
        rows = [{"run": r, "chrF": m["chrf"], "EM": m["exact_match"], "val_loss": m["val_loss"],
                 "trainable_M": m["trainable"] / 1e6, "active_M": m["active"] / 1e6,
                 "sec_per_step": m.get("sec_per_step"), "peak_mem_gb": m.get("peak_mem_gb"),
                 "maxvio_mean": sum(m["load"]["maxvio"]) / len(m["load"]["maxvio"]) if "load" in m else None}
                for r, m in self.metrics.items()]
        df = pd.DataFrame(rows)
        wl.log_table("part3/comparison", df, self.dir / "figures" / "comparison")
        return df

    def routing_figures(self):
        moe = {r: m["load"] for r, m in self.metrics.items() if "load" in m}
        fig_dir = self.dir / "figures"
        for r, load in moe.items():
            wl.log_figure(f"part3/load_{r}", wl.heatmap_fig(load["counts"], f"{r}: val expert load per layer"),
                          fig_dir / f"load_{r}.png")
        wl.log_figure("part3/maxvio", wl.layer_lines_fig({r: l["maxvio"] for r, l in moe.items()}, "MaxVio",
                                                         "val MaxVio per layer", hline=0, hline_label="balanced"),
                      fig_dir / "maxvio.png")

    def loss_curves(self):
        logs = {r: [json.loads(l) for l in (self.dir / r / "train_log.jsonl").read_text().splitlines()]
                for r in self.runs if (self.dir / r / "train_log.jsonl").exists()}
        series = {r: ([x["step"] for x in log], [x["loss"] for x in log]) for r, log in logs.items()}
        wl.log_figure("part3/train_loss", wl.curves_fig(series, "train loss (answer tokens)", "SFT train loss"),
                      self.dir / "figures" / "train_loss.png")
