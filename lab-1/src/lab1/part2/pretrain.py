"""Continued pretraining loop, one run per (strategy, seed).

    python -m lab1.part2.pretrain --strategy aux_1e-2 --seed 42 [--max_steps 5]

Trainable: attention (q, k, v, o) + routers, in fp32. Frozen: experts, embeddings, norms (fp16).
Every strategy uses the same data order (fixed by the seed), token budget, LR and schedule.
Outputs in <output_dir>/part2/runs/<strategy>-s<seed>/: train_log.jsonl, final.pt (last.pt while running).
"""

import argparse
import json
import math
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from ..common import load_config, out_dir
from ..data import block_batches
from . import wandb_log as wl
from .balancing import BiasBalancer, aux_loss, stats_from_counts, z_loss
from .router_stats import evaluate, patch_routers


def load_model(name, device, aux_coef=0.0, output_router_logits=False):
    """fp16, since a T4 has no fast bf16. The HF aux coefficient only works when passed here:
    changing model.config after loading has no effect. For training we keep it at 0 and use our own loss."""
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16, router_aux_loss_coef=aux_coef, output_router_logits=output_router_logits
    )
    return model.to(device)


def is_saved(name):
    """Parameters that can change in any run: attention projections and routers."""
    return ".self_attn." in name or ".mlp.gate." in name


def prepare_for_training(model, train_router):
    """Freeze everything except attention (+ router); trainable parameters go to fp32."""
    for name, p in model.named_parameters():
        p.requires_grad = ".self_attn." in name or (train_router and ".mlp.gate." in name)
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return [p for p in model.parameters() if p.requires_grad]


def trainable_state(model):
    return {n: p.detach().float().cpu() for n, p in model.named_parameters() if is_saved(n)}


@torch.no_grad()
def load_trainable(model, state):
    params = dict(model.named_parameters())
    for name, tensor in state.items():
        params[name].copy_(tensor)


def lr_factor(step, warmup, total, min_ratio):
    """Linear warmup, then cosine decay to min_ratio."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def train(model, data, cfg, strategy, seed, run_dir, max_steps=None, save=True, validate=True):
    p2, tc = cfg["part2"], cfg["part2"]["train"]
    s = p2["strategies"][strategy]
    device = next(model.parameters()).device
    n_layers, n_experts = model.config.num_hidden_layers, model.config.num_local_experts
    top_k = model.config.num_experts_per_tok
    mb, accum, seq_len = tc["micro_batch"], tc["grad_accum"], data["train"].shape[1]
    tokens_per_step = mb * accum * seq_len
    n_steps = tc["token_budget"] // tokens_per_step
    last_step = n_steps if max_steps is None else min(max_steps, n_steps)

    set_seed(seed)
    balancers = (
        [BiasBalancer(n_experts, tc["bias_update_rate"], device) for _ in range(n_layers)] if s["bias"] else None
    )
    routers = patch_routers(model, balancers)
    params = prepare_for_training(model, s["train_router"])
    opt = torch.optim.AdamW(params, lr=tc["lr"], betas=tuple(tc["betas"]), weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: lr_factor(st, tc["warmup_steps"], n_steps, tc["min_lr_ratio"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Data order depends only on the seed, so every strategy sees the same blocks in the same order.
    n_blocks = n_steps * mb * accum
    assert n_blocks <= len(data["train"]), f"budget needs {n_blocks} blocks, train pool has {len(data['train'])}"
    order = torch.randperm(len(data["train"]), generator=torch.Generator().manual_seed(seed))
    blocks = data["train"][order[:n_blocks]]

    log_path, last_path = run_dir / "train_log.jsonl", run_dir / "last.pt"
    start = 0
    if save and last_path.exists():
        ckpt = torch.load(last_path, weights_only=False)
        load_trainable(model, ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        if balancers:
            for b, bias in zip(balancers, ckpt["bias"]):
                b.bias.copy_(bias)
        start = ckpt["step"]
        kept = [line for line in log_path.read_text().splitlines() if json.loads(line)["step"] <= start]
        log_path.write_text("".join(line + "\n" for line in kept))
        print(f"resumed from step {start}")
    elif save:
        log_path.write_text("")

    print(f"[{strategy} s{seed}] {n_steps} steps x {tokens_per_step} tokens = {n_steps * tokens_per_step:,} tokens")
    if validate and start == 0:
        val_ppl = evaluate(model, routers, block_batches(data["val"], p2["eval"]["block_batch"]))["ppl"]
        if save:
            wl.log({"val/ppl": val_ppl}, step=0)
        print(f"step 0 val ppl {val_ppl:.3f}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start, speeds = time.time(), []
    for step in range(start, last_step):
        model.train()
        t0 = time.time()
        counts = torch.zeros(n_layers, n_experts, dtype=torch.long, device=device)
        rec = {"loss_lm": 0.0, "aux_per_layer": 0.0, "z_per_layer": 0.0, "lse_mean": 0.0}
        for micro in range(accum):
            i = (step * accum + micro) * mb
            ids = blocks[i : i + mb].long().to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(input_ids=ids, labels=ids, output_router_logits=False)
            assert out.aux_loss is None, "HF must not add its own aux loss"
            with torch.no_grad():  # count before backward (checkpoint recompute re-runs the routers)
                counts += torch.stack([torch.bincount(r.last_idx.reshape(-1), minlength=n_experts) for r in routers])
                rec["lse_mean"] += sum(
                    torch.logsumexp(r.last_logits.float(), -1).mean().item() for r in routers
                ) / (n_layers * accum)
            # Our losses, summed over layers (each router gets the paper-form gradient).
            # f_i comes from the step's running counts (all micro-batches so far), not just this
            # micro-batch: global-batch balancing as in EuroMoE pretraining (Qiu et al., 2025).
            with torch.set_grad_enabled(s["aux"] > 0):
                aux = sum(aux_loss(r.last_logits, top_k, n_experts, counts=c) for r, c in zip(routers, counts))
            with torch.set_grad_enabled(s["z"] > 0):
                z = sum(z_loss(r.last_logits) for r in routers)
            if step == start and micro == 0:
                assert aux.item() > 0, "aux loss is zero: router logits are not being captured"
            loss = out.loss
            if s["aux"] > 0:
                loss = loss + s["aux"] * aux
            if s["z"] > 0:
                loss = loss + s["z"] * z
            scaler.scale(loss / accum).backward()
            rec["loss_lm"] += out.loss.item() / accum
            rec["aux_per_layer"] += aux.item() / (n_layers * accum)
            rec["z_per_layer"] += z.item() / (n_layers * accum)

        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"]).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        lr = sched.get_last_lr()[0]
        sched.step()
        if balancers:  # sign-rule update after every optimizer step, on this step's counts
            for b, c in zip(balancers, counts):
                b.update(c)

        st = stats_from_counts(counts)
        speed = tokens_per_step / (time.time() - t0)
        speeds.append(speed)
        rec.update({
            "step": step + 1,
            "tokens": (step + 1) * tokens_per_step,
            "lr": lr,
            "grad_norm": grad_norm,
            "grad_scale": scaler.get_scale(),
            "tok_per_s": speed,
            "maxvio_mean": st["maxvio"].mean().item(),
            "maxvio_max": st["maxvio"].max().item(),
            "entropy_mean": st["entropy"].mean().item(),
            "maxvio_layers": st["maxvio"].tolist(),
        })
        if balancers:
            bias = torch.stack([b.bias for b in balancers])
            rec.update({"bias_abs_mean": bias.abs().mean().item(), "bias_min": bias.min().item(), "bias_max": bias.max().item()})
        if validate and ((step + 1) % tc["val_every"] == 0 or step + 1 == last_step):
            rec["val_ppl"] = evaluate(model, routers, block_batches(data["val"], p2["eval"]["block_batch"]))["ppl"]

        if save:  # save=False is the Stage 1 throughput test: no files, no W&B
            wl.log({f"train/{k}": v for k, v in rec.items() if k not in ("step", "maxvio_layers", "val_ppl")}, step=step + 1)
            wl.log_per_layer("train_maxvio", rec["maxvio_layers"], step=step + 1)
            if "val_ppl" in rec:
                wl.log({"val/ppl": rec["val_ppl"]}, step=step + 1)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        if (step + 1) % 10 == 0 or step == start:
            print(
                f"step {step + 1}/{n_steps} lm {rec['loss_lm']:.4f} aux/L {rec['aux_per_layer']:.4f} "
                f"maxvio {rec['maxvio_mean']:.3f} {speed:,.0f} tok/s"
                + (f" val_ppl {rec['val_ppl']:.3f}" if "val_ppl" in rec else "")
            )
        if save and (step + 1) % tc["ckpt_every"] == 0 and step + 1 < n_steps:
            torch.save({
                "model": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "step": step + 1,
                "bias": torch.stack([b.bias for b in balancers]) if balancers else None,
            }, last_path)

    summary = {
        "steps": last_step,
        "tokens": last_step * tokens_per_step,
        "minutes": (time.time() - t_start) / 60,
        "tok_per_s": sum(speeds[1:]) / max(1, len(speeds) - 1),  # skip the first (warmup) step
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
    }
    if save and last_step == n_steps:
        torch.save({
            "model": trainable_state(model),
            "bias": torch.stack([b.bias for b in balancers]).cpu() if balancers else None,
            "strategy": strategy, "seed": seed, "summary": summary, "config": cfg,
        }, run_dir / "final.pt")
        last_path.unlink(missing_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=None, help="smoke test: stop early, no final.pt")
    args = parser.parse_args()

    cfg = load_config()
    p2 = cfg["part2"]
    assert args.strategy in p2["strategies"], f"unknown strategy {args.strategy}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(out_dir(cfg, "part2", "data") / "data.pt")  # built by stage1_base
    name = f"{args.strategy}-s{args.seed}" + ("-smoke" if args.max_steps else "")
    run_dir = out_dir(cfg, "part2", "runs", name)

    wl.init_run(cfg, name, job_type="train", config={
        "strategy": args.strategy, "seed": args.seed, **p2["strategies"][args.strategy], **p2["train"],
        "model": p2["model"], "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    })
    model = load_model(p2["model"], device)
    summary = train(model, data, cfg, args.strategy, args.seed, run_dir, max_steps=args.max_steps)
    wl.log_summary(summary)
    print(json.dumps(summary, indent=2))
    if cfg["wandb"]["log_checkpoints"] and (run_dir / "final.pt").exists():
        wl.log_artifact(run_dir / "final.pt", f"ckpt-{name}", "model")
    wl.finish()


if __name__ == "__main__":
    main()
