"""Stage 3: everything after training.

    python -m lab1.part2.stage3_eval

1. Load the base model once; re-evaluate it (must equal Stage 1).
2. For every run with a final.pt: load its attention + router weights (+ bias), run the same evaluate()
   on UA test / FLORES uk / FLORES en, compute routing drift (Jaccard vs base top-k on the same tokens).
3. Main table (+ noise floor rows, seed means), heatmaps, MaxVio-vs-PPL scatter, per-layer plots,
   training curves, prediction-vs-result table.
Outputs: <output_dir>/part2/{eval,figures}/, W&B run "stage3-summary".
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from ..common import load_config, out_dir
from . import wandb_log as wl
from .balancing import BiasBalancer
from .pretrain import load_model, load_trainable, trainable_state
from .router_stats import eval_sets, evaluate_all, jaccard_drift, patch_routers

PREDICTIONS = Path(__file__).parents[3] / "report" / "part2_predictions.yaml"
DRIFT_SETS = ["ua_test", "flores_en"]


def flatten(results):
    flat = {}
    for run, sets in results.items():
        for s, m in sets.items():
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    flat[f"{run}/{s}/{k}"] = v
    return flat


def main_table(results, floor, run_info):
    rows = []
    for run, m in results.items():
        ua = m["ua_test"]
        rows.append({
            "run": run, "strategy": run_info.get(run, ("base", None))[0], "seed": run_info.get(run, (None, None))[1],
            "ua_ppl": ua["ppl"], "maxvio_mean": ua["maxvio_mean"], "maxvio_max": ua["maxvio_max"],
            "entropy": ua["entropy_mean"], "logsumexp": ua["lse_mean"],
            "drift_jaccard_ua": ua.get("drift_jaccard", 1.0), "drift_jaccard_en": m["flores_en"].get("drift_jaccard", 1.0),
            "flores_en_ppl": m["flores_en"]["ppl"],
            "nll_sent_uk": m["flores_uk"]["nll_per_seq_mean"], "nll_sent_en": m["flores_en"]["nll_per_seq_mean"],
            "js_uk_en": m["js_uk_en"]["mean"],
        })
    f = floor["ua_test"]
    rows.append({"run": "random router (mean)", "maxvio_mean": f["maxvio_mean"], "maxvio_max": f["maxvio_max_over_layers_mean"]})
    rows.append({"run": "random router (p95)", "maxvio_mean": f["maxvio_p95"], "maxvio_max": f["maxvio_max_over_layers_p95"]})
    return pd.DataFrame(rows)


def seed_table(df):
    runs = df[df["seed"].notna()]
    cols = ["ua_ppl", "maxvio_mean", "maxvio_max", "entropy", "drift_jaccard_ua", "flores_en_ppl"]
    out = runs.groupby("strategy")[cols].mean().reset_index()
    out["n_seeds"] = runs.groupby("strategy").size().values
    for c in ["ua_ppl", "maxvio_mean"]:
        per_seed = runs.groupby("strategy")[c].apply(lambda v: " / ".join(f"{x:.4g}" for x in v))
        out[f"{c}_per_seed"] = out["strategy"].map(per_seed)
    return out


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prediction_table(flat):
    if not PREDICTIONS.exists():
        return None
    rows = []
    for p in yaml.safe_load(PREDICTIONS.read_text()):
        measured = ", ".join(f"{flat[k]:.4g}" if k in flat else "n/a" for k in p.get("measure", []))
        rows.append({"id": p["id"], "claim": p["claim"], "predicted": p["predicted"], "measured": measured,
                     "verdict (fill in)": ""})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    p2 = cfg["part2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir, eval_dir, fig_dir = (out_dir(cfg, "part2", d) for d in ["base", "eval", "figures"])
    runs_dir = out_dir(cfg, "part2", "runs")
    run_dirs = sorted(d for d in runs_dir.iterdir() if (d / "final.pt").exists())
    print("runs:", [d.name for d in run_dirs])

    wl.init_run(cfg, "stage3-summary", job_type="eval", config={"runs": [d.name for d in run_dirs]})
    tok = AutoTokenizer.from_pretrained(p2["model"])
    data = torch.load(out_dir(cfg, "part2", "data") / "data.pt")
    floor = json.loads((base_dir / "noise_floor.json").read_text())
    sets = eval_sets(data, p2["eval"], tok.eos_token_id)

    model = load_model(p2["model"], device)
    routers = patch_routers(model)
    n_exp = routers[0].num_experts
    base_state = trainable_state(model)

    print("\n== base (sanity: must equal Stage 1) ==")
    base_metrics, base_raw = evaluate_all(model, routers, sets, eval_dir / "base")
    stage1 = json.loads((base_dir / "metrics_all.json").read_text())
    for s in sets:
        rel = abs(base_metrics[s]["ppl"] - stage1[s]["ppl"]) / stage1[s]["ppl"]
        assert rel < 1e-3, f"base {s} ppl differs from Stage 1 by {rel:.2e}"
    base_topk = {s: np.load(base_dir / s / "topk_uint8.npy") for s in DRIFT_SETS}
    for s in DRIFT_SETS:
        same = jaccard_drift(base_topk[s], base_raw[s]["topk"].numpy(), n_exp).mean().item()
        print(f"  base vs Stage 1 top-k Jaccard on {s}: {same:.4f} (should be ~1)")
    results, counts = {"base": base_metrics}, {"base": base_raw["ua_test"]["counts"]}
    run_info, layer_drift = {}, {}
    del base_raw

    for d in run_dirs:
        print(f"\n== {d.name} ==")
        ckpt = torch.load(d / "final.pt", weights_only=False)
        load_trainable(model, base_state)
        load_trainable(model, ckpt["model"])
        for i, r in enumerate(routers):
            r.balancer = None
            if ckpt["bias"] is not None:
                r.balancer = BiasBalancer(n_exp, 0.0, device)
                r.balancer.bias.copy_(ckpt["bias"][i])
        m, raw = evaluate_all(model, routers, sets, eval_dir / d.name)
        for s in DRIFT_SETS:
            drift = jaccard_drift(base_topk[s], raw[s]["topk"].numpy(), n_exp)
            m[s]["drift_jaccard"] = drift.mean().item()
            if s == "ua_test":
                layer_drift[d.name] = drift.tolist()
            print(f"  drift (Jaccard vs base) {s}: {drift.mean():.4f}")
        (eval_dir / d.name / "metrics_all.json").write_text(json.dumps(m, indent=2))
        results[d.name], counts[d.name] = m, raw["ua_test"]["counts"]
        run_info[d.name] = (ckpt["strategy"], ckpt["seed"])
        del raw

    print("\n== tables ==")
    flat = flatten(results)
    (eval_dir / "all_metrics.json").write_text(json.dumps(flat, indent=2))
    table = main_table(results, floor, run_info)
    print(table.to_string(index=False))
    wl.log_table("results/main", table, eval_dir / "main_table")
    seeds = seed_table(table)
    print(seeds.to_string(index=False))
    wl.log_table("results/by_strategy", seeds, eval_dir / "by_strategy")
    preds = prediction_table(flat)
    if preds is not None:
        print(preds.to_string(index=False))
        wl.log_table("results/predictions", preds, eval_dir / "predictions_vs_results")

    print("\n== figures ==")
    for name, c in counts.items():
        wl.log_figure(f"heatmap/{name}", wl.heatmap_fig(c, f"{name}, UA test"), fig_dir / f"heatmap_{name}_ua_test.png")
    runs = table[table["seed"].notna() | (table["run"] == "base")]
    f = floor["ua_test"]
    wl.log_figure("results/maxvio_vs_ppl", wl.scatter_fig(
        list(zip(runs["run"], runs["maxvio_mean"], runs["ua_ppl"])), "mean global MaxVio over layers (UA test)",
        "UA test perplexity", "balance vs quality", xband=(f["maxvio_mean"], f["maxvio_p95"])),
        fig_dir / "maxvio_vs_ppl.png")
    wl.log_figure("results/maxvio_per_layer", wl.layer_lines_fig(
        {r: results[r]["ua_test"]["per_layer"]["maxvio"] for r in results}, "global MaxVio", "UA test, per layer",
        hline=f["maxvio_p95"], hline_label="random router p95"), fig_dir / "maxvio_per_layer_all.png")
    wl.log_figure("results/entropy_per_layer", wl.layer_lines_fig(
        {r: results[r]["ua_test"]["per_layer"]["entropy"] for r in results}, "normalized load entropy",
        "UA test, per layer"), fig_dir / "entropy_per_layer_all.png")
    wl.log_figure("results/js_uk_en_per_layer", wl.layer_lines_fig(
        {r: results[r]["js_uk_en"]["per_layer"] for r in results}, "JS divergence", "FLORES uk vs en load, per layer"),
        fig_dir / "js_uk_en_all.png")
    if layer_drift:
        wl.log_figure("results/drift_per_layer", wl.layer_lines_fig(
            layer_drift, "Jaccard vs base top-8", "routing drift, UA test"), fig_dir / "drift_per_layer.png")

    logs = {d.name: read_log(d / "train_log.jsonl") for d in run_dirs}
    batch_floor = floor["train_batch"]["maxvio_p95"]
    for key, label in [("maxvio_mean", "batch MaxVio (mean over layers)"), ("maxvio_max", "batch MaxVio (max over layers)"),
                       ("entropy_mean", "batch load entropy"), ("loss_lm", "train LM loss"), ("lse_mean", "router logsumexp")]:
        series = {n: ([r["step"] for r in log], [r[key] for r in log]) for n, log in logs.items()}
        hline = batch_floor if key == "maxvio_mean" else None
        wl.log_figure(f"curves/{key}", wl.curves_fig(series, label, label + " during training", hline=hline,
                      hline_label="random router p95 (per layer)"), fig_dir / f"curve_{key}.png")
    val = {n: ([r["step"] for r in log if "val_ppl" in r], [r["val_ppl"] for r in log if "val_ppl" in r]) for n, log in logs.items()}
    wl.log_figure("curves/val_ppl", wl.curves_fig(val, "val perplexity", "validation perplexity", smooth=1),
                  fig_dir / "curve_val_ppl.png")
    bias_logs = {n: ([r["step"] for r in log], [r["bias_abs_mean"] for r in log]) for n, log in logs.items()
                 if "bias_abs_mean" in log[0]}
    if bias_logs:
        wl.log_figure("curves/bias_abs_mean", wl.curves_fig(bias_logs, "mean |bias|", "selection bias magnitude", smooth=1),
                      fig_dir / "curve_bias.png")

    wl.log_artifact(eval_dir, "stage3-eval", "analysis")
    wl.finish()
    print(f"\nStage 3 done. Tables in {eval_dir}, figures in {fig_dir}.")


if __name__ == "__main__":
    main()
