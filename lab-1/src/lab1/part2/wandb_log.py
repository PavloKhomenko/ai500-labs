"""Weights & Biases logging tools for Part 2 (+ the matplotlib figures we log).

Every figure / table is also saved locally, so the report never depends on W&B.
W&B runs online only when WANDB_API_KEY is set (or WANDB_MODE is given); otherwise it is disabled.
On Kaggle: os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


def init_run(cfg, name, job_type, config=None, group="part2", project=None):
    mode = os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "disabled")
    return wandb.init(
        project=project or cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        group=group,
        name=name,
        job_type=job_type,
        config=config or {},
        mode=mode,
        reinit="finish_previous",
    )


def log(metrics, step=None):
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_per_layer(prefix, values, step=None):
    log({f"{prefix}/layer_{i:02d}": float(v) for i, v in enumerate(values)}, step=step)


def log_summary(values):
    if wandb.run is not None:
        wandb.run.summary.update(values)


def log_figure(key, fig, path=None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    log({key: wandb.Image(fig)})
    plt.close(fig)


def log_table(key, df, path=None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path.with_suffix(".csv"), index=False)
        path.with_suffix(".md").write_text(df.to_markdown(index=False, floatfmt=".4g"))
    log({key: wandb.Table(dataframe=df)})


def log_artifact(path, name, type_):
    if wandb.run is None:
        return
    art = wandb.Artifact(name, type=type_)
    if path.is_dir():
        art.add_dir(str(path))
    else:
        art.add_file(str(path))
    wandb.log_artifact(art)


def finish():
    if wandb.run is not None:
        wandb.finish()


def heatmap_fig(counts, title):
    counts = np.asarray(counts, dtype=float)
    rel = counts / counts.mean(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(np.log2(np.clip(rel, 1 / 16, 16)), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xlabel("expert")
    ax.set_ylabel("layer")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2(load / mean load)")
    return fig


def layer_lines_fig(series, ylabel, title, hline=None, hline_label=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in series.items():
        ax.plot(range(len(values)), values, marker="o", ms=3, label=label)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", label=hline_label)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def curves_fig(series, ylabel, title, hline=None, hline_label=None, smooth=10):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, (steps, values) in series.items():
        v = np.asarray(values, dtype=float)
        if smooth > 1 and len(v) >= smooth:
            v = np.convolve(v, np.ones(smooth) / smooth, mode="valid")
            steps = steps[smooth - 1 :]
        ax.plot(steps, v, label=label)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", label=hline_label)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def scatter_fig(points, xlabel, ylabel, title, xband=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if xband is not None:
        ax.axvspan(*xband, color="gray", alpha=0.2, label="random-router floor (mean..p95)")
    for label, x, y in points:
        ax.scatter(x, y, s=40)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig
