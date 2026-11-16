import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


def init_run(cfg, name, job_type, config=None, group=None):
    mode = os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "disabled")
    return wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], group=group, name=name,
                      job_type=job_type, config=config or {}, mode=mode, reinit="finish_previous")


def define_token_axis(*keys):
    if wandb.run is not None:
        wandb.define_metric("tokens")
        for key in keys:
            wandb.define_metric(key, step_metric="tokens")


def log(metrics, step=None):
    if wandb.run is not None:
        wandb.log(metrics, step=step)


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
    if wandb.run is not None:
        art = wandb.Artifact(name, type=type_)
        art.add_file(str(path))
        wandb.log_artifact(art)


def finish():
    if wandb.run is not None:
        wandb.finish()
