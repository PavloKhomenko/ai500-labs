import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

from ..common import load_config
from .data import contaminated, encode_batch, eval_prompts, load_tokenizer

REPORT = Path(__file__).parents[3] / "report" / "figures" / "part2"


def profile(rows, tok, prompts, name, chunk=2000):
    records = []
    for start in range(0, len(rows), chunk):
        part = rows.select(range(start, min(start + chunk, len(rows))))
        convs = part["messages"]
        sources = part["source"] if "source" in part.column_names else [name] * len(convs)
        for m, src, item in zip(convs, sources, encode_batch(tok, convs)):
            records.append({
                "dataset": name,
                "source": src,
                "tokens": len(item["input_ids"]),
                "labeled": sum(t != -100 for t in item["labels"]),
                "turns": sum(x["role"] == "user" for x in m),
                "has_system": m[0]["role"] == "system",
                "ends_assistant": m[-1]["role"] == "assistant",
                "contaminated": contaminated(m, prompts),
            })
    return pd.DataFrame(records)


def summarize(df, by, max_len):
    g = df.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "mean": g.tokens.mean(),
        "median": g.tokens.median(),
        "p90": g.tokens.quantile(0.9),
        "p99": g.tokens.quantile(0.99),
        "max": g.tokens.max(),
        "dropped": g.tokens.apply(lambda s: int((s > max_len).sum())),
        "dropped_%": g.tokens.apply(lambda s: 100 * (s > max_len).mean()),
        "turns": g.turns.mean(),
        "system_%": 100 * g.has_system.mean(),
        "not_end_assistant_%": 100 * (1 - g.ends_assistant.mean()),
        "labeled_%": 100 * g.labeled.sum() / g.tokens.sum(),
        "contaminated": g.contaminated.sum(),
    })
    out["share_before_%"] = 100 * out["n"] / out["n"].sum()
    kept = df[df.tokens <= max_len].groupby(by).size()
    out["share_after_%"] = 100 * kept / kept.sum()
    return out.reset_index()


def conv_keys(rows):
    return {json.dumps(m, sort_keys=True) for m in rows["messages"]}


def main():
    cfg = load_config()["part2"]
    d, max_len, seed = cfg["data"], cfg["max_len"], load_config()["seed"]
    tok, prompts = load_tokenizer(cfg), eval_prompts(cfg)
    smol_train = load_dataset(d["smoltalk"], split="train")
    smol_test = load_dataset(d["smoltalk"], split="test")
    every = load_dataset(d["everyday"])
    sample = smol_train.shuffle(seed=seed).select(range(d["eda_sample"]))

    df = pd.concat([
        profile(sample, tok, prompts, "smoltalk_train"),
        profile(smol_test, tok, prompts, "smoltalk_test"),
        profile(every["train_sft"], tok, prompts, "everyday_train"),
        profile(every["test_sft"], tok, prompts, "everyday_test"),
    ])
    by_dataset = summarize(df, "dataset", max_len)
    by_source = summarize(df[df.dataset == "smoltalk_train"], "source", max_len)

    usable = (df.tokens <= max_len) & df.ends_assistant & ~df.contaminated
    kept = df[(df.dataset == "smoltalk_train") & usable]
    n, every_kept = d["n_train"], df[(df.dataset == "everyday_train") & usable]
    budget = pd.DataFrame([{
        "n_train_smoltalk": n,
        "n_everyday": len(every_kept),
        "tokens": int(kept.tokens.mean() * n + every_kept.tokens.sum()),
        "labeled_tokens": int(kept.labeled.mean() * n + every_kept.labeled.sum()),
        "mean_tokens_per_conv_kept": kept.tokens.mean(),
    }])

    train_keys = conv_keys(smol_train)
    overlap = pd.DataFrame([
        {"pair": "smoltalk_test vs smoltalk_train", "exact_conv_overlap": len(conv_keys(smol_test) & train_keys)},
        {"pair": "everyday_test vs smoltalk_train", "exact_conv_overlap": len(conv_keys(every["test_sft"]) & train_keys)},
        {"pair": "everyday_train vs smoltalk_train", "exact_conv_overlap": len(conv_keys(every["train_sft"]) & train_keys)},
        {"pair": "everyday_test vs everyday_train", "exact_conv_overlap": len(conv_keys(every["test_sft"]) & conv_keys(every["train_sft"]))},
        {"pair": "smoltalk_test vs everyday_train", "exact_conv_overlap": len(conv_keys(smol_test) & conv_keys(every["train_sft"]))},
    ])

    REPORT.mkdir(parents=True, exist_ok=True)
    sections = [("By dataset", by_dataset), ("smoltalk train sample by source", by_source),
                ("Expected training set", budget), ("Train / validation overlap", overlap)]
    text = f"# Part 2 EDA (templated with default system message, max_len {max_len}, smoltalk train sample {d['eda_sample']}, seed {seed})\n"
    for title, table in sections:
        text += f"\n## {title}\n\n{table.to_markdown(index=False, floatfmt='.1f')}\n"
    (REPORT / "eda.md").write_text(text)
    print(text)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.logspace(np.log10(df.tokens.min()), np.log10(df.tokens.max()), 60)
    for name, g in df.groupby("dataset"):
        axes[0].hist(g.tokens, bins=bins, histtype="step", density=True, label=name)
    axes[0].set(xscale="log", xlabel="tokens per conversation (templated)", ylabel="density")
    axes[0].axvline(max_len, color="k", ls="--", label=f"max_len {max_len}")
    axes[0].legend()
    by_source.set_index("source")["dropped_%"].sort_values().plot.barh(ax=axes[1])
    axes[1].set(xlabel=f"% dropped (> {max_len} tokens)", title="smoltalk train sample")
    fig.tight_layout()
    fig.savefig(REPORT / "eda_lengths.png", dpi=150)


if __name__ == "__main__":
    main()
