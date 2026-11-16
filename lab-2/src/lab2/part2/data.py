import json
import random
import re
from collections import Counter

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

EOT, SOR, EOR = 100257, 100264, 100265
ASSISTANT = re.compile(r"<\|start_of_role\|>assistant<\|end_of_role\|>(.*?<\|end_of_text\|>)", re.S)


def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    tok.chat_template = open(hf_hub_download(cfg["instruct"], "chat_template.jinja")).read()
    assert tok.convert_tokens_to_ids(["<|end_of_text|>", "<|start_of_role|>", "<|end_of_role|>"]) == [EOT, SOR, EOR]
    assert tok.pad_token_id is not None and tok.pad_token_id != EOT
    tok.padding_side = "right"
    return tok


def encode_batch(tok, convs):
    texts = [tok.apply_chat_template(m, tokenize=False) for m in convs]
    enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for text, ids, offsets in zip(texts, enc["input_ids"], enc["offset_mapping"]):
        spans = [m.span(1) for m in ASSISTANT.finditer(text)]
        labels = [t if any(a <= s and e <= b for a, b in spans) else -100 for t, (s, e) in zip(ids, offsets)]
        out.append({"input_ids": ids, "labels": labels})
    return out


def decode_mask(tok, item):
    return tok.decode([t for t in item["labels"] if t != -100])


def eval_prompts(cfg):
    d = cfg["data"]
    ifeval = load_dataset(d["ifeval"], split="train")["prompt"]
    alpaca = json.load(open(hf_hub_download(d["alpaca_eval"], "alpaca_eval.json", repo_type="dataset")))
    return {p.strip() for p in ifeval} | {x["instruction"].strip() for x in alpaca}


def contaminated(messages, prompts):
    return any(m["role"] == "user" and m["content"].strip() in prompts for m in messages)


def conv_key(messages):
    return json.dumps(messages, sort_keys=True)


def select(rows, tok, prompts, max_len, n=None, seed=0, exclude=frozenset(), chunk=1000):
    order = list(range(len(rows)))
    if n is not None:
        random.Random(seed).shuffle(order)
    kept, stats = [], Counter()
    for start in range(0, len(order), chunk):
        convs = rows.select(order[start:start + chunk])["messages"]
        stats["seen"] += len(convs)
        ok = []
        for m in convs:
            if not m or m[-1]["role"] != "assistant":
                stats["not_ending_assistant"] += 1
            elif contaminated(m, prompts):
                stats["contaminated"] += 1
            elif conv_key(m) in exclude:
                stats["in_val"] += 1
            else:
                ok.append(m)
        for m, item in zip(ok, encode_batch(tok, ok)):
            if len(item["input_ids"]) > max_len:
                stats["too_long"] += 1
            else:
                kept.append(item | {"messages": m})
        if n is not None and len(kept) >= n:
            break
    kept = kept[:n] if n is not None else kept
    stats["kept"] = len(kept)
    return kept, dict(stats)


def build_splits(cfg, tok, seed):
    d, max_len, prompts = cfg["data"], cfg["max_len"], eval_prompts(cfg)
    smol_train = load_dataset(d["smoltalk"], split="train")
    smol_test = load_dataset(d["smoltalk"], split="test")
    every = load_dataset(d["everyday"])
    parts = {
        "val_smoltalk": select(smol_test, tok, prompts, max_len, d["n_val_smoltalk"], seed),
        "val_everyday": select(every["test_sft"], tok, prompts, max_len),
    }
    val_keys = {conv_key(x["messages"]) for k in parts for x in parts[k][0]}
    parts["train_smoltalk"] = select(smol_train, tok, prompts, max_len, d["n_train"], seed, val_keys)
    parts["train_everyday"] = select(every["train_sft"], tok, prompts, max_len, exclude=val_keys)
    stats = {k: v[1] for k, v in parts.items()}
    train = parts["train_smoltalk"][0] + parts["train_everyday"][0]
    val = parts["val_smoltalk"][0] + parts["val_everyday"][0]
    return train, val, stats


def make_batches(lengths, max_tokens, seed, shuffle=True):
    rng = random.Random(seed)
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], rng.random()))
    batches, cur, cur_max = [], [], 0
    for i in order:
        if cur and (len(cur) + 1) * max(cur_max, lengths[i]) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, lengths[i])
    batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(items, pad_id):
    n = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), n), pad_id)
    labels = torch.full((len(items), n), -100)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    for i, x in enumerate(items):
        k = len(x["input_ids"])
        ids[i, :k] = torch.tensor(x["input_ids"])
        labels[i, :k] = torch.tensor(x["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def prepare(cfg, tok, path, seed):
    if (path / "data.pt").exists():
        return torch.load(path / "data.pt", weights_only=False)
    train, val, stats = build_splits(cfg, tok, seed)
    t = cfg["train"]
    data = {
        "train": train,
        "val": val,
        "stats": stats,
        "train_batches": make_batches([len(x["input_ids"]) for x in train], t["micro_tokens"], seed),
        "val_batches": make_batches([len(x["input_ids"]) for x in val], t["micro_tokens"], seed, shuffle=False),
    }
    path.mkdir(parents=True, exist_ok=True)
    torch.save(data, path / "data.pt")
    (path / "batches.json").write_text(json.dumps({"train": data["train_batches"], "val": data["val_batches"]}))
    return data
