"""Ukrainian corpus loading, cleaning, train/val/test splits.
A reuses the test split to pick long documents (>= scale * l_train tokens) for Part 1.

Split rule: a document's split is a function of its text hash, so identical texts can never
land in two splits (web crawls repeat the same page under different ids).
"""

import hashlib
import json
import unicodedata
from collections import Counter

import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def normalize(text):
    return unicodedata.normalize("NFC", text).strip()


def doc_key(text):
    """Dedup key: md5 of the normalized first 1000 characters."""
    return hashlib.md5(text[:1000].encode("utf-8")).hexdigest()


def split_of(key, test_buckets, val_buckets):
    bucket = int(key, 16) % 1000
    if bucket < test_buckets:
        return "test"
    if bucket < test_buckets + val_buckets:
        return "val"
    return "train"


def encode_doc(tokenizer, text, max_tokens):
    """[BOS] + tokens + [EOS]; EOS only if the document was not truncated."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > max_tokens - 2:
        return [tokenizer.bos_token_id] + ids[: max_tokens - 1]
    return [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]


def pack(docs, seq_len):
    """Concatenate token lists in order and cut into [n_blocks, seq_len]; the remainder is dropped."""
    flat = torch.tensor([t for doc in docs for t in doc], dtype=torch.int32)
    n = len(flat) // seq_len
    return flat[: n * seq_len].view(n, seq_len)


def iter_shard(corpus, shard):
    """Rows of one Kobza parquet shard, in file order. Each shard is one ~5 GB row group, so
    HF streaming would decode it all before the first row; we download it once and read batches."""
    path = hf_hub_download(corpus, shard, repo_type="dataset")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["text", "source"]):
        yield from batch.to_pylist()


def build_kobza_splits(corpus, tokenizer, dcfg):
    """Read one Kobza shard (sources are interleaved inside it) until the train split has
    dcfg["train_pool_tokens"] tokens. Returns ({split: token lists}, {split: texts}, report)."""
    ds = iter_shard(corpus, dcfg["shard"])
    docs = {"train": [], "val": [], "test": []}
    keys = {"train": set(), "val": set(), "test": set()}
    texts = {"train": [], "val": [], "test": []}
    sources = {s: Counter() for s in docs}
    n_tokens = Counter()
    seen, n_short, n_dup = set(), 0, 0
    for row in ds:
        text = normalize(row["text"])
        if len(text) < dcfg["min_doc_chars"]:
            n_short += 1
            continue
        key = doc_key(text)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        split = split_of(key, dcfg["test_buckets"], dcfg["val_buckets"])
        # Cut very long texts before tokenizing (~8 chars per token is a safe upper bound).
        ids = encode_doc(tokenizer, text[: dcfg["max_doc_tokens"] * 8], dcfg["max_doc_tokens"])
        docs[split].append(ids)
        keys[split].add(key)
        texts[split].append(text)
        sources[split][row.get("source", "?")] += 1
        n_tokens[split] += len(ids)
        if n_tokens["train"] >= dcfg["train_pool_tokens"]:
            break

    # No document may appear in two splits.
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"]) and not (keys["val"] & keys["test"])
    report = {
        "corpus": f"{corpus}/{dcfg['shard']}",
        "dropped_short": n_short,
        "dropped_duplicates": n_dup,
        "splits": {
            s: {"docs": len(docs[s]), "tokens": n_tokens[s], "sources": dict(sources[s])} for s in docs
        },
    }
    return docs, texts, report


def load_flores(name, tokenizer):
    """FLORES+ dev + devtest for uk and en, aligned by id. Each sentence: [BOS] + tokens (no EOS)."""
    out = {}
    for lang, config in [("uk", "ukr_Cyrl"), ("en", "eng_Latn")]:
        rows = []
        for split in ["dev", "devtest"]:
            ds = load_dataset(name, config, split=split)
            rows += [(split, r["id"], normalize(r["text"])) for r in ds]
        rows.sort(key=lambda r: (r[0], r[1]))
        out[lang] = rows
    assert [r[:2] for r in out["uk"]] == [r[:2] for r in out["en"]], "FLORES uk/en not aligned"
    texts = {lang: [r[2] for r in rows] for lang, rows in out.items()}
    ids = {
        lang: [[tokenizer.bos_token_id] + tokenizer(t, add_special_tokens=False).input_ids for t in sents]
        for lang, sents in texts.items()
    }
    return texts, ids


def flores_in_train(flores_texts, train_texts):
    """Count FLORES sentences that appear verbatim inside any train document."""
    big = "\n".join(train_texts)
    return {lang: sum(s in big for s in sents) for lang, sents in flores_texts.items()}


def prepare_part2_data(cfg, tokenizer, path):
    """Build (or load) the Part 2 data file: packed train/val/test blocks + FLORES token ids."""
    data_file, report_file = path / "data.pt", path / "split_report.json"
    if data_file.exists():
        return torch.load(data_file), json.loads(report_file.read_text())

    dcfg = cfg["part2"]["data"]
    docs, texts, report = build_kobza_splits(cfg["data"]["corpus"], tokenizer, dcfg)
    flores_texts, flores_ids = load_flores(dcfg["flores"], tokenizer)

    report["flores_sentences_in_train"] = flores_in_train(flores_texts, texts["train"])
    data = {s: pack(docs[s], dcfg["seq_len"]) for s in docs}
    data["flores"] = flores_ids
    for s in ["train", "val", "test"]:
        report["splits"][s]["blocks"] = len(data[s])
    for lang in ["uk", "en"]:
        report[f"flores_{lang}"] = {"sentences": len(flores_ids[lang]), "tokens": sum(map(len, flores_ids[lang]))}

    torch.save(data, data_file)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (path / "flores_texts.json").write_text(json.dumps(flores_texts, ensure_ascii=False))
    return data, report


def block_batches(blocks, batch_size):
    """Packed blocks -> (input_ids, attention_mask) batches; no padding."""
    for i in range(0, len(blocks), batch_size):
        ids = blocks[i : i + batch_size].long()
        yield ids, torch.ones_like(ids)


def sentence_batches(seqs, batch_size, pad_id):
    """Variable-length sentences -> right-padded batches, in the original order."""
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        width = max(map(len, chunk))
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, s in enumerate(chunk):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        yield ids, mask
