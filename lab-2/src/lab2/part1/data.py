import hashlib

import numpy as np
import torch
from datasets import load_dataset
from transformers.convert_slow_tokenizer import bytes_to_unicode


def doc_key(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def token_bytes(tokenizer):
    inverse = {c: b for b, c in bytes_to_unicode().items()}
    out = torch.zeros(len(tokenizer), dtype=torch.long)
    special = set(tokenizer.all_special_ids)
    for token, idx in tokenizer.get_vocab().items():
        if idx not in special:
            out[idx] = len(token) if all(c in inverse for c in token) else len(token.encode())  # 1 char = 1 byte
    return out


def stream_texts(data_files, seed, buffer_size, min_chars=0, fmt="json"):
    ds = load_dataset(fmt, data_files=data_files, split="train", streaming=True).shuffle(seed=seed, buffer_size=buffer_size)
    for row in ds:
        if len(row["text"]) >= min_chars:
            yield row["text"]


class EvalSamples:
    def __init__(self, tokenizer, seq_len):
        self.tok, self.seq_len = tokenizer, seq_len
        self.tbytes = token_bytes(tokenizer)

    def build(self, texts, target_tokens, min_docs=0):
        ids, nbytes, keys, total = [], [], [], 0
        for text in texts:
            doc = self.tok(text, add_special_tokens=False).input_ids[: self.seq_len - 1]
            if not doc:
                continue
            ids.append(torch.tensor([self.tok.eos_token_id] + doc, dtype=torch.int32))
            nbytes.append(int(self.tbytes[doc].sum()))
            keys.append(doc_key(text))
            total += len(doc)
            if total >= target_tokens and len(ids) >= min_docs:
                break
        return {"ids": ids, "bytes": torch.tensor(nbytes), "keys": keys, "tokens": total}

    def build_all(self, dcfg, seed):
        pile = f"hf://datasets/{dcfg['pile']}"
        kobza = f"hf://datasets/{dcfg['kobza']}/{dcfg['kobza_shard']}"
        buf, mc = dcfg["shuffle_buffer"], dcfg["min_doc_chars"]
        return {
            "en": self.build(stream_texts(f"{pile}/{dcfg['test_file']}", seed, buf, mc), dcfg["en_tokens"], dcfg["min_docs"]),
            "uk": self.build(stream_texts(kobza, seed, buf, mc, fmt="parquet"), dcfg["uk_tokens"], dcfg["min_docs"]),
            "val": self.build(stream_texts(f"{pile}/{dcfg['val_file']}", seed, buf), dcfg["val_tokens"]),
        }

    @staticmethod
    def same(a, b):
        return a["keys"] == b["keys"] and all(torch.equal(x, y) for x, y in zip(a["ids"], b["ids"]))


class TrainStream:
    def __init__(self, tokenizer, seq_len):
        self.tok, self.seq_len = tokenizer, seq_len

    def build(self, dcfg, seed, out, batch_docs=1000):
        texts = stream_texts(f"hf://datasets/{dcfg['pile']}/{dcfg['train_file']}", seed, dcfg["shuffle_buffer"])
        n_calib = dcfg["calib_tokens"] // self.seq_len
        n_rows = n_calib + dcfg["train_tokens"] // self.seq_len
        rows = np.lib.format.open_memmap(out / "rows.npy", mode="w+", dtype=np.uint16, shape=(n_rows, self.seq_len))
        keys, buf, filled, batch = [], [], 0, []
        for text in texts:
            batch.append(text)
            if len(batch) < batch_docs:
                continue
            for doc in self.tok(batch, add_special_tokens=False).input_ids:
                buf.extend(doc + [self.tok.eos_token_id])
            keys += [doc_key(t) for t in batch]
            batch = []
            while len(buf) >= self.seq_len and filled < n_rows:
                rows[filled] = buf[: self.seq_len]
                buf, filled = buf[self.seq_len :], filled + 1
            if filled == n_rows:
                break
        rows.flush()
        np.save(out / "calib.npy", rows[:n_calib])
        np.save(out / "train.npy", rows[n_calib:])
        (out / "rows.npy").unlink()
        (out / "train_keys.txt").write_text("\n".join(keys))
        return n_calib, n_rows - n_calib

    @staticmethod
    def load(out):
        return (torch.from_numpy(np.load(out / "calib.npy").astype(np.int64)),
                torch.from_numpy(np.load(out / "train.npy").astype(np.int64)))


def overlap(train_keys, samples):
    train_keys = set(train_keys)
    return {name: sum(k in train_keys for k in s["keys"]) for name, s in samples.items()}
