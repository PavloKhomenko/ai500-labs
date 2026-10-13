import json
import math
import random
import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..part2 import wandb_log as wl
from .rope import RoPEScaling

FILLER = {"uk": "Трава зелена. Небо блакитне. Сонце жовте. Ось так. Туди і назад. ",
          "en": "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "}
NEEDLE = {"uk": "Пароль: {k}. Запам'ятай його. {k} — це пароль. ",
          "en": "The pass key is {k}. Remember it. {k} is the pass key. "}
QUESTION = {"uk": "Який пароль? Пароль:", "en": "What is the pass key? The pass key is"}


class LongContextEvaluator:
    def __init__(self, model, tokenizer, p1, seed=42):
        self.model, self.tok, self.p1, self.seed = model.eval(), tokenizer, p1, seed
        self.rope = RoPEScaling.from_config(model.config, p1["scale"])
        self.l_train = p1["l_train"]
        self.windows = [self.l_train * k for k in (1, 2, 4)]
        self.device = next(model.parameters()).device

    def set_method(self, method, seq_len=None, verbose=True):
        self.rope.apply(self.model, method, seq_len, verbose)

    @torch.inference_mode()
    def token_nll(self, ids):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            h = self.model.model(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
        labels, chunk = ids[0, 1:], self.p1["lm_head_chunk"]
        return torch.cat([
            F.cross_entropy(self.model.lm_head(h[i : i + chunk]).float(), labels[i : i + chunk], reduction="none")
            for i in range(0, len(labels), chunk)
        ]).double().cpu()

    def doc_nll(self, ids, method):
        if method != "dynamic_ntk":
            nll = self.token_nll(ids)
            return nll, torch.stack([nll[:w].sum() for w in self.windows])
        parts, sums = [], []
        for k, w in enumerate(self.windows):
            self.set_method(method, seq_len=w, verbose=False)
            nll = self.token_nll(ids[:, : w + 1])
            sums.append(nll.sum())
            parts.append(nll[(self.windows[k - 1] if k else 0) :])
        return torch.cat(parts), torch.stack(sums)

    def loss_by_position(self, docs, n_bytes, method, lang=""):
        if method != "dynamic_ntk":
            self.set_method(method)
        total, win_nll = torch.zeros(self.windows[-1], dtype=torch.float64), []
        t0 = time.time()
        for d, row in enumerate(docs):
            nll, sums = self.doc_nll(row[None].long().to(self.device), method)
            total += nll
            win_nll.append(sums)
            if d == 0:
                sec = time.time() - t0
                print(f"[{method} {lang}] first document {sec:.1f}s -> ~{sec * len(docs) / 60:.1f} min for {len(docs)}")
        bins = (total / len(docs)).view(-1, self.p1["bin_size"]).mean(1)
        bpb = torch.stack(win_nll).sum(0) / (math.log(2) * n_bytes.double().sum(0))
        res = {"bin_nll": bins.tolist(), "bpb": dict(zip(map(str, self.windows), bpb.tolist())),
               "minutes": (time.time() - t0) / 60}
        for i, v in enumerate(res["bin_nll"]):
            wl.log({f"loss/{lang}": v, "position": (i + 0.5) * self.p1["bin_size"]})
        wl.log_summary({f"bpb/{lang}_{w}": v for w, v in res["bpb"].items()} | {f"minutes/loss_{lang}": res["minutes"]})
        return res

    def passkey_prompt(self, lang, key, length, depth, filler):
        enc = lambda t: self.tok(t, add_special_tokens=False).input_ids
        needle, question = enc(" " + NEEDLE[lang].format(k=key)), enc(" " + QUESTION[lang])
        n_fill = length - 1 - len(needle) - len(question)
        cut = int(n_fill * depth)
        return [self.tok.bos_token_id] + filler[:cut] + needle + filler[cut:n_fill] + question

    @torch.inference_mode()
    def passkey(self, method, langs=("uk", "en")):
        p1 = self.p1
        keys = [random.Random(self.seed + i).randint(10000, 99999) for i in range(p1["passkey_keys"])]
        if method != "dynamic_ntk":
            self.set_method(method)
        rows, t0 = [], time.time()
        for lang in langs:
            filler = self.tok(FILLER[lang] * (max(p1["passkey_lengths"]) // 4), add_special_tokens=False).input_ids
            for length in p1["passkey_lengths"]:
                if method == "dynamic_ntk":
                    self.set_method(method, seq_len=length)
                for depth in p1["passkey_depths"]:
                    hits = 0
                    for key in keys:
                        ids = torch.tensor([self.passkey_prompt(lang, key, length, depth, filler)], device=self.device)
                        assert ids.shape[1] == length
                        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                                      max_new_tokens=p1["passkey_new_tokens"],
                                                      pad_token_id=self.tok.eos_token_id)
                        found = re.search(r"\d{5}", self.tok.decode(out[0, length:]))
                        hits += bool(found) and found.group() == str(key)
                    rows.append({"lang": lang, "length": length, "depth": depth, "acc": hits / len(keys)})
                    print(f"passkey {method} {lang} L={length} depth={depth}: {hits}/{len(keys)}")
        df = pd.DataFrame(rows)
        wl.log_table("passkey", df)
        wl.log_summary({f"passkey/{r.lang}_{r.length}_d{r.depth}": r.acc for r in df.itertuples()}
                       | {"minutes/passkey": (time.time() - t0) / 60})
        return df

    def evaluate(self, method, data, out_dir, tag=None, passkey=True):
        tag = tag or method
        res = {"loss": {lang: self.loss_by_position(data[lang], data[f"{lang}_bytes"], method, lang) for lang in ["uk", "en"]}}
        (out_dir / f"loss_{tag}.json").write_text(json.dumps(res["loss"], indent=2))
        if passkey:
            res["passkey"] = self.passkey(method)
            res["passkey"].to_json(out_dir / f"passkey_{tag}.json", orient="records", indent=2)
        return res


class ResultsReport:
    def __init__(self, eval_dir, fig_dir, l_train, bin_size):
        self.eval_dir, self.fig_dir, self.l_train, self.bin_size = eval_dir, fig_dir, l_train, bin_size
        self.losses = {f.stem[len("loss_"):]: json.loads(f.read_text()) for f in sorted(eval_dir.glob("loss_*.json"))}
        self.passkeys = {f.stem[len("passkey_"):]: pd.read_json(f) for f in sorted(eval_dir.glob("passkey_*.json"))}

    def loss_plot(self, lang):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for tag, res in self.losses.items():
            bins = res[lang]["bin_nll"]
            ax.plot([(i + 0.5) * self.bin_size for i in range(len(bins))], bins, marker="o", ms=3, label=tag)
        ax.axvline(self.l_train, color="gray", ls="--", label="L_train")
        ax.set(xlabel="position", ylabel="mean nll (nats)", title=f"teacher-forced loss by position, {lang}")
        ax.legend(fontsize=8)
        fig.savefig(self.fig_dir / f"loss_by_position_{lang}.png", dpi=150, bbox_inches="tight")
        wl.log({f"loss_by_position/{lang}": wl.wandb.Image(fig)})
        return fig

    def bpb_table(self):
        df = pd.DataFrame([{"method": t, "lang": lang, **{f"bpb@{w}": v for w, v in r[lang]["bpb"].items()}}
                           for t, r in self.losses.items() for lang in ["uk", "en"]])
        wl.log_table("bpb", df, self.eval_dir / "bpb")
        return df

    def passkey_table(self):
        df = pd.concat([d.assign(method=t) for t, d in self.passkeys.items()])
        df = df.pivot_table(index=["method", "lang"], columns=["length", "depth"], values="acc")
        df.columns = [f"{length}@{depth}" for length, depth in df.columns]
        df = df.reset_index()
        wl.log_table("passkey", df, self.eval_dir / "passkey_table")
        return df
