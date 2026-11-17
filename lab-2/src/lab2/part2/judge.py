import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..common import load_config, out_dir

MODELS = ["base", "instruct", "sft_a", "sft_b"]
SFT, OPPONENTS = ["sft_a", "sft_b"], ["base", "instruct"]

SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to "
    "the user question displayed below. You should choose the assistant that follows the user's instructions and "
    "answers the user's question better. Your evaluation should consider factors such as the helpfulness, "
    "relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the two responses and provide a short explanation of at most three sentences. Avoid any position "
    "biases and ensure that the order in which the responses were presented does not influence your decision. "
    "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the "
    "assistants. Be as objective as possible. After providing your explanation, output your final verdict by "
    'strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and '
    '"[[C]]" for a tie.'
)
USER = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
)
VERDICT = re.compile(r"\[\[([ABC])\]\]")


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def sample_prompts(cfg, judge):
    rows = json.load(open(hf_hub_download(cfg["data"]["alpaca_eval"], "alpaca_eval.json", repo_type="dataset")))
    idx = sorted(random.Random(judge["seed"]).sample(range(len(rows)), judge["n_prompts"]))
    return [{"id": i, "instruction": rows[i]["instruction"]} for i in idx]


def cmd_generate(cfg, judge, root, args):
    import torch
    from transformers import AutoTokenizer

    from .evaluate import generate
    from .model import load_adapter, load_base

    sft_root = Path(args.sft_root)
    tok = AutoTokenizer.from_pretrained(sft_root / "sft_a")
    if args.model in ("base", "instruct"):
        model = load_base({**cfg, "base": cfg[args.model]}).eval()
    else:
        model = load_adapter(cfg, sft_root / args.model)
    prompts = sample_prompts(cfg, judge)
    convs = [[{"role": "user", "content": p["instruction"]}] for p in prompts]
    start = time.time()
    outs = generate(model, tok, convs, judge["max_new_tokens"], batch_size=judge["gen_batch_size"])
    minutes = (time.time() - start) / 60
    with open(root / f"answers_{args.model}.jsonl", "w") as f:
        for p, o in zip(prompts, outs):
            f.write(json.dumps({**p, "model": args.model, **o}, ensure_ascii=False) + "\n")
    print(json.dumps({"model": args.model, "n": len(outs), "stopped": sum(o["stopped"] for o in outs) / len(outs),
                      "empty": sum(not o["answer"].strip() for o in outs) / len(outs), "minutes": round(minutes, 1),
                      "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}))


def make_client(judge):
    from openai import OpenAI
    return OpenAI(base_url=judge["base_url"], api_key=os.environ[judge["api_key_env"]])


def ask(client, judge, question, answer_a, answer_b):
    extra = {"reasoning_effort": judge["reasoning_effort"]} if judge.get("reasoning_effort") else {}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(question=question, answer_a=answer_a, answer_b=answer_b)}]
    for attempt in range(judge["max_retries"]):
        try:
            r = client.chat.completions.create(model=judge["model"], messages=messages, temperature=0,
                                               max_tokens=judge["max_tokens"], **extra)
            text = r.choices[0].message.content or ""
            found = VERDICT.findall(text)
            usage = r.usage
            return {"verdict": found[-1] if found else "?", "text": text,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}
        except Exception as e:
            wait = min(120, 5 * 2 ** attempt)
            print(f"  judge call failed ({type(e).__name__}: {str(e)[:120]}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("judge call failed after all retries; rerun `judge` later to resume")


def cmd_judge(cfg, judge, root, args):
    answers = {m: {r["id"]: r for r in read_jsonl(root / f"answers_{m}.jsonl")} for m in MODELS}
    missing = [m for m in MODELS if not answers[m]]
    assert not missing, f"generate the answers first: {missing}"
    path = root / "verdicts.jsonl"
    done = {(v["sft"], v["opponent"], v["id"], v["order"]) for v in read_jsonl(path)}
    jobs = [(s, o, i, order) for s in SFT for o in OPPONENTS for i in sorted(answers[s])
            for order in ["sft_first", "opponent_first"] if (s, o, i, order) not in done]
    print(f"{len(done)} verdicts already done, {len(jobs)} to go, judge {judge['model']}, "
          f"{judge['workers']} parallel workers", flush=True)
    client, lock, counter = make_client(judge), threading.Lock(), [0]

    def run(job):
        s, o, i, order = job
        first, second = (s, o) if order == "sft_first" else (o, s)
        res = ask(client, judge, answers[s][i]["instruction"], answers[first][i]["answer"], answers[second][i]["answer"])
        preferred = {"A": first, "B": second, "C": "tie"}.get(res["verdict"], "invalid")
        row = {"sft": s, "opponent": o, "id": i, "order": order, "first": first, "second": second,
               "preferred": "sft" if preferred == s else "opponent" if preferred == o else preferred, **res}
        with lock:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            counter[0] += 1
            if counter[0] % 25 == 0:
                print(f"  {counter[0]}/{len(jobs)}", flush=True)
        time.sleep(judge["min_interval_s"])

    with open(path, "a") as f, ThreadPoolExecutor(max_workers=judge["workers"]) as pool:
        list(pool.map(run, jobs))


def cmd_report(cfg, judge, root, args):
    rows = read_jsonl(root / "verdicts.jsonl")
    by_prompt = {}
    for v in rows:
        by_prompt.setdefault((v["sft"], v["opponent"], v["id"]), {})[v["order"]] = v["preferred"]
    table = {}
    for s in SFT:
        for o in OPPONENTS:
            c = Counter()
            for (s2, o2, _), orders in by_prompt.items():
                if (s2, o2) != (s, o) or len(orders) < 2:
                    continue
                a, b = orders["sft_first"], orders["opponent_first"]
                c["win" if a == b == "sft" else "loss" if a == b == "opponent" else "tie"] += 1
                c["consistent"] += a == b
                c["invalid"] += (a == "invalid") + (b == "invalid")
            n = c["win"] + c["tie"] + c["loss"]
            table[f"{s}_vs_{o}"] = {"win": c["win"], "tie": c["tie"], "loss": c["loss"], "n_prompts": n,
                                   "order_consistency": c["consistent"] / n if n else None, "invalid_calls": c["invalid"]}
    tokens_in = sum(v["prompt_tokens"] for v in rows)
    tokens_out = sum(v["completion_tokens"] for v in rows)
    cost = tokens_in / 1e6 * judge["price_in_per_mtok"] + tokens_out / 1e6 * judge["price_out_per_mtok"]
    summary = {"judge_model": judge["model"], "judge_base_url": judge["base_url"], "n_calls": len(rows),
               "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": round(cost, 4),
               "results": table, "system_prompt": SYSTEM, "user_template": USER}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"judge {judge['model']} | {len(rows)} calls | {tokens_in:,} in + {tokens_out:,} out tokens | ${cost:.4f}")
    print(f"{'pair':22s} | {'W / T / L':>11s} | {'order consistency':>17s} | invalid")
    for k, r in table.items():
        cons = f"{r['order_consistency']:.2f}" if r["order_consistency"] is not None else "n/a"
        print(f"{k:22s} | {r['win']:>3d} /{r['tie']:>3d} /{r['loss']:>3d} | {cons:>17s} | {r['invalid_calls']}")


def cmd_list_models(cfg, judge, root, args):
    for m in make_client(judge).models.list():
        print(m.id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["generate", "judge", "report", "list-models"])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--sft_root", default=None, help="folder with sft_a/ and sft_b/ (adapters + tokenizer)")
    args = ap.parse_args()
    full = load_config()
    cfg = full["part2"]
    root = out_dir(full, "part2", "judge")
    {"generate": cmd_generate, "judge": cmd_judge, "report": cmd_report,
     "list-models": cmd_list_models}[args.command](cfg, cfg["judge"], root, args)


if __name__ == "__main__":
    main()
