import argparse
import json
import time
from pathlib import Path

import yaml

from ..common import load_config, out_dir

PROMPTS = Path(__file__).parents[3] / "configs" / "chat_prompts.yaml"
MODELS = ["base", "instruct", "sft_a", "sft_b"]


def load(cfg, name, sft_root):
    from transformers import AutoTokenizer

    from .model import load_named

    tok = AutoTokenizer.from_pretrained(Path(sft_root) / "sft_a")
    return load_named(cfg, name, sft_root), tok


def transcripts(model, tok, prompts, max_new_tokens, batch_size):
    from .evaluate import generate

    convs = {p["id"]: [] for p in prompts}
    turns = {p["id"]: [] for p in prompts}
    for k in range(max(len(p["turns"]) for p in prompts)):
        active = [p for p in prompts if k < len(p["turns"])]
        for p in active:
            convs[p["id"]].append({"role": "user", "content": p["turns"][k]})
        outs = generate(model, tok, [convs[p["id"]] for p in active], max_new_tokens, batch_size=batch_size)
        for p, o in zip(active, outs):
            convs[p["id"]].append({"role": "assistant", "content": o["answer"]})
            n_new = len(tok(o["answer"], add_special_tokens=False).input_ids)
            turns[p["id"]].append({"user": p["turns"][k], "answer": o["answer"], "stopped": o["stopped"],
                                   "new_tokens": n_new})
    return [{"id": p["id"], "kind": p["kind"], "turns": turns[p["id"]]} for p in prompts]


def to_markdown(name, rows, max_new_tokens):
    lines = [f"# Transcripts: {name}", "",
             f"Greedy decoding, at most {max_new_tokens} new tokens per assistant turn, cut at the first end-of-turn "
             f"token; chat template of the official instruct model; no system turn (the template adds its default "
             f"system message).", ""]
    for r in rows:
        lines += [f"## {r['id']}: {r['kind']}", ""]
        for t in r["turns"]:
            limit = "" if t["stopped"] else ", stopped by the token limit"
            lines += [f"**User:** {t['user']}", "", f"**Assistant** ({t['new_tokens']} tokens{limit}):", "",
                      t["answer"].strip() or "_(empty answer)_", ""]
    return "\n".join(lines)


def chat_loop(model, tok, max_new_tokens=512):
    from .evaluate import generate

    history = []
    print("Type a message. /reset clears the history, /quit ends the chat.")
    while True:
        text = input("you> ").strip()
        if text == "/quit":
            return history
        if text == "/reset":
            history = []
            print("(history cleared)")
            continue
        if not text:
            continue
        history.append({"role": "user", "content": text})
        out = generate(model, tok, [history], max_new_tokens, batch_size=1)[0]
        history.append({"role": "assistant", "content": out["answer"]})
        print(f"model> {out['answer']}" + ("" if out["stopped"] else "\n[stopped by the token limit]"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["transcripts", "demo"])
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--sft_root", required=True, help="folder with sft_a/ and sft_b/ (adapters + tokenizer)")
    args = ap.parse_args()
    full = load_config()
    cfg, chat = full["part2"], full["part2"]["chat"]
    model, tok = load(cfg, args.model, args.sft_root)
    if args.command == "demo":
        chat_loop(model, tok, chat["max_new_tokens"])
        return
    prompts = yaml.safe_load(PROMPTS.read_text())["prompts"]
    start = time.time()
    rows = transcripts(model, tok, prompts, chat["max_new_tokens"], chat["batch_size"])
    root = out_dir(full, "part2", "transcripts")
    (root / f"{args.model}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (root / f"{args.model}.md").write_text(to_markdown(args.model, rows, chat["max_new_tokens"]))
    turns = [t for r in rows for t in r["turns"]]
    print(json.dumps({"model": args.model, "prompts": len(rows), "turns": len(turns),
                      "stopped": sum(t["stopped"] for t in turns) / len(turns),
                      "empty": sum(not t["answer"].strip() for t in turns) / len(turns),
                      "minutes": round((time.time() - start) / 60, 1)}))


if __name__ == "__main__":
    main()
