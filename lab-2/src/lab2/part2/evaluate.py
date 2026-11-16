import math
from collections import Counter

import torch
import torch.nn.functional as F

from .data import EOT, collate


def device_of(model):
    return next(model.parameters()).device


@torch.no_grad()
def val_metrics(model, items, batches, pad_id):
    was_training, device = model.training, device_of(model)
    model.eval()
    tot = Counter()
    for b in batches:
        batch = {k: v.to(device) for k, v in collate([items[i] for i in b], pad_id).items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        lg, lb = logits[:, :-1][mask].float(), labels[mask]
        pred = lg.argmax(-1)
        eot = lb == EOT
        tot["nll"] += F.cross_entropy(lg, lb, reduction="sum").item()
        tot["n"] += mask.sum().item()
        tot["correct"] += (pred == lb).sum().item()
        tot["eot"] += eot.sum().item()
        tot["eot_hit"] += (pred[eot] == EOT).sum().item()
    model.train(was_training)
    loss = tot["nll"] / tot["n"]
    return {"val_loss": loss, "val_ppl": math.exp(loss), "val_acc": tot["correct"] / tot["n"],
            "eot_top1": tot["eot_hit"] / max(1, tot["eot"])}


@torch.no_grad()
def generate(model, tok, conversations, max_new_tokens, batch_size=10):
    was_training, device = model.training, device_of(model)
    model.eval()
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in conversations]
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                 eos_token_id=EOT, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:].tolist():
            stopped = EOT in row
            out.append({"stopped": stopped, "answer": tok.decode(row[:row.index(EOT)] if stopped else row)})
    tok.padding_side = "right"
    model.train(was_training)
    return out


def stop_rate(model, tok, items, n, max_new_tokens):
    convs = [x["messages"][:-1] for x in items[:n]]
    outs = generate(model, tok, convs, max_new_tokens)
    examples = [{"user": c[-1]["content"][:500], **o} for c, o in zip(convs[:3], outs[:3])]
    return sum(o["stopped"] for o in outs) / len(outs), examples
