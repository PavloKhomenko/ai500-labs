import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..common import load_config, out_dir
from .evaluate import val_metrics
from .model import load_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", type=Path)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()
    full = load_config()
    data = torch.load(out_dir(full, "part2", "data") / "data.pt", weights_only=False)
    tok = AutoTokenizer.from_pretrained(args.adapter)
    model = load_adapter(full["part2"], args.adapter, device="cuda" if torch.cuda.is_available() else "cpu")
    measured = val_metrics(model, data["val"], data["val_batches"], tok.pad_token_id)["val_loss"]
    logged = json.loads((args.adapter / "summary.json").read_text())["final_val"]["val_loss"]
    print(json.dumps({"logged_val_loss": logged, "measured_val_loss": measured, "abs_diff": abs(measured - logged)}))
    assert abs(measured - logged) < args.tol


if __name__ == "__main__":
    main()
