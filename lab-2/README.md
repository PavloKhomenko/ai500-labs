# AI500 Lab 2, Part 1: selection variants of mamba-130m

Run `notebooks/lab2-part1.ipynb` top to bottom on one CUDA GPU (Part 1 ran on an H100).
Settings and seeds: `config.yaml`. Code: `src/lab2/part1/` (`data`, `evaluate`, `variants`, `recovery`).
Large outputs go to `outputs/` (override with `LAB2_OUTPUT_DIR`); report tables/figures to `report/figures/part1/`.
W&B tracking is enabled when `WANDB_API_KEY` is set (project `ai500-lab2-part1`).

# AI500 Lab 2, Part 2: LoRA SFT of Granite-4.0-H-1B-Base

Code: `src/lab2/part2/` (`data`, `eda`, `model`, `evaluate`, `train`, `verify`). Settings: `part2:` in `config.yaml`.
EDA (lengths, drops at 2,048 tokens, per-source mix, overlap/contamination): `report/figures/part2/eda.md`.

Kaggle (2 x T4), in a notebook:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
!git clone https://github.com/PavloKhomenko/ai500-labs.git
!bash ai500-labs/lab-2/scripts/part2.sh setup
!bash ai500-labs/lab-2/scripts/part2.sh timing a      # 20 steps: sec/step, peak memory, projected hours
!bash ai500-labs/lab-2/scripts/part2.sh train ab      # a on GPU 0, b on GPU 1
!bash ai500-labs/lab-2/scripts/part2.sh verify a
```

Outputs go to `/kaggle/working/outputs/part2/sft_{a,b}/` (adapter, tokenizer with chat template, `train_config.yaml`,
`batches.json`, `logs.csv`, `summary.json`) and are also uploaded as W&B model artifacts `sft_a` / `sft_b`
(project `ai500-lab2-part2`).
