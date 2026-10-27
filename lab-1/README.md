# AI500 Lab 1: Long Context and Mixture of Experts

## Part 2: MoE load balancing (EuroMoE-2.6B-A0.6B)

The constants for every stage are in `config.yaml` → `part2`. Outputs go to `$LAB1_OUTPUT_DIR` (default `/kaggle/working/outputs`) → `part2/{data,base,runs,eval,figures}`.

### Kaggle setup (GPU T4 x2, internet on)
```python
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")            # FLORES+ is gated: accept its terms on HF first
os.environ["WANDB_API_KEY"] = s.get_secret("WANDB_API_KEY")  # without it W&B is disabled; local files are still written
```
```bash
!pip install -q -e . "transformers==5.18.0" "datasets==5.1.0" wandb tabulate   # pin as in uv.lock; keep Kaggle's torch
!python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"   # must print 5.18.0
!pytest -q tests/                        # CPU unit + smoke tests (~10 s)
```

### Stage 1: no training (~35–45 min)
Data splits, tokenizer checks, HF aux-loss check, model card, router-patch equivalence, base-model evaluation, noise floor, gradient check, throughput test, γ check.
```bash
!python -m lab1.part2.stage1_base
```
Afterwards:
- Set `train.token_budget` (and `micro_batch` / `grad_accum`) from the throughput numbers.
- Fill in `report/part2_predictions.yaml`. Do this **before** Stage 2.

### Stage 2: training (~35–55 min per run at 5M tokens)
Smoke test first (5 steps, no `final.pt`):
```bash
!python -m lab1.part2.pretrain --strategy bias --seed 42 --max_steps 5
```
Run two jobs at a time, one per T4. A run that is interrupted resumes from `last.pt` when you start it again.
```bash
!CUDA_VISIBLE_DEVICES=0 nohup python -m lab1.part2.pretrain --strategy none --seed 42 > log_none42.txt 2>&1 &
!CUDA_VISIBLE_DEVICES=1 nohup python -m lab1.part2.pretrain --strategy bias --seed 42 > log_bias42.txt 2>&1 &
```
Full list: `none` (42, 43), `aux_1e-3` (42), `aux_1e-2` (42), `bias` (42, 43), `frozen_router` (42), optionally `aux_z` (42).

Each run writes these files to `runs/<strategy>-s<seed>/`:
- `train_log.jsonl`, written every step
- `final.pt`, which holds the attention and router weights plus the bias (~260 MB). It is also uploaded as a W&B artifact.

Save `/kaggle/working/outputs` as a dataset between sessions.

### Stage 3: after training (~20–30 min)
```bash
!python -m lab1.part2.stage3_eval
```
Produces:
- `eval/main_table.md`, `eval/by_strategy.md` and `eval/predictions_vs_results.md`
- `eval/all_metrics.json`, which lists the keys `measure` can refer to
- heatmaps, the MaxVio-vs-PPL scatter, per-layer plots and training curves in `figures/`
