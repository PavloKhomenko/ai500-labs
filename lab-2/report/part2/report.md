# AI500 Lab 2, Part 2: Chat fine-tuning of Granite-4.0-H-1B-Base

## 2.1 Model choice

| | Granite-4.0-H-1B-Base | Granite-4.0-H-1B (official instruct) |
|---|---|---|
| Hub id | `ibm-granite/granite-4.0-h-1b-base` | `ibm-granite/granite-4.0-h-1b` |
| Layers | 40: 36 Mamba-2 + 4 attention (no positional encoding) | same |
| Total params | 1.466B (1.477B with the SFT-b adapters) | 1.466B |
| Licence | Apache-2.0 | Apache-2.0 |

Why this model:
- It is the default of the assignment and fits the 1-3B range.
- Most of its layers (36 of 40) are Mamba-2.
- It has an official instruct version, which gives an upper reference point.
- The fused Mamba-2 kernels compile on a T4 with Triton 3.6. The model trains in fp16 within 16 GB with LoRA (peak 11.1-12.8 GB, §2.4).

## 2.2 Chat template, system turns and loss mask

- **Template.** The base model ships no chat template. We copy `chat_template.jinja` from the instruct repository into our tokenizer (`lab2.part2.data.load_tokenizer`). The same tokenizer is used for training and for every Granite model we evaluate, including base and instruct in IFEval, in the judge answers and in the transcripts.
- **Special tokens.** End-of-turn is `<|end_of_text|>` (id 100257). The padding token is a different id, so the end-of-turn token stays in the labels.
- **System turns.** We do not add or remove system turns.
  - Conversations that have a system turn (15% of smol-smoltalk, see `report/figures/part2/eda.md`) keep it.
  - All other conversations, all evaluation prompts and the transcripts get the template's default system message.
  - Training and evaluation therefore see the same prompt format.
- **Loss mask.** The loss is on assistant tokens only: the assistant content plus the end-of-turn token. The assistant header and the newline after end-of-turn are excluded. The decoded label mask of one training example is stored in `summary.json` (`decoded_label_mask`): it starts at the first word of the assistant answer and contains no role header.

## 2.3 Data

| Split | Source | Conversations |
|---|---|---|
| train | smol-smoltalk train (5,000 of 6,000 seen) + everyday-conversations train (2,002) | 7,002 |
| validation | smol-smoltalk test (250) + everyday-conversations test (106) | 356 |

- **Train size:** 4.29M tokens, of which 3.20M are labelled (assistant) tokens.
- **What was dropped:**
  - conversations longer than 2,048 tokens: 78 train, 6 validation;
  - conversations that do not end with an assistant turn: 4 smoltalk and 255 everyday train, 13 everyday validation;
  - training conversations that overlap the validation set: 3;
  - training conversations that contain an IFEval or AlpacaEval prompt: 2.
- **Evaluation sets are disjoint from training:** IFEval (541 prompts) and the 100 AlpacaEval prompts (`alpaca_eval.json`, `random.Random(42).sample`).
- **Duplicates:** about 25 everyday conversations occur more than once in the training set. We left them in.

## 2.4 LoRA SFT

**Hyperparameters.** Both configurations use the same values (`config.yaml`, `part2:`):
- LoRA r = 16, alpha = 32, dropout 0.05;
- lr 2e-4, warmup 3%, one epoch = 187 optimizer steps;
- token-limit batches of 2,048 tokens with length bucketing, grad accumulation 16;
- frozen base in fp16, LoRA weights in fp32, fp16 autocast with a GradScaler;
- seed 42, so both configurations see identical batches (`batches.json`).

| | SFT-a | SFT-b |
|---|---|---|
| Target modules | `in_proj` | `in_proj`, `input_linear`, `output_linear`, `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable params | 4.60M (0.31%) | 15.08M (1.02%) |
| Time per optimizer step | 22.6 s | 25.6 s |
| Throughput | 1,018 tok/s | 898 tok/s |
| Peak memory | 11.1 GB | 12.8 GB |
| Train time | 1.76 h | 1.95 h |
| Skipped fp16 steps (GradScaler) | 4, at the start | 2, at the start |
| Mean grad norm | 0.21 | 0.25 |
| Val loss, start → end | 1.167 → 0.845 | 1.167 → 0.831 |
| Stop rate, 20 val prompts, start → end | 0.65 → 0.95 | 0.65 → 0.95 |

- **Hardware:** Kaggle 2 × T4, SFT-a on GPU 0 and SFT-b on GPU 1 in parallel.
- **Kernel path:** `KernelCounter` counted the fused kernels during training for both runs (`fused_kernel_path: true`): `mamba2_split_conv1d_scan_combined` 107,244 calls, `mamba2_chunk_scan` 60,480, `causal_conv1d_fn` 60,480. The time per step agrees with this.
- **Adapter gradients:** at steps 3-5 every adapter had a non-zero gradient (`grad_checks`: 0 problems). No adapter is placed on `out_proj`.
- **Validation loss** at steps 0 / 20 / 100 / 187:
  - SFT-a: 1.167 / 0.913 / 0.849 / 0.845
  - SFT-b: 1.167 / 0.877 / 0.835 / 0.831

  SFT-b is lower at every evaluation step.

## 2.5 Evaluation setup

All four models use:
- the same tokenizer and chat template;
- greedy decoding;
- the frozen weights loaded with `dtype=float16` (checked with `model.dtype`);
- the SFT models as base + PEFT adapter.

For generation we confirmed that all five Mamba-2 kernel entry points in transformers took the fused path (`is_new_implementation = True`). The log had no "falling back to … reference PyTorch implementation" warning.

- **IFEval:** lm-evaluation-harness 0.4.13, task `ifeval`, all 541 prompts.
  - Flags: `--apply_chat_template`, `max_gen_toks 1280`, `--log_samples`, seed 42.
  - Model args: `tokenizer=<sft_a dir>,dtype=float16`, plus `peft=<adapter dir>` for the SFT models.
  - Stop share: answers shorter than 1,280 tokens, counted from the logged samples with the same tokenizer.
- **Pairwise judge:** `src/lab2/part2/judge.py`, settings in `part2.judge` of `config.yaml`.
  - 100 AlpacaEval prompts, seed 42; answers up to 512 new tokens, cut at the first end-of-turn token.
  - Judge model: `gpt-4o-mini-2024-07-18`, temperature 0. Prompt: the MT-Bench pairwise prompt, given in full in `judge/summary.json` (`system_prompt`, `user_template`).
  - Each SFT model is judged against base and against instruct, in both answer orders: 2 SFT × 2 opponents × 100 prompts × 2 orders = 800 calls.
  - A win counts only if the SFT model is preferred in both orders, a loss only if the opponent is preferred in both. Anything else is a tie.
  - Cost: 654,544 input + 75,850 output tokens = **$0.14**.
- **Validation loss:** the same 356 conversations and the same batches as in training, loss on assistant tokens only. For base it reproduces the step-0 value of the SFT runs (1.1671).
- **Compute for evaluation:**

  | Model | IFEval time | GPU | Batch size |
  |---|---|---|---|
  | instruct | 45.7 min | Kaggle T4 | 16 |
  | SFT-a | 45.8 min | Kaggle T4 | 16 |
  | SFT-b | 59.5 min | Kaggle T4 | 16 |
  | base | 14.2 + 14.5 min, two halves (prompts 0-270 and 271-540), merged from the per-sample logs | Colab | 32 |

## 2.6 Results

| | Base | SFT-a (`in_proj`) | SFT-b (+ MLP + attention) | Instruct (official) |
|---|---|---|---|---|
| Validation loss | 1.167 | 0.845 | **0.831** | 0.976 |
| IFEval prompt-level strict | 0.118 ± 0.014 | 0.436 ± 0.021 | 0.477 ± 0.021 | **0.738 ± 0.019** |
| IFEval instruction-level strict | 0.209 | 0.555 | 0.597 | **0.811** |
| Share of answers stopping before 1,280 tokens | 0.53 (incl. 0.37 empty) | 0.78 | 0.83 | **0.93** |
| Trainable params | n/a | 4.60M | 15.08M | n/a |
| Judge vs base, W / T / L | n/a | 94 / 6 / 0 | **97 / 3 / 0** | n/a |
| Judge vs instruct, W / T / L | n/a | 15 / 15 / 70 | 15 / 17 / 68 | n/a |
| Time per training step | n/a | 22.6 s | 25.6 s | n/a |
| Peak training memory | n/a | 11.1 GB | 12.8 GB | n/a |

More numbers:
- **IFEval loose accuracy (prompt / instruction):**
  - base 0.144 / 0.235
  - SFT-a 0.464 / 0.584
  - SFT-b 0.510 / 0.629
  - instruct 0.765 / 0.831
- **Mean IFEval answer length (tokens):**
  - base 620
  - SFT-a 466
  - SFT-b 394
  - instruct 377
- **Judge order consistency** (share of prompts with the same winner in both orders):
  - SFT-a: 0.95 vs base, 0.86 vs instruct
  - SFT-b: 0.97 vs base, 0.84 vs instruct
  - 0 unparseable verdicts.
- **Share of judge answers stopping before 512 tokens:**
  - base 0.82, of which 0.77 are empty
  - SFT-a 0.77
  - SFT-b 0.80
  - instruct 0.71
- **Files:**
  - `ifeval/summary.json` and `ifeval/<model>/`
  - `judge/summary.json`, `judge/verdicts.jsonl`, `judge/answers_<model>.jsonl`
  - `val_loss.json`, `summary_table.md`

## 2.7 Write-up

**Which set of target modules gives the better chat model at the same budget?** Configuration b: `in_proj` + MLP + attention.

Same data, same order, same number of steps and same seed. SFT-b beats SFT-a on every measurement:
- **validation loss:** 0.831 vs 0.845;
- **IFEval prompt-level strict:** +4.1 points (0.477 vs 0.436); instruction-level +4.2 points;
- **stop share:** 0.83 vs 0.78; answers are also shorter (394 vs 466 tokens);
- **judge vs base:** 97 vs 94 wins; vs instruct the two are about equal (15 W each, 68 vs 70 losses).

Caveats and cost:
- **Small IFEval gap.** The IFEval difference is about two standard errors of either score, and the two models answer the same prompts, so the gap is suggestive but not strong.
- **Judge vs instruct does not separate them.** Against instruct the judge results are identical within noise.
- **Transcripts agree with the numbers.** SFT-b solves the time arithmetic in p05 correctly (2 h 35 min); SFT-a gets 3 h 10 min. SFT-b keeps a coherent three-turn dialogue in p07. SFT-a repeats the same "Café Lumière" paragraph three times in p07 turn 2 instead of three new names.
- **Cost of b:** 3.3× the trainable parameters, +13% time per step, +1.7 GB peak memory.

**How close does SFT come to the official instruct model?** Not close on instruction following:
- **IFEval prompt-level strict:** SFT-b recovers 0.359 of the 0.620 gap between base (0.118) and instruct (0.738), about 58%.
- **Judge:** SFT-b loses 68 of 100 prompts to instruct and wins only 15.
- **Why:** we trained one epoch on 7k conversations (3.2M labelled tokens) with rank-16 adapters. The official model had a much larger SFT and alignment pipeline behind it.

## 2.8 Failure cases and negative results

- **Lower validation loss does not mean a better chat model.** Both SFT models have lower validation loss than instruct (0.83-0.85 vs 0.98), yet the judge prefers instruct in about 70% of the prompts, and instruct is 26 points ahead on IFEval. The validation set has the same distribution as our training data, so validation loss measures fit to smol-smoltalk / everyday style, not general chat quality.
- **Base output is empty or degenerate.** Validation loss is computed only on the reference answers. On free generation the base model:
  - produces an empty answer in 37% of IFEval prompts and 77% of judge prompts;
  - often repeats a junk token (`myModalLabel`, `stringWithFormat`) or the prompt until the limit.

  Its "stop share" of 0.53 therefore mostly counts empty answers, not finished ones. Most judge wins against base are against an empty answer (the judge text says so). The 94-97 wins therefore show that SFT learned to answer at all, not that its answers are good.
- **The judge prefers the first position.** Against instruct, the judge chose the first answer 218 times and the second 177 times. Both orders and the "win in both" rule remove most of this bias, which is why the tie counts are 15-17.
- **SFT models stop less often than instruct on IFEval.** 17-22% of SFT answers reach the 1,280-token limit, vs 7% for instruct. Many long IFEval prompts ("at least 600 words", repetition constraints) push the SFT models into loops.
- **Errors in the transcripts:**
  - instruct gets the time arithmetic in p05 wrong (3 h 35 min);
  - instruct gives a 20-item list when asked for a short packing list (p10);
  - in p07 instruct picks a name it suggested in the first turn, not one of "these" three.
- **Multi-GPU Triton failure.** Our first run of base IFEval on Kaggle with `--device cuda:1` failed with `ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)` (`ifeval/base_run1_failure.txt`). The Mamba-2 Triton kernels launch on the current CUDA device (cuda:0) while the weights sat on cuda:1. Fix: one process per GPU with `CUDA_VISIBLE_DEVICES=<i>` and `--device cuda:0`.
- **Prompts were fixed after training.** The 10 transcript prompts were committed (`d822fa0`, `configs/chat_prompts.yaml`) after the adapters had been trained, but before any SFT output was generated or read. They were not tuned on the SFT models.

## 2.9 Chat demo and transcripts

- **Demo:** `python -m lab2.part2.chat demo --model sft_b --sft_root <dir>` runs a multi-turn loop that keeps the full history. Commands: `/reset`, `/quit`.
- **Transcripts:** `python -m lab2.part2.chat transcripts --model <name> --sft_root <dir>` generates them for the 10 fixed prompts.
  - The prompts cover a fact with a length limit, strict JSON, an explanation, a poem with a negative constraint, arithmetic and a practical email.
  - 4 of them are three-turn dialogues with scripted follow-ups (code, rewriting for an audience, naming, everyday talk).
  - Each turn is fed the model's own earlier answers.
- **Stop rate in the transcripts (18 assistant turns per model):**
  - instruct, SFT-a and SFT-b: 18/18;
  - base: 13/18, all of them empty answers. The other 5 turns hit the 512-token limit.
- **Attachment:** `transcripts/{base,sft_a,sft_b,instruct}.md`, with the raw `.jsonl` alongside.

## 2.10 Reproduction

```bash
bash scripts/part2.sh setup
python -m lab2.part2.judge generate --model sft_b --sft_root outputs/part2   # repeat for base, instruct, sft_a
python -m lab2.part2.judge judge        # needs OPENAI_API_KEY
python -m lab2.part2.judge report
python -m lab2.part2.chat transcripts --model sft_b --sft_root outputs/part2
lm_eval --model hf --model_args pretrained=ibm-granite/granite-4.0-h-1b-base,tokenizer=outputs/part2/sft_a,dtype=float16,peft=outputs/part2/sft_b \
  --tasks ifeval --apply_chat_template --batch_size 16 --device cuda:0 --seed 42 --log_samples --output_path outputs/part2/ifeval/sft_b
```

**Versions:**
- torch 2.10.0+cu126, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0;
- mamba_ssm 2.3.2.post1, causal_conv1d 1.7.0, triton 3.6.0;
- lm_eval 0.4.13, openai ≥ 1.40.

Dependencies are in `pyproject.toml` and `uv.lock`; settings and seeds are in `config.yaml`.
