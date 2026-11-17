# Part 2 summary table

Granite-4.0-H-1B, fp16 on a T4, greedy decoding, chat template of the official instruct model with its default system message, one tokenizer for all four rows.

| | Base | SFT-a (`in_proj`) | SFT-b (+ MLP + attention) | Instruct (official) |
|---|---|---|---|---|
| Val loss (356 conv., assistant tokens) | 1.167 | 0.845 | **0.831** | 0.976 |
| IFEval prompt-level strict | 0.118 ± 0.014 | 0.436 ± 0.021 | 0.477 ± 0.021 | **0.738 ± 0.019** |
| IFEval instruction-level strict | 0.209 | 0.555 | 0.597 | **0.811** |
| IFEval prompt / inst loose | 0.144 / 0.235 | 0.464 / 0.584 | 0.510 / 0.629 | **0.765 / 0.831** |
| IFEval stop share (< 1280 tokens) | 0.53 (0.37 empty) | 0.78 | 0.83 | **0.93** |
| Judge stop share (< 512 tokens, 100 prompts) | 0.82 (0.77 empty) | 0.77 | 0.80 | 0.71 |
| Trainable params | n/a | 4.60M | 15.08M | n/a |
| Judge vs base, W / T / L | n/a | 94 / 6 / 0 | **97 / 3 / 0** | n/a |
| Judge vs instruct, W / T / L | n/a | 15 / 15 / 70 | 15 / 17 / 68 | n/a |
| Time per optimizer step | n/a | 22.6 s | 25.6 s | n/a |
| Peak training memory | n/a | 11.1 GB | 12.8 GB | n/a |

Notes
- IFEval: lm-evaluation-harness 0.4.13, all 541 prompts, `--apply_chat_template`, max_gen_toks 1280, stderr from the harness formula. Base was run in two halves (0-270, 271-540) and merged from the per-sample logs; per-model numbers in `ifeval/summary.json`.
- Stop share counts answers shorter than the generation limit; "empty" answers stop immediately and are counted as stopped, so for base the stop share overstates how often it ends a real answer.
- Judge: gpt-4o-mini-2024-07-18, MT-Bench pairwise prompt, 100 AlpacaEval prompts (seed 42), both orders, a win only when preferred in both orders; order consistency 0.84-0.97, 0 invalid verdicts, 800 calls, $0.14.
- Training numbers (one epoch, 187 steps, LoRA r=16, alpha=32) come from the SFT runs' `summary.json`.
