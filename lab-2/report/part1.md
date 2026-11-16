# Part 1: selection variants of mamba-130m

## Pre-registered prediction (written before the zero-shot V1–V3 evaluation)

Variants (static = replaced by its calibration average, frozen input-independent):

| variant | Δ | B | C |
|---|---|---|---|
| V1 | static | static | static |
| V2 | selective | static | static |
| V3 | static | selective | static |
| V4 | selective | selective | selective (stock model) |

**Predicted zero-shot English perplexity ordering (best → worst): V4 < V2 < V3 < V1.**

Reasoning:
- V4 is the unmodified pretrained model, so it is best by construction.
- V1 removes all input dependence at once, so it is worst.
- Between V2 and V3: Δ controls how much the state is reset or kept per token (the gating role),
  and the Mamba paper names it the most important selective parameter. Without it, a pretrained
  model loses its ability to forget/ignore tokens. So keeping a selective Δ (V2) should hurt less than
  keeping only a selective B (V3). This matches the Table 7 ordering (V4 8.71 < V2 9.81 < V3 10.15 < V1 10.93),
  although Table 7 trains from scratch and here the surgery is zero-shot on a pretrained model.
- Expected gap size: the zero-shot gaps should be much larger than in Table 7 (perplexities far above 10.56
  for V1–V3), because the rest of the network was co-adapted to input-dependent Δ/B/C. Recovery training
  should close most of the gap and move the ordering closer to Table 7.
