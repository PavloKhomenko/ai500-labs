# AI500 Lab 1 Report

## Setup
Models (total / active params), hardware, training time per run, transformers / torch versions.

## Pre-registered predictions

## Part 1: Context extension
Loss-vs-position plot, passkey table, KV-cache prediction vs measured, which method we would ship.

## Part 2: MoE load balancing

### 2.1 Model used

| | EuroMoE-2.6B-A0.6B-2512 (`MixtralForCausalLM`) |
|---|---|
| Layers / hidden size | 24 / 1024; GQA with 8 query and 2 KV heads, head_dim 128; RoPE θ = 500k; context 4,096 |
| Experts | **64 routed, 0 shared, top-8**; SwiGLU experts with intermediate size 512 |
| Gating | softmax over 64 logits → top-8 → renormalize the 8 weights; router jitter 0 |
| Pretraining balancing (EuroMoE report) | aux load-balancing loss 0.01, computed over the global batch, + router z-loss 0.001 |
| Parameters (counted) | **2.61B total**: experts 2.42B, attention 62.9M, routers 1.57M |
| Active parameters | **0.50B** (0.63B if the tied LM head is counted separately, which matches the published "0.6B") |

### 2.2 Data

**Source.** One shard of Kobza (`Goader/kobza`, file `000_ukr_corpus_00000.parquet`, ~1M documents). All five Kobza sources are interleaved within it. Documents are read in file order until the train split holds 10M tokens.

**Preparation:**
1. NFC normalization; drop documents under 200 characters (359) and exact duplicates (4).
2. **Split by document hash** (md5 mod 1000 → 3% test, 1% val, rest train), so identical texts never cross splits.
3. EuroMoE tokenizer; `[BOS] + doc + [EOS]`, truncated to 2,048 tokens, packed into 1,024-token blocks.
4. **FLORES+** dev + devtest: 2,009 aligned uk/en sentences, padding masked. **None of them appear in train.**

| Split | Docs | Tokens | 1,024-token blocks | Used for |
|---|---|---|---|---|
| train | 14,424 | 10,000,213 (5.0M used) | 9,765 | gradients only |
| val | 139 | 105,197 | 102 | validation PPL every 100 steps |
| test | 442 | 321,938 | 314 | main table: PPL and routing |
| FLORES uk / en | 2,009 / 2,009 | 81,644 / 56,342 predicted | — | language comparison, English forgetting |


**Tokenizer fertility** (FLORES): uk 2.17 tokens/word (40.6 per sentence), en 1.32 (28.0 per sentence). The same content costs 1.45× more tokens in Ukrainian. We therefore compare languages by **total NLL per sentence**, not per-token perplexity. Kobza is built from web crawls that probably overlap EuroMoE's pretraining data; this affects all runs equally.

### 2.3 Instrumentation and metrics

**Instrumentation.** We replace each layer's router with a subclass of `MixtralTopKRouter`. It stores its logits and selected experts, and can optionally select experts with a bias. With the bias off, it reproduces the original model exactly (max abs logit difference **0.0**).

**Metrics.** For one layer, with $T$ tokens, $N = 64$ experts, $k = 8$, logits $\ell(x)$, probabilities $p(x) = \mathrm{softmax}(\ell(x))$ and selected set $S(x)$:

$$f_i = \frac{1}{Tk}\sum_x \mathbf 1[i\in S(x)],\qquad \mathrm{MaxVio} = \frac{\max_i c_i - \bar c}{\bar c} = N\max_i f_i - 1,\qquad H = -\frac{\sum_i f_i\log f_i}{\log N},\qquad \mathrm{LSE} = \frac1T\sum_x \log\sum_j e^{\ell_j(x)}$$

- MaxVio ranges from 0 (perfect balance) to $N/k - 1 = 7$ (one expert in the top-8 of *every* token).
- **Routing drift** is the mean Jaccard overlap between the base and the trained top-8 sets on the same tokens.
- **Language difference** is the Jensen–Shannon divergence between the uk and en load distributions per layer (base 2, range 0–1).

**Noise floor** (uniform random top-8 router, 1,000 draws at the same token count):

| Set | Tokens | MaxVio per layer, mean (p95) | Max over 24 layers, mean (p95) |
|---|---|---|---|
| training batch | 8,192 | 0.070 (0.093) | 0.099 (0.119) |
| UA test | 321,536 | 0.011 (0.014) | 0.015 (0.018) |
| FLORES uk / en | 83,653 / 58,351 | 0.022 / 0.026 | 0.030 / 0.038 |

### 2.4 Base model

- **UA test:** perplexity 7.516, mean MaxVio **5.70** (max over layers 6.96), entropy 0.881. That is about **500× the random-router floor** (0.011).
- **English is much more balanced:** on FLORES en, MaxVio is 1.52 and entropy 0.952, vs 5.46 and 0.878 on FLORES uk.
- **Per sentence, Ukrainian costs more:** total NLL 112.4 vs 88.7 nats.

| Layers | MaxVio | Entropy | Overloaded experts (> 2× mean) | Rarely used experts (< 0.5× mean) |
|---|---|---|---|---|
| early 0–7 | 4.79 | 0.909 | 4.8 | 17.1 |
| middle 8–15 | 5.67 | 0.926 | 5.1 | 14.5 |
| late 16–23 | **6.64** | 0.809 | 7.6 | **31.5** |

- Layers 0 and 20–23 have MaxVio of 6.7–7.0, close to the maximum of 7. One expert there is in the top-8 of **~99% of tokens**, so it acts as a de facto shared expert.
- Layers 6–7 are the most balanced (MaxVio 1.3–1.7).
- The uk/en difference is **U-shaped**: JS divergence is 0.18 in early layers, 0.10 in middle layers and 0.41 in late layers, with 0.56 at layer 0 and 0.80 at layer 23.

<p><img src="figures/part2/heatmap_base_ua_test.png" width="100%"></p>
<p><img src="figures/part2/heatmap_base_flores_uk.png" width="49%"> <img src="figures/part2/heatmap_base_flores_en.png" width="49%"></p>
<p><img src="figures/part2/base_maxvio_per_layer.png" width="49%"> <img src="figures/part2/base_js_uk_en.png" width="49%"></p>

### 2.5 Training

**What is trained.** Attention (q, k, v, o) and the routers are trained: 64.5M parameters, kept in fp32. The experts, embeddings and norms are frozen in fp16. Full fine-tuning with AdamW would need ~42 GB; this setup peaks at **9.4 GB** on a 15 GB T4.

**Objective.** $\mathcal L = \mathcal L_{\text{LM}} + \alpha\sum_{l=1}^{24}\mathcal L_{\text{aux}}^{(l)} + \beta\sum_{l=1}^{24}\mathcal L_{z}^{(l)}$, with these terms:

- **Auxiliary load-balancing loss** (Switch / GShard, in paper form):
  $$\mathcal L_{\text{aux}}^{(l)} = N\sum_{i=1}^{N} f_i\,P_i,\qquad P_i = \frac1T\sum_x p_i(x)$$
  - $f_i$ is a count and carries no gradient; the gradient reaches the router through $P_i$.
  - The loss equals **1 at perfect balance**. It is summed over layers, so each router receives the paper-form gradient α.
  - $f_i$ is computed from the expert counts of the whole optimizer step so far (all micro-batches), not one micro-batch. This is global-batch balancing, as in EuroMoE's pretraining (Qiu et al., 2025).
  - *Check against HF.* `load_balancing_loss_func` equals $k$ at balance and pools all layers into one distribution. On the same batch it gives 8.40, against our per-layer mean of 2.03, a ratio of 4.1 rather than 8. Pooling across layers hides per-layer imbalance, because different layers overload different experts.
  - *Assignment check.* With the coefficient passed to `from_pretrained`, the HF aux loss is non-zero, and $\mathcal L(0.02) - \mathcal L(0.01) = 0.01\cdot\mathcal L_{\text{aux}}$ (ratio 1.000002). Changing `config.router_aux_loss_coef` after loading leaves the loss unchanged (2.27755 in both cases). For training, the HF aux loss is disabled and only ours is used.
- **Router z-loss** (ST-MoE): $\mathcal L_z^{(l)} = \frac1T\sum_x\big(\log\sum_j e^{\ell_j(x)}\big)^2$.
- **Auxiliary-loss-free bias** (Wang et al., 2024):
  - Selection uses $S(x) = \mathrm{TopK}(\ell(x) + b,\,k)$.
  - Gate weights come from the **unbiased** softmax, $g_i = p_i/\sum_{j\in S(x)}p_j$.
  - After every optimizer step, $b_i \leftarrow b_i + \gamma\,\mathrm{sign}(\bar c - c_i)$, using that step's counts.
  - $b$ is a float32 buffer outside the optimizer, added to the logits.
  - γ = 1e-3, chosen before training. The bias can move at most γ·610 = 0.61 per run, about 10× the median gap between the 8th and 9th logit (0.061).

| Run | Strategy | α (aux) | β (z) | Bias | Router trained | Seeds |
|---|---|---|---|---|---|---|
| A | no balancing (control) | 0 | 0 | off | yes | 42, 43 |
| B | aux, weak | 1e-3 | 0 | off | yes | 42 |
| C | aux, strong | 1e-2 | 0 | off | yes | 42 |
| D | bias (aux-loss-free) | 0 | 0 | γ = 1e-3 | yes | 42, 43 |
| E | frozen router | 0 | 0 | off | **no** | 42 |
| F | aux + z-loss | 1e-3 | 1e-3 | off | yes | 42 |

**Constants, identical for every run** (`config.yaml` → `part2.train`):

| | |
|---|---|
| Sequence length | 1,024 |
| Batch | micro-batch 2 × gradient accumulation 4 = **8,192 tokens/step**. Micro-batch 4 runs out of memory. |
| Token budget | **4,997,120 tokens = 610 steps** |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0, gradient clipping 1.0 |
| Learning rate | 3e-5 (attention and router), 30 warmup steps, cosine decay to 3e-6 |
| Precision | fp16 autocast + GradScaler; gradient checkpointing with `use_reentrant=False` |

### 2.6 Training curves (W&B)

<p><img src="figures/part2/curves_maxvio_mean.png" width="49%"> <img src="figures/part2/curves_maxvio_max.png" width="49%"></p>
<p><img src="figures/part2/curves_val_ppl.png" width="49%"> <img src="figures/part2/curves_loss_lm.png" width="49%"></p>
<p><img src="figures/part2/curves_lse_mean.png" width="49%"> <img src="figures/part2/curves_bias_abs_mean.png" width="49%"></p>

### 2.7 Results

**Main table** (UA test; the drift columns give the Jaccard overlap with the base model's top-8, where 1 = unchanged):

| Run | UA PPL | Δ PPL vs A | Mean MaxVio | Max MaxVio | Entropy | LSE | Drift UA | Drift EN | FLORES-en PPL | NLL/sent uk | NLL/sent en | JS uk/en |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 7.516 | +3.4% | 5.70 | 6.96 | 0.881 | 0.014 | 1 | 1 | 23.65 | 112.4 | 88.7 | 0.229 |
| **A** none (s42 / s43) | 7.272 / 7.273 | — | 5.73 / 5.72 | 6.97 | 0.880 | 0.011 / 0.014 | 0.914 | 0.863 | 25.31 / 24.85 | 105.7 | 90.6 / 90.1 | 0.210 |
| B aux 1e-3 | 7.291 | +0.3% | 5.57 | 6.96 | 0.898 | 0.066 | 0.872 | 0.855 | 25.43 | 105.7 | 90.8 | 0.210 |
| C aux 1e-2 | 7.567 | +4.1% | 4.57 | 6.91 | 0.939 | 0.196 | 0.718 | 0.814 | 25.86 | 106.7 | 91.2 | 0.187 |
| **D** bias (s42 / s43) | 7.725 / 7.734 | **+6.3%** | **3.89 / 3.89** | 6.87 | **0.948** | 0.004 / 0.007 | **0.635** | 0.708 | 25.49 / 24.83 | 112.5 / 112.9 | 90.8 / 90.1 | 0.192 |
| E frozen router | 7.273 | +0.0% | 5.72 | 6.97 | 0.880 | 0.013 | 0.926 | 0.872 | 25.13 | 105.7 | 90.4 | 0.210 |
| F aux + z | 7.291 | +0.3% | 5.57 | 6.96 | 0.898 | 0.033 | 0.873 | 0.854 | 25.52 | 105.7 | 90.9 | 0.210 |
| random router, mean (p95) | — | — | 0.011 (0.014) | 0.015 (0.018) | — | — | — | — | — | — | — | — |

<p><img src="figures/part2/results_maxvio_vs_ppl.png" width="60%"></p>

*Balance vs quality: one dot per run, both seeds shown. The random-router floor is at x ≈ 0.01, on the left edge.*

<p><img src="figures/part2/results_maxvio_per_layer.png" width="49%"> <img src="figures/part2/results_drift_per_layer.png" width="49%"></p>
<p><img src="figures/part2/results_entropy_per_layer.png" width="49%"> <img src="figures/part2/results_js_uk_en_per_layer.png" width="49%"></p>

**Heatmaps after training** (UA test, same scale as the base heatmap above):

<p><img src="figures/part2/heatmap_none-s42.png" width="49%"> <img src="figures/part2/heatmap_frozen_router-s42.png" width="49%"></p>
<p><img src="figures/part2/heatmap_aux_1e-3-s42.png" width="49%"> <img src="figures/part2/heatmap_aux_z-s42.png" width="49%"></p>
<p><img src="figures/part2/heatmap_aux_1e-2-s42.png" width="49%"> <img src="figures/part2/heatmap_bias-s42.png" width="49%"></p>
<p><img src="figures/part2/heatmap_none-s43.png" width="49%"> <img src="figures/part2/heatmap_bias-s43.png" width="49%"></p>

**Comparisons**

1. **Base vs A (effect of training alone).** UA perplexity improves by 3.2% (7.516 → 7.272), and Ukrainian NLL per sentence drops from 112.4 to 105.7. Balance does **not** get worse: MaxVio goes from 5.70 to 5.72–5.73. English is partly forgotten: FLORES-en perplexity rises from 23.65 to 24.85–25.31.
2. **A vs B / C / D (effect of balancing, the main result).** Balance improves in the order B < C < D (−3%, −20%, −32% mean MaxVio; entropy 0.898 / 0.939 / 0.948), and perplexity gets worse in the same order (+0.3%, +4.1%, +6.3%). In both C and D, each 1.0 drop in MaxVio costs about 0.25 perplexity (C: −1.16 for +0.30; D: −1.84 for +0.46), so they sit on roughly one trade-off line. D's per-sentence NLL on FLORES uk (112.5–112.9) is back at the base level: in that metric, D cancelled the whole gain from training.
3. **C vs D.** D balances more (3.89 vs 4.57) and moves routing further from the base (Jaccard 0.635 vs 0.718). It doesn't inflate the router logits (LSE 0.004 vs 0.196), but it costs more perplexity (7.73 vs 7.57). C reached its balance in ~200 steps and then stayed put. D was still changing at the end (§2.6), so a longer run would give D more balance and probably more cost.
4. **A vs E (router learning vs attention-induced drift).** With the router frozen, 7.4% of top-8 choices still change (Jaccard 0.926), against 8.6% in A (0.914). Most of A's drift comes from attention changing the router's input, not from the router learning.
5. **UA vs EN.** Training brings uk and en routing slightly closer (JS 0.229 → 0.210); balancing brings them closer still (C 0.187, D 0.192). On English the strategies differ by less than the seed spread: FLORES-en perplexity is 24.83–25.86, while A's two seeds alone differ by 0.46.
6. **Seeds.** For A and D, the second seed changes UA perplexity by 0.001 and 0.009, and MaxVio by 0.010 and 0.002. The C and D effects (+0.30 and +0.46 perplexity) are 30–50× larger than the larger of these. B's +0.019 is only 2–19× the seed spread, so with one seed it is the least certain result.
7. **The z-loss** (F vs B) halves the router logsumexp (0.066 → 0.033) and changes nothing else measurable.
8. **The late layers stay unbalanced.** Max-over-layers MaxVio is 6.87–6.97 for *every* run. Balancing helps the early and middle layers most: under D, layer 6 drops to 0.4 and layers 2, 8 and 14 to 1.7–1.9. Layers 20–23 stay at 6.5–6.9 under every strategy.

**Which strategy balances best, and what does it cost?** The bias update balances best: mean MaxVio −32%, the highest entropy, and the same result on both seeds. It costs the most: +6.3% UA perplexity (worse than the untrained base), and its validation perplexity was still rising when training ended. The aux loss at 1e-2 gives two-thirds of that improvement for two-thirds of the cost. At 1e-3 it is nearly free and does almost nothing. With experts frozen and a 5M-token budget, **no strategy gave balance without a perplexity cost.** A likely reason, which we did not measure, is that balancing redirects Ukrainian tokens to experts that cannot adapt, because the experts are frozen.


### 2.9 Failure cases and negative results

- **D did not converge.** The sign rule moves each bias by exactly γ per step, so 610 steps cap the bias at 0.61. That cap was reached, and both MaxVio and the perplexity cost were still changing at the end. A longer run, or a larger γ, would be needed to see D's end state.
- **No strategy fixes the late-layer "shared" experts** (MaxVio ≈ 6.9 in layers 20–23).
- **The z-loss at 1e-3 had no effect** on balance or perplexity; it only reduced the logsumexp.
- **Limited replication.** B, C, E and F have one seed each, and the budget is only 5M tokens with frozen experts. The effects are smaller than in full continued pretraining would likely be, and the trade-off may look different when experts can adapt.
- **Inference precision.** The model was pretrained in bf16 and run in fp16 on the T4. No non-finite values appeared in any run.

## Part 3: MixLoRA (bonus)

## Prediction-error table

## Failure cases and negative results

## Contributions

## AI tools disclosure
