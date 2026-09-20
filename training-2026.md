---
layout: distill
title: "How Training Changed"
# permalink: /main/
description: "The recipe for training a frontier model in 2026 differs from LLaMA 3's in four ways that matter for systems: the matmuls run in fp8 (and increasingly fp4), the optimizer is usually Muon rather than Adam, the model is trained to predict more than one token at a time, and a growing share of the compute is spent after pretraining in reinforcement learning, where the expensive part is generating text rather than taking gradient steps. This section works out what each of these costs and what it does to the rooflines of Sections 5 and 6."
date: 2026-09-20
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Aghyad Deeb
    url: "https://github.com/aghyad-deeb"
    affiliations:
      name: "Independent; written with Claude"

section_number: 15

previous_section_url: "../attention"
previous_section_name: "Part 14: KV Cache"

next_section_url: ../applied-frontier
next_section_name: "Part 16: DeepSeek"

giscus_comments: false

bibliography: update.bib

toc:
  - name: "The 2026 Recipe at a Glance"
  - name: "Low Precision: fp8 Everywhere, fp4 Arriving"
    subsections:
    - name: "The fp8 recipe"
    - name: "What fp8 does to the rooflines"
    - name: "fp4 pretraining"
    - name: "Quantization-aware training for the weights you ship"
  - name: "Muon Replaces Adam"
    subsections:
    - name: "The update"
    - name: "What it costs"
    - name: "Distributing Muon"
  - name: "Multi-Token Prediction"
  - name: "Reinforcement Learning Is a Systems Problem"
    subsections:
    - name: "The loop"
    - name: "Where the time goes"
    - name: "Colocated or disaggregated"
    - name: "Asynchrony and the long tail"
    - name: "Two engines, one policy"
  - name: "Long-Context Training"
  - name: "What Should You Take Away from this Section?"
  - name: "Worked Problems"

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
---

_[Section 5](https://jax-ml.github.io/scaling-book/training) and [Section 6](https://jax-ml.github.io/scaling-book/applied-training) describe training as it was done for LLaMA 3: bf16 matmuls, AdamW, a next-token loss, and a parallelism plan chosen so that the weight gathers hide behind the FLOPs. All of that still applies. But if you read a 2026 technical report you'll find four things that weren't in the LLaMA 3 paper, and each of them changes a number somewhere in this book. Here they are, in the order they appear in a training run, followed by a short note on how runs reach a million tokens of context._

## The 2026 Recipe at a Glance

The labs disclose the following about how they train. Blank cells are things they didn't say.

| Model | Tokens | Batch | Seq. length | Matmul precision | Optimizer | Parallelism | Cluster |
| :---- | -----: | ----: | ----------: | :--------------- | :-------- | :---------- | :------ |
| LLaMA 3 405B<d-cite key="llama3"></d-cite> | 15T | 16M | 8k | bf16 | AdamW | TP8, CP16, PP16, DP | 16k H100 |
| DeepSeek-V3<d-cite key="DeepSeek3"></d-cite> | 14.8T | 63M | 4k | fp8 | AdamW | EP64, PP16, DP2 | 2,048 H800 |
| Kimi K2<d-cite key="kimik2"></d-cite> | 15.5T | 67M | 4k | bf16 (fp8 storage) | MuonClip | EP16, PP16, ZeRO-1 | H800 |
| GLM-4.5<d-cite key="glm45"></d-cite> | 23T | 16M to 64M | 4k | | Muon | | |
| GLM-5<d-cite key="glm5"></d-cite> | 28.5T | | 4k | | Muon | | |
| Nemotron 3 Ultra<d-cite key="nemotron3ultra"></d-cite> | 20T | | 4k | NVFP4 | | CP32, TP8, EP128, PP2 | GB200 |
| DeepSeek-V4-Pro<d-cite key="deepseekv4"></d-cite> | 33T | 94M | 4k to 1M | fp8 | Muon + AdamW | | |
| Kimi K3<d-cite key="kimik3"></d-cite> | | | 8k to 64k | fp8 activations | Per-head Muon | PP, EP, ZeRO-1, CP | |

Three things to notice before we get into the details. Batch sizes are four to six times larger than LLaMA 3's, which as [Section 5](https://jax-ml.github.io/scaling-book/training) explained is what lets you spread a run over more chips while staying compute-bound. Of the labs that say, only Kimi K2 kept bf16 matmuls, and only DeepSeek-V3 and Nemotron 3 Nano kept Adam. And the labs have become much less forthcoming about clusters and costs than they were in 2024: DeepSeek-V3 is still the only frontier open pretraining run with a published per-token GPU-hour figure and cost breakdown (2.79M H800-hours; Llama 4 and gpt-oss give totals of 7.38M and 2.1M H100-hours with no breakdown).

## Low Precision: fp8 Everywhere, fp4 Arriving

### The fp8 recipe

DeepSeek-V3 was the first open frontier model with a published end-to-end fp8 recipe<d-cite key="DeepSeek3"></d-cite>, and most labs have since adopted it. The important design decisions, in the order you would hit them implementing it:

* **Which matmuls.** All three matmuls of every linear layer (the forward pass, the activation gradient, and the weight gradient) take fp8 inputs and produce bf16 or fp32 outputs. Everything else stays in bf16 or fp32: embeddings, the output head, the router, normalization, and the attention score and value matmuls. Master weights and gradient accumulators are fp32. Adam moments are bf16.
* **Which fp8.** E4M3 for everything, including gradients. Earlier work used E5M2 for the backward pass to get more range; DeepSeek gets the range from scaling instead.
* **How to scale.** This is the part that matters. A single scale per tensor is too coarse: activations have outlier channels that would force everything else to underflow. Activations get one scale per 1x128 tile (per token, per 128 channels); weights get one per 128x128 block. The scales are computed on the fly from the max absolute value of each tile, not from a running history. The 1x128 tile for activations isn't negotiable: the report says block-wise 128x128 quantization of activation *gradients* made a 16B MoE diverge after 300B tokens.
* **How to accumulate.** The H800's fp8 tensor cores keep only about 14 bits of accumulator precision. DeepSeek accumulates 128 elements at a time on the tensor core, then adds the partial sum into fp32 registers on the CUDA cores, and overlaps the two by running two warpgroups so that one promotes while the other multiplies. The report says this keeps tensor-core utilization high; the roofline cost is a slightly lower issue rate per warpgroup, not idle tensor cores.
* **What it buys.** A 2x higher FLOPs ceiling, half the activation bytes in the matmuls, and fp8 dispatch in the MoE AllToAll (combine stays bf16, since it sums). The relative loss error against a bf16 baseline was under 0.25%.

Qwen3.5 describes essentially the same pipeline with "runtime monitoring preserving BF16 in sensitive layers", DeepSeek-V4 and V4.1 inherit it with small changes (a power-of-two scale format from V3.1 on, 32x32 blocks in V4.1), and Nemotron went to four bits, discussed next. Kimi K2 is the interesting holdout: it stores some activations in fp8 to save memory but does *not* compute in fp8, citing "potential risks of performance degradation that we observed during preliminary study". Its 15.5T-token run was bf16 matmuls with fp32 gradient accumulation.<d-footnote>The other thing K2 tells you is what that costs in memory: bf16 parameters plus an fp32 gradient accumulation buffer for a 1.04T-parameter model is about 6TB, spread over a 256-GPU model-parallel group, or 24GB per GPU before optimizer state and activations. The report puts total state at about 30GB per GPU.</d-footnote>

### What fp8 does to the rooflines

Every roofline in this book is a ratio of FLOPs to bytes. fp8 doubles $C$ and halves the bytes of whatever tensors are actually stored in fp8. So the effect on each roofline depends on whether the bytes in its denominator halved too:

| Roofline | Bytes in the denominator | Halved by fp8? | Threshold |
| :------- | :----------------------- | :------------: | :-------- |
| Decode, weight loading ([Section 7](https://jax-ml.github.io/scaling-book/inference)) | weights | yes if weights are fp8 | unchanged |
| Tensor parallelism ([Section 5](https://jax-ml.github.io/scaling-book/training)) | activations moved over ICI | yes if activations are sent in fp8 | unchanged |
| FSDP weight gathers | weights moved over ICI | only if you gather the fp8 copy | doubles otherwise |
| Expert-parallel AllToAll ([Section 13](../moe)) | dispatched tokens | dispatch yes, combine no | 1.5x tighter (bytes fall to 3/4, FLOPs double) |
| Attention ([Section 14](../attention)) | KV cache | yes if KV is fp8 | unchanged |
| Anything measured against HBM bandwidth with bf16 tensors | | no | doubles |

For a concrete example, take the H800 DeepSeek trained on. Its NVLink operational intensity goes from `990e12 / 200e9 = 4,950` in bf16 to 9,900 in fp8, its InfiniBand intensity from `990e12 / 50e9 = 19,800` to 39,600, and its HBM intensity from `990e12 / 3.35e12 = 296` to 591. On a GB200 the NVLink number goes from 2,780 to 5,560 and the HBM number from 312 to 625; on TPU7x the per-axis ICI intensity goes from 12,800 to 25,600. The GPU didn't change; the FLOPs ceiling did. If you run fp8 matmuls but keep bf16 activations in HBM, every "am I compute-bound" question in this book gets harder by 2x. So fp8 is a free 2x on the FLOPs and on nothing else; each roofline improves only where its bytes come down too.

Let's also redo the utilization estimate from Question 7 of [Section 4](https://jax-ml.github.io/scaling-book/transformers) with better numbers. DeepSeek-V3 spent 2,664K H800-hours on 14.8T tokens at 37B active parameters, or `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs. An H800 has the same tensor cores as an H100, 1.98e15 dense fp8 FLOPs/s, so the utilization was `3.3e24 / (2.664e6 * 3600 * 1.98e15) = 17%`. DeepSeek's later hardware paper gives the steady-state figure directly: 385 TFLOP/s per GPU counting attention FLOPs, which they report as 39% of *bf16* peak and which is 19% of the fp8 peak the matmuls actually ran at.<d-cite key="deepseek_isca"></d-cite> (Section 4 used 1.51e15, which is the PCIe H800's rate, and got 22%. Either way it is well under the 40 to 50% the book assumes for dense bf16 training on TPUs, and [Section 13](../moe) told you why: the cross-node expert AllToAll.)

<p markdown=1 class="takeaway">**Takeaway:** fp8 training uses E4M3 everywhere with per-1x128 activation scales and per-128x128 weight scales, fp32 accumulation every 128 elements, and bf16 or fp32 for everything that is not a linear layer. It doubles the FLOPs ceiling, so every roofline whose byte term did not also halve becomes 2x harder to satisfy. DeepSeek-V3's fp8 utilization was about 17%.</p>

### fp4 pretraining

The next halving is here, at least on NVIDIA hardware. Blackwell's tensor cores run fp4 at twice the fp8 rate on GB200 and three times on GB300<d-cite key="nvfp4"></d-cite>, and the format NVIDIA pushes is NVFP4: E2M1 elements (one sign bit, two exponent bits, one mantissa bit) with an E4M3 scale per 16 elements and an fp32 scale per tensor, 4.5 bits per value all in. The alternative, MXFP4, uses a power-of-two scale per 32 elements and is what OpenAI, DeepSeek and Kimi K3 use for shipped weights.

The pretraining recipe<d-cite key="nvfp4"></d-cite> is the fp8 recipe with three additions, each of which exists because four bits is not many:

1. Weights are scaled in 16x16 blocks and activations and gradients in 1x16 blocks, the same shape logic as DeepSeek's 128x128 and 1x128, just finer.
2. A random Hadamard transform is applied to the inputs of the weight-gradient matmul to spread outliers across a block before quantizing, and gradients are rounded stochastically rather than to nearest. Forward tensors are rounded to nearest; stochastic rounding in the forward pass hurt.
3. About 16% of the linear layers (the first two and last eight blocks of the 62-block test model) stay in higher precision, along with everything the fp8 recipe already kept in high precision.

With that, NVIDIA trained a 12B hybrid Mamba-Transformer on 10T tokens with a relative loss gap under 1% for most of the run, widening to about 1.5% near the end, and matching fp8 on downstream evaluations. The first frontier-scale models to use it are Nemotron 3 Super (120B, 25T tokens) and Nemotron 3 Ultra (550B, 20T tokens)<d-cite key="nemotron3super"></d-cite><d-cite key="nemotron3ultra"></d-cite>, which keep the final 15% of layers, the attention projections, the MoE latent projections, the multi-token-prediction heads (MTP, below) and the embeddings in bf16 or MXFP8; Ultra reports a training loss gap under 0.4%.

Be careful with the claim "trained in fp4", because it is often wrong. DeepSeek-V4 ships fp4 expert weights but was pretrained in fp8; the fp4 comes from quantization-aware training in the post-training stage, discussed next. Kimi K3 likewise. gpt-oss's expert weights were quantized to MXFP4 during post-training (OpenAI doesn't detail the method). As of this writing, the Nemotron 3 models are the only frontier-scale open models whose *pretraining* matmuls ran in four bits.

What does fp4 do to the rooflines? The same thing fp8 did, again. On a GB300, fp4 dense compute is 15e15 FLOPs/s against 8e12 bytes/s of HBM and 900e9 bytes/s of NVLink egress:

| GB300 NVL72, per GPU | bf16 | fp8 | fp4 |
| :------------------- | ---: | --: | --: |
| Dense FLOPs/s | 2.5e15 | 5e15 | 15e15 |
| HBM intensity $C / W_\text{hbm}$ | 312 | 625 | 1,875 |
| NVLink intensity $C / W_\text{egress}$ | 2,800 | 5,600 | 16,700 |
| Minimum expert width for compute-bound EP (fp8 dispatch, [Section 13](../moe)) | 1,400 | 2,800 | 8,300 |

At fp4 compute, the NVL72's NVLink sits where InfiniBand sat for fp8 on the H800: a 2,048-wide expert's AllToAll is four times communication-bound even inside the rack. Rubin (35e15 dense fp4 against 1.5e12 egress) is worse. If fp4 training takes hold, expect experts to get wider, or the AllToAll to be hidden behind attention as [Section 13](../moe) described, or both. Each chip generation since v5p has raised $\alpha$; low precision raises it again without touching a single link.

### Quantization-aware training for the weights you ship

What became standard in 2026 is not fp4 pretraining but fp8 pretraining followed by quantization-aware training (QAT): during post-training the expert weights are quantized to four bits *inside the training loop*, so that the model learns to live with the rounding, and those are the weights that ship.

| Model | Shipped expert weights | Method | Result |
| :---- | :--------------------- | :----- | :----- |
| Kimi K2 Thinking<d-cite key="kimik2thinking"></d-cite> | int4 (group 32) | QAT in post-training | 594GB vs 1,029GB; about 2x generation speed |
| GLM-5<d-cite key="glm5"></d-cite> | int4 | QAT in supervised fine-tuning (SFT), bit-identical train and inference kernels | |
| DeepSeek-V4<d-cite key="deepseekv4"></d-cite> | MXFP4 (plus the sparse-attention indexer) | QAT in post-training; native fp4 in RL rollouts | 865GB vs 1,606GB |
| Kimi K3<d-cite key="kimik3"></d-cite> | MXFP4 weights, MXFP8 activations | QAT through SFT and RL | 1,561GB for 2.78T parameters |

The sizes in the last column are the checkpoints as shipped, with attention, shared experts and the vision tower in bf16. With everything but the experts in fp8, which is how Sections 13, 14 and 16 count serving memory, K3 is 1.42TB and V4-Pro 0.83TB.

| gpt-oss<d-cite key="gptoss"></d-cite> | MXFP4 | quantized during post-training | fits an 80GB GPU |

DeepSeek's implementation has a neat trick. The fp32 master weights are quantized to MXFP4, then *dequantized to fp8* for the actual matmul. Because E4M3 has two more exponent bits than E2M1, the fp4-to-fp8 conversion is exact, so the existing fp8 training kernels run unmodified and the only new code is the quantizer and a straight-through estimator in the backward pass. The forward pass sees exactly the weights that will be served.

That last point is the systems reason this matters. [Section 13](../moe) showed that four-bit expert weights halve the chips needed to hold a model and double the batch you can fit; that is why it is worth doing. But there is a second reason that appears once you get to reinforcement learning: if the policy's rollouts are generated by an inference engine using the quantized weights while the trainer uses the full-precision ones, the two disagree, and RL with a mismatched sampler is off-policy. Kimi K3's report says QAT was chosen so that "rollout and training share the same quantization scheme", and DeepSeek-V4 runs its RL rollouts on the native fp4 weights for the same reason. (GLM-5's "bitwise-identical" int4 kernel serves a narrower purpose: making its SFT-stage QAT match the shipped weights; its RL rollouts run in fp8.) We come back to this below.

<p markdown=1 class="takeaway">**Takeaway:** fp4 pretraining works (NVFP4, with about 15% of layers kept in higher precision and a loss gap under 1%) but is so far confined to NVIDIA's own models. The common 2026 pattern is fp8 pretraining followed by quantization-aware post-training to 4-bit expert weights, which halves serving memory and, if the RL rollouts also use the 4-bit weights, removes a train-serve mismatch.</p>

## Muon Replaces Adam

### The update

Adam has been the default optimizer for a decade. In 2025 it lost that position, at least among the labs training open MoEs. Kimi K2, Kimi K3, GLM-4.5 and GLM-5, DeepSeek-V4 and V4.1, and Qwen's experimental Qwen3.8-Next all use Muon<d-cite key="muon"></d-cite><d-cite key="moonlight"></d-cite> for their matrix parameters. DeepSeek-V3, Nemotron 3 Nano and Llama used AdamW; Qwen and the larger Nemotrons don't say.

Muon is SGD with momentum plus one extra step. For a weight matrix $W$ with gradient $G$:

$$M_t = \mu M_{t-1} + G_t, \qquad O_t = \text{NewtonSchulz}(M_t) \cdot \sqrt{\max(n, m)} \cdot 0.2, \qquad W_t = W_{t-1} - \eta \left( O_t + \lambda W_{t-1} \right)$$

The Newton-Schulz step orthogonalizes the momentum matrix: it replaces $M$ with the nearest matrix whose singular values are all one, so every direction in the update gets the same step size regardless of how large its gradient was. It is computed with five iterations of a fixed quintic polynomial in $X X^\top$, entirely in bf16 matmuls.<d-footnote>Newton-Schulz iteration is $X \leftarrow a X + b (X X^\top) X + c (X X^\top)^2 X$ with $(a, b, c) = (3.4445, -4.7750, 2.0315)$, five times, on the momentum matrix scaled to unit norm. DeepSeek-V4 uses a "hybrid" ten iterations, eight with those coefficients and two with $(2, -1.5, 0.5)$ to land exactly on singular value one. The $\sqrt{\max(n, m)} \cdot 0.2$ factor matches the update's root-mean-square to what Adam would produce, so Adam learning rates transfer; DeepSeek uses 0.18.</d-footnote> The claimed benefit is token efficiency: Moonlight's scaling-law fits on sub-2B models found Muon needs about half the training FLOPs of AdamW to reach the same loss, and Kimi K2's 15.5T-token run "with zero loss spikes" is the largest published evidence that it holds up at scale (as evidence of stability, that is; there is no AdamW twin of K2 to compare against). Vectors, biases, norms, embeddings and the output head still use AdamW.

Muon has a stability problem of its own: attention logits explode, more often than with Adam. Where keys are materialized per head, normalizing queries and keys fixes it (GLM-4.5 on grouped-query attention; DeepSeek-V4 normalizes queries and KV entries directly). For MLA the keys are never materialized per head at inference time, so Kimi's answer is QK-Clip: after each step, for any head whose maximum attention logit $\ell_\text{max}$ exceeded a threshold $\tau = 100$, scale that head's query and key projection weights down by $\sqrt{\tau / \ell_\text{max}}$. In K2, 12.7% of heads triggered the clip at some point in the first 70,000 steps, after which none did; the mechanism switches itself off. Kimi K3, GLM-5 and DeepSeek-V4.1 also orthogonalize each attention head's block of the projection matrices separately ("per-head Muon" or "Muon Split"), which GLM reports is enough to keep logits stable with no clipping at all; K3 keeps the clip as well.

### What it costs

Very little, which is the reason it was adopted so fast.

**FLOPs.** Newton-Schulz on an $m \times n$ matrix costs about $4 m n^2 + 2 n^3$ per iteration for $m \geq n$. For DeepSeek-V3's 44,718 expert matrices of shape 7168x2048 that is `5 * (4 * 7168 * 2048^2 + 2 * 2048^3) = 6.9e11` FLOPs each, `3.1e16` in total per step, plus a rounding error for attention and dense layers. A training step on a 63M-token batch is `6 * 37e9 * 63e6 = 1.4e19` FLOPs. The optimizer is **0.23%** of the step, or 0.46% with DeepSeek-V4's ten iterations. Keller Jordan's original estimate for a 405B-scale model was 0.5%. It's also embarrassingly parallel across matrices, and MoEs make it friendlier to the hardware still: hundreds of experts per layer share a shape, so their Newton-Schulz iterations batch into one large matmul.

**Memory.** Adam keeps two moments per parameter, Muon one, so with both in fp32 the optimizer state goes from 8 bytes per parameter to 4, and [Section 5](https://jax-ml.github.io/scaling-book/training)'s rule that model plus optimizer state is 10 bytes per parameter (2 for bf16 weights plus 8) becomes 6. For a 1T-parameter model that's 4TB saved, before ZeRO-1 shards it across data parallelism. The Moonlight paper puts it plainly: "the additional memory used by the Muon optimizer is half of Distributed AdamW." (DeepSeek-V3 kept its Adam moments in bf16, 4 bytes in all, so against that recipe a Muon momentum in fp32 saves nothing; the comparison above is Moonlight's like-for-like one.)

### Distributing Muon

Muon's one real cost is in distributing it. Adam's update is elementwise, which is why ZeRO-1 can shard the optimizer state arbitrarily: every rank updates its own slice of every tensor. Newton-Schulz needs the *whole* matrix. So under ZeRO, each rank must gather the full momentum matrix for the parameters it owns, orthogonalize, and scatter the result back.

Moonlight's distributed algorithm does a ReduceScatter of the fp32 gradient (4 bytes per parameter), gathers the sharded momentum in bf16 for the Newton-Schulz step (2 bytes), and AllGathers the updated fp32 parameters (4 bytes), for 10 bytes per parameter per step against Adam's 8: at most 1.25x the communication, once per step, not per layer, so it doesn't touch the per-layer rooflines of [Section 5](https://jax-ml.github.io/scaling-book/training). GLM-5 and Kimi K3 both point out that the naive version gathers the *entire* parameter buffer on every rank; GLM-5 restricts the AllGather to the shards each rank owns, and K3 goes further and fetches them point-to-point. DeepSeek-V4 caps the width of its ZeRO group so that no matrix is split across too many ranks, packs matrices onto ranks with a knapsack heuristic, and when data parallelism exceeds that width simply recomputes the update redundantly in the extra groups. It also rounds MoE gradients to bf16 stochastically before the data-parallel reduce, halving that traffic.

<p markdown=1 class="takeaway">**Takeaway:** Muon orthogonalizes the momentum matrix with five bf16 Newton-Schulz iterations. It costs about 0.2 to 0.5% of training FLOPs, halves optimizer memory (6 bytes per parameter with weights instead of 10), adds at most 25% to once-per-step optimizer communication because the whole matrix is needed, and in exchange trains a given loss in roughly half the tokens at small scale. QK-Clip or per-head orthogonalization is needed to keep attention logits from exploding.</p>

## Multi-Token Prediction

Nearly every DeepSeek, GLM, Qwen, Nemotron and MiniMax flagship since 2025 trains with a multi-token prediction (MTP) objective<d-cite key="mtp"></d-cite> (Kimi K2 didn't; Qwen added it with Qwen3-Next; DeepSeek-V4.1-Flash dropped it in favor of a drafter trained afterwards): in addition to predicting token $t+1$ from the trunk's representation of token $t$, a small extra module predicts token $t+2$. DeepSeek's version<d-cite key="DeepSeek3"></d-cite> is one additional Transformer block that takes the trunk's final hidden state for position $t$, concatenates the embedding of the (ground-truth) token $t+1$, projects back to $D$, runs the block, and applies the shared output head. Its loss is weighted 0.3 for most of training and 0.1 at the end. Depth one is the DeepSeek, Kimi and GLM-4.5 default; MiniMax, LongCat, MiMo and Nemotron train 2 to 7 modules, and GLM-5 trains three steps with shared parameters so that the module is used at inference the way it was trained.

**Cost in training.** One extra block on a 61-block model is 1.6% more block FLOPs, and the extra pass through the 129,280-wide output head is another 2.5%: **about 4% of forward FLOPs** for DeepSeek-V3, or about 14B parameters in the checkpoint that aren't used for next-token prediction at all. The labs report it improves the main loss, and take the 4%.

**Payoff at inference.** The MTP block is a draft model that shares the trunk's computation, the embedded drafter head that [Section 7](https://jax-ml.github.io/scaling-book/inference)'s Appendix D described (it already cites DeepSeek-V3 for it). What's new is how well it works at scale. Decode is memory-bound, so verifying two or three drafted tokens in one forward pass costs almost the same bytes as generating one. What you get:

| System | Draft tokens per step | Measured acceptance length | Speedup |
| :----- | --------------------: | -------------------------: | :------ |
| DeepSeek-V3 paper | 1 | 85 to 90% second-token acceptance | 1.8x tokens/s |
| SGLang, V3 on H200, 2 requests per GPU | 3 / 4 | 2.18 / 2.44 | +60% throughput |
| SGLang, V3, 128 requests per GPU | 1 | | +14% |
| GLM-5 vs DeepSeek-V3.2, 4 steps | 4 | 2.76 vs 2.55 | |
| GLM-5.3, EAGLE-style draft from the MTP head | 6 (5 steps) | 3.5 | 252 to 920 tok/s per H200 |
| DeepSeek-V4-Flash with DSpark (5 parallel draft positions) | 5 | | +60 to 85% per-user speed over depth-1 MTP (Pro: +57 to 78%) |

The pattern is familiar from [Section 7](https://jax-ml.github.io/scaling-book/inference): speculation buys the most when the batch is small and the step is bandwidth-bound (+60% at two requests per GPU), and least when the batch is large and the MoE FLOPs are already busy (+14% at 128). The place it matters most in 2026 is one we have not discussed yet: RL rollouts, where a few very long sequences finish last and everyone waits. GLM-5's report says MTP "provides disproportionately large benefits on the long tail" there.

<p markdown=1 class="takeaway">**Takeaway:** Multi-token prediction adds about 4% to training FLOPs and gives the served model a built-in speculative decoder with acceptance lengths of 2.2 to 3.5 tokens, worth 1.6 to 1.8x in tokens per second at small batch and much less at large batch.</p>

## Reinforcement Learning Is a Systems Problem

Of the four changes in this section, this is the only one with no analog in the original book. In LLaMA 3, post-training was a small fraction of compute: some supervised fine-tuning, some preference optimization on pairs. DeepSeek-R1<d-cite key="deepseekr1"></d-cite> showed in January 2025 that running reinforcement learning against verifiable rewards (does the code pass the tests, is the answer to the math problem right) for tens of thousands of steps produces models that reason at length, and every frontier model since has a "thinking" mode trained this way. DeepSeek-V3.2 says its RL budget exceeded 10% of pretraining compute (up from 5.5% for R1; no other lab publishes the ratio). And RL compute isn't spent the way pretraining compute is.

### The loop

One RL step, in the GRPO (group relative policy optimization)<d-cite key="grpo"></d-cite> form that most labs use:

1. Take a batch of prompts. For each, sample $G$ responses from the current policy (R1 uses 16; this $G$ is GRPO's letter and has nothing to do with the group factor of [Section 4](https://jax-ml.github.io/scaling-book/transformers)). This is *generation*: autoregressive decode, thousands to tens of thousands of tokens per response.
2. Score each response with a reward (a verifier, a test harness, occasionally a reward model). Compute each response's advantage as its reward minus the mean over its group, divided by the group's standard deviation.
3. Take a few gradient steps on the policy with a clipped policy-gradient loss weighted by the advantages. This is *training*: a forward and backward pass over the generated tokens, $6 N_\text{active}$ FLOPs per token, plus a forward pass of the frozen reference model for the KL term if the recipe keeps one (R1 does, at $2 N_\text{active}$ more; GLM-5, Magistral and DAPO drop it), with the usual parallelism.
4. Copy the new weights to whatever is doing the generating, and repeat.

{% include figure.liquid path="assets/img/rl-loop.svg" class="img-fluid" caption="<b>Figure:</b> one reinforcement-learning step. Generation dominates the wall clock because it is memory-bound decode and because the step cannot finish before its longest response does. The two placements of trainer and generator, and the two fixes for the tail, are discussed below." %}

R1-Zero's numbers make the scale concrete: 32 questions times 16 samples is a `32 * 16 = 512`-response training batch; a rollout of 8,192 responses is `8192 / 512 = 16` mini-batches; responses run up to 32,768 tokens (65,536 later in the run); 10,400 steps, on 512 H800s for about 198 hours. That's 101K GPU-hours, and the whole R1 pipeline was 147K GPU-hours, or 5.5% of V3's pretraining. MiniMax-M1's RL ran three weeks on 512 H800s. These aren't small runs, and by 2026 they are larger: the fraction is over 10% for V3.2, and Kimi K3 trains RL at a million tokens of context.

### Where the time goes

Almost all of it goes to step 1. The published breakdowns agree: generation is 60 to 90% of RL wall-clock (up to 81% in the HybridFlow paper's baseline<d-cite key="verl"></d-cite>, over 90% in ByteDance's measurements, 65 to 72% in NVIDIA's on GB200). That isn't because generation has more FLOPs. Let's count for an R1-style rollout of 8,192 responses averaging 10,000 tokens, 82M tokens in total, on a DeepSeek-V3-class model:

* **Generation FLOPs:** `2 * 37e9 * 82e6 = 6e18`. **Training FLOPs:** three times that, `1.8e19`.
* **Training time** at 30% of 512 H800s' fp8 peak: `1.8e19 / (512 * 1.98e15 * 0.3) = 60 s`.
* **Generation time** if every GPU decoded at DeepSeek's production rate of 1,850 tokens/s: `82e6 / (512 * 1850) = 86 s`.

So even in the best case generation takes longer than training with a third of the FLOPs, because decode is memory-bound: those 86 seconds are `6e18 / (86 * 512 * 1.98e15) = 7%` of fp8 peak, a quarter of the 30% we assumed for the training pass. And the best case isn't what happens. Response lengths are wildly skewed (APRIL measures a standard deviation of 4,000 to 4,500 tokens with a tail near the maximum), and a synchronous system can't train until the *last* response finishes. A 65,536-token response at 20 tokens per second takes **55 minutes**. Meanwhile the batch drains, the per-GPU decode batch shrinks, and the memory-bound step gets no more efficient. This is the straggler problem, and every 2026 RL system is a design for dealing with it.

There's a memory cost to notice too. Those 82M in-flight tokens have KV caches: `82e6 * 35e3 = 2.9TB` in fp8 for a V3-class model at the peak, if every sequence were resident at once, 6GB per GPU across 512. At 64k-token responses, or a million-token agent context, it is an order of magnitude more, and it competes with the trainer's weights and optimizer state for HBM. Kimi K3 offloads training state to NVMe during rollouts and keeps an external KV pool in host DRAM for exactly this reason.

### Colocated or disaggregated

There are two ways to arrange the trainer and the generator, and the field uses both.

**Colocated:** the same GPUs alternate between training and generating. Kimi K2 and K3, DeepSeek-V4.1 and slime's synchronous mode work this way. During generation the trainer's weights and optimizer state are offloaded to host memory (K2) or NVMe (K3) and the GPUs run an inference engine. The price is the switch: the inference engine needs the new weights in its own layout. K2's "checkpoint engine" broadcasts the full 1.04TB of fp8 parameters to every node and lets each inference rank pull its shard, in 16 seconds on 256 H20s, which is `1.04e12 / 16 = 65GB/s` into every node (the report's design target was under 30 seconds). Kimi k1.5 quotes under a minute to switch from training to inference and about ten seconds back. The benefit is that no GPU sits idle waiting for the other side, which is why K3's report says colocation keeps a 1M-context RL experiment "within a few hundred GPUs".

**Disaggregated:** separate pools for training and generation, with weights pushed across the network every few gradient steps. AReaL<d-cite key="areal"></d-cite>, PipelineRL, Magistral<d-cite key="magistral"></d-cite>, GLM's agentic RL and MiniMax-M2 do this. AReaL found that giving three quarters of the GPUs to generation beat an even split, which tells you the ratio of the two workloads. Magistral broadcasts weights GPU-to-GPU in under five seconds and never lets the generators wait for the trainers. The benefit is that the two sides can run different software, different parallelism, even different hardware (slime notes its rollout engines "can even run on different GPU models or vendors").

Which is better depends on what is being generated. slime<d-cite key="slime"></d-cite>, the framework behind every GLM model since GLM-4.5, uses colocated synchronous training for reasoning RL and disaggregated asynchronous training for agentic RL, where a rollout may involve minutes of tool execution in a sandbox during which a colocated GPU would be idle.

### Asynchrony and the long tail

Two ideas remove the straggler problem, and most 2026 systems use both.

**Partial rollouts** (Kimi k1.5, K2, K3; APRIL; DeepSeek-V4's generation service). Stop the generation phase when a fraction $\lambda$ of responses have finished, save the unfinished ones (with their KV caches), train on what completed, and resume the rest in the next iteration under the new weights. The resumed tokens were sampled from an older policy, so the loss must tolerate some staleness, which brings us to:

**Asynchronous training** (AReaL, PipelineRL, Magistral, slime, DeepSeek-V4.1, Qwen3.5). Let generation run continuously and train on whatever has finished, accepting that a response may have been sampled by a policy several steps old. AReaL measured how much staleness the loss can absorb: up to 8 policy versions costs nothing measurable, as long as the loss uses the *behavior* policy's probabilities for importance weighting. PipelineRL goes further and updates the generators' weights *in flight*, mid-sequence, keeping the KV cache; Magistral found recomputing the cache after a weight update was unnecessary. AReaL reports 2.6x higher training throughput than a synchronous system; DeepSeek-V4.1 says "asynchronous training is now enabled for nearly all our RL" tasks.

One subtlety that DeepSeek-V4 documents: if you cancel and restart unfinished responses instead of resuming them, short responses are more likely to survive, and the model learns to be terse. Their generation service keeps a token-level write-ahead log so a preempted request resumes exactly where it stopped.

### Two engines, one policy

The last problem took the longest to notice. The trainer (Megatron or FSDP, bf16 or fp8) and the generator (vLLM or SGLang, often with fused kernels and quantized weights) compute slightly different probabilities for the same token. The RL loss assumes the sampled tokens came from the policy it is updating. They did not, quite, so nominally on-policy RL is off-policy, and it can diverge.<d-cite key="offpolicy_secret"></d-cite> MiniMax traced their instability to the output head, where activations are large; running it in fp32 raised the correlation between trainer and generator probabilities from about 0.9 to 0.99, and Meta's ScaleRL study found the same fp32-head fix worth a sizeable gain in final reward.

The fixes in production 2026 systems, in increasing order of thoroughness:

* **Importance weighting with the generator's probabilities.** Have the inference engine return the log-probabilities it actually sampled with, and use them as the behavior policy in the loss, with truncation or masking of tokens whose ratio is too far from one (truncated importance sampling; DeepSeek's off-policy sequence masking; GLM's IcePop; GSPO's sequence-level ratios).
* **Routing replay.** For MoEs, force the trainer to use the exact experts the generator chose (DeepSeek's "Keep Routing", Qwen3.5's "rollout router replay"). Otherwise a small probability difference flips a top-$k$ decision and the two engines run different networks. GLM-5 found the same thing for sparse attention: a nondeterministic top-$k$ in the indexer "caused drastic performance degradation during RL after only a few steps".
* **Identical numerics.** Train with the quantization the generator uses (Kimi K3's MXFP4 QAT through RL, DeepSeek-V4's native fp4 rollouts).

Each of these is a systems decision with a modeling consequence.

<p markdown=1 class="takeaway">**Takeaway:** RL post-training passed 10% of pretraining compute at DeepSeek by V3.2, and 60 to 90% of its wall-clock is memory-bound generation whose step time is set by the longest response. The 2026 systems answer with partial rollouts, asynchronous training tolerant of several steps of staleness, weight synchronization in seconds for trillion-parameter policies, and numerics deliberately matched between the training and inference engines.</p>

## Long-Context Training

[Section 14](../attention) explained why attention is affordable at long context in 2026 architectures. Here's how the training runs actually reach a million tokens: briefly, and at the end.

| Model | Schedule | Tokens at long length |
| :---- | :------- | :-------------------- |
| LLaMA 3 405B | 8k, then six stages to 128k | about 800B |
| DeepSeek-V3 | 4k, then 32k, then 128k | 2 phases of 1,000 steps (about 125B); 119K of 2.79M GPU-hours |
| Kimi K2 | 4k throughout; anneal 400B at 4k + 60B at 32k; YaRN to 128k | 60B |
| GLM-5 | 4k, then 32k, 128k, 200k | 1T, 500B, 50B |
| DeepSeek-V4 | 4k, 16k, 64k, then 1M | dense attention for the first 1T tokens; sparse from 64k |
| DeepSeek-V4.1 | 64k from the start (sparse), 1M from 34T of 45T tokens | 11T |
| Kimi K3 | 8k, 64k in pretraining; 256k to 1M in the cooldown | undisclosed |
| Nemotron 3 Ultra | 4k, then 1M | 33B |

Two patterns. First, until DeepSeek-V4.1 nobody trained at long length for long: a few percent of tokens, at the end, with YaRN<d-cite key="yarn"></d-cite> to stretch the position encoding. DeepSeek-V3's context extension was 4% of its GPU-hours. Second, the labs that designed for long context from the start (V4.1 at 64k from token one, K3 at 8k to 64k with no position encoding in its full-attention layers) can afford it only because their attention is sparse or linear, per [Section 14](../attention).

Context parallelism, which [Section 5](https://jax-ml.github.io/scaling-book/training) mentioned in a note, is now routine. LLaMA 3 used 16-way CP for 128k with an AllGather of keys and values rather than ring attention, on the argument that with GQA "the time complexity of attention computation is an order of magnitude larger than all-gather", $O(S^2)$ against $O(S)$; that argument only gets stronger with MLA's small latent. Kimi K3 needed a separate scheme for its linear layers, which computes each shard's state transition locally and then reconciles the shards with one fixed-size AllGather and a prefix scan rather than a sequential hand-off, and DeepSeek-V4 needed a two-stage exchange so that its 4:1 and 128:1 token compression works across shard boundaries. Nemotron 3 Ultra's million-token phase ran with 32-way context, 8-way tensor, 128-way expert and 2-way pipeline parallelism on GB200s, which is a nice summary of how many axes a 2026 training job has.

<p markdown=1 class="takeaway">**Takeaway:** Long context is a short phase at the end of pretraining, a few percent of tokens with YaRN to stretch the positions, unless the attention is sparse or linear enough to train at 64k or more from the start. Context parallelism is routine, and for linear-attention layers it is a state hand-off rather than a KV AllGather.</p>

## What Should You Take Away from this Section?

* fp8 training is the standard: E4M3, 1x128 activation and 128x128 weight scales, fp32 accumulation, high precision for embeddings, heads, routers, norms and attention. It doubles the FLOPs ceiling, and every roofline whose byte term didn't also halve gets 2x harder. DeepSeek-V3 reached about 17% of fp8 peak.

* fp4 pretraining exists (NVFP4, Nemotron 3) but the common pattern is fp8 pretraining plus quantization-aware post-training to 4-bit expert weights, which halves serving memory and can be matched exactly by the RL sampler.

* Muon costs 0.2 to 0.5% of FLOPs and half of Adam's optimizer memory, needs whole-matrix gathers once per step (at most 1.25x Adam's optimizer communication), and needs QK-Norm, QK-Clip or per-head orthogonalization to keep attention logits from exploding. It's now the majority choice among open MoE trainers.

* Multi-token prediction adds about 4% to forward FLOPs and provides a speculative decoder with acceptance lengths of 2.2 to 3.5, worth 1.6 to 1.8x at small batch.

* RL passed 10% of pretraining compute at DeepSeek and is dominated by memory-bound generation with a long tail of response lengths. Partial rollouts, asynchronous training (staleness up to 8 steps is fine), fast weight broadcast, and numerically matched engines are the standard toolkit.

* Long context is trained at the end, on a few percent of tokens, with YaRN; only models with sparse or linear attention train at 64k or more from the start.

## Worked Problems

**Question 1 [fp8 accounting]:** DeepSeek-V3 reports 180K H800-hours per trillion tokens. Using 1.98e15 fp8 FLOPs/s per GPU and 37B active parameters, what utilization does that imply? If you add the 4% MTP overhead and count attention score FLOPs at 4k context (about 14% of the matmul FLOPs for this model, from [Section 14](../attention)'s crossover of 19k tokens), what is the utilization of *executed* FLOPs?

{% details Click here for the answer. %}

Per trillion tokens the useful FLOPs are `6 * 37e9 * 1e12 = 2.2e23`. The available FLOPs are `180e3 * 3600 * 1.98e15 = 1.28e24`. Utilization is **17%**. Adding 18% for MTP and attention brings the executed FLOPs to `2.6e23` and the utilization to about 20%. Whichever way you count it, the run spent most of its time on something other than matmuls. [Section 13](../moe) identified the cross-node AllToAll as the main culprit; the fp32 accumulation dance and the pipeline bubbles of a 16-stage pipeline are the others.

{% enddetails %}

**Question 2 [Muon's bill for Kimi K2]:** Kimi K2 has 384 routed experts plus 1 shared expert per layer, each three 7168x2048 matrices, in 60 MoE layers, and was trained with a 67M-token batch and 32B active parameters. Estimate the Newton-Schulz FLOPs per step as a fraction of training FLOPs, and the optimizer-state memory saved relative to AdamW.

{% details Click here for the answer. %}

Expert matrices: `60 * (384 + 1) * 3 = 69,300`, each costing `5 * (4 * 7168 * 2048^2 + 2 * 2048^3) = 6.9e11` FLOPs to orthogonalize, so `4.8e16` per step. Training FLOPs per step are `6 * 32e9 * 67e6 = 1.3e19`. The optimizer is **0.37%** of the step. Memory: AdamW would keep 8 bytes of fp32 moments per parameter, `8.3TB` for 1.04T parameters; Muon keeps 4, `4.2TB`. Spread over K2's 256-GPU model-parallel group that's 16GB per GPU saved before ZeRO-1 shards the optimizer state across data parallelism (the report says it does, or offloads it to the host on small clusters), against the roughly 30GB per GPU of total state that the report quotes.

{% enddetails %}

**Question 3 [what fp8 does to the book's constants]:** [Section 12](https://jax-ml.github.io/scaling-book/gpus) says FSDP across H100 nodes is compute-bound above 2,475 tokens per GPU and tensor parallelism inside a node is compute-bound below $Y = F W_\text{nvlink} / C$; [Section 5](https://jax-ml.github.io/scaling-book/training) says 850 tokens per chip and $Y = 3F / 2550$ on TPU v5p. Redo both for fp8 matmuls, (a) on H100 nodes and on a GB200 NVL72, if weights and activations are gathered in bf16, (b) the same if they are gathered in fp8, and (c) on TPU7x.

{% details Click here for the answer. %}

(a) On an H100 with fp8 matmuls $C = 1.98e15$, so the cross-node FSDP threshold is `1.98e15 / 400e9 = 4,950` tokens per GPU (twice the book's 2,475) and TP is compute-bound below `Y = 28672 * 450e9 / 1.98e15 = 6.5`-way for LLaMA 3-70B's $F = 28672$. Eight-way TP inside the node, the book's default, is now 1.2x communication-bound. On a GB200 NVL72 the NVLink intensity in fp8 is `5e15 / 900e9 = 5,560`, so TP is compute-bound below `28672 / 5560 = 5.2`-way (the two-matmul convention, as in Section 12), and FSDP across racks needs `5e15 / 3.6e12 = 1,390` tokens per GPU.

(b) Gathering fp8 tensors halves the bytes, so every threshold halves: 2,475 tokens per GPU and 13-way TP on H100, 10-way TP and 694 tokens per GPU on GB200. Sending activations in fp8 is what keeps 8-way TP alive in the fp8 era, and Nemotron 3 Ultra in the table above runs exactly that, TP8 inside its NVLink domain.

(c) TPU7x in fp8 has $\alpha = 4.61e15 / 1.8e11 = 25{,}600$ per axis. Gathering bf16 tensors, FSDP needs `25600 / 3 = 8,500` tokens per chip (ten times v5p's 850) and TP is compute-bound below `Y = 3 * 28672 / 25600 = 3.4`-way: two-way, at most. With fp8 gathers, 4,300 tokens per chip and 6.7-way TP, still five times tighter than v5p in bf16. This is why tensor parallelism on a 2026 TPU is two- or four-way and the recipe is EP plus pipeline plus a very large batch, while on GPUs, where NVLink grew with the chip and $\alpha$ stayed near 2,500, eight-way TP survives.

{% enddetails %}

**Question 4 [MTP as a drafter]:** SGLang measured an acceptance length of 2.44 tokens for DeepSeek-V3 with 4 draft tokens per step at 2 requests per GPU, and a throughput gain of 60%. What does the ratio of those two numbers tell you about the step time? At 128 requests per GPU, with a single draft token, the gain was 14%. Using the decode roofline from [Section 13](../moe), explain why.

{% details Click here for the answer. %}

If the step time were unchanged, 2.44 accepted tokens per step would be a 2.44x gain. A 1.6x gain means the step got `2.44 / 1.6 = 1.5x` slower: verifying 5 tokens (4 drafts plus the base) per sequence instead of 1 isn't free even at batch 2, because the drafter's own forward pass and the attention over 5 query positions add work. At 128 requests per GPU with one draft token, verification puts `128 * 2 = 256` tokens through the MoE per step. The compute-bound threshold for V3's experts on an H200 with fp8 weights and fp8 matmuls is $E/k \cdot (\text{bytes}/2) \cdot C / W_\text{hbm} = 32 \cdot 0.5 \cdot 412 = 6{,}600$ tokens (with $E/k = 256/8$ and the H200's fp8 intensity `1.98e15 / 4.8e12 = 412`), so the step is still memory-bound in principle, but the expert matmuls, the dispatch and the attention all scale with the token count while the weight bytes don't, and at 128 sequences those token-proportional terms are already most of the step. Speculation only pays for the part of the step that is fixed cost, and at large batch that part is small.

{% enddetails %}

**Question 5 [an RL step]:** You are running GRPO on a Kimi K2-class model (1T total, 32B active, 35kB of KV per token) with 512 prompts, 16 samples each, responses averaging 12k tokens with a maximum of 64k, on 1,024 H800s. (a) How many tokens are generated per step and how much KV cache do they need? (b) If the generator decodes at 1,850 tokens/s per GPU when fully batched, how long does generation take, and how long does the training pass take at 30% of fp8 peak? (c) How long does the longest response take at 20 tokens/s, and what does that do to (b) in a synchronous system?

{% details Click here for the answer. %}

(a) `512 * 16 = 8,192` responses of 12k tokens is `98M` tokens per step, and `98e6 * 35e3 = 3.4TB` of fp8 KV cache at the peak, 3.4GB per GPU. Manageable per GPU, but note that 3.4TB is more than three times the 1.04TB of fp8 weights, and in a colocated system it lives alongside them: `1.04e12 / 1024 = 1GB` of weights and 3.4GB of cache per GPU if both are fully sharded.

(b) Generation: `98e6 / (1024 * 1850) = 52 s`. Training: `6 * 32e9 * 98e6 = 1.9e19` FLOPs at `1024 * 1.98e15 * 0.3 = 6.1e17` FLOPs/s is `31 s`. Generation already takes longer than training with a third of the FLOPs.

(c) The 64k-token response takes `65536 / 20 = 3,300 s`, or 55 minutes, during which the batch drains and the decode step gets no cheaper. A synchronous step is therefore closer to an hour than to 83 seconds, and over 95% of the GPU-time is spent waiting on a handful of sequences. This single calculation is the reason partial rollouts and asynchronous training exist.

{% enddetails %}

**Question 6 [weight sync]:** Kimi K2's checkpoint engine broadcasts 1.04TB of fp8 weights to 256 GPUs in 16 seconds. What effective bandwidth is that into each node, and how does it compare with the node's 400Gb/s network links? If a training step takes a minute, what fraction is the sync, and what happens to that fraction in an asynchronous system with a staleness of 4 steps?

{% details Click here for the answer. %}

Every node receives the full 1.04TB (each inference rank then pulls its shard from the node's copy), so the effective ingress is `1.04e12 / 16 = 65GB/s` per node, matching the body text. A node with 8 x 400Gb/s links has 400GB/s of ingress, so the broadcast uses about a sixth of the network; K2's report explains that the limit is the host PCIe fabric, which is why it pipelines host-to-device copies with the broadcast. At one step per minute the sync is 27% of the step if nothing overlaps it. With asynchronous training the generators keep running on stale weights while the new ones stream in, so the sync costs no wall-clock at all, only staleness, which AReaL's results say is harmless up to about 8 steps.

{% enddetails %}

**Question 7 [fp4 on GB300, open-ended]:** GB300 delivers 15e15 dense fp4 FLOPs/s per GPU with 8e12 bytes/s of HBM and 900GB/s of NVLink egress. For a DeepSeek-V4-Pro-shaped model (384 experts of width 3,072, top-6) served with fp4 expert weights and fp4 matmuls, compute the decode batch needed to be compute-bound in the experts, the maximum tensor-parallel degree that stays compute-bound for a dense 28,672-wide MLP, and whether 64-way expert parallelism inside the rack is compute-bound with fp8 dispatch. Then argue which of these numbers matter for a decode server that is, per [Section 14](../attention), going to be bound by KV cache reads anyway.

<h3 markdown=1 class="next-section">That's all for Section 15. Section 16 puts the last three sections to work on DeepSeek's published serving system and on training a V4-class model on a TPU7x pod: click [here](../applied-frontier).</h3>
