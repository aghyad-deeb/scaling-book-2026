---
layout: distill
title: "How Training Changed"
# permalink: /main/
description: "The recipe for training a frontier model in 2026 differs from LLaMA 3's in four ways that matter for systems: the matmuls run in fp8 (and increasingly fp4), the optimizer is usually Muon rather than Adam, the model is trained to predict more than one token at a time, and a growing share of the compute goes to reinforcement learning after pretraining, where the expensive part is generating text rather than taking gradient steps. This section works out what each of these costs and what it does to the rooflines of Sections 5 and 6."
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
next_section_name: "Part 16: Serving and Training DeepSeek"

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

## The 2026 Recipe at a Glance

[Section 5](https://jax-ml.github.io/scaling-book/training) and [Section 6](https://jax-ml.github.io/scaling-book/applied-training) describe training the way LLaMA 3 did it: bf16 matmuls, AdamW, a next-token loss, and a parallelism plan chosen so that the weight gathers hide behind the FLOPs. All of that still applies, but a typical 2026 technical report contains four things the LLaMA 3 paper didn't: fp8 (or fp4) matmuls, the Muon optimizer, multi-token prediction, and a large reinforcement-learning phase after pretraining. Each of them changes a number somewhere in this book. We'll take them in the order they show up in a training run, then look briefly at how runs get to a million tokens of context. First, here's what the reports actually tell us (a blank cell means the report doesn't say):

| Model | Tokens | Batch | Seq. length | Matmul precision | Optimizer | Parallelism | Cluster |
| :---- | -----: | ----: | ----------: | :--------------- | :-------- | :---------- | :------ |
| LLaMA 3 405B<d-cite key="llama3"></d-cite> | 15T | 16M | 8k | bf16 | AdamW | TP8, PP16, DP (CP16 added at 128k) | 16k H100 |
| DeepSeek-V3<d-cite key="DeepSeek3"></d-cite> | 14.8T | 63M | 4k | fp8 | AdamW | EP64, PP16, DP2 | 2,048 H800 |
| Kimi K2<d-cite key="kimik2"></d-cite> | 15.5T | 67M | 4k | bf16 (fp8 storage) | MuonClip | EP16, PP16, ZeRO-1 | H800 |
| GLM-4.5<d-cite key="glm45"></d-cite> | 23T | 16M to 64M | 4k | | Muon | | |
| GLM-5<d-cite key="glm5"></d-cite> | 28.5T | | 4k | | Muon | | |
| Nemotron 3 Ultra<d-cite key="nemotron3ultra"></d-cite> | 20T | 25M | 8k | NVFP4 | AdamW | CP32, TP8, EP128, PP2 (1M phase; main-phase layout not disclosed) | GB200 (stated for the 1M phase) |
| DeepSeek-V4-Pro<d-cite key="deepseekv4"></d-cite> | 33T | 94M | 4k to 1M | fp8 | Muon + AdamW | | |
| Kimi K3<d-cite key="kimik3"></d-cite> | | | 8k to 64k | fp8 activations | Per-head Muon | PP, EP, ZeRO-1, CP | |

EP, TP, PP, CP and DP are expert, tensor, pipeline, context and data parallelism (see [Sections 5](https://jax-ml.github.io/scaling-book/training) and [13](../moe)), and ZeRO-1 shards the optimizer state across data parallelism<d-cite key="zero"></d-cite>. Look at the batch column first. The Chinese labs' batches are four to six times LLaMA 3's (Nemotron 3's 25M is 1.6x), and as [Section 5](https://jax-ml.github.io/scaling-book/training) showed, a bigger batch is exactly what lets a run spread over more GPUs and stay compute-bound. Now look at the precision and optimizer columns. Of the 2025 and 2026 rows that say, only Kimi K2 kept bf16 matmuls, and only DeepSeek-V3 and Nemotron 3 Ultra kept Adam (as did Nemotron 3 Nano, which is too small for the table). Finally, notice how many cluster cells are blank. The newer reports say much less about clusters and costs than the 2024 ones did. That matters because we'll want a per-token GPU-hour figure later, and DeepSeek-V3 (2.79M H800-hours) is still the only frontier open pretraining run that publishes one along with a cost breakdown.<d-footnote>Llama 4 gives 5.0M H100-hours for Scout and 2.38M for Maverick, and gpt-oss gives 2.1M, but none breaks the total down per token or per stage.</d-footnote>

## Low Precision: fp8 Everywhere, fp4 Arriving

### The fp8 recipe

The recipe to know is DeepSeek-V3's<d-cite key="DeepSeek3"></d-cite>. It was the first end-to-end fp8 recipe published for an open frontier model, and most of the models above have since adopted it. Here are its main decisions, in the order we'd hit them:

* **Which matmuls?** All three matmuls of every linear layer (the forward pass, the activation gradient and the weight gradient) take fp8 inputs and produce bf16 or fp32 outputs. Everything else stays in bf16 or fp32: embeddings, the output head, the router, normalization, and the attention score and value matmuls. Master weights and gradient accumulators are fp32, and the Adam moments are bf16.
* **Which fp8?** E4M3 for everything, including the gradients. Earlier work used E5M2 in the backward pass for more range, but DeepSeek gets its range from the scaling instead.
* **How do we scale?** This is the most important decision. A single scale per tensor is too coarse, because activations have outlier channels that would force everything else to underflow. So activations get one scale per 1x128 tile (per token, per 128 channels) and weights one per 128x128 block. The scales are computed on the fly from each tile's max absolute value, with no running history. The 1x128 tile for activations is necessary: the report says block-wise 128x128 quantization of the activation *gradients* made a 16B MoE diverge after 300B tokens.
* **How do we accumulate?** The H800's fp8 tensor cores keep only about 14 bits of accumulator precision. DeepSeek accumulates 128 elements at a time on the tensor core, adds the partial sum into fp32 registers on the CUDA cores, and overlaps the two by running two warpgroups (one promotes while the other multiplies). The report says this keeps tensor-core utilization high: each warpgroup issues tensor-core instructions slightly less often, but the tensor cores themselves don't sit idle.
* **What does it buy?** A 2x higher FLOPs ceiling, half the activation bytes in the matmuls, and fp8 dispatch in the MoE AllToAll (the combine stays bf16, since it sums). The relative loss error against a bf16 baseline was under 0.25%.

Qwen3.5 describes essentially the same pipeline (with "runtime monitoring preserving BF16 in sensitive layers"), and DeepSeek-V4 and V4.1 inherit it with small changes (a power-of-two scale format from V3.1 on, and 32x32 blocks in V4.1). Nemotron went all the way to four bits, which we'll get to next. Kimi K2 is the exception. It stores some activations in fp8 to save memory but does *not* compute in fp8, citing "potential risks of performance degradation that we observed during preliminary study", so its whole 15.5T-token run used bf16 matmuls with fp32 gradient accumulation.<d-footnote>K2 also tells us what that costs in memory: bf16 parameters plus an fp32 gradient accumulation buffer for a 1.04T-parameter model is about 6TB, or 24GB per GPU over a 256-GPU model-parallel group, before optimizer state and activations (the report puts the total state at about 30GB per GPU).</d-footnote>

### What fp8 does to the rooflines

Every roofline in this book is a ratio of FLOPs to bytes. fp8 doubles $C$ and halves the bytes of whichever tensors are stored in fp8, so its effect on each roofline depends on whether the bytes in the denominator halved too:

| Roofline | Bytes in the denominator | Halved by fp8? | Threshold |
| :------- | :----------------------- | :------------: | :-------- |
| Decode, weight loading ([Section 7](https://jax-ml.github.io/scaling-book/inference)) | weights | yes if weights are fp8 | unchanged |
| Tensor parallelism ([Section 5](https://jax-ml.github.io/scaling-book/training)) | activations moved over NVLink (ICI on a TPU) | yes if activations are sent in fp8 | unchanged |
| FSDP weight gathers | weights moved over InfiniBand or NVLink (ICI on a TPU) | only if you gather the fp8 copy | doubles otherwise |
| Expert-parallel AllToAll ([Section 13](../moe)) | dispatched tokens | dispatch yes, combine no | 1.5x tighter (bytes fall to 3/4, FLOPs double) |
| Attention ([Section 14](../attention)) | KV cache | yes if KV is fp8 | unchanged |
| Anything measured against HBM bandwidth with bf16 tensors | bf16 weights or activations read from HBM | no | doubles |

Let's put in numbers for the H800 that DeepSeek trained on. In bf16 its NVLink operational intensity is `990e12 / 200e9 = 4,950` (6,190 at the 160GB/s DeepSeek actually achieves). Switch the matmuls to fp8 and $C$ doubles while the link doesn't, so the intensity goes to 9,900 (12,400). Since none of the links got any faster, the same doubling happens to every intensity in this book:

* H800 InfiniBand: `990e12 / 50e9 = 19,800` in bf16, 39,600 in fp8.
* H800 HBM: `990e12 / 3.35e12 = 296` in bf16, 591 in fp8.
* GB200: NVLink 2,780 to 5,560, HBM 312 to 625.
* TPU7x: per-axis ICI 12,800 to 25,600.

While we're here, let's redo the utilization estimate from Question 7 of [Section 4](https://jax-ml.github.io/scaling-book/transformers) with better numbers. DeepSeek-V3 spent 2,664K H800-hours on 14.8T tokens at 37B active parameters, or `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs. An H800 has the same tensor cores as an H100 (1.98e15 dense fp8 FLOPs/s), so the utilization was `3.3e24 / (2.664e6 * 3600 * 1.98e15) = 17%`. DeepSeek's later hardware paper gives the steady-state figure directly: 385 TFLOP/s per GPU counting attention FLOPs, which they report as 39% of *bf16* peak, or 19.5% of the fp8 peak the matmuls actually ran at.<d-cite key="deepseek_isca"></d-cite><d-footnote>Section 4 got 22% because it used 1.51e15, the PCIe H800's rate.</d-footnote> Either way, that's far below the 40 to 50% we assumed for dense bf16 training on TPUs in earlier sections. As [Section 13](../moe) showed, most of the missing time goes to the cross-node expert AllToAll.

<p markdown=1 class="takeaway">**Takeaway:** fp8 training uses E4M3 everywhere, with per-1x128 activation scales and per-128x128 weight scales, fp32 accumulation every 128 elements, and bf16 or fp32 for everything that isn't a linear layer. It doubles the FLOPs ceiling, so every roofline whose byte term didn't also halve becomes 2x harder to satisfy. DeepSeek-V3's fp8 utilization was about 17%.</p>

### fp4 pretraining

Can we halve again? On NVIDIA hardware, yes. Blackwell's tensor cores run fp4 at twice the fp8 rate on GB200 and three times on GB300<d-cite key="nvfp4"></d-cite>. NVIDIA's format is NVFP4. Each element is E2M1 (one sign bit, two exponent bits and one mantissa bit), every 16 elements share an E4M3 scale, and each tensor has an fp32 scale, which comes to 4.5 bits per value. The alternative, MXFP4, uses a power-of-two scale per 32 elements, and it's what OpenAI, DeepSeek and Kimi K3 use for the weights they ship.

NVIDIA's pretraining recipe<d-cite key="nvfp4"></d-cite> is essentially the fp8 recipe plus three additions, each needed because four bits isn't very many:

1. Weights are scaled in 16x16 blocks, and activations and gradients in 1x16 blocks (the same shape logic as DeepSeek's 128x128 and 1x128, just finer).
2. A random Hadamard transform is applied to the inputs of the weight-gradient matmul, spreading outliers across a block before quantizing, and gradients are rounded stochastically rather than to nearest. (Forward tensors still round to nearest, since stochastic rounding there turned out to hurt.)
3. About 16% of the linear layers (the first two and last eight blocks of the 62-block test model) stay in higher precision, along with everything the fp8 recipe already kept there.

**Does it work?** With that recipe, NVIDIA trained a 12B hybrid Mamba-Transformer on 10T tokens with a relative loss gap under 1% for most of the run (it widens to about 1.5% near the end), and the model matched fp8 on downstream evaluations. So far only NVIDIA itself has used it at frontier scale, for Nemotron 3 Super (120B, 25T tokens) and Nemotron 3 Ultra (550B, 20T tokens)<d-cite key="nemotron3super"></d-cite><d-cite key="nemotron3ultra"></d-cite>. Both keep a fair chunk of the model in bf16 or MXFP8: the final 15% of layers, the attention projections, the MoE latent projections, the multi-token-prediction (MTP) heads we'll get to below, and the embeddings. Ultra reports a training loss gap under 0.4%.

DeepSeek-V4 ships fp4 weights too, but they come from a different process. DeepSeek-V4 was pretrained in fp8, and its fp4 expert weights come from quantization-aware training during post-training, which we discuss next. The same goes for Kimi K3, and gpt-oss's expert weights were quantized to MXFP4 during post-training (OpenAI doesn't describe the method). So when a model is described as "trained in fp4", ask which matmuls ran in four bits, and when. For the Nemotron 3 models the answer is the *pretraining* matmuls, and they're the only frontier-scale open models we know of where that's true.

fp4 does to the rooflines what fp8 did, one more time. On a GB300, dense fp4 compute is 15e15 FLOPs/s against 8e12 bytes/s of HBM and 900e9 bytes/s of NVLink egress:

| GB300 NVL72, per GPU | bf16 | fp8 | fp4 |
| :------------------- | ---: | --: | --: |
| Dense FLOPs/s | 2.5e15 | 5e15 | 15e15 |
| HBM intensity $C / W_\text{hbm}$ | 312 | 625 | 1,875 |
| NVLink intensity $C / W_\text{egress}$ | 2,780 | 5,560 | 16,700 |
| Minimum expert width for compute-bound EP (fp8 dispatch, [Section 13](../moe)) | 1,390 | 2,780 | 8,330 |

Look at the last two rows. At fp4, the NVL72's NVLink intensity (16,700) sits roughly where the H800's InfiniBand sat in bf16 (19,800), and the minimum expert width is 8,330, so a 2,048-wide expert's AllToAll is about four times communication-bound even inside the rack. Rubin (35e15 dense fp4 against 1.5e12 egress) is worse still. The fixes are the two [Section 13](../moe) gave us: make the experts wider, or hide the AllToAll behind attention, or both. If fp4 training catches on, we should expect to see a lot more of each. Every hardware generation since the H100 and v5p has raised $\alpha = C / W$ (the FLOPs-to-bandwidth ratio from [Section 5](https://jax-ml.github.io/scaling-book/training)), and lower precision raises it again with no change to the links, so the communication rooflines in this book keep getting harder to satisfy.

### Quantization-aware training for the weights you ship

If fp4 pretraining isn't standard yet, what is? The common recipe is fp8 pretraining followed by quantization-aware training (QAT). During post-training, the expert weights are quantized to four bits *inside the training loop*, so the model learns to live with the rounding, and those quantized weights are what ships.

| Model | Shipped expert weights | Method | Result |
| :---- | :--------------------- | :----- | :----- |
| Kimi K2 Thinking<d-cite key="kimik2thinking"></d-cite> | int4 (group 32) | QAT in post-training | 594GB vs 1,029GB; about 2x generation speed |
| GLM-5<d-cite key="glm5"></d-cite> | int4 | QAT in supervised fine-tuning (SFT), bit-identical train and inference kernels | |
| DeepSeek-V4<d-cite key="deepseekv4"></d-cite> | MXFP4 (plus the sparse-attention indexer) | QAT in post-training; native fp4 in RL rollouts | 0.87TB (Pro, fp4 experts) vs 1.6TB (fp8 Base) |
| Kimi K3<d-cite key="kimik3"></d-cite> | MXFP4 weights, MXFP8 activations | QAT through SFT and RL | 1.56TB for 2.78T parameters |
| gpt-oss<d-cite key="gptoss"></d-cite> | MXFP4 | post-training quantization (method not published) | about 61GB, fits one 80GB GPU |

The last column is the checkpoint size as downloaded. MXFP4 costs 4.25 bits per parameter once its scales are included, and attention, the shared experts and (for K3) the vision tower stay in bf16. These are the sizes Sections 13, 14 and 16 use whenever they count serving memory.

**Does this need new fp4 training kernels?** Not with DeepSeek's trick. The fp32 master weights are quantized to MXFP4 and then *dequantized to fp8* for the actual matmul. E4M3 has two more exponent bits than E2M1, so the fp4-to-fp8 conversion is exact, and the existing fp8 training kernels run unmodified. The only new code is the quantizer (plus a straight-through estimator in the backward pass), and the forward pass sees exactly the weights that will be served.

**Why go to this trouble?** As [Section 13](../moe) showed, four-bit expert weights halve the number of GPUs needed to hold a model and double the batch that fits, which is reason enough to do QAT. A second reason, and the one that makes the exact match matter, only shows up in reinforcement learning. If the rollouts come from an inference engine using the quantized weights while the trainer uses the full-precision ones, the two disagree, and RL with a mismatched sampler is off-policy without anyone noticing. Kimi K3's report says QAT was chosen so that "rollout and training share the same quantization scheme", and DeepSeek-V4 runs its RL rollouts on the native fp4 weights for the same reason.<d-footnote>GLM-5's "bitwise-identical" int4 kernel serves a narrower purpose: it makes the SFT-stage QAT match the shipped weights, and GLM-5's RL rollouts run in fp8.</d-footnote> We'll come back to this below.

<p markdown=1 class="takeaway">**Takeaway:** fp4 pretraining works (NVFP4, with about 15% of layers kept in higher precision and a loss gap under 1%), but so far only NVIDIA's own models use it. The common 2026 pattern is fp8 pretraining followed by quantization-aware post-training to 4-bit expert weights. That halves serving memory and, if the RL rollouts also use the 4-bit weights, removes a mismatch between training and serving.</p>

## Muon Replaces Adam

### The update

Now the optimizer column. Every earlier section of this book takes Adam for granted, but most of the rows in our table say Muon<d-cite key="muon"></d-cite><d-cite key="moonlight"></d-cite>. Among the open MoEs trained from 2025 on, Kimi K2 and K3, GLM-4.5 and GLM-5, DeepSeek-V4 and V4.1, and Qwen's experimental Qwen3.8-Flash-Next all use it for their matrix parameters. DeepSeek-V3, Llama 4 and NVIDIA's Nemotron 3 models used AdamW, and Qwen doesn't say. So we'd better understand what Muon is and what it costs.

At its core, Muon is SGD with momentum plus one extra step, and it's only for matrices. For a weight matrix $W$ with gradient $G$:

$$M_t = \mu M_{t-1} + G_t, \qquad O_t = \text{NewtonSchulz}(M_t) \cdot \sqrt{\max(n, m)} \cdot 0.2, \qquad W_t = W_{t-1} - \eta \left( O_t + \lambda W_{t-1} \right)$$

The Newton-Schulz step orthogonalizes the momentum matrix. It replaces $M$ with the nearest matrix whose singular values are all one, so every direction in the update gets the same step size, whatever its gradient was. It's computed with five iterations of a fixed quintic polynomial in $X X^\top$, entirely in bf16 matmuls, and AdamW is still used for vectors, biases, norms, embeddings and the output head.<d-footnote>The Newton-Schulz iteration is $X \leftarrow a X + b (X X^\top) X + c (X X^\top)^2 X$ with $(a, b, c) = (3.4445, -4.7750, 2.0315)$, applied five times to the momentum matrix scaled to unit norm. DeepSeek-V4 uses a "hybrid" ten iterations, eight with those coefficients and two with $(2, -1.5, 0.5)$, to land exactly on singular value one. The $\sqrt{\max(n, m)} \cdot 0.2$ factor matches the update's root-mean-square to what Adam would produce, so that Adam learning rates transfer over (DeepSeek uses 0.18).</d-footnote>

**Why bother?** The claimed benefit is token efficiency: reaching a given loss in fewer tokens. Moonlight's scaling-law fits on sub-2B models found that Muon needs about half the training FLOPs of AdamW to reach the same loss. Kimi K2's 15.5T-token run "with zero loss spikes" is the largest published evidence that Muon is *stable* at scale, but no AdamW version of K2 has been published to compare against, so it tells us nothing about efficiency.

Muon has a stability problem of its own: attention logits explode more often than with Adam. Where the keys are materialized per head, normalizing the queries and keys fixes it (GLM-4.5 does this with grouped-query attention, and DeepSeek-V4 normalizes queries and KV entries directly). MLA (multi-head latent attention, see [Section 14](../attention)) never materializes per-head keys at inference time, so Kimi's answer is QK-Clip. After each step, for any head whose maximum attention logit $\ell_\text{max}$ exceeded a threshold $\tau = 100$, it scales that head's compressed query and key projections down by $\sqrt{\tau / \ell_\text{max}}$ and its rotary query projection by $\tau / \ell_\text{max}$ (the shared rotary key is left alone). In K2, 12.7% of heads triggered the clip at some point in the first 70,000 steps and none did afterwards, so the mechanism switches itself off. Kimi K3, GLM-5 and DeepSeek-V4.1 also orthogonalize each attention head's block of the projection matrices separately ("per-head Muon" or "Muon Split"). GLM reports that this alone keeps the logits stable with no clipping at all (K3 keeps the clip as well).

### What it costs

Muon turns out to cost very little, which is a large part of why so many rows in our table use it. Let's count the FLOPs and the memory.

**FLOPs:** Newton-Schulz on an $m \times n$ matrix costs about $4 m n^2 + 2 n^3$ FLOPs per iteration for $m \geq n$. DeepSeek-V3 used AdamW, but its shapes make a good test case, so let's price Muon as if V3 had used it. Its 44,718 expert matrices of shape 7168x2048 cost `5 * (4 * 7168 * 2048^2 + 2 * 2048^3) = 6.9e11` FLOPs each, or `3.1e16` per step in total (the attention and dense layers add a little, not enough to change the answer). A training step on a 63M-token batch is `6 * 37e9 * 63e6 = 1.4e19` FLOPs, so the optimizer is **0.23%** of the step, or 0.46% with DeepSeek-V4's ten iterations. That's next to nothing. (Keller Jordan's original estimate for a 405B-scale model was 0.5%, in the same ballpark.) It's also embarrassingly parallel across matrices, and in an MoE the hundreds of experts per layer share a shape, so their iterations batch into one large matmul.

**Memory:** Adam keeps two moments per parameter and Muon keeps one. With both in fp32, the optimizer state drops from 8 bytes per parameter to 4, so [Section 5](https://jax-ml.github.io/scaling-book/training)'s 10 bytes per parameter for model plus optimizer state becomes 6 (2 for the bf16 weights plus 4 for the momentum). For a 1T-parameter model that saves 4TB before ZeRO-1 shards the state across data parallelism, which is the halving Moonlight reports ("the additional memory used by the Muon optimizer is half of Distributed AdamW").<d-footnote>This only holds when comparing fp32 to fp32, as Moonlight does. DeepSeek-V3 kept its Adam moments in bf16, 4 bytes in all, so against that recipe an fp32 Muon momentum saves nothing.</d-footnote>

### Distributing Muon

The catch is in distributing Muon. Adam's update is elementwise, so under ZeRO-1 every rank simply updates its own slice of every tensor. Newton-Schulz needs the *whole* matrix, so each rank has to gather the full momentum matrix for the parameters it owns, orthogonalize it, and scatter the result back.

**How much communication is that?** Moonlight's distributed algorithm does a ReduceScatter of the fp32 gradient (4 bytes per parameter), gathers the sharded momentum in bf16 for the Newton-Schulz step (2 bytes), and AllGathers the updated fp32 parameters (4 bytes), for 10 bytes per parameter per step against Adam's 8. That's at most 1.25x the communication, and it happens once per step rather than once per layer, so it doesn't touch the per-layer rooflines of [Section 5](https://jax-ml.github.io/scaling-book/training). Implemented naively, this gathers the *entire* parameter buffer on every rank, as GLM-5 and Kimi K3 both point out. GLM-5 restricts the AllGather to the shards each rank owns, and K3 fetches them point-to-point. DeepSeek-V4 caps the width of its ZeRO group so that no matrix is split across too many ranks, and packs matrices onto ranks with a knapsack heuristic. When data parallelism is wider than that, the extra groups recompute the update redundantly. It also rounds its MoE gradients to bf16 stochastically before the data-parallel reduce, which halves that traffic.

<p markdown=1 class="takeaway">**Takeaway:** Muon orthogonalizes the momentum matrix with five bf16 Newton-Schulz iterations. It costs about 0.2 to 0.5% of training FLOPs, halves the optimizer memory (6 bytes per parameter with weights instead of 10), and adds at most 25% to the once-per-step optimizer communication, because it needs whole matrices. In exchange it reaches a given loss in roughly half the tokens, at least at small scale. It needs QK-Clip or per-head orthogonalization to keep the attention logits from exploding.</p>

## Multi-Token Prediction

What if, besides the next token, we asked the model to predict the one after that too? That's multi-token prediction (MTP)<d-cite key="mtp"></d-cite>, the third change on our list. Alongside predicting token $t+1$ from the trunk's representation of token $t$, a small extra module predicts token $t+2$. DeepSeek's version<d-cite key="DeepSeek3"></d-cite> is one additional Transformer block. It takes the trunk's final hidden state for position $t$, concatenates the embedding of the (ground-truth) token $t+1$, projects back to $D$, runs the block, and applies the shared output head. Its loss is weighted 0.3 for most of training and 0.1 at the end.

MTP is close to universal now: the DeepSeek, GLM, Qwen, Nemotron and MiniMax flagships from 2025 on all train with it.<d-footnote>Kimi K2 didn't, Qwen only added it with Qwen3-Next, and DeepSeek-V4.1-Flash dropped it in favor of a drafter trained afterwards<d-cite key="deepseekv41"></d-cite>.</d-footnote> Depth one is the default for DeepSeek, Kimi K3 and GLM-4.5, while MiniMax, LongCat, MiMo and Nemotron train 2 to 7 modules. GLM-5 trains three steps with shared parameters, so the module is used at inference the way it was trained.

**What does it cost in training?** On DeepSeek-V3, one extra block on top of 61 is 1.6% more block FLOPs, and the extra pass through the 129,280-wide output head adds another 2.5%, so MTP costs **about 4% of forward FLOPs**. It also puts about 14B parameters in the checkpoint that aren't used for next-token prediction at all. The reports say it improves the main loss, so the 4% is worth paying, and we get something back at inference.

**What does it buy at inference?** A draft model that comes with the checkpoint. The MTP block shares the trunk's computation, so it's exactly the embedded drafter head that Appendix D of [Section 7](https://jax-ml.github.io/scaling-book/inference) describes (that appendix even uses DeepSeek-V3 as its example). Decode is memory-bound, so verifying two or three drafted tokens in one forward pass costs almost the same bytes as generating one, and the whole question is how many of the drafted tokens get accepted. Here are the measurements we have:

| System | Draft tokens per step | Measured acceptance length | Speedup |
| :----- | --------------------: | -------------------------: | :------ |
| DeepSeek-V3 paper | 1 | about 1.85 to 1.9 (85 to 90% second-token acceptance) | 1.8x tokens/s |
| SGLang, V3 on H200, 2 requests per GPU | 3 / 4 | 2.18 / 2.44 | +60% throughput |
| SGLang, V3, 128 requests per GPU | 1 | | +14% |
| GLM-5 vs DeepSeek-V3.2, 4 steps | 4 | 2.76 vs 2.55 | not reported |
| GLM-5.3, EAGLE-style draft from the MTP head | 6 (5 steps) | 3.5 (simulated in the published run) | not reported (3.7ms per token at batch 1, 14.2ms at batch 16, on 8 H200s) |
| DeepSeek-V4-Flash with DSpark<d-cite key="dspark"></d-cite> (its 5-position parallel drafter) | 5 | not reported | +60 to 85% per-user speed over depth-1 MTP (Pro: +57 to 78%) |

This is the pattern from [Section 7](https://jax-ml.github.io/scaling-book/inference): speculation helps most when the batch is small and the step is bandwidth-bound (+60% at two requests per GPU), and least when the batch is large and the MoE FLOPs are already busy (+14% at 128). In 2026 it matters most somewhere we haven't discussed yet: RL rollouts, where a few very long sequences finish last and everything else waits for them. GLM-5's report says MTP "provides disproportionately large benefits on the long tail" there.

<p markdown=1 class="takeaway">**Takeaway:** multi-token prediction adds about 4% to training FLOPs and gives the served model a built-in speculative decoder with acceptance lengths of 2.2 to 3.5 tokens. That's worth 1.6 to 1.8x in tokens per second at small batch, and much less at large batch.</p>

## Reinforcement Learning Is a Systems Problem

So you've pretrained your model in fp8 with Muon and an MTP head, and now you want it to reason. This is the last of our four changes, and nothing earlier in the book covers it. For LLaMA 3, post-training was a small fraction of compute (some supervised fine-tuning and some preference optimization on pairs), so we could safely ignore it. Since DeepSeek-R1<d-cite key="deepseekr1"></d-cite> in January 2025, that's no longer true. Running reinforcement learning against verifiable rewards (does the code pass the tests? is the answer to the math problem right?) for tens of thousands of steps produces a model that reasons at length, and every frontier model since has a "thinking" mode trained this way. **How much compute is that?** DeepSeek-V3.2 says its post-training budget, dominated by RL, exceeded 10% of pretraining compute, up from 5.5% for R1.<d-footnote>No other report publishes the ratio.</d-footnote> What matters more for the rooflines is that RL compute isn't spent the way pretraining compute is, so let's work out where it goes.

### The loop

Here's one RL step, in the GRPO (group relative policy optimization)<d-cite key="grpo"></d-cite> form that most of these models use:

1. Take a batch of prompts and, for each one, sample $G$ responses from the current policy (R1 uses 16; this $G$ is GRPO's letter and has nothing to do with the group factor of [Section 4](https://jax-ml.github.io/scaling-book/transformers)). This is *generation*: autoregressive decode, thousands to tens of thousands of tokens per response.
2. Score each response with a reward (a verifier, a test harness, occasionally a reward model), and compute each response's advantage as its reward minus the mean over its group, divided by the group's standard deviation.
3. Take a few gradient steps on the policy with a clipped policy-gradient loss weighted by the advantages. This is *training*: a forward and backward pass over the generated tokens at $6 N_\text{active}$ FLOPs per token, with the usual parallelism. Recipes that keep a KL term, like R1, also run the frozen reference model forward at $2 N_\text{active}$ more, while GLM-5, Magistral and DAPO drop it<d-cite key="dapo"></d-cite>.
4. Copy the new weights to whatever does the generating, and repeat.

{% include figure.liquid path="assets/img/rl-loop.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> one reinforcement-learning step. Generation dominates the wall clock because it is memory-bound decode and because the step cannot finish before its longest response does. The two placements of trainer and generator, and the two fixes for the tail, are discussed below." %}

R1-Zero's numbers give a sense of scale. Each training batch is `32 * 16 = 512` responses (32 questions with 16 samples each), and a rollout of 8,192 responses is `8192 / 512 = 16` mini-batches. Responses run up to 32,768 tokens (65,536 later in the run). The run took 10,400 steps on 512 H800s for about 198 hours, or 101K GPU-hours. The whole R1 pipeline took 147K, and `147e3 / 2.664e6 = 5.5%` of V3's pretraining hours is the 5.5% above. MiniMax-M1's RL<d-cite key="minimaxm1"></d-cite> ran for three weeks on 512 H800s. These are real training runs, and the 2026 ones are bigger still: Kimi K3 trains RL at a million tokens of context.

*Before reading on, try to guess which of the four steps above dominates the wall clock, and why.*

### Where the time goes

Most of it goes to step 1, generation. The published breakdowns put generation at 65% to over 90% of RL wall-clock: 81% in the HybridFlow paper's baseline<d-cite key="verl"></d-cite>, over 90% in APRIL's measurements, and 65 to 72% in NVIDIA's on GB200. You might guess that's because generation has more FLOPs, but it doesn't. Let's count, for an R1-style rollout of 8,192 responses averaging 10,000 tokens (82M tokens in total) on a DeepSeek-V3-class model:

* **Generation FLOPs:** `2 * 37e9 * 82e6 = 6e18`. **Training FLOPs:** three times that, `1.8e19`.
* **Training time** at 30% of 512 H800s' fp8 peak: `1.8e19 / (512 * 1.98e15 * 0.3) = 60 s`.
* **Generation time** if every GPU decoded at DeepSeek's production rate of 1,850 tokens/s: `82e6 / (512 * 1850) = 86 s`.

So even in the best case, generation takes longer than training despite having a third of the FLOPs. Decode is memory-bound: those 86 seconds are `6e18 / (86 * 512 * 1.98e15) = 7%` of fp8 peak, a quarter of the 30% we assumed for training. In practice we don't get the best case, because response lengths are very skewed and a synchronous system can't train until the *last* response finishes. APRIL, ByteDance's rollout-scheduling system, measures a standard deviation of 4,000 to 4,500 tokens with a tail near the maximum, and a 65,536-token response at 20 tokens per second takes **55 minutes**. As the other responses finish, the per-GPU decode batch shrinks, but a memory-bound step costs about as much with a small batch as with a large one, so throughput per GPU falls as the batch drains. This is the straggler problem, and nearly every 2026 RL system is built around it.

There's a memory cost too. Those 82M in-flight tokens all have KV caches: `82e6 * 35e3 = 2.9TB` in fp8 for a V3-class model at the peak, if every sequence were resident at once, or 6GB per GPU across 512. With 64k-token responses, or a million-token agent context, it's an order of magnitude more, and it competes with the trainer's weights and optimizer state for HBM. That's why Kimi K3 offloads its training state to NVMe during rollouts and keeps an external KV pool in host DRAM.

### Colocated or disaggregated

**Where do the trainer and the generator live?** There are two arrangements in practice.

**Colocated:** the same GPUs alternate between training and generating, so no GPU ever sits idle waiting for the other side. Kimi K2 and K3, DeepSeek-V4.1 and slime's synchronous mode work this way, and K3's report says this is what keeps a 1M-context RL experiment "within a few hundred GPUs". What we pay for it is the switch. During generation the trainer's weights and optimizer state are offloaded to host memory (K2) or NVMe (K3), and an inference engine runs on the GPUs instead, which needs the new weights in its own layout. K2's "checkpoint engine" broadcasts the full 1.04TB of fp8 parameters to every node and lets each inference rank pull its own shard. On 256 H20s this takes 16 seconds, or `1.04e12 / 16 = 65GB/s` into every node.<d-footnote>The K2 report itself quotes under 30 seconds. The 16 seconds is a later checkpoint-engine benchmark on H20s, the export-market Hopper with the H800's networking and a fraction of its FLOPs.</d-footnote> Kimi k1.5<d-cite key="kimik15"></d-cite> quotes under a minute to switch from training to inference and about ten seconds back.

**Disaggregated:** training and generation get their own pools of GPUs, and the new weights are pushed across the network every few gradient steps. The two sides then no longer have to agree on anything. They can run different software, different parallelism, or even different hardware (slime notes its rollout engines "can even run on different GPU models or vendors"). Since generation is the slow side, it should get most of the GPUs, and AReaL<d-cite key="areal"></d-cite> found exactly that: putting three quarters of the GPUs on generation beat an even split. Since every update now crosses the network, the weight push has to be fast: Magistral<d-cite key="magistral"></d-cite> broadcasts weights GPU-to-GPU in under five seconds and never lets the generators wait for the trainers. PipelineRL, GLM's agentic RL and MiniMax-M2 are built this way too.

**Which should we pick?** It depends on what's being generated. slime<d-cite key="slime"></d-cite>, the framework behind every GLM model from GLM-4.5 on, uses colocated synchronous training for reasoning RL but disaggregated asynchronous training for agentic RL. An agentic rollout may spend minutes running tools in a sandbox, and a colocated GPU would sit idle the whole time.

### Asynchrony and the long tail

**How do we get rid of the stragglers?** Two ideas do the job, and most 2026 systems use both.

**Partial rollouts** (Kimi k1.5, K2 and K3, APRIL, and DeepSeek-V4's generation service). Stop the generation phase once a fraction $\lambda$ of the responses have finished, save the unfinished ones with their KV caches, train on what completed, and resume the rest in the next iteration under the new weights. The resumed tokens were sampled from an older policy, so the loss has to tolerate some staleness, which brings us to:

**Asynchronous training** (AReaL, PipelineRL, Magistral, slime, DeepSeek-V4.1, Qwen3.5). Let generation run continuously and train on whatever has finished, accepting that a response may have been sampled by a policy several steps old. How much staleness can the loss absorb? AReaL measured it: up to about 8 policy versions has minimal measurable cost, as long as the loss uses the *behavior* policy's probabilities for importance weighting (its decoupled PPO objective). PipelineRL goes further and updates the generators' weights *in flight*, mid-sequence, keeping the KV cache (Magistral found that recomputing the cache after a weight update was unnecessary). AReaL reports 2.6x the training throughput of a synchronous system, and DeepSeek-V4.1 says "asynchronous training is now enabled for nearly all our RL" tasks.

DeepSeek-V4 documents one subtlety. If unfinished responses are cancelled and restarted instead of resumed, short responses are more likely to survive, and the model learns to be terse. Its generation service keeps a token-level write-ahead log so that a preempted request resumes exactly where it stopped.

### Two engines, one policy

There's one more problem. The trainer (Megatron<d-cite key="megatron"></d-cite>, in bf16 or fp8) and the generator (vLLM or SGLang, often with fused kernels and quantized weights) compute slightly different probabilities for the same token. The RL loss assumes the sampled tokens came from exactly the policy it's updating. They didn't quite, so RL that is nominally on-policy is actually slightly off-policy, and off-policy RL can diverge.<d-cite key="offpolicy_secret"></d-cite> MiniMax traced its instability to the output head, where the activations are large. Running the head in fp32 raised the correlation between trainer and generator probabilities from about 0.9 to 0.99, and Meta's ScaleRL study found the same fp32-head fix was worth a sizeable gain in final reward.

There are three fixes, in increasing order of thoroughness, and each appears in production 2026 systems:

* **Importance weighting with the generator's probabilities.** The inference engine returns the log-probabilities it sampled with, and the loss uses them as the behavior policy, truncating or masking any tokens whose ratio is too far from one (truncated importance sampling, DeepSeek's off-policy sequence masking, GLM's IcePop, and GSPO's sequence-level ratios are all versions of this).
* **Routing replay.** For MoEs, the trainer is forced to use exactly the experts the generator chose (DeepSeek's "Keep Routing", Qwen3.5's "rollout router replay"). Otherwise a tiny probability difference can flip a top-$k$ decision, and the two engines end up running different networks. GLM-5 found the same thing for sparse attention, where a nondeterministic top-$k$ in the indexer "caused drastic performance degradation during RL after only a few steps".
* **Identical numerics.** Train with the same quantization the generator uses (Kimi K3's MXFP4 QAT through RL, DeepSeek-V4's native fp4 rollouts). This is the QAT connection promised above.

<p markdown=1 class="takeaway">**Takeaway:** post-training, most of it RL, passed 10% of pretraining compute at DeepSeek by V3.2, and 60 to 90% of its wall-clock is memory-bound generation whose step time is set by the longest response. The 2026 systems we've looked at deal with this through partial rollouts, asynchronous training that tolerates several steps of staleness, weight synchronization in seconds for trillion-parameter policies, and numerics deliberately matched between the training and inference engines.</p>

## Long-Context Training

[Section 14](../attention) showed why attention is affordable at long context in 2026 architectures. But how do training runs actually get to a million tokens? Mostly by training at long length briefly, and only at the end:

| Model | Schedule | Tokens trained at the long lengths |
| :---- | :------- | :-------------------- |
| LLaMA 3 405B | 8k, then six stages to 128k | about 800B |
| DeepSeek-V3 | 4k, then 32k, then 128k | about 125B (2 phases of 1,000 steps) |
| Kimi K2 | 4k throughout; anneal 400B at 4k + 60B at 32k; YaRN to 128k | 60B |
| GLM-5 | 4k, then 32k, 128k, 200k | 1T at 32k, 500B at 128k, 50B at 200k |
| DeepSeek-V4 | 4k, 16k, 64k, then 1M | not disclosed (sparse attention from the 64k stage; dense for the first 1T tokens) |
| DeepSeek-V4.1 | 64k from the start (sparse), 1M from 34T of 45T tokens | 11T |
| Kimi K3 | 8k, 64k in pretraining; 256k to 1M in the cooldown | undisclosed |
| Nemotron 3 Ultra | 8k, then 1M (92% of the 1M-phase steps at 1M, 8% at 4k) | 33B (about 30B at 1M) |

There are two things to notice in this table. First, until DeepSeek-V4.1, none of these models spent much of its training at long length. The long-context phase is a few percent of the tokens, it comes at the very end, and it relies on YaRN<d-cite key="yarn"></d-cite> to stretch the position encoding rather than on training at that length from the start. DeepSeek-V3's context extension, for example, was 119K of its 2.79M GPU-hours, about 4%. Second, the models that *were* designed for long context from the start (V4.1 at 64k from the first token, K3 at 8k to 64k with no position encoding in its full-attention layers) can afford it only because their attention is sparse or linear, as we saw in [Section 14](../attention).

**How is a million-token sequence sharded across chips?** With context parallelism (CP), which [Section 5](https://jax-ml.github.io/scaling-book/training) only mentions in a note but which is now routine. LLaMA 3 used 16-way CP for 128k. Rather than ring attention, it simply AllGathers the keys and values, on the argument that with GQA "the time complexity of attention computation is an order of magnitude larger than all-gather" ($O(S^2)$ against $O(S)$). That argument only gets stronger with MLA, whose latent is smaller still. Kimi K3 needed a separate scheme for its linear-attention layers. Each shard computes its state transition locally, and one fixed-size AllGather plus a prefix scan reconciles the shards, so no shard waits on a sequential hand-off from the one before it. DeepSeek-V4 needed a two-stage exchange so that its 4:1 and 128:1 token compression works across shard boundaries. Nemotron 3 Ultra's million-token phase ran with 32-way context, 8-way tensor, 128-way expert and 2-way pipeline parallelism on GB200s, which gives a sense of how many axes a 2026 training job has.

<p markdown=1 class="takeaway">**Takeaway:** long context is a short phase at the end of pretraining (a few percent of tokens, with YaRN to stretch the positions), unless the attention is sparse or linear enough to train at 64k or more from the start. Context parallelism is routine, and for linear-attention layers it exchanges fixed-size states rather than keys and values.</p>

## What Should You Take Away from this Section?

* fp8 training is now standard: E4M3, 1x128 activation and 128x128 weight scales, fp32 accumulation, and high precision for embeddings, heads, routers, norms and attention. It doubles the FLOPs ceiling, and every roofline whose byte term didn't also halve gets 2x harder. DeepSeek-V3 reached about 17% of fp8 peak.

* fp4 pretraining exists (NVFP4, Nemotron 3), but the common pattern is fp8 pretraining plus quantization-aware post-training to 4-bit expert weights, which halves serving memory and can be matched exactly by the RL sampler.

* Muon costs 0.2 to 0.5% of FLOPs and half of Adam's optimizer memory, needs whole-matrix gathers once per step (at most 1.25x Adam's optimizer communication), and needs QK-Norm, QK-Clip or per-head orthogonalization to keep the attention logits from exploding. It's now the majority choice among the open MoEs.

* Multi-token prediction adds about 4% to forward FLOPs and gives a speculative decoder with acceptance lengths of 2.2 to 3.5, worth 1.6 to 1.8x at small batch.

* Post-training passed 10% of pretraining compute at DeepSeek, and it's dominated by memory-bound generation with a long tail of response lengths. The standard toolkit is partial rollouts, asynchronous training (staleness up to about 8 steps costs little), fast weight broadcast, and numerically matched engines.

* Long context is trained at the end, on a few percent of tokens, with YaRN. Only models with sparse or linear attention train at 64k or more from the start.

## Worked Problems

**Question 1 [fp8 accounting]:** DeepSeek-V3 reports 180K H800-hours per trillion tokens. What utilization does that imply? If we add the 4% MTP overhead and count the attention score FLOPs at 4k context (about 14% of the matmul FLOPs for this model, from [Section 14](../attention)'s crossover of 19k tokens), what is the utilization of *executed* FLOPs? *Assume 1.98e15 fp8 FLOPs/s per GPU and 37B active parameters.*

{% details Click here for the answer, once you've thought about it! %}

The useful FLOPs per trillion tokens are `6 * 37e9 * 1e12 = 2.2e23`, and the available FLOPs are `180e3 * 3600 * 1.98e15 = 1.28e24`, so the utilization is **17%**. Adding 18% for MTP and attention brings the executed FLOPs to `2.6e23` and the utilization to about 20%. Either way, the run spent most of its time on something other than matmuls. As [Section 13](../moe) showed, the cross-node AllToAll is the main culprit, and the fp32 accumulation and the bubbles of a 16-stage pipeline are the others.

{% enddetails %}

**Question 2 [Muon's bill for Kimi K2]:** What did Muon cost Kimi K2? K2 has 384 routed experts plus 1 shared expert per layer, each three 7168x2048 matrices, in 60 MoE layers, and was trained with a 67M-token batch and 32B active parameters. Estimate the Newton-Schulz FLOPs per step as a fraction of training FLOPs, and the optimizer-state memory saved relative to AdamW.

{% details Click here for the answer. %}

There are `60 * (384 + 1) * 3 = 69,300` expert matrices, each costing `5 * (4 * 7168 * 2048^2 + 2 * 2048^3) = 6.9e11` FLOPs to orthogonalize, so `4.8e16` per step. Training FLOPs per step are `6 * 32e9 * 67e6 = 1.3e19`, so the optimizer is **0.37%** of the step, still next to nothing. For memory, AdamW would keep 8 bytes of fp32 moments per parameter, `8.3TB` for 1.04T parameters, while Muon keeps 4, or `4.2TB`. Over K2's 256-GPU model-parallel group that saves 16GB per GPU, against the roughly 30GB per GPU of total state the report quotes. This is before ZeRO-1 shards the optimizer state across data parallelism, which the report says K2 does (on small clusters it offloads the state to the host instead).

{% enddetails %}

**Question 3 [what fp8 does to this book's constants]:** [Section 12](https://jax-ml.github.io/scaling-book/gpus) says FSDP across H100 nodes is compute-bound above 2,475 tokens per GPU and tensor parallelism inside a node is compute-bound below $Y = F W_\text{nvlink} / C$, and [Section 5](https://jax-ml.github.io/scaling-book/training) gives 850 tokens per chip and $Y = 3F / 2550$ on TPU v5p. Redo both for fp8 matmuls (a) on H100 nodes and on a GB200 NVL72, with weights and activations gathered in bf16, (b) the same with them gathered in fp8, and (c) on TPU7x.

{% details Click here for the answer. %}

(a) On an H100 with fp8 matmuls, $C = 1.98e15$, so the cross-node FSDP threshold is `1.98e15 / 400e9 = 4,950` tokens per GPU (twice the 2,475 we started with), and TP is compute-bound below `Y = F * W_nvlink / C = 28672 * 450e9 / 1.98e15 = 6.5`-way for LLaMA 3-70B's $F = 28672$ (using Section 12's dense-MLP rule, which counts two matmuls per token). So eight-way TP inside the node, which used to be the default, is now 1.2x communication-bound. On a GB200 NVL72 the NVLink intensity in fp8 is `5e15 / 900e9 = 5,560`, so TP is compute-bound below `28672 / 5560 = 5.2`-way, and FSDP across racks needs `5e15 / 3.6e12 = 1,390` tokens per GPU.

(b) Gathering fp8 tensors halves the bytes, so every threshold halves too: 2,475 tokens per GPU and 13-way TP on H100, and 10-way TP and 694 tokens per GPU on GB200. So sending activations in fp8 is what would keep 8-way TP compute-bound in the fp8 era.<d-footnote>Nemotron 3 Ultra's million-token phase did use TP8 on GB200, but that was a long-context layout for a model whose expert matmuls are NVFP4 and whose attention projections are bf16, and the report doesn't say what precision its TP collectives use. Treat it as consistent with this rule rather than proof of it.</d-footnote>

(c) TPU7x in fp8 has $\alpha = 4.61e15 / 1.8e11 = 25{,}600$ per axis. Gathering bf16 tensors, FSDP needs `25600 / 3 = 8,500` tokens per chip (ten times v5p's 850), and TP is compute-bound below `Y = 3 * 28672 / 25600 = 3.4`-way, so two-way at most. With fp8 gathers we get 4,270 tokens per chip and 6.7-way TP, still five times tighter than v5p in bf16. This is why tensor parallelism on a 2026 TPU is two- or four-way, and the recipe is EP plus pipeline plus a very large batch. On GPUs, where NVLink grew with the GPU so that bf16 $\alpha$ stayed near 2,500 (about 5,000 in fp8), eight-way TP survives as long as the activations are sent in fp8, as in (b).

{% enddetails %}

**Question 4 [MTP as a drafter]:** SGLang measured an acceptance length of 2.44 tokens for DeepSeek-V3 with 4 draft tokens per step at 2 requests per GPU, and a throughput gain of 60%. What does the ratio of those two numbers tell us about the step time? At 128 requests per GPU, with a single draft token, the gain was only 14%. Using the decode roofline from [Section 13](../moe), can you explain why?

{% details Click here for the answer. %}

If the step time were unchanged, 2.44 accepted tokens per step would give a 2.44x gain. A 1.6x gain means the step got `2.44 / 1.6 = 1.5x` slower. Verifying 5 tokens per sequence (4 drafts plus the base) instead of 1 isn't free even at batch 2, because the drafter's own forward pass and the attention over 5 query positions both add work. Now the large batch. At 128 requests per GPU with one draft token, verification puts `128 * 2 = 256` tokens through the MoE per step. The compute-bound threshold for V3's experts on an H200 is $B_\text{crit} = (E/k) \cdot (C / W_\text{hbm}) \cdot (\text{bits per weight} / \text{bits per activation})$, which is [Section 7](https://jax-ml.github.io/scaling-book/inference)'s rule with the MoE factor from [Section 13](../moe). With $E/k = 256/8 = 32$, the H200's fp8 intensity of `1.98e15 / 4.8e12 = 412`, fp8 weights and bf16 activations, that's `32 * 412 * 0.5 = 6,600` tokens (13,200 with fp8 activations), so the step is still memory-bound in principle. But the expert matmuls, the dispatch and the attention all scale with the token count while the weight bytes don't, and at 128 sequences those token-proportional terms are already most of the step. Speculation only pays for the fixed-cost part of the step, and at large batch that part is small.

{% enddetails %}

**Question 5 [an RL step]:** Say we're running GRPO on a Kimi K2-class model (1T total, 32B active, 35kB of KV per token) with 512 prompts and 16 samples each, responses averaging 12k tokens with a maximum of 64k, on 1,024 H800s. (a) How many tokens are generated per step, and how much KV cache do they need? (b) If the generator decodes at 1,850 tokens/s per GPU when fully batched, how long does generation take, and how long does the training pass take at 30% of fp8 peak? (c) How long does the longest response take at 20 tokens/s, and what does that do to (b) in a synchronous system?

{% details Click here for the answer. %}

(a) `512 * 16 = 8,192` responses of 12k tokens is `98M` tokens per step, which need `98e6 * 35e3 = 3.4TB` of fp8 KV cache at the peak, or 3.4GB per GPU. That much per GPU is fine on its own, but compare it with the weights. 3.4TB is more than three times the 1.04TB of fp8 weights, and if the trainer and generator are colocated the two have to share HBM: `1.04e12 / 1024 = 1GB` of weights and 3.4GB of cache per GPU if both are fully sharded.

(b) Generation takes `98e6 / (1024 * 1850) = 52 s`. Training is `6 * 32e9 * 98e6 = 1.9e19` FLOPs at `1024 * 1.98e15 * 0.3 = 6.1e17` FLOPs/s, or `31 s`. So generation already takes longer than training, with a third of the FLOPs.

(c) The 64k-token response takes `65536 / 20 = 3,300 s`, or 55 minutes, forty times the 83 seconds of (b). In a synchronous system the whole step takes that long, so over 95% of GPU-time is spent waiting on a handful of sequences while the decode batch on every GPU shrinks toward one. That's terrible! This one calculation is most of the reason partial rollouts and asynchronous training exist.

{% enddetails %}

**Question 6 [weight sync]:** Kimi K2's checkpoint engine broadcasts 1.04TB of fp8 weights to 256 GPUs in 16 seconds. What effective bandwidth is that into each node, and how does it compare with the node's 400Gb/s network links? If a training step takes a minute, what fraction of it is the sync, and what happens to that fraction in an asynchronous system with a staleness of 4 steps?

{% details Click here for the answer, once you've thought about it! %}

Every node receives the full 1.04TB (each inference rank then pulls its shard from the node's copy), so the effective ingress is `1.04e12 / 16 = 65GB/s` per node, the number quoted above. A node with 8 x 400Gb/s links has 400GB/s of ingress, so the broadcast uses only about a sixth of the network. K2's report explains that the real limit is the host PCIe fabric, so it pipelines the host-to-device copies with the broadcast. At one step per minute the sync is 27% of the step if nothing overlaps it, which is a lot. With asynchronous training the generators keep running on stale weights while the new ones stream in, so the sync costs no wall-clock time at all, only staleness, which AReaL's results say is harmless up to about 8 steps.

{% enddetails %}

**Question 7 [fp4 on GB300, open-ended]:** GB300 delivers 15e15 dense fp4 FLOPs/s per GPU with 8e12 bytes/s of HBM and 900GB/s of NVLink egress. Say we serve a DeepSeek-V4-Pro-shaped model (384 experts of width 3,072, top-6) with fp4 expert weights and fp4 matmuls. What decode batch do we need to be compute-bound in the experts? What's the maximum tensor-parallel degree that stays compute-bound for a dense 28,672-wide MLP? Is 64-way expert parallelism inside the rack compute-bound with fp8 dispatch? Then argue which of these numbers actually matter for a decode server that, as [Section 14](../attention) showed, is going to be bound by KV cache reads anyway.

<h3 markdown=1 class="next-section">That's it for Section 15! Section 16 puts the last three sections to work on DeepSeek's published serving system, on training a V4-class model on GB200 NVL72 racks, and on the same run on a TPU7x pod. For that, click [here](../applied-frontier).</h3>
