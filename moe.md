---
layout: distill
title: "How to Think About Mixture of Experts"
# permalink: /main/
description: "Nearly every frontier open-weight model released since this book was written is a Mixture of Experts, and most of them are far sparser than the MoEs of 2024. This section extends the Transformer math of Section 4 and the parallelism rooflines of Sections 5 and 7 to MoEs: how to count their parameters and FLOPs, why sparse models want enormous batches, what expert parallelism costs on TPUs and GPUs, and why routing has become a systems decision as much as a modeling one."
date: 2026-09-20
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Aghyad Deeb
    url: "https://github.com/aghyad-deeb"
    affiliations:
      name: "Independent; written with Claude"

section_number: 13

previous_section_url: ".."
previous_section_name: "2026 Update: Outline"

next_section_url: ../attention
next_section_name: "Part 14: KV Cache"

giscus_comments: false

bibliography: update.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "What Changed Since 2024?"
  - name: "MoE Transformer Math"
  - subsections:
    - name: "Counting parameters"
    - name: "Counting FLOPs"
    - name: "Fine-grained experts"
  - name: "Routing Is a Systems Decision"
  - name: "What Sparsity Does to Memory"
  - name: "The Batch-Size Rooflines"
  - name: "The Expert-Parallelism Rooflines"
  - subsections:
    - name: "Expert parallelism on TPUs"
    - name: "Expert parallelism on GPUs"
    - name: "Combining expert parallelism with FSDP"
  - name: "What the Frontier Labs Actually Do"
  - name: "What Should You Take Away from this Section?"
  - name: "Worked Problems"

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

_[Section 4](https://jax-ml.github.io/scaling-book/transformers) treated Mixture of Experts (MoE) models as a two-paragraph aside. That was reasonable in early 2025, when LLaMA 3 was the reference open model and it was dense. It isn't anymore. Nearly every open-weight model near the frontier today, above about 30B parameters, is an MoE (Mistral's dense 128B Medium 3.5 is the exception), and the way they are sparse has changed: hundreds of small experts, a handful active per token, and a total parameter count that has run away from the active count by a factor of roughly 20 to 33, with $E/k$ sparsities of 32 to 64. This section works through the math of that design and what it does to every roofline in the book. As with the rest of the book, we care less about why MoEs are a good modeling idea and more about what they cost to run._

## What Changed Since 2024?

Here is a table you should build for yourself, in the spirit of the big table of open-source LLMs that [Section 6](https://jax-ml.github.io/scaling-book/applied-training) recommends you build. Every row comes from the model's `config.json` or its technical report.<d-footnote>Total and active counts are the ones the labs report. Conventions differ: DeepSeek and Kimi count the embedding and unembedding parameters in "active", GLM and Gemma do not (GLM-5's 40B is 42B with them, Gemma 4's 3.8B is 4.4B), and the MTP module is usually excluded from both. The active count always includes the shared experts, which run for every token. The script that accompanies this chapter recomputes every row from the config files and lands within 2% of each reported total. We list the routed expert width $F$ (called <code>moe_intermediate_size</code> in most configs), which for MoEs is the number that matters for rooflines. Kimi K3 routes a 3,584-wide latent rather than the full 7,168-wide residual, which we discuss below.</d-footnote>

| Model | Released | Total | Active | Experts $E$ | Active $k$ | Shared | Expert width $F$ | $D$ | $L$ | $E/k$ |
| :---- | :------: | ----: | -----: | ----------: | ---------: | -----: | ---------------: | --: | --: | ----: |
| Mixtral 8x22B<d-cite key="mixtral"></d-cite> | Apr 2024 | 141B | 39B | 8 | 2 | 0 | 16,384 | 6,144 | 56 | 4 |
| DeepSeek-V3<d-cite key="DeepSeek3"></d-cite> | Dec 2024 | 671B | 37B | 256 | 8 | 1 | 2,048 | 7,168 | 61 | 32 |
| Llama 4 Maverick<d-cite key="llama4"></d-cite> | Apr 2025 | 400B | 17B | 128 | 1 | 1 | 8,192 | 5,120 | 48 | 128 |
| Qwen3-235B-A22B<d-cite key="qwen3"></d-cite> | Apr 2025 | 235B | 22B | 128 | 8 | 0 | 1,536 | 4,096 | 94 | 16 |
| Kimi K2<d-cite key="kimik2"></d-cite> | Jul 2025 | 1.04T | 32B | 384 | 8 | 1 | 2,048 | 7,168 | 61 | 48 |
| gpt-oss-120b<d-cite key="gptoss"></d-cite> | Aug 2025 | 117B | 5.1B | 128 | 4 | 0 | 2,880 | 2,880 | 36 | 32 |
| GLM-5<d-cite key="glm5"></d-cite> | Feb 2026 | 744B | 40B | 256 | 8 | 1 | 2,048 | 6,144 | 78 | 32 |
| Qwen3.5-397B-A17B<d-cite key="qwen35"></d-cite> | Feb 2026 | 397B | 17B | 512 | 10 | 1 | 1,024 | 4,096 | 60 | 51 |
| DeepSeek-V4-Pro<d-cite key="deepseekv4"></d-cite> | Apr 2026 | 1.6T | 49B | 384 | 6 | 1 | 3,072 | 7,168 | 61 | 64 |
| Kimi K3<d-cite key="kimik3"></d-cite> | Jul 2026 | 2.8T | 104B | 896 | 16 | 2 | 3,072 | 7,168 | 93 | 56 |
| Qwen3.8-2.4T-A95B<d-cite key="qwen38"></d-cite> | Aug 2026 | 2.4T | 95B | 512 | 10 | 1 | 2,048 | 8,192 | 92 | 51 |

A few things jump out.

* **Total parameters grew 20x in two years. Active parameters grew about 2.5x.** Mixtral 8x22B to Kimi K3 is 141B to 2.8T total, but only 39B to 104B active. Nearly all of the growth went into parameters that any given token never touches.
* **Sparsity went from 4 to somewhere between 32 and 64.** The ratio $E/k$ (the book calls this the sparsity) was 4 for Mixtral. For everything released since DeepSeek-V3 it is 16 or more, and the 2026 models sit between 32 and 64.
* **Experts got narrow.** Mixtral's experts were 16,384 wide, the same as a dense MLP. DeepSeek, Kimi, GLM and Qwen use experts between 1,024 and 3,072 wide, with hundreds of them per layer. This "fine-grained" design<d-cite key="deepseekmoe"></d-cite> has won, with Llama 4 the exception (128 experts of width 8,192, one active per token).
* **Most labs stop dense at about 30B.** Qwen3.6-27B and Gemma 4 31B are the largest recent dense open models from their labs. Mistral is the exception, with dense 123B (Devstral 2) and 128B (Medium 3.5) open weights. Everyone else above that is sparse.

{% include figure.liquid path="assets/img/moe-total-vs-active.svg" class="img-fluid" caption="<b>Figure:</b> total versus active parameters for open-weight models released between 2023 and 2026. Dense models sit on the diagonal. The frontier MoEs of 2026 sit roughly 20 to 33x above it. Note that the active parameter counts of the largest 2026 models are still smaller than dense LLaMA 3 405B." %}

The reason this matters for us is simple. Every roofline in this book compares something proportional to FLOPs (which scale with *active* parameters) against something proportional to bytes (which scale with *total* parameters, for weights). An MoE with sparsity 32 decouples those two quantities by a factor of 32. Rules of thumb like "FSDP is compute-bound above 850 tokens per chip" or "decode is compute-bound above batch 240" were derived for dense models where the two coincide. The book already sketched the fix in passing: Question 8 of [Section 4](https://jax-ml.github.io/scaling-book/transformers) and Question 5 of [Section 7](https://jax-ml.github.io/scaling-book/inference) multiply the decode roofline by $E/k$, and [Section 12](https://jax-ml.github.io/scaling-book/gpus) does the same for data parallelism and derives a GPU expert-parallel roofline. This section works those results out for fine-grained experts, for expert parallelism on TPUs, and for EP combined with FSDP.

## MoE Transformer Math

Let's fix notation. An MoE layer replaces the single MLP block of [Section 4](https://jax-ml.github.io/scaling-book/transformers) with $E$ *routed experts*, each a gated MLP with its own three matrices $W_\text{in1}[D, F]$, $W_\text{in2}[D, F]$, $W_\text{out}[F, D]$, plus (usually) a small number $E_s$ of *shared experts* that every token passes through. A *router* $W_r[D, E]$ scores each token against each expert, and the top $k$ scores select which experts the token visits. The outputs of the $k$ experts are combined with weights derived from the router scores and added to the shared expert output.

{% include figure.liquid path="assets/img/moe-layer.svg" class="img-fluid" caption="<b>Figure:</b> a fine-grained MoE layer with one shared expert and $E$ routed experts, of which $k$ are active per token. The router is a single $[D, E]$ matmul. Note that the routed experts are much narrower than the shared expert or a dense MLP, and that the expensive part of the layer from a systems standpoint is not any matmul but the two AllToAlls that move tokens to their experts and back." %}

**Note [sigmoid routing and shared experts]:** The original MoE papers<d-cite key="moe"></d-cite><d-cite key="switch"></d-cite> softmax the router scores across experts and pick the top $k$. DeepSeek-V3 and most 2026 models instead apply a sigmoid to each score independently, pick the top $k$, then renormalize the selected weights to sum to one. DeepSeek-V4 uses a square-root-of-softplus. The reason to care is that these choices interact with load balancing, discussed below. The shared expert is a plain dense MLP that every token uses. It is always active, so from a systems standpoint it is just a dense layer that happens to live next to the MoE, and it can be sharded like one.

### Counting parameters

Per MoE layer, the parameter count is

$$P_\text{MoE layer} = \underbrace{(E + E_s) \cdot 3 D F}_{\text{experts}} + \underbrace{E D}_{\text{router}}$$

and the number of parameters any single token touches is

$$P_\text{MoE layer, active} = (k + E_s) \cdot 3 D F + E D$$

The router is negligible: for DeepSeek-V3 it is `256 * 7168 = 1.8M` parameters per layer against 11.3B for the experts. The total model is

$$P_\text{total} = L_\text{MoE} (E + E_s) 3DF + L_\text{dense} \cdot 3 D F_\text{dense} + L \cdot P_\text{attn} + 2 D V$$

where several models keep the first one to three layers dense (`first_k_dense_replace` in DeepSeek-style configs) because routing on barely-embedded tokens works poorly.<d-footnote>DeepSeek-V4 went a different direction and made those first layers MoE as well, but with <em>hash routing</em>: the expert is chosen by a hash of the token id rather than by a learned router<d-cite key="hashlayers"></d-cite>. This removes the router entirely from those layers, which is nice for a systems person because the dispatch pattern is known before the layer runs.</d-footnote> Attention here is Multi-head Latent Attention (MLA), which [Section 14](../attention) takes apart; for now take its parameter count as given.

Let's do the accounting for DeepSeek-V3, following the same pattern we used for LLaMA 3-70B in [Section 6](https://jax-ml.github.io/scaling-book/applied-training). From the [config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json): $D=7168$, $L=61$ (3 dense, 58 MoE), $E=256$, $E_s=1$, $k=8$, $F=2048$, $F_\text{dense}=18432$, $V=129280$.

| param | formula | count |
| :---- | :------ | ----: |
| Routed + shared experts | 58 layers * (256 + 1) * 3 * 7168 * 2048 | **656.5e9** |
| Dense MLPs | 3 layers * 3 * 7168 * 18432 | **1.2e9** |
| Attention (MLA) | 61 layers * 187e6 | **11.4e9** |
| Vocab | 2 * 129280 * 7168 | **1.9e9** |
| **Total** | | **671e9** |

That's the reported 671B on the nose!<d-footnote>The Hugging Face repository holds 685B parameters. The extra 14B is the multi-token prediction module (an extra Transformer block plus its own copy of the embedding and output head) that DeepSeek trains alongside the main model and that serving engines use as a speculative decoder. We discuss MTP in <a href="../training-2026">Section 15</a>.</d-footnote> Now the active count, replacing $E + E_s = 257$ with $k + E_s = 9$:

| param | formula | count |
| :---- | :------ | ----: |
| Active experts | 58 * 9 * 3 * 7168 * 2048 | **23.0e9** |
| Dense MLPs | (as above) | **1.2e9** |
| Attention | (as above) | **11.4e9** |
| Vocab | (as above) | **1.9e9** |
| **Active total** | | **37.4e9** |

Also a match. Notice one thing that differs from the dense picture in [Section 6](https://jax-ml.github.io/scaling-book/applied-training): there, the MLP was 80% of the parameters and attention 17%. Here attention is *30% of the active parameters*, because the MoE MLP per token is only 9 experts of width 2,048, or 18,432 effective width, against a 7,168-wide residual stream. Attention has become a first-class cost again, and [Section 14](../attention) is devoted to it.

<p markdown=1 class="takeaway">**Takeaway:** For an MoE, total parameters are $L (E + E_s) 3DF$ to within a few percent, and active parameters are $L (k + E_s) 3DF$ plus attention. The ratio of the two is *not* $E/k$: shared experts and attention are counted in both, so DeepSeek-V3 has $E/k = 32$ but a total-to-active ratio of 18. When you see "sparsity" quoted, check which one is meant.</p>

### Counting FLOPs

The FLOPs story is short: the $6 \cdot \text{params} \cdot \text{tokens}$ rule from [Section 4](https://jax-ml.github.io/scaling-book/transformers) holds if you plug in *active* parameters. Routing adds $2BDE$ FLOPs per layer, which is tiny. Each token does its $k$ expert matmuls and nothing else. So DeepSeek-V3 costs `6 * 37e9 = 2.2e11` FLOPs per token to train, one-eleventh of LLaMA 3 405B's `6 * 405e9 = 2.4e12`, while holding 65% more parameters.

This is the whole point of the design, and it shows in the pretraining budgets of the models people actually use:

| Model | Active params | Tokens | Training FLOPs ($6 \cdot \text{active params} \cdot \text{tokens}$) |
| :---- | ------------: | -----: | -------------------------------------: |
| LLaMA 3 405B | 405B | 15T | 3.6e25 |
| LLaMA 3 70B | 70B | 15T | 6.3e24 |
| DeepSeek-V3 | 37B | 14.8T | 3.3e24 |
| Kimi K2 | 32B | 15.5T | 3.0e24 |
| Qwen3-235B-A22B | 22B | 36T | 4.8e24 |
| DeepSeek-V4-Pro | 49B | 33T | 9.7e24 |

DeepSeek-V3 was trained for half the compute of LLaMA 3 70B. Kimi K2 holds a trillion parameters and cost less to train than LLaMA 3 70B did. Even DeepSeek-V4-Pro, at 33T tokens, is a quarter of LLaMA 3 405B.<d-footnote>Not every lab publishes token counts anymore. Kimi K3 and Qwen3.8 do not, and gpt-oss reports only "2.1M H100-hours" for the 120B model. If you assume 40% utilization of an H100's bf16 FLOPs, that is <code>2.1e6 * 3600 * 9.9e14 * 0.4 = 3e24</code> FLOPs, or about 100T tokens at 5.1B active parameters, which seems high, so the utilization or the token count is probably lower than that guess.</d-footnote>

You have probably noticed that the ratio of training tokens to active parameters is huge by Chinchilla<d-cite key="chinchilla"></d-cite> standards: 400 tokens per active parameter for DeepSeek-V3, 1,600 for Qwen3-235B, against the 20 that is compute-optimal for a dense model you only train once. These models are deliberately overtrained because inference cost scales with active parameters, and a small active count is worth paying for with extra training tokens. The total parameter count is what buys back the quality.

<p markdown=1 class="takeaway">**Takeaway:** Training FLOPs for an MoE are $6 \cdot \text{active params} \cdot \text{tokens}$. The frontier open models of 2025 and 2026 were trained for 3e24 to 1e25 FLOPs, less than LLaMA 3 405B, while carrying 2 to 7 times more parameters.</p>

### Fine-grained experts

Why did everyone converge on hundreds of experts of width 1,024 to 3,072 rather than eight of width 16,384? The modeling argument<d-cite key="deepseekmoe"></d-cite><d-cite key="finegrainedmoe"></d-cite> is combinatorial: choosing 8 of 256 experts gives the model $\binom{256}{8} \approx 4 \times 10^{14}$ distinct MLP configurations per token, against $\binom{8}{2} = 28$ for Mixtral, at the same active FLOPs. DeepSeekMoE's ablations found that splitting each expert four ways and activating four times as many beat the coarse design at equal active FLOPs, and its 16B model matched LLaMA 2 7B with about 40% of the compute.

The systems price is that every matmul got smaller. A dense LLaMA 3-70B MLP is a `[B, 8192] x [8192, 28672]` matmul. A DeepSeek-V3 expert is `[t_e, 7168] x [7168, 2048]`, where $t_e$ is the number of tokens that landed on that particular expert. Two things follow:

1. **The weight matrix is narrow.** $F = 2048$ is eight 256-wide MXU tiles on TPU v6e or TPU7x (Ironwood), which is fine. But if you were to shard that expert with tensor parallelism eight ways, each shard would be 256 wide, a single tile, and any padding or imbalance costs you a large fraction of the MXU. Tensor parallelism *within* experts is therefore unattractive, and we'll mostly shard *across* experts instead.

2. **The batch dimension is $t_e$, not $B$.** On average $t_e = Bk/E$. For the MXU to be efficient we need $t_e$ to be at least the systolic array height (128 or 256), and for the matmul to be compute-bound rather than weight-loading-bound we need $t_e$ to exceed the arithmetic intensity of the chip, which is 240 on TPU v5e and 311 on TPU7x in bf16 ([Section 2](https://jax-ml.github.io/scaling-book/tpus)). With $E/k = 32$, that means $B$ must exceed roughly `32 * 240 = 7680` tokens per layer-step before the experts stop being memory-bound. We'll return to this number, because it's the single most important consequence of sparsity for inference.

<p markdown=1 class="takeaway">**Takeaway:** Fine-grained experts turn one big matmul into $E$ small ones with an effective batch of $Bk/E$ each. Every efficiency argument in this book that depended on the batch being large must now be made with $Bk/E$ in place of $B$.</p>

## Routing Is a Systems Decision

In a dense model, the compiler knows the shape of every matmul before the program runs. In an MoE, the shape of every expert matmul is decided at runtime by the router, and so is the communication pattern. This makes three properties of the router matter to us as systems people.

**Load balance.** If expert 17 receives twice the average number of tokens, then whichever chip holds expert 17 has twice as much work, and every other chip waits for it. The efficiency of an expert-parallel layer is roughly $\text{mean load} / \text{max load}$ across chips. The old solutions were an auxiliary loss that pushes the router toward uniform usage<d-cite key="switch"></d-cite><d-cite key="stmoe"></d-cite>, and a *capacity factor*: give each expert a fixed buffer of $c \cdot Bk/E$ slots and drop tokens that overflow it<d-cite key="gshard"></d-cite>. Dropping tokens hurts quality and the auxiliary loss fights the language modeling loss, so the 2025 generation replaced both. DeepSeek-V3 adds a per-expert bias to the router score that is used only for the top-$k$ selection, and nudges the bias up for underloaded experts and down for overloaded ones after every step<d-cite key="auxlossfree"></d-cite>. Kimi K3 goes further and sets each bias from a quantile of the router scores so that every expert receives exactly its target load, computing the quantile from a histogram that is AllReduced across the cluster.<d-cite key="kimik3"></d-cite> None of the models in the table drop tokens. Instead, expert matmuls are *ragged* (in JAX, `jax.lax.ragged_dot`), and balance is enforced by training-time bookkeeping.

At inference time balance is handled differently: hot experts are simply duplicated. DeepSeek's serving system holds 32 redundant copies of routed experts per 32-GPU prefill unit and rebalances which experts are duplicated based on observed load<d-cite key="eplb"></d-cite>. Kimi K3's training system does the same thing *during training*, planning redundant experts per micro-batch so that every rank receives exactly $(B/Z) \cdot k$ token copies, which has the pleasant side effect that every matmul shape is static and the host never has to synchronize with the device to learn how big the next matmul is.<d-footnote>Static shapes matter more than they sound. In a conventional MoE implementation the host must wait for the router to finish to know the size of each expert's input before it can launch the expert kernels, which stalls the pipeline at every layer. With guaranteed balance the shapes are known in advance. XLA users will recognize this as the same reason dynamic shapes are painful on TPU.</d-footnote>

**Locality.** The router also decides how far each token's activations have to travel. DeepSeek-V3 constrains each token's 8 experts to lie on at most 4 of the 8 nodes that hold the experts (`n_group = 8`, `topk_group = 4` in the config), by first picking the 4 best nodes by summed affinity and then the 8 best experts within them. That halves the number of cross-node copies of each token from 8 to 4 and is a purely systems-motivated modification of the routing function. DeepSeek-V4 removed the constraint, and the same report describes a fused dispatch, expert-matmul and combine kernel that hides most of the AllToAll behind compute, which is presumably what made the constraint unnecessary.<d-cite key="deepseekv4"></d-cite>

**Latent routing.** Kimi K3 and NVIDIA's Nemotron 3 Super<d-cite key="nemotron3super"></d-cite> project each token down before dispatch: $z = W_\downarrow x \in \mathbb{R}^{\ell}$ with $\ell = D/2$ for K3 (3,584 of 7,168) and $\ell = D/4$ for Nemotron. The routed experts operate on $z$, and a $W_\uparrow$ maps the combined result back to $D$. The shared experts still see the full $x$. This halves (or quarters) the bytes that cross the network in both AllToAlls and shrinks each routed expert's matrices to $[\ell, F]$. It doesn't change the *ratio* of expert FLOPs to dispatch bytes, since both scale with $\ell$, but it does cut the absolute cost, and if you're communication-bound the absolute cost is what you feel.

<p markdown=1 class="takeaway">**Takeaway:** The router determines the shape of every expert matmul and the pattern of every AllToAll. Modern models keep it balanced with per-expert biases rather than dropped tokens, sometimes constrain where tokens can go to save network bandwidth, and sometimes shrink what gets sent, all of it for the hardware's sake.</p>

## What Sparsity Does to Memory

The obvious cost of a 2.8T-parameter model is that it is 2.8TB in fp8. Let's see what that does to us.

**Inference.** During inference we hold one copy of the weights. Here's the minimum number of chips needed just to *store* the weights of a few models, before a single byte of KV cache:

| Model | Weights | TPU v5e (16GB) | TPU v6e (32GB) | TPU7x (192GB) | H100 (80GB) | H200 (141GB) | B200 (180GB) |
| :---- | ------: | -------------: | -------------: | ------------: | ----------: | -----------: | -----------: |
| DeepSeek-V3, fp8 | 671GB | 42 | 21 | 4 | 9 | 5 | 4 |
| Kimi K2, fp8 | 1.04TB | 65 | 33 | 6 | 13 | 8 | 6 |
| DeepSeek-V4-Pro, fp4 experts | 0.83TB | 52 | 26 | 5 | 11 | 6 | 5 |
| Kimi K3, fp4 experts | 1.42TB | 89 | 45 | 8 | 18 | 11 | 8 |
| Kimi K3, fp8 | 2.78TB | 174 | 87 | 15 | 35 | 20 | 16 |
| Qwen3.8-2.4T, fp8 | 2.4TB | 150 | 75 | 13 | 30 | 18 | 14 |

On the inference chips of 2024 (v5e, H100) these models don't fit on a node, so you have *no choice* but to shard the weights across many chips, and [Section 7](https://jax-ml.github.io/scaling-book/inference) taught us that the only sharding we can afford at decode time is model parallelism (moving activations, not weights). Expert parallelism is a form of model parallelism, which is why it's the default for MoE serving. The 2026 hardware (TPU7x, B200, GB200 NVL72 with 72 GPUs and 13.4TB of HBM), on the other hand, was sized so that a trillion-parameter model *does* fit in one scale-up domain, which is no accident.

Note also how much fp4 matters here. Kimi K3 and DeepSeek-V4 both ship their expert weights in a 4-bit microscaling format (MXFP4) trained with quantization-aware training, halving the weight footprint relative to fp8 and doubling the batch size you can fit. [Section 15](../training-2026) discusses how that is done.

**Training.** During training we store weights, gradients and optimizer state. [Section 5](https://jax-ml.github.io/scaling-book/training) used 10 bytes per parameter (bf16 weights plus two fp32 Adam moments). For DeepSeek-V3 that is 6.7TB and for Kimi K3 it would be 28TB (K3 actually trained with Muon, which [Section 15](../training-2026) shows brings the figure to 6 bytes and 17TB; we keep Section 5's 10 here to compare like with like). That sounds terrifying, but spread across a TPU7x pod of 9,216 chips it is 3GB per chip, and across 2,048 GPUs it is 14GB. Training memory for weights isn't a per-chip problem at pod scale. It's a problem in three other ways:

* You cannot train these models on a small cluster. The "how few chips can I train on" exercise from [Section 6](https://jax-ml.github.io/scaling-book/applied-training) gives `28e12 / 192e9 = 146` TPU7x chips for Kimi K3 before any activations; with that section's two bf16 checkpoints per layer, a 60M-token batch adds `2 * 7168 * 60e6 * 2 * 93 = 160TB` of activations, so the real floor is about `188e12 / 192e9 = 980` chips. LLaMA 3-70B's weights and optimizer state needed `0.7e12 / 96e9 = 8` v5p chips (that section found 117 once activation checkpoints were included). Activations still dominate for an MoE, but by 6x rather than 15x, because sparsity grew the weights by $E/k$ without growing the activations.
* The optimizer state is 32 to 64 times larger *relative to the compute you do per step* than for a dense model. FSDP's AllGather and ReduceScatter traffic is proportional to weight bytes, and the FLOPs that hide it are proportional to active parameters. We'll see this inflate the FSDP roofline by $E/k$.
* Checkpoints are enormous. A Kimi K3 checkpoint with optimizer state is 17 to 28TB depending on the optimizer, written every few hours.

Activations, by contrast, look like those of a dense model whose MLP width is $(k + E_s) F$. There's nothing new there, except that the ragged expert inputs are awkward to checkpoint (Kimi K3 devotes a section of its report to a custom activation manager for exactly this reason).

<p markdown=1 class="takeaway">**Takeaway:** An MoE's weights are $E/k$ times larger than a dense model with the same per-token cost. At inference this forces model parallelism across many chips just to hold the weights, and makes 4-bit weights worth real money. In training it inflates every roofline that compares weight bytes to FLOPs by $E/k$, while per-chip memory at pod scale stays manageable.</p>

## The Batch-Size Rooflines

Now let's redo the two batch-size rooflines that dense-model intuition gets most wrong.

**Decode.** In [Section 7](https://jax-ml.github.io/scaling-book/inference) we found that a generate step is compute-bound once the batch (in tokens) exceeds the chip's arithmetic intensity $B_\text{crit} = C / W_\text{hbm}$, times $\text{bytes}/2$ for weights not in bf16. That calculation assumed every weight loaded from HBM is used by every token; Question 5 of [Section 7](https://jax-ml.github.io/scaling-book/inference) already noted that an MoE inflates it by $E/k$, and here is the derivation. In an MoE layer we must load *all* $E$ experts (some token in the batch will want each of them) but each token only does FLOPs against $k$:

$$\begin{align*}
T_\text{math} &= \frac{2 \cdot B \cdot k \cdot 3DF}{C} \\
T_\text{HBM} &= \frac{E \cdot 3DF \cdot \text{bytes}}{W_\text{hbm}}
\end{align*}$$

so we are compute-bound when

$$B > \frac{E}{k} \cdot \frac{\text{bytes}}{2} \cdot \frac{C}{W_\text{hbm}} = \frac{E}{k} \cdot B_\text{crit, dense}$$

Here's what that gives for a few models, with weights in bf16 (halve every number for fp8 weights with bf16 FLOPs, as in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference)):

| Model | $E/k$ | TPU v5e (240) | TPU7x (311) | H100 (296) | B200 (281) |
| :---- | ----: | ------------: | ----------: | ---------: | ---------: |
| DeepSeek-V3, GLM-5, gpt-oss | 32 | 7,700 | 10,000 | 9,500 | 9,000 |
| Kimi K2 | 48 | 11,500 | 14,900 | 14,200 | 13,500 |
| Qwen3.8-2.4T | 51 | 12,300 | 15,900 | 15,100 | 14,400 |
| Kimi K3 | 56 | 13,400 | 17,400 | 16,500 | 15,700 |
| DeepSeek-V4-Pro | 64 | 15,400 | 19,900 | 18,900 | 18,000 |

To be compute-bound while decoding DeepSeek-V3 you need to be generating for about ten thousand sequences *at once, in one model replica*. That's a remarkably large batch. Ten thousand sequences at 32k context with DeepSeek-V3's 35kB-per-token KV cache ([Section 14](../attention)) is 11TB of KV cache. So for any realistic deployment, **MoE decode is memory-bound on weights, on KV cache, or both**, and the arithmetic intensity of the MXU is irrelevant. What you can do is spread the $E \cdot 3DF$ bytes of weights over as many chips as possible so that each chip's share is small, and then you are bound by KV cache bandwidth instead. That's what DeepSeek does with 144-way expert parallelism at decode, and it is why [Section 14](../attention) spends so long on shrinking the KV cache.

The same thing shows up in the MXU: at batch 256 with 64-way expert parallelism, each chip's 4 DeepSeek-V3 experts see `256 * 8 / 256 = 8` tokens apiece. Eight rows of a 256-row systolic array.

**Training with FSDP.** In [Section 5](https://jax-ml.github.io/scaling-book/training) we found that FSDP is compute-bound when the per-chip batch exceeds $C / W_\text{ici}$, which is 2,550 for TPU v5p per ICI axis, or 850 with three axes. The derivation compared per-layer FLOPs $\propto B \cdot 3DF$ against per-layer weight-gather bytes $\propto 3DF$. For an MoE the FLOPs are $\propto B k \cdot 3DF$ and the gathered bytes are $\propto E \cdot 3DF$, so

$$\frac{B}{X} > \frac{E}{k} \cdot \frac{C}{M_X \cdot W_\text{ici}}$$

For DeepSeek-V3 on TPU v5p with three axes that's `32 * 850 = 27,200` tokens per chip! On a full 8,960-chip pod that's a 244M-token batch, four times what DeepSeek actually trained with. For Kimi K3 it is `56 * 850 = 47,600`. **Pure FSDP for a fine-grained MoE is off the table.** We need to shard the experts some other way, and that's what expert parallelism is for.

<p markdown=1 class="takeaway">**Takeaway:** Both the decode batch roofline and the FSDP training roofline are inflated by $E/k$. At sparsity 32 to 64, decode needs ten to twenty thousand concurrent tokens per replica to be compute-bound, which never happens, and FSDP alone needs tens of thousands of tokens per chip, which is impractical. Every MoE deployment therefore shards experts across chips.</p>

## The Expert-Parallelism Rooflines

Expert parallelism (EP) shards the expert dimension: $W_\text{in}[E_Z, D, F]$, $W_\text{out}[E_Z, F, D]$ over an axis $Z$ of the mesh, so that each chip holds $E/Z$ experts. Since tokens live on whatever chip processed the previous layer, we have to move each token's activations to the chips holding its $k$ experts and move the results back. That's two AllToAlls per layer ([Section 3](https://jax-ml.github.io/scaling-book/sharding)).

{% details Here's the full algorithm. %}

<div markdown=1 class="algorithm">

**Expert-parallel MoE layer (forward pass):**

1. Scores[B<sub>Z</sub>, E] = In[B<sub>Z</sub>, D] \*<sub>D</sub> W<sub>r</sub>[D, E] (*router; tiny*)
2. Idx[B<sub>Z</sub>, k], Gate[B<sub>Z</sub>, k] = **TopK**(Scores) (*decides the AllToAll pattern*)
3. Send[B<sub>Z</sub>, k, D] = **Gather**(In, Idx) (*k copies of each token*)
4. Recv[(Bk/Z), D] = **AllToAll**<sub>Z</sub>(Send) (*dispatch: each token goes to the chips owning its experts*)
5. Hidden[(Bk/Z), F] = **RaggedDot**(Recv, W<sub>in1</sub>[E<sub>Z</sub>, D, F]) \* **RaggedDot**(Recv, W<sub>in2</sub>[E<sub>Z</sub>, D, F]) (*grouped by local expert*)
6. Out'[(Bk/Z), D] = **RaggedDot**(Hidden, W<sub>out</sub>[E<sub>Z</sub>, F, D])
7. Back[B<sub>Z</sub>, k, D] = **AllToAll**<sub>Z</sub>(Out') (*combine: results return to the token's home chip*)
8. Out[B<sub>Z</sub>, D] = Σ<sub>k</sub> Gate \* Back + SharedExpert(In) (*weighted sum*)

</div>

The backward pass is the transpose: two more AllToAlls carrying gradients of the same shapes. If the shared expert is present it is computed locally (it is replicated or tensor-sharded like a dense MLP) and overlaps with the AllToAlls.

{% enddetails %}

Like tensor parallelism, this layout moves *activations* rather than weights, so its communication cost is proportional to $B$ and independent of $E$. Unlike tensor parallelism, each token is sent $k$ times (once per expert), so the bytes are $k$ times a plain activation resharding.<d-footnote>Implementations send one copy per <em>destination chip</em> rather than per expert, so if two of a token's experts share a chip only one copy travels. For $Z \gg k$ this is a small correction. DeepSeek-V3's node-limited routing exploits exactly this: by confining a token's 8 experts to 4 nodes it guarantees at most 4 cross-node copies.</d-footnote>

### Expert parallelism on TPUs

Let's compute the roofline in the book's notation. Take a global batch of $B$ tokens, EP over $Z$ chips arranged as a sub-mesh whose longest axis has length $A$, and expert width $F$. Following [Section 5](https://jax-ml.github.io/scaling-book/training) we consider the forward pass, and we count all three expert matmuls this time rather than two, since for narrow experts the gating einsum isn't something to ignore.<d-footnote>Section 5 dropped the gating matmul and used $4BDF$ FLOPs per layer to keep the algebra clean. For MoEs with $F = 2048$ we keep all three matrices ($6BDF$). DeepSeek's own overlap analysis in the V4 report also uses the three-matmul count, so the thresholds below match theirs. Be aware of what the choice does to the numbers: counting all three matrices makes every EP threshold 1.5x more permissive than Section 5's two-matmul convention, and fp8 dispatch another 1.33x, so under Section 5's $4BDF$, bf16 model you would double every $F$ in the tables below (v5p at $A = 4$ becomes 2,550, which DeepSeek-V3's 2,048-wide experts don't clear).</d-footnote> Per chip:

$$T_\text{math} = \frac{6 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

Each chip dispatches $Bk/Z$ token copies of $D$ elements at $b_d$ bytes each, and receives the same number back at $b_c$ bytes each. From [Section 3](https://jax-ml.github.io/scaling-book/sharding), an AllToAll that moves $M$ bytes in total, $M/N$ per chip, on a mesh whose longest axis is $A$ costs $(M/N) \cdot A / (4 W_\text{ici})$ per chip, so

$$T_\text{comms} = \frac{(b_d + b_c) \cdot B \cdot k \cdot D}{Z} \cdot \frac{A}{4 W_\text{ici}}$$

Setting $T_\text{math} > T_\text{comms}$, the batch, $k$, $D$ and $Z$ all cancel:

$$F > \frac{(b_d + b_c)}{24} \cdot A \cdot \frac{C}{W_\text{ici}}$$

With bf16 activations both ways ($b_d + b_c = 4$) this is $F > A \cdot \alpha / 6$ where $\alpha = C / W_\text{ici}$ is the ICI operational intensity from [Section 5](https://jax-ml.github.io/scaling-book/training). With fp8 dispatch and bf16 combine, which is what DeepSeek does, it is $F > A \cdot \alpha / 8$.

This is a strange and useful result. **Whether expert parallelism is compute-bound depends on the expert width $F$ and the longest axis of the EP mesh, and on nothing else.** Not the batch size, not $k$, not the number of experts, not even $Z$ except through $A$. It's the analog of [Section 5](https://jax-ml.github.io/scaling-book/training)'s tensor-parallel result $F > Y \cdot \alpha / M_Y$ (which in this three-matmul convention reads $F > 2 Y \alpha / (3 M_Y)$), with the AllToAll's factor of four discount and the wraparound torus doing the rest.

Here's the minimum expert width, with fp8 dispatch, for the chips in this book:

| Chip | $\alpha = C / W_\text{ici}$ (bf16) | $A = 2$ | $A = 4$ | $A = 8$ | $A = 16$ |
| :--- | ---------------------------------: | ------: | ------: | ------: | -------: |
| TPU v5e | 2,190 | 550 | 1,100 | 2,200 | 4,400 |
| TPU v5p | 2,550 | 640 | 1,280 | 2,550 | 5,100 |
| TPU v6e | 5,110 | 1,280 | 2,560 | 5,110 | 10,200 |
| TPU7x | 12,800 | 3,200 | 6,400 | 12,800 | 25,600 |

These thresholds assume a wraparound link (a bidirectional ring) on the longest EP axis. [Section 2](https://jax-ml.github.io/scaling-book/tpus) explains that v5e and v6e only have one on a full axis of 16, and v5p and TPU7x only on axes that are multiples of 4 inside a full cube, so for $A < 16$ on the 2D chips and for $A = 2$ on any chip, double the threshold: a 4x4 block on v6e needs about 5,100, not 2,560, and a 2x2x2 on TPU7x about 6,400, not 3,200.

Compare against the expert widths in the first table: 2,048 for DeepSeek-V3, Kimi K2 and GLM-5; 3,072 for DeepSeek-V4-Pro and Kimi K3; 1,024 for Qwen3.5.

* On **TPU v5p**, a 4x4x4 cube of 64-way EP ($A = 4$) needs $F > 1280$. Every model in the table clears it except Qwen3.5, whose 1,024-wide experts fall 20% short. A 4x4x8 (128-way) needs 2,550, which DeepSeek-V3's 2,048 doesn't quite reach.
* On **TPU v6e**, a 2D torus, a 4x4 (16-way) EP block has no wraparound and needs $F > 5100$. DeepSeek-style experts are communication-bound by 2.5x; Qwen3.5's 1,024-wide experts by 5x. Even a 2x2 block ($A = 2$, no wraparound, only 4-way EP) needs 2,560, which DeepSeek's experts miss by 25%.
* On **TPU7x**, even a 2x2x2 cube (no wraparound) needs $F > 6400$, and so does a 4x4x4 cube with its wraparound. **No fine-grained MoE in the table is compute-bound under expert parallelism on Ironwood** by this simple model. DeepSeek-V3's experts fall short by a factor of 3 on a cube; Kimi K3's by 2.

This is the first place in the book where a new chip makes a roofline *worse*. TPU7x has 5x the bf16 FLOPs/s of v5p (2.3e15 against 4.59e14) but the same 9e10 bytes/s per ICI link.<d-cite key="tpu7x"></d-cite> Its ICI operational intensity is therefore 5x higher, 12,800 against 2,550, and every roofline in [Section 5](https://jax-ml.github.io/scaling-book/training) that compares FLOPs to ICI bytes tightens by that factor. Tensor parallelism of a dense model becomes communication-bound beyond $Y > 3 F / 12800$, or about 6-way for LLaMA 3-70B. FSDP needs $12800 / 3 = 4270$ tokens per chip instead of 850. And expert parallelism, which for narrow experts was already the tightest of the three, needs the AllToAll to be hidden behind something other than the expert matmuls it serves.<d-footnote>Ironwood's HBM bandwidth did grow, from 2.8e12 to 7.4e12 bytes/s, so its HBM arithmetic intensity only doubled, from 164 to 311. The chip is designed around per-chip memory bandwidth and fp8 compute, which is to say around inference and around large-batch training, and its ICI has become the scarce resource for model-parallel training. If you run in fp8, $C$ doubles again and so does every threshold in this table.</d-footnote>

**What hides an AllToAll if not the expert matmuls?** The rest of the layer. Attention in these models is substantial (30% of active parameters for DeepSeek-V3, plus the quadratic term at long context), and with a pipeline schedule that interleaves two micro-batches you can run micro-batch 2's attention while micro-batch 1's dispatch is in flight. This is what DeepSeek's DualPipe schedule does<d-cite key="dualpipe"></d-cite>, and why Kimi K2 chose "the smallest feasible EP" of 16 so that the attention compute of one micro-batch is long enough to cover the AllToAll of another.<d-cite key="kimik2"></d-cite> The roofline above is the worst case where nothing else overlaps; the labs live in the space between it and perfect overlap.

There's also an alternative layout worth knowing. Instead of dispatching tokens to experts, we can **AllGather the tokens across the EP axis** (as tensor parallelism does), let each chip run its local experts on whichever of the gathered tokens routed to them, and ReduceScatter the results. Then

$$T_\text{comms} = \frac{4 \cdot B \cdot D}{M_Z \cdot W_\text{ici}} \qquad T_\text{math} = \frac{6 B k D F}{Z C}$$

which is compute-bound when $Z < 1.5 \cdot M_Z \cdot k F / \alpha$: this is tensor parallelism with an effective MLP width of $kF$. For DeepSeek-V3 ($kF = 16384$) on v5p with three axes that allows $Z < 29$, and on TPU7x $Z < 6$. Which layout is cheaper? The AllToAll dispatch costs $(b_d + b_c) k A / (4 Z)$ per token per axis-unit of bandwidth; the gather costs $4 / M_Z$. On a 4x4x4 cube with $k = 8$ the AllToAll is `3 * 8 * 4 / (4 * 64) = 0.375` against `4 / 3 = 1.33`, so dispatching is 3.5x cheaper. On a single ring of 16, it is `3 * 8 * 16 / (4 * 16) = 6` against `4`, and gathering wins by 1.5x. **Dispatch on a cube, gather on a ring.**

<p markdown=1 class="takeaway">**Takeaway:** On TPUs, expert parallelism with AllToAll dispatch is compute-bound when $F > A \cdot \alpha / 8$ (fp8 dispatch), where $A$ is the longest axis of the EP mesh and $\alpha = C / W_\text{ici}$. This is independent of batch, $k$, $E$ and $Z$. On TPU v5p a 64-way EP cube works for 2,048-wide experts; on TPU7x, whose ICI intensity is 5x higher, the same experts are 3x communication-bound and the AllToAll must be overlapped with attention or another micro-batch.</p>

### Expert parallelism on GPUs

[Section 12](https://jax-ml.github.io/scaling-book/gpus) already worked out the GPU version, so we'll just restate it in the same three-matmul convention. On a switched network every byte a GPU sends leaves through its own egress port, so there's no $A/4$ factor:

$$T_\text{comms} = \frac{(b_d + b_c) \cdot B \cdot k \cdot D}{Z \cdot W_\text{egress}} \quad \Rightarrow \quad F > \frac{(b_d + b_c)}{6} \cdot \frac{C}{W_\text{egress}}$$

With fp8 dispatch and bf16 combine this is $F > \alpha / 2$, which is exactly the condition DeepSeek states in the V4 report for hiding the AllToAll behind expert compute.<d-cite key="deepseekv4"></d-cite>

| Network | $W_\text{egress}$ per GPU | $\alpha$ (bf16 FLOPs) | Minimum $F$ (fp8 dispatch) |
| :------ | ------------------------: | --------------------: | -------------------------: |
| H100 NVLink (in node) | 450 GB/s | 2,200 | 1,100 |
| GB200 NVL72 NVLink (72 GPUs) | 900 GB/s | 2,800 | 1,400 |
| H100 InfiniBand (across nodes) | 50 GB/s | 19,800 | 9,900 |
| B200 InfiniBand (across nodes) | 50 GB/s | 45,000 | 22,500 |

So within an NVLink domain, fine-grained experts are fine, and a GB200 NVL72 rack can run 64- or 72-way expert parallelism at full compute efficiency for anything 1,400 wide or more. Across InfiniBand, nothing is: DeepSeek-V3's experts fall short by 5x on H800s, and that is before you account for the fact that GPUs rarely achieve their peak network bandwidth on all-to-all traffic ([Section 12](https://jax-ml.github.io/scaling-book/gpus)). This is the whole reason DeepSeek-V3's training used node-limited routing, dedicated 20 of the H800's 132 streaming multiprocessors (SMs) to communication, and overlapped everything with DualPipe. It's also why the NVL72 rack, not the 8-GPU node, became the natural unit of MoE serving in 2025 and 2026: for the first time a trillion-parameter model's experts fit in one domain where the AllToAll is cheap.

### Combining expert parallelism with FSDP

In practice EP is one axis of a larger mesh. Suppose we do EP over $Z$ chips and FSDP over $X$, for $N = XZ$ total (this is [Section 5](https://jax-ml.github.io/scaling-book/training)'s $N$ for the chip count; in [Section 4](https://jax-ml.github.io/scaling-book/transformers) and in Sections 14 and 16, $N$ is the number of attention heads). Each chip holds $E/Z$ experts and gathers their weights over $X$ each layer, so the FSDP traffic per layer is proportional to $(E/Z) \cdot 3DF$ rather than $E \cdot 3DF$. Redoing the FSDP roofline:

$$\frac{B}{N} > \frac{E}{k Z} \cdot \frac{C}{M_X \cdot W_\text{ici}}$$

**Expert parallelism of degree $Z$ divides the FSDP inflation factor $E/k$ by $Z$.** Choose $Z = E/k$ and the MoE has the same FSDP roofline as a dense model. For DeepSeek-V3 that is $Z = 32$; for Kimi K3, $Z = 56$; for DeepSeek-V4-Pro, $Z = 64$. Notice that these are the sizes of EP that the labs actually use (DeepSeek-V3 trained with EP64), and notice that $Z = 64$ is a 4x4x4 TPU cube, which we just showed is compute-bound for the AllToAll on v5p. The two rooflines meet in the same place, which suggests the model is roughly right.

Pipeline parallelism helps too, exactly as in [Section 12](https://jax-ml.github.io/scaling-book/gpus): $P$ pipeline stages divide the weight bytes each chip must gather by $P$ without adding meaningful communication. DeepSeek-V3's EP64 x PP16 x DP2 layout has each GPU holding `256 / 64 = 4` experts for `61 / 16` layers, which is why 2-way ZeRO-1 data parallelism was all they needed.

<p markdown=1 class="takeaway">**Takeaway:** Expert parallelism of degree $Z$ shrinks the MoE FSDP penalty from $E/k$ to $E/(kZ)$. Setting $Z \approx E/k$ (32 to 64 for current models) recovers the dense-model roofline, and coincidentally is also about the largest EP that stays compute-bound on a TPU v5p cube or inside a GB200 NVL72 rack.</p>

## What the Frontier Labs Actually Do

Let's check the theory against the published training configurations we have.

**DeepSeek-V3** (2,048 H800s, 2024)<d-cite key="DeepSeek3"></d-cite>: 64-way EP across 8 nodes, 16-way pipeline parallelism with DualPipe, 2-way ZeRO-1 data parallelism. Batch 15,360 sequences of 4,096 tokens, so `62.9M` tokens per step or `30.7k` tokens per GPU. FP8 dispatch, bf16 combine. Node-limited routing to 4 nodes per token. The cross-node AllToAll is about 5x communication-bound by our roofline, hidden partly by DualPipe's overlap with the attention and MLP of the other micro-batch. The achieved utilization, which you computed in Question 7 of [Section 4](https://jax-ml.github.io/scaling-book/transformers), was about 22% of fp8 peak by that section's H800 figure and 17% by the H800's actual dense fp8 rate ([Section 15](../training-2026) redoes it), and now you know where the rest went.

**Kimi K2** (H800s, 2025)<d-cite key="kimik2"></d-cite>: 16-way pipeline parallelism with virtual stages, 16-way EP, ZeRO-1 data parallelism, bf16 weights with fp32 gradient accumulation, and fp8 *storage* (not compute) for a few insensitive activations. The report is explicit that EP was set to "the smallest feasible" value so that the attention compute of one micro-batch covers the EP AllToAll of another under a standard one-forward-one-backward (1F1B) pipeline schedule, and that a smaller EP group "relaxes expert-balance constraints". Note that with $Z = 16 < E/k = 48$, their FSDP-equivalent inflation was 3x, absorbed by the pipeline stages.

**Kimi K3** (2026)<d-cite key="kimik3"></d-cite>: pipeline parallelism with virtual stages, EP, ZeRO-1, pipeline-aware ZeRO-2 gradient sharding, and context parallelism for the linear-attention layers. Two things are new. Their expert-parallel dispatch library, MoonEP, plans redundant experts every micro-batch so that each rank receives exactly $(B/Z) \cdot k$ token copies, with a proof that at most $E/Z$ redundant expert slots per rank always suffice, which makes every expert matmul a static shape. And the shared experts are replicated across EP ranks and run on a separate stream while the AllToAll is in flight, which is the "hide it behind something else" strategy in its purest form.

**Gemma 4 26B-A4B** (TPU v6e, 2026)<d-cite key="gemma4"></d-cite> is the one TPU MoE data point we have: 128 experts of width 704 with 8 active, trained on 6,144 TPU v6e chips. Its expert width is well below the 5,100 that a 4x4 EP block on v6e would need, and its sparsity of 16 is mild, so one would expect it to have leaned on FSDP over many chips with a small EP factor and a large batch. Google says only that the optimizer state is ZeRO-3 (FSDP-style) sharded under GSPMD, and gives no expert-parallel degree.

The common thread: nobody runs expert parallelism unoverlapped. Everybody picks the EP degree that their network can hide behind the rest of the layer, and then fills in the remaining parallelism with pipelining and FSDP.

## What Should You Take Away from this Section?

* An MoE has $L (E + E_s) 3DF$ parameters and $L (k + E_s) 3DF$ active parameters plus attention. Training costs $6 \cdot \text{active params} \cdot \text{tokens}$ FLOPs. The 2026 frontier open models have total-to-active ratios of roughly 20 to 33 ($E/k$ of 32 to 64) and were trained for 3e24 to 1e25 FLOPs.

* Every roofline that compares weight bytes to FLOPs is inflated by $E/k$: decode needs $E/k \cdot B_\text{crit}$ tokens per step to be compute-bound (10k to 20k for current models, so decode is always memory-bound), and pure FSDP needs $E/k \cdot 850$ tokens per chip on v5p (27k+, so nobody does it).

* Expert parallelism moves activations, costs two AllToAlls per layer, and is compute-bound when $F > A \alpha / 8$ on TPUs (fp8 dispatch, $A$ = longest EP axis) or $F > \alpha / 2$ on GPUs. On v5p and inside a GB200 NVL72, 2,048-wide experts clear this at 64-way EP. Across InfiniBand or on TPU7x they do not, and the AllToAll must be overlapped with attention or another micro-batch.

* EP of degree $Z$ cuts the FSDP inflation to $E/(kZ)$; $Z \approx E/k$ recovers dense-model behavior. Pipeline parallelism cuts it further.

* The router is a systems component: it is kept balanced with per-expert biases instead of dropped tokens, it may be constrained to limit network fan-out, and its inputs may be projected to a latent to shrink what is sent.

* Hardware moved to meet the models: the GB200 NVL72 rack and the 192GB TPU7x chip both exist so that a trillion-parameter MoE fits in one fast domain. But TPU7x's ICI didn't scale with its FLOPs, and every model-parallel roofline in this book is 5x tighter on it than on v5p.

## Worked Problems

**Question 1 [Kimi K2 parameters]:** Using the [Kimi K2 config](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json) ($D = 7168$, $L = 61$ with 1 dense layer, $F_\text{dense} = 18432$, $E = 384$, $E_s = 1$, $k = 8$, $F = 2048$, $V = 163840$, MLA attention with 64 heads and the same latent dimensions as DeepSeek-V3), compute the total and active parameter counts. *Attention per layer for this MLA configuration is about 101M parameters; take that as given.*

{% details Click here for the answer. %}

| param | formula | count |
| :---- | :------ | ----: |
| Experts | 60 * 385 * 3 * 7168 * 2048 | **1017e9** |
| Dense MLP | 1 * 3 * 7168 * 18432 | **0.4e9** |
| Attention | 61 * 101e6 | **6.2e9** |
| Vocab | 2 * 163840 * 7168 | **2.3e9** |
| **Total** | | **1026e9** |

Kimi reports 1.04T, so we are within 2%. The shipped checkpoint (1.03e12 bytes in fp8, with `num_nextn_predict_layers: 0`) holds exactly our count, so the remaining 1% is either the report's rounding or an unreleased one-layer MTP block of about 17B that the K3 report credits K2 with; the config can't tell us which. Active: `60 * 9 * 3 * 7168 * 2048 = 23.8e9` for the experts, plus the same 0.4 + 6.2 + 2.3 = 8.9B, for **32.7B**, against the reported 32B. Note that K2 has half the attention heads of DeepSeek-V3 (64 vs 128), so attention is only 19% of its active parameters.

{% enddetails %}

**Question 2 [gpt-oss-120b on one GPU]:** gpt-oss-120b has $E = 128$, $k = 4$, $F = D = 2880$, $L = 36$, and ships its expert weights in MXFP4 (0.5 bytes per parameter) with everything else in bf16. Its attention alternates 18 full layers with 18 sliding-window layers (128 tokens), with 8 KV heads of dimension 64.

(a) Does it fit on one 80GB H100? How much room is left for KV cache?

(b) What is the lower bound on decode step time at small batch, and what batch size would be needed for the expert matmuls to be compute-bound (bf16 FLOPs)?

(c) At 8k context, how many sequences fit in the remaining memory with a bf16 KV cache, and what is the step time and throughput at that batch?

{% details Click here for the answer. %}

(a) Expert weights: `36 * 128 * 3 * 2880 * 2880 = 114.7e9` parameters at 0.5 bytes is `57.3GB`. Everything else (attention, router, embeddings) is about `2.1e9` parameters in bf16, `4.2GB`. Total `61.5GB`, leaving about `18.5GB` for KV cache and activations. It fits, which was clearly the design target.

(b) Loading 61.5GB from HBM at 3.35e12 bytes/s takes `18.4ms`, so at batch 1 the best you can do is about 54 tokens/s. To be compute-bound in the experts we need $B > (E/k) \cdot (\text{bytes}/2) \cdot C / W_\text{hbm} = 32 \cdot 0.25 \cdot 296 = 2,370$ tokens per step. Note that 4-bit weights cut this number by 4x relative to bf16 weights (which would need 9,500), which is a big part of why 4-bit matters for MoEs.

(c) Only the 18 full-attention layers hold a KV cache that grows with context: `2 * 8 * 64 * 18 * 2 bytes = 36.9kB` per token. The sliding layers hold at most 128 tokens each, `18 * 128 * 2048 bytes = 4.7MB` per sequence, fixed. At 8k context each sequence costs `8192 * 36.9e3 + 4.7e6 = 307MB`, so about `18.5e9 / 307e6 = 60` sequences fit. At batch 60 the step loads `61.5GB` of weights and `18.4GB` of KV, `24ms`, for a throughput of about `2,500` tokens/s on one GPU. We are nowhere near the 2,370-token batch that would make the experts compute-bound, so the GPU is loading weights and KV cache the entire time.

{% enddetails %}

**Question 3 [expert parallelism on TPU v6e]:** You want to train Qwen3-235B-A22B ($E = 128$, $k = 8$, $F = 1536$, $D = 4096$) on a TPU v6e 16x16 slice (256 chips, 2D torus, $\alpha = 5110$). What is the largest EP degree that stays compute-bound with fp8 dispatch? What about with the gather-based layout? What batch size does the FSDP roofline then require?

{% details Click here for the answer. %}

For AllToAll dispatch we need $F > A \alpha / 8$, so $A < 8 \cdot 1536 / 5110 = 2.4$. Only $A = 2$ works, a 2x2 block: **4-way EP**. For the gather layout we need $Z < 1.5 M_Z k F / \alpha = 1.5 \cdot 2 \cdot 8 \cdot 1536 / 5110 = 7.2$, so 4-way again (rounding down to a 2x2). Either way, ICI limits us to 4-way EP for these narrow experts on v6e.

With $Z = 4$, the FSDP inflation is $E / (kZ) = 128 / 32 = 4$, so we need `4 * 5110 / 2 = 10,200` tokens per chip over the remaining 64-way FSDP axis, or a global batch of `256 * 10,200 = 2.6M` tokens. That's a normal batch size for a model this size, so this works, but note that the large batch did all the work; EP contributed a factor of four. Qwen's 1,536-wide experts are simply too narrow for a 2D torus of this generation to spread widely.

{% enddetails %}

**Question 4 [Llama 4's odd choice]:** Llama 4 Maverick uses 128 experts of width 8,192 with $k = 1$ plus one shared expert, in every other layer. Compare it with DeepSeek-V3 on (a) the largest compute-bound EP cube on TPU v5p with bf16 dispatch and (b) the decode batch needed to be compute-bound on TPU v5e. What tradeoff did Meta make?

{% details Click here for the answer. %}

(a) With bf16 both ways the condition is $F > A \alpha / 6$, so $A < 6 \cdot 8192 / 2550 = 19$. Any cube up to $A = 16$ works, so Maverick could be expert-parallel across all 128 experts (a 4x4x8 mesh, $A = 8$) with plenty of margin. DeepSeek-V3's 2,048-wide experts allow only $A < 4.8$, so a 4x4x4 cube and no more.

(b) The decode roofline is $E/k \cdot B_\text{crit}$: `128 * 240 = 30,700` tokens per step for Maverick against `32 * 240 = 7,700` for DeepSeek-V3.

Meta chose wide, communication-friendly experts at the cost of a 4x higher batch requirement in decode. DeepSeek chose the reverse. Given that decode is memory-bound for both at any realistic batch, DeepSeek's choice costs little at inference and buys the combinatorial quality of fine-grained routing, which is probably why the rest of the field followed DeepSeek rather than Meta.

{% enddetails %}

**Question 5 [Kimi K3 on a TPU7x pod, hard]:** You want to pretrain Kimi K3 ($E = 896$, $k = 16$, $E_s = 2$, routed expert width $F = 3072$ on a latent of $\ell = 3584$, $D = 7168$, 93 layers of which 92 are MoE) on a full TPU7x pod of 9,216 chips with a 60M-token batch, in bf16 with fp8 dispatch. Take $\alpha = 12800$ and three ICI axes.

(a) Can you use pure FSDP?

(b) Pick an EP degree that makes FSDP compute-bound. Is the EP AllToAll itself compute-bound?

(c) Roughly what fraction of peak could you achieve if the AllToAll were not overlapped with anything? Routed experts are about 47% of K3's active FLOPs (the shared experts, latent projections and attention are the other 53%).

{% details Click here for the answer. %}

(a) The per-chip batch is `60e6 / 9216 = 6,500` tokens. Pure FSDP needs $E/k \cdot \alpha / 3 = 56 \cdot 4270 = 239,000$ tokens per chip, forty times what we have.

(b) With EP $Z$, we need `(56 / Z) * 4270 < 6500`, so $Z > 37$: a 4x4x4 cube ($Z = 64$, $A = 4$) works, giving a threshold of `(56 / 64) * 4270 = 3,740` tokens per chip. For the AllToAll we need $F > A \alpha / 8 = 4 \cdot 12800 / 8 = 6,400$. K3's experts are 3,072 wide, so the dispatch is about **2.1x communication-bound** on a cube, or 2.1x again on a 2x2x2 ($A = 2$ but no wraparound, so the threshold is also 6,400), and in any case $Z = 8$ leaves FSDP at `(56 / 8) * 4270 = 29,900` tokens per chip, which we don't have. The latent projection doesn't help the ratio: it halves the bytes and the FLOPs together.

(c) If the routed-expert time is 2.1x its compute time and routed experts are 47% of FLOPs, the step takes `0.53 + 0.47 * 2.1 = 1.5` times its compute-bound duration, so about 66% of whatever utilization you would otherwise reach. In practice you would overlap the dispatch with the attention layers and the shared experts (which K3 runs on a separate stream for exactly this reason) and recover most of that. But this is the calculation that tells you Ironwood's ICI, not its MXU, is what you'll be fighting.

{% enddetails %}

**Question 6 [DeepSeek-V3 on TPU v5p]:** Repeat the previous exercise for DeepSeek-V3 ($E = 256$, $k = 8$, $F = 2048$) on a full TPU v5p pod (8,960 chips, $\alpha = 2550$) with DeepSeek's actual 63M-token batch. Is there a configuration where both the FSDP and EP rooflines are satisfied without any overlap tricks? How does this compare with what DeepSeek achieved on 2,048 H800s?

**Question 7 [tokens per expert]:** For the configuration you found in Question 6, how many tokens does each expert on each chip see per step? Is the MXU well utilized? Now answer the same question for DeepSeek's decode deployment (144-way EP and about 88 sequences per GPU, as [Section 16](../applied-frontier) derives from the published throughput). What does this tell you about where the FLOPs go during decode?

**Question 8 [choose a chip]:** Suppose you had to serve Qwen3.8-2.4T-A95B ($E = 512$, $k = 10$, $F = 2048$, 2.4TB of fp8 weights) and could choose between a GB200 NVL72 rack (72 GPUs, 186GB and 8e12 bytes/s each, 900GB/s NVLink egress) and a 4x4x4 TPU7x cube (64 chips, 192GB and 7.4e12 bytes/s each). Which holds the weights with more room for KV cache? On which is 64-way expert parallelism compute-bound for the AllToAll? Which gives lower per-step weight-loading time at small batch?

<h3 markdown=1 class="next-section">That's all for Section 13. For Section 14, on what happened to attention, click [here](../attention).</h3>
