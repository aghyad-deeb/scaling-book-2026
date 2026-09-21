---
layout: distill
title: "How to Think About Mixture of Experts"
# permalink: /main/
description: "Nearly every frontier open-weight model released since this book was written is a Mixture of Experts, and most of them are far sparser than the MoEs of 2024. This section extends the Transformer math of Section 4 and the parallelism rooflines of Sections 5 and 7 to MoEs: how to count their parameters and FLOPs, why sparse models want enormous batches, what expert parallelism costs on GPUs and TPUs, and why routing has become a systems decision as much as a modeling one."
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
    subsections:
    - name: "Counting parameters"
    - name: "Counting FLOPs"
    - name: "Fine-grained experts"
  - name: "Routing Is a Systems Decision"
  - name: "What Sparsity Does to Memory"
  - name: "The Batch-Size Rooflines"
  - name: "The Expert-Parallelism Rooflines"
    subsections:
    - name: "Expert parallelism on GPUs"
    - name: "The same rooflines on a TPU torus"
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

_[Section 4](https://jax-ml.github.io/scaling-book/transformers) treated Mixture of Experts (MoE) models as a two-paragraph aside. That was reasonable in early 2025, when LLaMA 3 was the reference open model and it was dense. It isn't anymore. Nearly every open-weight model near the frontier today, above about 30B parameters, is an MoE, and the way they are sparse has changed: hundreds of small experts, a handful active per token, and a total parameter count that has run away from the active count by a factor of roughly 20 to 33, with $E/k$ sparsities of 32 to 64. This section works through the math of that design and what it does to every roofline in the book. As with the rest of the book, we care less about why MoEs are a good modeling idea and more about what they cost to run._

## What Changed Since 2024?

Here is a table you should build for yourself, in the spirit of the big table of open-source LLMs that [Section 6](https://jax-ml.github.io/scaling-book/applied-training) recommends you build. Every row comes from the model's `config.json` or its technical report.<d-footnote>Total and active counts are the ones the labs report. Conventions differ: DeepSeek and Kimi count the embedding and unembedding parameters in "active", GLM and Gemma do not (GLM-5's 40B is 42B with them, Gemma 4's 3.8B is 4.4B), and the multi-token prediction (MTP) module is usually excluded from both. The active count always includes the shared experts, which run for every token. The script that accompanies this chapter recomputes every row from the config files and lands within 2% of each technical-report total. We list the routed expert width $F$ (called <code>moe_intermediate_size</code> in most configs), which for MoEs is the number that matters for rooflines. Kimi K3 routes a 3,584-wide latent rather than the full 7,168-wide residual, which we discuss below.</d-footnote>

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

{% include figure.liquid path="assets/img/moe-total-vs-active.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> total versus active parameters for open-weight models released between 2023 and 2026. Dense models sit on the diagonal. The frontier MoEs of 2026 sit roughly 20 to 33x above it. Note that the active parameter counts of the largest 2026 models are still smaller than dense LLaMA 3 405B." %}

The reason this matters for us is simple. Every roofline in this book compares something proportional to FLOPs (which scale with *active* parameters) against something proportional to bytes (which scale with *total* parameters, for weights). An MoE with sparsity 32 decouples those two quantities by a factor of 32. Rules of thumb like "FSDP is compute-bound above 2,500 tokens per GPU across nodes" ([Section 12](https://jax-ml.github.io/scaling-book/gpus)) or "decode is compute-bound above batch 300" were derived for dense models where the two coincide. The book already sketched the fix in passing: Question 8 of [Section 4](https://jax-ml.github.io/scaling-book/transformers) and Question 5 of [Section 7](https://jax-ml.github.io/scaling-book/inference) multiply the decode roofline by $E/k$, and [Section 12](https://jax-ml.github.io/scaling-book/gpus) does the same for data parallelism and derives a GPU expert-parallel roofline. This section works those results out for fine-grained experts, for expert parallelism (EP) on the machines the models actually ran on (the H800's cut-down NVLink, the GB200 NVL72 rack, InfiniBand between them) and on TPU tori, and for EP combined with FSDP.

## MoE Transformer Math

Let's fix notation. An MoE layer replaces the single MLP block of [Section 4](https://jax-ml.github.io/scaling-book/transformers) with $E$ *routed experts*, each a gated MLP with its own three matrices $W_\text{in1}[D, F]$, $W_\text{in2}[D, F]$, $W_\text{out}[F, D]$, plus (usually) a small number $E_s$ of *shared experts* that every token passes through. A *router* $W_r[D, E]$ scores each token against each expert, and the top $k$ scores select which experts the token visits. The outputs of the $k$ experts are combined with weights derived from the router scores and added to the shared expert output.

{% include figure.liquid path="assets/img/moe-layer.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> a fine-grained MoE layer with one shared expert and $E$ routed experts, of which $k$ are active per token. The router is a single $[D, E]$ matmul. Note that the routed experts are much narrower than the shared expert or a dense MLP, and that the expensive part of the layer from a systems standpoint is not any matmul but the two AllToAlls that move tokens to their experts and back." %}

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
| **Active total** | | **37.5e9** |

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

1. **The weight matrix is narrow.** A `[t_e, 7168] x [7168, 2048]` matmul with a few hundred tokens is a skinny problem: on an H800 it is at most a few dozen Tensor Core tiles of work, not enough to keep 132 streaming multiprocessors busy on its own, which is why the labs wrote grouped-matmul kernels (DeepGEMM, and the ragged kernels in every MoE framework) that run all of a GPU's local experts as one launch. Shard that expert with tensor parallelism eight ways and each shard is 256 wide, a single tile, and any padding or imbalance costs you a large fraction of the unit. The same holds on a TPU, where $F = 2048$ is eight 256-wide MXU tiles and an eight-way shard is one tile. Tensor parallelism *within* experts is therefore unattractive, and we'll mostly shard *across* experts instead.

2. **The batch dimension is $t_e$, not $B$.** On average $t_e = Bk/E$. For the matmul unit to be efficient we need $t_e$ to fill its tile (64 rows for a Hopper Tensor Core instruction, 128 on Blackwell, 128 or 256 on a TPU systolic array), and for the matmul to be compute-bound rather than weight-loading-bound we need $t_e$ to exceed the accelerator's arithmetic intensity, which is 296 on an H800 or H100, 206 on an H200, 281 on a B200 and 311 on TPU7x in bf16 ([Section 12](https://jax-ml.github.io/scaling-book/gpus), [Section 2](https://jax-ml.github.io/scaling-book/tpus)). With $E/k = 32$, that means $B$ must exceed roughly `32 * 296 = 9,500` tokens per layer-step on an H800 before the experts stop being memory-bound. We'll return to this number, because it's the single most important consequence of sparsity for inference.

<p markdown=1 class="takeaway">**Takeaway:** Fine-grained experts turn one big matmul into $E$ small ones with an effective batch of $Bk/E$ each. Every efficiency argument in this book that depended on the batch being large must now be made with $Bk/E$ in place of $B$.</p>

## Routing Is a Systems Decision

In a dense model, the compiler knows the shape of every matmul before the program runs. In an MoE, the shape of every expert matmul is decided at runtime by the router, and so is the communication pattern. This makes three properties of the router matter to us as systems people.

**Load balance.** If expert 17 receives twice the average number of tokens, then whichever GPU holds expert 17 has twice as much work, and every other GPU waits for it. The efficiency of an expert-parallel layer is roughly $\text{mean load} / \text{max load}$ across GPUs. The old solutions were an auxiliary loss that pushes the router toward uniform usage<d-cite key="switch"></d-cite><d-cite key="stmoe"></d-cite>, and a *capacity factor*: give each expert a fixed buffer of $c \cdot Bk/E$ slots and drop tokens that overflow it<d-cite key="gshard"></d-cite>. Dropping tokens hurts quality and the auxiliary loss fights the language modeling loss, so the 2025 generation replaced both. DeepSeek-V3 adds a per-expert bias to the router score that is used only for the top-$k$ selection, and nudges the bias up for underloaded experts and down for overloaded ones after every step<d-cite key="auxlossfree"></d-cite>. Kimi K3 goes further and sets each bias from a quantile of the router scores so that every expert receives exactly its target load, computing the quantile from a histogram that is AllReduced across the cluster.<d-cite key="kimik3"></d-cite> None of the models in the table drop tokens. Instead, expert matmuls are *ragged* (in JAX, `jax.lax.ragged_dot`), and balance is enforced by training-time bookkeeping.

At inference time balance is handled differently: hot experts are simply duplicated. DeepSeek's serving system holds 32 redundant copies of routed experts per 32-GPU prefill unit and rebalances which experts are duplicated based on observed load<d-cite key="eplb"></d-cite>. Kimi K3's training system does the same thing *during training*, planning redundant experts per micro-batch so that every rank receives exactly $Bk/Z$ token copies, where $Z$ is the number of GPUs the experts are spread over (the expert-parallel degree we define below), which has the pleasant side effect that every matmul shape is static and the host never has to synchronize with the device to learn how big the next matmul is.<d-footnote>Static shapes matter more than they sound. In a conventional MoE implementation the host must wait for the router to finish to know the size of each expert's input before it can launch the expert kernels, which stalls the pipeline at every layer. With guaranteed balance the shapes are known in advance. XLA users will recognize this as the same reason dynamic shapes are painful on TPU.</d-footnote>

**Locality.** The router also decides how far each token's activations have to travel. DeepSeek-V3 constrains each token's 8 experts to lie on at most 4 of the 8 nodes that hold the experts (`n_group = 8`, `topk_group = 4` in the config), by first picking the 4 best nodes by summed affinity and then the 8 best experts within them. That halves the number of cross-node copies of each token from 8 to 4 and is a purely systems-motivated modification of the routing function. DeepSeek-V4 removed the constraint, and the same report describes a fused dispatch, expert-matmul and combine kernel that hides most of the AllToAll behind compute, which is presumably what made the constraint unnecessary.<d-cite key="deepseekv4"></d-cite>

**Latent routing.** Kimi K3 and NVIDIA's Nemotron 3 Super<d-cite key="nemotron3super"></d-cite> project each token down before dispatch: $z = W_\downarrow x \in \mathbb{R}^{\ell}$ with $\ell = D/2$ for K3 (3,584 of 7,168) and $\ell = D/4$ for Nemotron. The routed experts operate on $z$, and a $W_\uparrow$ maps the combined result back to $D$. The shared experts still see the full $x$. This halves (or quarters) the bytes that cross the network in both AllToAlls and shrinks each routed expert's matrices to $[\ell, F]$. It doesn't change the *ratio* of expert FLOPs to dispatch bytes, since both scale with $\ell$, but it does cut the absolute cost, and if you're communication-bound the absolute cost is what you feel.

<p markdown=1 class="takeaway">**Takeaway:** The router determines the shape of every expert matmul and the pattern of every AllToAll. Modern models keep it balanced with per-expert biases rather than dropped tokens, sometimes constrain where tokens can go to save network bandwidth, and sometimes shrink what gets sent, all of it for the hardware's sake.</p>

## What Sparsity Does to Memory

The obvious cost of a 2.8T-parameter model is that it is 2.8TB in fp8. Let's see what that does to us.

**Inference.** During inference we hold one copy of the weights. Here's the minimum number of GPUs (or TPUs) needed just to *store* the weights of a few models, before a single byte of KV cache:

| Model | Weights | H800/H100 (80GB) | H200 (141GB) | B200 (180GB) | GB200 (186GB) | TPU7x (192GB) | TPU v6e (32GB) | TPU v5e (16GB) |
| :---- | ------: | ---------------: | -----------: | -----------: | ------------: | ------------: | -------------: | -------------: |
| DeepSeek-V3, fp8 | 671GB | 9 | 5 | 4 | 4 | 4 | 21 | 42 |
| Kimi K2, fp8 | 1.04TB | 13 | 8 | 6 | 6 | 6 | 33 | 65 |
| DeepSeek-V4-Pro, MXFP4 experts (as shipped) | 0.87TB | 11 | 7 | 5 | 5 | 5 | 28 | 55 |
| Kimi K3, MXFP4 experts (as shipped) | 1.56TB | 20 | 12 | 9 | 9 | 9 | 49 | 98 |
| Kimi K3, fp8 | 2.78TB | 35 | 20 | 16 | 15 | 15 | 87 | 174 |
| Qwen3.8-2.4T, fp8 | 2.4TB | 30 | 18 | 14 | 13 | 13 | 75 | 150 |

Now compare those counts with the size of a fast domain. An 8-GPU H800 node has 640GB of HBM and holds none of these models. An 8-GPU H200 node (1.13TB) holds DeepSeek-V3 with room for cache and Kimi K2 with none. An 8-GPU B200 node (1.44TB) holds V4-Pro's fp4 checkpoint (0.87TB) with room for cache, but not K3's (1.56TB), which needs an 8-GPU GB300 node (2.3TB), and that is exactly what Kimi's vLLM recipe asks for.<d-footnote>MXFP4 with a group of 32 and one E8M0 scale per group costs 4.25 bits per parameter, not 4, and only the routed experts are 4-bit: Kimi K3 keeps attention, shared experts, latent projections and the output head in bf16 (about 0.11TB), DeepSeek-V4-Pro keeps them in fp8 (about 0.03TB). The idealized 0.5 bytes per parameter would give 1.42TB and 0.83TB; the shipped checkpoints are 1.56TB and 0.87TB, and we use the shipped sizes throughout.</d-footnote> A GB200 NVL72 rack (72 GPUs, 13.4TB) holds every model in the table with terabytes to spare, and so does a 64-chip TPU7x cube (12.3TB). On the 2024 hardware these models don't fit in a node, so you have *no choice* but to shard the weights across nodes, and [Section 7](https://jax-ml.github.io/scaling-book/inference) taught us that the only sharding we can afford at decode time is model parallelism (moving activations, not weights). Expert parallelism is a form of model parallelism, which is why it's the default for MoE serving, and on H800s it has to cross InfiniBand.

Note also how much fp4 matters here. Kimi K3 and DeepSeek-V4 both ship their expert weights in a 4-bit microscaling format (MXFP4) trained with quantization-aware training, halving the weight footprint relative to fp8 and doubling the batch size you can fit. [Section 15](../training-2026) discusses how that is done.

**Training.** During training we store weights, gradients and optimizer state. [Section 5](https://jax-ml.github.io/scaling-book/training) used 10 bytes per parameter (bf16 weights plus two fp32 Adam moments). For DeepSeek-V3 that is 6.7TB and for Kimi K3 it would be 28TB (K3 actually trained with Muon, which [Section 15](../training-2026) shows brings the figure to 6 bytes and 17TB; we keep Section 5's 10 here to compare like with like). That sounds terrifying, but K3's 28TB spread across 2,048 GPUs is 14GB per GPU, and across a 9,216-chip TPU7x pod 3GB per chip. Kimi K2's report gives the one measured number we have: bf16 weights plus fp32 gradient buffers for its 1.04T parameters come to about 6TB, which over its 256-GPU model-parallel group is 24GB per GPU, and about 30GB per GPU of total state once the sharded optimizer is added.<d-cite key="kimik2"></d-cite> Training memory for weights isn't a per-GPU problem at cluster scale. It's a problem in three other ways:

* You cannot train these models on a small cluster. The "how few chips can I train on" exercise from [Section 6](https://jax-ml.github.io/scaling-book/applied-training) gives `28e12 / 80e9 = 350` H800s for Kimi K3 before any activations, or `28e12 / 192e9 = 146` TPU7x chips; with that section's two bf16 checkpoints per layer, a 60M-token batch adds `2 * 7168 * 60e6 * 2 * 93 = 160TB` of activations, so the real floor is about `188e12 / 80e9 = 2,350` H800s or 980 TPU7x chips. LLaMA 3-70B's weights and optimizer state needed `0.7e12 / 96e9 = 8` v5p chips (that section found 117 once activation checkpoints were included). Activations still dominate for an MoE, but by 6x rather than 15x, because sparsity grew the weights by $E/k$ without growing the activations.
* The optimizer state is 32 to 64 times larger *relative to the compute you do per step* than for a dense model. FSDP's AllGather and ReduceScatter traffic is proportional to weight bytes, and the FLOPs that hide it are proportional to active parameters. We'll see this inflate the FSDP roofline by $E/k$.
* Checkpoints are enormous. A Kimi K3 checkpoint with optimizer state is 17 to 28TB depending on the optimizer, written every few hours.

Activations, by contrast, look like those of a dense model whose MLP width is $(k + E_s) F$. There's nothing new there, except that the ragged expert inputs are awkward to checkpoint (Kimi K3 devotes a section of its report to a custom activation manager for exactly this reason).

<p markdown=1 class="takeaway">**Takeaway:** An MoE's weights are $E/k$ times larger than a dense model with the same per-token cost. At inference this forces model parallelism across many GPUs just to hold the weights (across nodes, on anything older than an NVL72 rack), and makes 4-bit weights worth real money. In training it inflates every roofline that compares weight bytes to FLOPs by $E/k$, while per-GPU memory at cluster scale stays manageable.</p>

## The Batch-Size Rooflines

Now let's redo the two batch-size rooflines that dense-model intuition gets most wrong.

**Decode.** In [Section 7](https://jax-ml.github.io/scaling-book/inference) we found that a generate step is compute-bound once the batch (in tokens) exceeds the accelerator's arithmetic intensity $B_\text{crit} = C / W_\text{hbm}$, times $\text{bytes}/2$ for weights not in bf16. That calculation assumed every weight loaded from HBM is used by every token; Question 5 of [Section 7](https://jax-ml.github.io/scaling-book/inference) already noted that an MoE inflates it by $E/k$, and here is the derivation. In an MoE layer we must load *all* $E$ experts (some token in the batch will want each of them) but each token only does FLOPs against $k$:

$$\begin{align*}
T_\text{math} &= \frac{2 \cdot B \cdot k \cdot 3DF}{C} \\
T_\text{HBM} &= \frac{E \cdot 3DF \cdot \text{bytes}}{W_\text{hbm}}
\end{align*}$$

so we are compute-bound when

$$B > \frac{E}{k} \cdot \frac{\text{bytes}}{2} \cdot \frac{C}{W_\text{hbm}} = \frac{E}{k} \cdot B_\text{crit, dense}$$

Here's what that gives for a few models, with weights in bf16 (halve every number for fp8 weights with bf16 FLOPs, as in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference)):

| Model | $E/k$ | H800/H100 (296) | H200 (206) | B200 (281) | GB200 (312) | TPU7x (311) | TPU v5e (240) |
| :---- | ----: | --------------: | ---------: | ---------: | ----------: | ----------: | ------------: |
| DeepSeek-V3, GLM-5, gpt-oss | 32 | 9,500 | 6,600 | 9,000 | 10,000 | 10,000 | 7,700 |
| Kimi K2 | 48 | 14,200 | 9,900 | 13,500 | 15,000 | 14,900 | 11,500 |
| Qwen3.8-2.4T | 51 | 15,100 | 10,500 | 14,400 | 15,900 | 15,900 | 12,300 |
| Kimi K3 | 56 | 16,600 | 11,500 | 15,700 | 17,500 | 17,400 | 13,400 |
| DeepSeek-V4-Pro | 64 | 18,900 | 13,200 | 18,000 | 20,000 | 19,900 | 15,400 |

To be compute-bound while decoding DeepSeek-V3 you need to be generating for about ten thousand sequences *at once, in one model replica*. That's a remarkably large batch. Ten thousand sequences at 32k context with DeepSeek-V3's 35kB-per-token KV cache ([Section 14](../attention)) is 11TB of KV cache. So for any realistic deployment, **MoE decode is memory-bound on weights, on KV cache, or both**, and the arithmetic intensity of the Tensor Core is irrelevant. What you can do is spread the $E \cdot 3DF$ bytes of weights over as many GPUs as possible so that each GPU's share is small, and then you are bound by KV cache bandwidth instead. That's what DeepSeek does with 144-way expert parallelism at decode, and it is why [Section 14](../attention) spends so long on shrinking the KV cache.

What does each expert actually see in that deployment? DeepSeek's decode step has about 88 sequences on each of 144 H800s ([Section 16](../applied-frontier)), so `144 * 88 * 8 = 101k` token-expert pairs per step spread over 256 routed experts plus 32 redundant copies: about `101e3 / 288 = 350` tokens per expert copy. Each expert matmul is a `[350, 7168] x [7168, 2048]` problem, above the H800's intensity of 296, so on its own it would be compute-bound. The step as a whole is nowhere near it, because the weight bytes and the network are shared across all of it. Only in the small-batch corner where a single sequence's latency matters do the experts themselves starve: at 8 sequences per GPU it's `8 * 8 * 144 / 288 = 32` rows per expert, half a Tensor Core tile.

**Training with FSDP.** In [Section 12](https://jax-ml.github.io/scaling-book/gpus) we found that FSDP on GPUs is compute-bound when the per-GPU batch exceeds $C / W_\text{collective}$, where $W_\text{collective}$ is a GPU's NVLink egress inside a node (2,200 tokens on an H100) and a whole node's InfiniBand egress across nodes (`990e12 / 400e9 = 2,475`). Section 12's 2,475 is at the H100's bf16 peak; DeepSeek ran fp8 matmuls, which doubles every FSDP threshold in this chapter (4,950 per GPU for a dense model), and we quote both where it matters. [Section 5](https://jax-ml.github.io/scaling-book/training)'s TPU version is $C / (M_X W_\text{ici})$, where $M_X$ is the number of ICI axes the gather spans: 850 on v5p with three axes. Either way the derivation compared per-layer FLOPs $\propto B \cdot 3DF$ against per-layer weight-gather bytes $\propto 3DF$. For an MoE the FLOPs are $\propto B k \cdot 3DF$ and the gathered bytes are $\propto E \cdot 3DF$, so

$$\frac{B}{N} > \frac{E}{k} \cdot \frac{C}{W_\text{collective}}$$

For DeepSeek-V3 across H800 nodes that's `32 * 2475 = 79,200` tokens per GPU at bf16 peak (158,400 at the fp8 peak the run used), and DeepSeek had `62.9e6 / 2048 = 30,700`: 2.6x (5.2x) short. For Kimi K3 it's `56 * 2475 = 139,000`. A GB200 NVL72 rack does better, because a rack of 72 GPUs pulls each weight shard in once through `72 * 50 = 3,600GB/s` of InfiniBand egress and shares it over NVLink, so the threshold is `2.5e15 / 3.6e12 = 694` tokens per GPU for a dense model and `32 * 694 = 22,200` for DeepSeek-V3, still more than anyone runs. On TPU v5p it's `32 * 850 = 27,200` tokens per chip, a 244M-token batch on a full pod. **Pure FSDP for a fine-grained MoE is off the table on every machine.** We need to shard the experts some other way, and that's what expert parallelism is for.

<p markdown=1 class="takeaway">**Takeaway:** Both the decode batch roofline and the FSDP training roofline are inflated by $E/k$. At sparsity 32 to 64, decode needs ten to twenty thousand concurrent tokens per replica to be compute-bound, which never happens, and FSDP alone needs tens of thousands of tokens per GPU, which is impractical. Every MoE deployment therefore shards experts across GPUs.</p>

## The Expert-Parallelism Rooflines

Expert parallelism (EP) shards the expert dimension: $W_\text{in}[E_Z, D, F]$, $W_\text{out}[E_Z, F, D]$ over an axis $Z$ of the mesh, so that each GPU holds $E/Z$ experts. Since tokens live on whatever GPU processed the previous layer, we have to move each token's activations to the GPUs holding its $k$ experts and move the results back. That's two AllToAlls per layer ([Section 3](https://jax-ml.github.io/scaling-book/sharding)).

{% details Here's the full algorithm. %}

<div markdown=1 class="algorithm">

**Expert-parallel MoE layer (forward pass):**

1. Scores[B<sub>Z</sub>, E] = In[B<sub>Z</sub>, D] \*<sub>D</sub> W<sub>r</sub>[D, E] (*router; tiny*)
2. Idx[B<sub>Z</sub>, k], Gate[B<sub>Z</sub>, k] = **TopK**(Scores) (*decides the AllToAll pattern*)
3. Send[B<sub>Z</sub>, k, D] = **Gather**(In, Idx) (*k copies of each token*)
4. Recv[(Bk/Z), D] = **AllToAll**<sub>Z</sub>(Send) (*dispatch: each token goes to the GPUs owning its experts*)
5. Hidden[(Bk/Z), F] = **RaggedDot**(Recv, W<sub>in1</sub>[E<sub>Z</sub>, D, F]) \* **RaggedDot**(Recv, W<sub>in2</sub>[E<sub>Z</sub>, D, F]) (*grouped by local expert*)
6. Out'[(Bk/Z), D] = **RaggedDot**(Hidden, W<sub>out</sub>[E<sub>Z</sub>, F, D])
7. Back[B<sub>Z</sub>, k, D] = **AllToAll**<sub>Z</sub>(Out') (*combine: results return to the token's home GPU*)
8. Out[B<sub>Z</sub>, D] = Σ<sub>k</sub> Gate \* Back + SharedExpert(In) (*weighted sum*)

</div>

The backward pass is the transpose: two more AllToAlls carrying gradients of the same shapes. If the shared expert is present it is computed locally (it is replicated or tensor-sharded like a dense MLP) and overlaps with the AllToAlls.

{% enddetails %}

Like tensor parallelism, this layout moves *activations* rather than weights, so its communication cost is proportional to $B$ and independent of $E$. Unlike tensor parallelism, each token is sent $k$ times (once per expert), so the bytes are $k$ times a plain activation resharding.<d-footnote>Implementations send one copy per <em>destination GPU</em> rather than per expert, so if two of a token's experts share a GPU only one copy travels. For $Z \gg k$ this is a small correction. DeepSeek-V3's node-limited routing exploits exactly this: by confining a token's 8 experts to 4 nodes it guarantees at most 4 cross-node copies.</d-footnote>

### Expert parallelism on GPUs

Let's compute the roofline in the book's notation, on the machine the models were actually built on. Take a global batch of $B$ tokens, EP over $Z$ GPUs, and expert width $F$. Following [Section 5](https://jax-ml.github.io/scaling-book/training) we consider the forward pass, and we count all three expert matmuls this time rather than two, since for narrow experts the gating einsum isn't something to ignore.<d-footnote>Section 5 dropped the gating matmul and used $4BDF$ FLOPs per layer to keep the algebra clean. For MoEs with $F = 2048$ we keep all three matrices ($6BDF$). DeepSeek's own overlap analysis in the V4 report also uses the three-matmul count, so the thresholds below match theirs. Be aware of what the choice does to the numbers: counting all three matrices makes every EP threshold 1.5x more permissive than Section 5's two-matmul convention, and fp8 dispatch another 1.33x, so under Section 5's $4BDF$, bf16 model you would double every $F$ in the tables below.</d-footnote> Per GPU:

$$T_\text{math} = \frac{6 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

Each GPU dispatches $Bk/Z$ token copies of $D$ elements at $b_d$ bytes each, and receives the same number back at $b_c$ bytes each. On a switched network (NVLink inside the domain, InfiniBand across it) every byte a GPU sends leaves through its own egress port at $W_\text{egress}$ bytes per second, so as in [Section 12](https://jax-ml.github.io/scaling-book/gpus)

$$T_\text{comms} = \frac{(b_d + b_c) \cdot B \cdot k \cdot D}{Z \cdot W_\text{egress}}$$

Setting $T_\text{math} > T_\text{comms}$, the batch, $k$, $D$ and $Z$ all cancel:

$$F > \frac{(b_d + b_c)}{6} \cdot \frac{C}{W_\text{egress}} = \frac{(b_d + b_c)}{6} \cdot \alpha$$

where $\alpha = C / W_\text{egress}$ is the network operational intensity of [Section 12](https://jax-ml.github.io/scaling-book/gpus). With bf16 activations both ways ($b_d + b_c = 4$) this is $F > 2\alpha / 3$. With fp8 dispatch and bf16 combine, which is what DeepSeek does, it's $F > \alpha / 2$, which is exactly the condition DeepSeek states in the V4 report for hiding the AllToAll behind expert compute.<d-cite key="deepseekv4"></d-cite>

This is a strange and useful result. **Whether expert parallelism is compute-bound depends on the expert width $F$ and on which network the AllToAll crosses, and on nothing else.** It doesn't depend on the batch size, on $k$, on $E$, or on the EP degree $Z$, except that $Z$ decides whether you are still inside the NVLink domain. It's the analog of [Section 12](https://jax-ml.github.io/scaling-book/gpus)'s tensor-parallel result $Y < F W / C$, and like that result it turns a question about parallelism into a question about one architectural constant.

Here's the minimum expert width for the GPUs the 2026 models run on, with fp8 dispatch and bf16 combine. The two columns matter because $\alpha$ doubles when the matmuls run in fp8: the same AllToAll now has half as much compute to hide behind.

| Network | $W_\text{egress}$ per GPU | $\alpha$ (bf16 FLOPs) | Min $F$, bf16 matmuls | Min $F$, fp8 matmuls |
| :------ | ------------------------: | --------------------: | --------------------: | -------------------: |
| H100 NVLink (8-GPU node) | 450GB/s | 2,200 | 1,100 | 2,200 |
| H800 NVLink (8-GPU node), spec | 200GB/s | 4,950 | 2,475 | 4,950 |
| H800 NVLink, as DeepSeek measures it | 160GB/s | 6,190 | 3,090 | 6,190 |
| HGX B200 NVLink (8-GPU node) | 900GB/s | 2,500 | 1,250 | 2,500 |
| GB200 NVL72 NVLink (72 GPUs) | 900GB/s | 2,780 | 1,390 | 2,780 |
| H800/H100 InfiniBand, one 400Gb/s NIC per GPU | 50GB/s | 19,800 | 9,900 | 19,800 |
| HGX B200 InfiniBand, 400Gb/s per GPU | 50GB/s | 45,000 | 22,500 | 45,000 |
| GB200 NVL72 InfiniBand, 400Gb/s per GPU | 50GB/s | 50,000 | 25,000 | 50,000 |
| GB300 NVL72 InfiniBand, 800Gb/s per GPU (Blackwell Ultra, the 2026 refresh of the rack) | 100GB/s | 25,000 | 12,500 | 25,000 |

The H800 rows deserve a word, because the H800 is the GPU DeepSeek-V3 was trained and served on and it isn't an H100. It has the same Tensor Cores and HBM, but NVIDIA cut its NVLink to 400GB/s bidirectional, 200GB/s each way in this book's convention, against the H100's 900 and 450.<d-cite key="deepseek_isca"></d-cite><d-footnote>Section 12 of the original book quotes 300GB/s for the H800. DeepSeek's own hardware paper says "the NVLink bandwidth in H800 SXM nodes is reduced from 900 GB/s to 400 GB/s", and its V3 report measures "160GB/s, roughly 3.2 times that of IB (50GB/s)" in practice. We use 200 as the spec figure and 160 as the achieved one. Section 4's Question 7 also used an fp8 rate of 1.51e15 for the H800, which is the PCIe part; the SXM part DeepSeek used has the H100's 1.98e15.</d-footnote> Compare the table against the expert widths in the first table: 2,048 for DeepSeek-V3, Kimi K2 and GLM-5; 3,072 for DeepSeek-V4-Pro and Kimi K3; 1,024 for Qwen3.5.

* **Inside an H100 or B200 node**, 2,048-wide experts clear the bf16 threshold with room to spare (1,100 on an H100, 1,250 on a B200) but fall just short of the fp8 one (2,200 on an H100, 2,500 on a B200: 7% and 18% short). Inside an **H800 node they miss even with bf16 matmuls** (2,475 spec, 3,090 measured: 17% to 34% short), and with the fp8 matmuls DeepSeek used the threshold is 4,950 to 6,190, 2.4x to 3x above them, before a single token leaves the node.
* **Inside a GB200 NVL72 rack**, 72-way expert parallelism is compute-bound for anything wider than 1,390 with bf16 matmuls, or 2,780 with fp8. DeepSeek-V4-Pro's and Kimi K3's 3,072-wide experts clear both. DeepSeek-V3's 2,048 clears the first and misses the second by 26%, which is fine for serving, where the step is memory-bound anyway, and a nuisance for training.
* **Across InfiniBand, nothing is close.** DeepSeek-V3's experts fall short by `9900 / 2048 = 4.8x` with bf16 matmuls and 9.7x with the fp8 matmuls the run actually used. (The InfiniBand rows assume the fat tree keeps its full 50GB/s per GPU above the node, as [Section 12](https://jax-ml.github.io/scaling-book/gpus) does. DeepSeek describes a two-layer, eight-plane fat tree; Kimi's cluster is RoCE at the same 400Gb/s per GPU; neither says whether the upper tier is oversubscribed, as Llama 3's was at 7:1 above the pod, so treat these as the best case.) That's before you account for the fact that GPUs rarely reach peak bandwidth on all-to-all traffic: DeepSeek's DeepEP kernels measure 153 to 158GB/s over NVLink and 43 to 58GB/s over InfiniBand on H800s, depending on the kernel version.<d-cite key="deepep"></d-cite> This is why the DeepSeek-V3 paper spends so many pages on communication; we check the numbers against their layout below.

It's also why the NVL72 rack, not the 8-GPU node, became the natural unit of MoE serving in 2025 and 2026: for the first time a trillion-parameter model's experts fit in one domain where the AllToAll is cheap. Cross the rack boundary and the egress per GPU drops from 900GB/s to 50, an 18x cliff, so a 72-way EP that spills into a second rack is worse than a 64-way one that doesn't.

**What hides an AllToAll if not the expert matmuls?** The rest of the layer. Attention in these models is substantial (30% of active parameters for DeepSeek-V3, plus the quadratic term at long context), and with a pipeline schedule that interleaves two micro-batches you can run micro-batch 2's attention while micro-batch 1's dispatch is in flight. This is what DeepSeek's DualPipe schedule does<d-cite key="dualpipe"></d-cite>, and it is the reason Kimi K2 kept EP at 16, as we'll see below. The roofline above is the worst case where nothing else overlaps; the labs live in the space between it and perfect overlap.

There's one alternative layout to rule out. Instead of dispatching tokens to experts, we can **AllGather the tokens across the EP group** (as tensor parallelism does), let each GPU run its local experts on whichever of the gathered tokens routed to them, and ReduceScatter the results. Then

$$T_\text{comms} = \frac{4 \cdot B \cdot D}{W_\text{egress}} \qquad T_\text{math} = \frac{6 B k D F}{Z C}$$

which is compute-bound when $Z < 1.5 \cdot k F / \alpha$: this is tensor parallelism with an effective MLP width of $kF$. For DeepSeek-V3 ($kF = 16384$) that allows $Z < 11$ over H100 NVLink, so it works inside a node, and $Z < 1.2$ over InfiniBand, so it never works across nodes. Per local token the dispatch layout moves $(b_d + b_c) k D = 3kD$ bytes and the gather layout $4ZD$, so dispatch is cheaper for any $Z > 0.75k$, which is $Z > 6$ for $k = 8$. On a switched network there is no regime where gathering wins for an EP group large enough to matter, and nobody uses it.

<p markdown=1 class="takeaway">**Takeaway:** On GPUs, expert parallelism with AllToAll dispatch is compute-bound when $F > \alpha / 2$ with fp8 dispatch and bf16 matmuls, or $F > \alpha$ with fp8 matmuls, where $\alpha = C / W_\text{egress}$ of whichever network the tokens cross. This is independent of batch, $k$, $E$ and $Z$. Inside an H100 node or a GB200 NVL72 rack, 2,048- to 3,072-wide experts clear it with bf16 matmuls; with fp8 matmuls only the 3,072-wide experts of V4-Pro and K3 do, and 2,048 misses by 7% (H100), 18% (B200) and 26% (GB200). Inside an H800 node 2,048 misses even with bf16 matmuls; across InfiniBand they miss by 5 to 10x, and the AllToAll must be overlapped with attention or another micro-batch.</p>

### The same rooflines on a TPU torus

On a TPU the AllToAll doesn't leave through a private port; it hops across a torus. From [Section 3](https://jax-ml.github.io/scaling-book/sharding), an AllToAll that moves $M/N$ bytes per chip on a mesh whose longest axis is $A$ costs $(M/N) \cdot A / (4 W_\text{ici})$ per chip, so

$$T_\text{comms} = \frac{(b_d + b_c) \cdot B \cdot k \cdot D}{Z} \cdot \frac{A}{4 W_\text{ici}} \quad \Rightarrow \quad F > \frac{(b_d + b_c)}{24} \cdot A \cdot \frac{C}{W_\text{ici}}$$

With fp8 dispatch and bf16 combine this is $F > A \cdot \alpha_\text{ici} / 8$: the same shape as the GPU result, with the factor of four from the bidirectional ring and the longest axis $A$ of the EP sub-mesh taking the place of the switch. Here's the minimum expert width, with fp8 dispatch and bf16 matmuls, for the TPUs in this book (double it for fp8 matmuls, as on GPUs):

| Chip | $\alpha = C / W_\text{ici}$ (bf16) | $A = 2$ | $A = 4$ | $A = 8$ | $A = 16$ |
| :--- | ---------------------------------: | ------: | ------: | ------: | -------: |
| TPU v5e | 2,190 | 550 | 1,100 | 2,200 | 4,400 |
| TPU v5p | 2,550 | 640 | 1,280 | 2,550 | 5,100 |
| TPU v6e | 5,110 | 1,280 | 2,560 | 5,110 | 10,200 |
| TPU7x | 12,800 | 3,200 | 6,400 | 12,800 | 25,600 |

These thresholds assume a wraparound link (a bidirectional ring) on the longest EP axis. [Section 2](https://jax-ml.github.io/scaling-book/tpus) explains that v5e and v6e only have one on a full axis of 16, and v5p and TPU7x only on axes that are multiples of 4 inside a full cube, so for $A < 16$ on the 2D chips and for $A = 2$ on any chip, double the threshold: a 4x4 block on v6e needs about 5,100, not 2,560, and a 2x2x2 on TPU7x about 6,400, not 3,200.

* On **TPU v5p**, a 4x4x4 cube of 64-way EP ($A = 4$) needs $F > 1280$. Every model in the table clears it except Qwen3.5, whose 1,024-wide experts fall 20% short. A 4x4x8 (128-way) needs 2,550, which DeepSeek-V3's 2,048 doesn't quite reach.
* On **TPU v6e**, a 2D torus, a 4x4 (16-way) EP block has no wraparound and needs $F > 5100$. DeepSeek-style experts are communication-bound by 2.5x; Qwen3.5's 1,024-wide experts by 5x. Even a 2x2 block ($A = 2$, no wraparound, only 4-way EP) needs 2,560, which DeepSeek's experts miss by 25%.
* On **TPU7x**, even a 2x2x2 cube (no wraparound) needs $F > 6400$, and so does a 4x4x4 cube with its wraparound. **No fine-grained MoE in the table is compute-bound under expert parallelism on Ironwood** by this simple model. DeepSeek-V3's experts fall short by a factor of 3 on a cube; Kimi K3's by 2.

This is the first place in the book where a new chip makes a roofline *worse*. TPU7x has 5x the bf16 FLOPs/s of v5p (2.3e15 against 4.59e14) but the same 9e10 bytes/s per ICI link.<d-cite key="tpu7x"></d-cite> Its ICI operational intensity is therefore 5x higher, 12,800 against 2,550, and every roofline in [Section 5](https://jax-ml.github.io/scaling-book/training) that compares FLOPs to ICI bytes tightens by that factor. Tensor parallelism of a dense model becomes communication-bound beyond $Y > 3 F / 12800$, or about 6-way for LLaMA 3-70B. FSDP needs $12800 / 3 = 4270$ tokens per chip instead of 850. And expert parallelism, which for narrow experts was already the tightest of the three, needs the AllToAll to be hidden behind something other than the expert matmuls it serves.<d-footnote>Ironwood's HBM bandwidth did grow, from 2.8e12 to 7.4e12 bytes/s, so its HBM arithmetic intensity only doubled, from 164 to 311. The chip is designed around per-chip memory bandwidth and fp8 compute, which is to say around inference and around large-batch training, and its ICI has become the scarce resource for model-parallel training. GPUs went the other way: NVLink bandwidth doubled with each generation's FLOPs, so $\alpha_\text{nvlink}$ stayed between 2,200 and 2,800 from H100 to GB200, and the scarce resource on GPUs is the InfiniBand link out of the node.</d-footnote>

The gather layout also behaves differently on a torus. Its cost is $4BD / (M_Z W_\text{ici})$, compute-bound when $Z < 1.5 \cdot M_Z \cdot k F / \alpha$: for DeepSeek-V3 on v5p with three axes that allows $Z < 29$, and on TPU7x $Z < 6$. Per token per axis-unit of bandwidth the AllToAll costs $(b_d + b_c) k A / (4 Z)$ and the gather $4 / M_Z$. On a 4x4x4 cube with $k = 8$ the AllToAll is `3 * 8 * 4 / (4 * 64) = 0.375` against `4 / 3 = 1.33`, so dispatching is 3.5x cheaper. On a single ring of 16, it is `3 * 8 * 16 / (4 * 16) = 6` against `4`, and gathering wins by 1.5x. **Dispatch on a cube, gather on a ring.** There's no GPU analog of this, because a switch has no rings.

<p markdown=1 class="takeaway">**Takeaway:** On TPUs the same roofline picks up the torus factor: expert parallelism is compute-bound when $F > A \cdot \alpha_\text{ici} / 8$ (fp8 dispatch, bf16 matmuls), where $A$ is the longest axis of the EP mesh. On TPU v5p a 64-way EP cube works for 2,048-wide experts; on TPU7x, whose ICI intensity is 5x higher, the same experts are 3x communication-bound and the AllToAll must be overlapped with attention or another micro-batch, exactly as across InfiniBand on GPUs.</p>

### Combining expert parallelism with FSDP

In practice EP is one axis of a larger layout. Suppose we do EP over $Z$ GPUs and data parallelism with sharded weights (FSDP, or ZeRO-3) over $X$, for $N = XZ$ total (this is [Section 5](https://jax-ml.github.io/scaling-book/training)'s $N$ for the accelerator count; in [Section 4](https://jax-ml.github.io/scaling-book/transformers) and in Sections 14 and 16, $N$ is the number of attention heads). Each GPU holds $E/Z$ experts and gathers their weights over $X$ each layer, so the FSDP traffic per layer is proportional to $(E/Z) \cdot 3DF$ rather than $E \cdot 3DF$. Redoing the FSDP roofline, with $P$ pipeline stages also dividing the weight bytes each GPU must gather (as in [Section 12](https://jax-ml.github.io/scaling-book/gpus)):

$$\frac{B}{N} > \frac{E}{k Z P} \cdot \frac{C}{W_\text{collective}}$$

**Expert parallelism of degree $Z$ divides the FSDP inflation factor $E/k$ by $Z$.** Choose $Z = E/k$ and the MoE has the same FSDP roofline as a dense model. For DeepSeek-V3 that is $Z = 32$; for Kimi K3, $Z = 56$; for DeepSeek-V4-Pro, $Z = 64$. Notice that these are the sizes of EP that the labs actually use (DeepSeek-V3 trained with EP64), and notice that $Z = 64$ or 72 is also the largest EP that fits inside one GB200 NVL72 rack, where we just showed the AllToAll is compute-bound. The two rooflines meet in the same place, which suggests the model is roughly right.

Let's check it against the one fully published GPU layout. DeepSeek-V3 ran EP64 across 8 H800 nodes, PP16, and 2-way data parallelism with ZeRO-1 (optimizer state sharded, weights replicated, so the per-step cost is a gradient AllReduce rather than a per-layer gather). Plugging into the formula as if it were FSDP: `256 / (8 * 64 * 16) * 2475 = 77` tokens per GPU needed (155 in fp8) against the 30,700 they had. With EP and PP carrying almost all of the model parallelism, data parallelism had nothing left to do, which is why a plain 2-way ZeRO-1 was enough and why the run's difficulty was the AllToAll, not the weight traffic. On a GB200 NVL72 cluster the same arithmetic gives 72-way EP inside each rack and FSDP across racks a threshold of `(384 / (6 * 72)) * 1390 = 1,240` tokens per GPU (fp8 matmuls, bf16 weight gathers; 620 if the gathers are fp8) for DeepSeek-V4-Pro, against the 10,200 per GPU of a 94M-token batch on 9,216 GPUs. [Section 16](../applied-frontier) works that plan out in full.

On a v5p pod the same formula reads $B/N > E/(kZ) \cdot C / (M_X W_\text{ici})$, and $Z = 64$ (a 4x4x4 cube, which we showed is compute-bound for the AllToAll) gives `32 / 64 * 850 = 425` tokens per chip, against the 7,000 per chip of DeepSeek's batch on 8,960 chips.

<p markdown=1 class="takeaway">**Takeaway:** Expert parallelism of degree $Z$ shrinks the MoE FSDP penalty from $E/k$ to $E/(kZ)$, and pipeline parallelism divides it by $P$ again. Setting $Z \approx E/k$ (32 to 64 for current models) recovers the dense-model roofline, and coincidentally is also about the largest EP that fits inside a GB200 NVL72 rack or stays compute-bound on a TPU v5p cube. With EP and PP in place, the data-parallel axis of a 2026 training run is small and cheap.</p>

## What the Frontier Labs Actually Do

First, the hardware. This table is what the labs themselves say about where their models were trained and where they expect them to be served; "not disclosed" means the report and model card say nothing. Since DeepSeek-V3 no Chinese lab has published a GPU count or GPU-hours, and the Western labs that do (Meta, OpenAI, Mistral, Google) publish no layout. B300 and GB300 are the Blackwell Ultra parts; the H20 is NVIDIA's export-market Hopper for China; the MI355X is AMD's Blackwell-generation competitor.

| Model | Trained on | Published training layout | Serving hardware named in the lab's card or its day-0 recipes |
| :---- | :--------- | :------------------------ | :----------------------------- |
| DeepSeek-V3<d-cite key="DeepSeek3"></d-cite> | 2,048 H800 (8-GPU NVLink nodes, 400Gb/s InfiniBand per GPU), 2.79M GPU-hours | EP64 across 8 nodes, PP16 (DualPipe), ZeRO-1 DP2; fp8 | H800 nodes: EP32 prefill, EP144 decode ([Section 16](../applied-frontier)) |
| DeepSeek-V4-Pro, V4.1<d-cite key="deepseekv4"></d-cite> | not disclosed | Muon and AdamW; fp8 ([Section 15](../training-2026)) | 4 GB300 (model card) |
| Kimi K2<d-cite key="kimik2"></d-cite> | H800 cluster (8 x 400Gb/s RoCE per node), count not disclosed | PP16 with virtual stages, EP16, ZeRO-1; bf16 | 16 H200 or H20 for 1T weights; 8 H200 with int4 (K2 Thinking) |
| Kimi K3<d-cite key="kimik3"></d-cite> | not disclosed (RL on "a few hundred GPUs" per 1M-context experiment; evals on H20) | PP with virtual stages, EP, ZeRO-1, pipeline ZeRO-2, context parallelism | 8 GB300 or MI355X (the vLLM recipe the card links); 16 GB200 or 32 H200 (NVIDIA's Dynamo recipe)<d-cite key="dynamo"></d-cite> |
| GLM-4.5, GLM-5<d-cite key="glm5"></d-cite> | not disclosed | Muon; interleaved PP, pipeline ZeRO-2, CP groups; pretraining precision not stated (RL: bf16 training, fp8 rollouts) | GLM-4.5: 8 H200 or 16 H100 in bf16; GLM-5: 8 H200, plus seven domestic accelerators |
| Qwen3.5, Qwen3.8<d-cite key="qwen35"></d-cite> | not disclosed | native fp8 pipeline; otherwise unstated | TP8 on one B300 or MI355X node in fp4, two nodes in fp8, for the 2.4T model (vLLM day-0 post) |
| Nemotron 3 Super, Ultra<d-cite key="nemotron3ultra"></d-cite> | GB200 for the long-context phase; main phase not stated | Ultra: CP32, TP8, EP128, PP2 (long-context phase); NVFP4 | 8 GB200/B200/GB300, or 16 H100 |
| Llama 4 Scout, Maverick<d-cite key="llama4"></d-cite> | H100, 7.38M GPU-hours for the pair | not disclosed (Behemoth: fp8 on 32K GPUs at 390 TFLOP/s per GPU) | 8 H100 (Scout fits one) |
| gpt-oss-120b<d-cite key="gptoss"></d-cite> | H100, 2.1M GPU-hours | not disclosed | one 80GB H100 or MI300X with MXFP4 experts |
| Mistral Large 3<d-cite key="mistral3"></d-cite> | 3,000 H200 | not disclosed | 8 H200 or B200 in fp8; 8 H100 in fp4 |
| Gemma 4 26B-A4B<d-cite key="gemma4"></d-cite> | 6,144 TPU v6e | ZeRO-3 under GSPMD; EP degree not stated | consumer GPUs and workstations |

Three things to take from it. Every open frontier model outside Google and NVIDIA that discloses its training hardware trained on Hopper, and the two Chinese labs that disclose anything (DeepSeek for V3 and R1, Moonshot for Kimi K2; MiniMax names the H800 only for M1's RL run) trained on the H800, whose cut-down NVLink is the reason the DeepSeek-V3 paper reads the way it does. Only NVIDIA's own Nemotron is documented on Blackwell, and only for its long-context phase; Gemma is the only TPU-trained open model. And the *serving* recipes have already moved to Blackwell: DeepSeek's V4 card and the day-0 vLLM recipes for Kimi K3 and Qwen3.8 all name a GB300 or B300 node as the reference deployment for the 2026 models, while the 2025 models are still specified in H200s by the labs themselves. Note, though, that every Blackwell throughput number in this update comes from NVIDIA, SGLang, vLLM or a benchmark project; the only production serving figures any lab has published are still DeepSeek's from February 2025, on H800s, which is why [Section 16](../applied-frontier) is built on them. Now let's check the theory against the published training layouts.

**DeepSeek-V3** (2,048 H800s, 2024)<d-cite key="DeepSeek3"></d-cite>: 64-way EP across 8 nodes, 16-way pipeline parallelism with DualPipe, 2-way ZeRO-1 data parallelism. Batch 15,360 sequences of 4,096 tokens, so `62.9M` tokens per step or `30.7k` tokens per GPU. FP8 dispatch, bf16 combine. Node-limited routing to 4 nodes per token. The cross-node AllToAll is about 5x communication-bound by our roofline with bf16 matmuls and 10x with the fp8 matmuls the run used, hidden partly by DualPipe's overlap with the attention and MLP of the other micro-batch and by the 20 SMs reserved for it. The achieved utilization, which you computed in Question 7 of [Section 4](https://jax-ml.github.io/scaling-book/transformers), was about 22% of fp8 peak by that section's H800 figure and 17% by the H800's actual dense fp8 rate ([Section 15](../training-2026) redoes it); DeepSeek's own hardware paper reports 385 TFLOP/s per GPU in steady state, which is 19% of fp8 peak once attention FLOPs are counted, or the 39% of *bf16* peak they quote.<d-cite key="deepseek_isca"></d-cite> Now you know where the rest went.

**Kimi K2** (H800s with 8 x 400Gb/s RoCE per node rather than InfiniBand, 2025; RoCE is RDMA over Converged Ethernet, an Ethernet scale-out fabric with the same 50GB/s per GPU as the InfiniBand in this section's tables)<d-cite key="kimik2"></d-cite>: 16-way pipeline parallelism with virtual stages, 16-way EP, ZeRO-1 data parallelism, bf16 weights with fp32 gradient accumulation, and fp8 *storage* (not compute) for a few insensitive activations. The report is explicit that EP was set to "the smallest feasible" value so that the attention compute of one micro-batch covers the EP AllToAll of another under a standard one-forward-one-backward (1F1B) pipeline schedule, and that a smaller EP group "relaxes expert-balance constraints". It also says why it didn't use DualPipe: that schedule doubles the memory for parameters and gradients, which a 1T-parameter model on 80GB GPUs couldn't afford. Note that with $Z = 16 < E/k = 48$, their FSDP-equivalent inflation was 3x, absorbed by the pipeline stages.

**Kimi K3** (2026)<d-cite key="kimik3"></d-cite>: pipeline parallelism with virtual stages, EP, ZeRO-1, pipeline-aware ZeRO-2 gradient sharding, and context parallelism for the linear-attention layers. Two things are new. Their dispatch library, MoonEP, is the redundant-expert planner we described under Routing, with a proof that at most $E/Z$ redundant expert slots per rank always suffice. And the shared experts are replicated across EP ranks and run on a separate stream while the AllToAll is in flight, which is the "hide it behind something else" strategy in its purest form.

**Gemma 4 26B-A4B** (TPU v6e, 2026)<d-cite key="gemma4"></d-cite> is the one TPU MoE data point we have: 128 experts of width 704 with 8 active, trained on 6,144 TPU v6e chips. Its expert width is well below the 5,100 that a 4x4 EP block on v6e would need, and its sparsity of 16 is mild, so one would expect it to have leaned on FSDP over many chips with a small EP factor and a large batch. Google says only that the optimizer state is ZeRO-3 (FSDP-style) sharded under GSPMD, and gives no expert-parallel degree.

The common thread: nobody runs expert parallelism unoverlapped. Everybody picks the EP degree that their network can hide behind the rest of the layer, and then fills in the remaining parallelism with pipelining and FSDP.

## What Should You Take Away from this Section?

* An MoE has $L (E + E_s) 3DF$ parameters and $L (k + E_s) 3DF$ active parameters plus attention. Training costs $6 \cdot \text{active params} \cdot \text{tokens}$ FLOPs. The 2026 frontier open models have total-to-active ratios of roughly 20 to 33 ($E/k$ of 32 to 64) and were trained for 3e24 to 1e25 FLOPs.

* Every roofline that compares weight bytes to FLOPs is inflated by $E/k$: decode needs $E/k \cdot B_\text{crit}$ tokens per step to be compute-bound (10k to 20k for current models, so decode is always memory-bound), and pure FSDP needs $E/k \cdot 2{,}475$ tokens per GPU across H800 nodes (79k for DeepSeek-V3 at bf16 peak, 158k at the fp8 peak it used, so nobody does it; 22k on a GB200 rack, 27k on a v5p pod).

* Expert parallelism moves activations, costs two AllToAlls per layer, and is compute-bound when $F > \alpha / 2$ on GPUs (fp8 dispatch, bf16 matmuls; $F > \alpha$ with fp8 matmuls) where $\alpha = C / W_\text{egress}$ of the network crossed, or $F > A \alpha_\text{ici} / 8$ on TPUs ($A$ = longest EP axis). Inside an H100 node or a GB200 NVL72 rack, 2,048- to 3,072-wide experts clear this with bf16 matmuls, and only the 3,072-wide ones with fp8; inside an H800 node 2,048 misses even in bf16; across InfiniBand or on TPU7x they miss by 3 to 10x, and the AllToAll must be overlapped with attention or another micro-batch.

* EP of degree $Z$ cuts the FSDP inflation to $E/(kZ)$; $Z \approx E/k$ recovers dense-model behavior, and pipeline parallelism cuts it further, which is why DeepSeek-V3's data-parallel axis was only 2 wide.

* The router is a systems component: it is kept balanced with per-expert biases instead of dropped tokens, it may be constrained to limit network fan-out, and its inputs may be projected to a latent to shrink what is sent.

* Hardware moved to meet the models: the GB200 NVL72 rack and the 192GB TPU7x chip both exist so that a trillion-parameter MoE fits in one fast domain. NVLink kept pace with GPU FLOPs, so $\alpha_\text{nvlink}$ stayed near 2,500 and the cliff on GPUs is the InfiniBand link out of the domain. TPU7x's ICI didn't scale with its FLOPs, and every model-parallel roofline in this book is 5x tighter on it than on v5p.

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

**Question 3 [expert parallelism for narrow experts]:** You want to train Qwen3-235B-A22B ($E = 128$, $k = 8$, $F = 1536$, $D = 4096$) on 32 HGX B200 nodes (256 GPUs; NVLink 900GB/s per GPU, one 400Gb/s NIC per GPU, bf16 matmuls with fp8 dispatch). (a) What is the largest EP degree that stays compute-bound? What about with the gather-based layout? (b) What batch size does the FSDP roofline then require? (c) Redo (a) and (b) on a TPU v6e 16x16 slice (256 chips, 2D torus, $\alpha = 5110$).

{% details Click here for the answer. %}

(a) Inside a node $\alpha$ = `2.25e15 / 9e11 = 2,500` and the AllToAll needs $F > \alpha / 2 = 1{,}250$, which 1,536 clears: **8-way EP inside the node works**. Across InfiniBand $\alpha = 45{,}000$ and the threshold is 22,500, so the experts can't leave the node. For the gather layout we need $Z < 1.5 k F / \alpha = 1.5 \cdot 8 \cdot 1536 / 2500 = 7.4$, so 4-way; dispatch is the better layout here as everywhere on a switched fabric.

(b) With $Z = 8$ the FSDP inflation is $E / (kZ) = 128 / 64 = 2$. The cross-node FSDP threshold on an 8-GPU B200 node is $C / W_\text{node}$ = `2.25e15 / 4e11 = 5,625` tokens per GPU, so we need `2 * 5625 = 11,250` tokens per GPU, or a global batch of `256 * 11,250 = 2.9M` tokens (twice that with fp8 matmuls). That's a normal batch for a model this size, so the plan is EP8 inside each node and FSDP across the 32 nodes, with the batch doing most of the work.

(c) On v6e no EP block is compute-bound for 1,536-wide experts: a 2x2 block has no wraparound and needs $F > 2560$, a 4x4 needs 5,100, both above 1,536 (the gather layout on a 2x2 without wraparound allows only $Z < 3.6$, so it doesn't rescue 4-way either). You can run 4-way EP about 1.7x communication-bound and hide the AllToAll behind attention, or drop EP and use pure FSDP over the 16x16 torus, whose full axes do have wraparound: with inflation $E/k = 16$ that needs `16 * 5110 / 2 = 40,900` tokens per chip, a 10.5M-token global batch on 256 chips, 3.6x the GPU plan's 2.9M. Either way Qwen's 1,536-wide experts are too narrow for a 2D torus of this generation to spread, while an NVLink node spreads them eight ways without trying.

{% enddetails %}

**Question 4 [Llama 4's odd choice]:** Llama 4 Maverick uses 128 experts of width 8,192 with $k = 1$ plus one shared expert, in every other layer, and Meta trained it on H100s. Compare it with DeepSeek-V3 on (a) whether expert parallelism across InfiniBand is compute-bound with bf16 dispatch and bf16 matmuls, (b) the decode batch needed to be compute-bound on an H100, and (c) the same two numbers on TPU v5p and TPU v5e. What tradeoff did Meta make?

{% details Click here for the answer. %}

(a) With bf16 both ways the GPU condition is $F > 2\alpha / 3$, and across H100 InfiniBand $\alpha = 19{,}800$, so the threshold is 13,200. Maverick's 8,192-wide experts miss it by 1.6x (1.2x with fp8 dispatch), which is close enough to hide behind attention; DeepSeek-V3's 2,048-wide experts miss it by 6.4x.

(b) The decode roofline is $E/k \cdot B_\text{crit}$: `128 * 296 = 37,900` tokens per step for Maverick against `32 * 296 = 9,500` for DeepSeek-V3 on an H100.

(c) On a v5p cube the condition is $F > A \alpha / 6$, so $A < 6 \cdot 8192 / 2550 = 19$: any cube up to $A = 16$ works and Maverick could be expert-parallel across all 128 experts (a 4x4x8 mesh) with margin, while DeepSeek-V3's experts allow $A < 4.8$, a 4x4x4 cube and no more. On a v5e the decode batches are 30,700 against 7,700.

Meta chose wide, communication-friendly experts at the cost of a 4x higher batch requirement in decode. DeepSeek chose the reverse. Given that decode is memory-bound for both at any realistic batch, DeepSeek's choice costs little at inference and buys the combinatorial quality of fine-grained routing, which is probably why the rest of the field followed DeepSeek rather than Meta.

{% enddetails %}

**Question 5 [Kimi K3 on GB200 racks, hard]:** You want to pretrain Kimi K3 ($E = 896$, $k = 16$, $E_s = 2$, routed expert width $F = 3072$ on a latent of $\ell = 3584$, $D = 7168$, 93 layers of which 92 are MoE) on 128 GB200 NVL72 racks (9,216 GPUs, 5e15 fp8 FLOPs/s each, 900GB/s of NVLink egress inside the rack, one 400Gb/s NIC per GPU so 3.6TB/s of InfiniBand egress per rack) with a 60M-token batch, in fp8 with fp8 dispatch and bf16 weight gathers.

(a) Can you use pure FSDP?

(b) Pick an EP degree that makes FSDP compute-bound. Is the EP AllToAll itself compute-bound?

(c) Roughly what fraction of peak could you achieve if the AllToAll were not overlapped with anything? Routed experts are about 47% of K3's active FLOPs (the shared experts, latent projections and attention are the other 53%).

(d) Redo (a) to (c) on a full TPU7x pod of 9,216 chips, in bf16 matmuls with fp8 dispatch, with $\alpha = 12800$ and three ICI axes.

{% details Click here for the answer. %}

(a) The per-GPU batch is `60e6 / 9216 = 6,500` tokens. The FSDP threshold across racks is $C / W_\text{rack}$ = `5e15 / 3.6e12 = 1,390` tokens per GPU for a dense model in fp8, and pure FSDP for K3 needs $E/k$ times that, `56 * 1390 = 77,800`: twelve times what we have.

(b) With EP $Z$ inside a rack we need `(56 / Z) * 1390 < 6500`, so $Z > 12$; take $Z = 64$ (a rack has 72 GPUs, and 64 divides the 896 experts evenly), giving a threshold of `(56 / 64) * 1390 = 1,220` tokens per GPU, five times below what we have. For the AllToAll inside the rack we need $F > \alpha_\text{fp8} / 2$ = `(5e15 / 9e11) / 2 = 2,780` with fp8 matmuls and fp8 dispatch. K3's experts are 3,072 wide, so the dispatch is **compute-bound by 10%** with nothing overlapped at all. Both rooflines are satisfied.

(c) If the AllToAll is compute-bound, an unoverlapped dispatch costs no wall-clock by this model, so the answer is "whatever the rest of the step allows": the pipeline bubbles, the attention layers and the HBM traffic of the linear-attention state set the utilization, not the network.

(d) On TPU7x the per-chip batch is the same 6,500, and pure FSDP needs $E/k \cdot \alpha / 3 = 56 \cdot 4270 = 239{,}000$ tokens per chip, about 37 times what we have. With EP over a 4x4x4 cube ($Z = 64$, $A = 4$) the FSDP threshold is `(56 / 64) * 4270 = 3,740` tokens per chip, fine. But the AllToAll needs $F > A \alpha / 8 = 4 \cdot 12800 / 8 = 6{,}400$, so the dispatch is about **2.1x communication-bound** on a cube (a 2x2x2 has no wraparound and lands at the same 6,400, and $Z = 8$ would leave FSDP at 29,900 tokens per chip, which we don't have). The latent projection doesn't help the ratio: it halves the bytes and the FLOPs together. If the routed-expert time is 2.1x its compute time and routed experts are 47% of FLOPs, the step takes `0.53 + 0.47 * 2.1 = 1.5` times its compute-bound duration, so about 66% of whatever utilization you would otherwise reach. In practice you would overlap the dispatch with the attention layers and the shared experts (which K3 runs on a separate stream for exactly this reason) and recover most of that. With the same model, batch and chip count, the AllToAll is free on paper on the GPU cluster and 2.1x over budget on the TPU pod.

{% enddetails %}

**Question 6 [DeepSeek-V3 on its own cluster]:** DeepSeek-V3 ($E = 256$, $k = 8$, $F = 2048$) trained on 2,048 H800s with 64-way EP across 8 nodes, 16-way pipeline parallelism and 2-way ZeRO-1 data parallelism, a 63M-token batch, fp8 matmuls and fp8 dispatch. Check the three rooflines of this section against that layout: the data-parallel weight traffic, the in-node AllToAll (NVLink 200GB/s spec, 160 measured) and the cross-node AllToAll (50GB/s). Which one binds, by how much, and how does that square with the run's 17% utilization of fp8 peak and the 20 SMs DeepSeek reserved for communication? Then repeat the exercise for the same model on a full TPU v5p pod (8,960 chips, $\alpha = 2550$): is there a configuration where both the FSDP and EP rooflines are satisfied without any overlap tricks?

{% details Click here for the answer. %}

Data-parallel weight traffic: with EP64 and PP16 the FSDP-style threshold is `256 / (8 * 64 * 16) * 4950 = 155` tokens per GPU at fp8 peak against 30,700, so it never binds, which is why 2-way ZeRO-1 sufficed. The in-node AllToAll needs $F > 4{,}950$ (spec) or 6,190 (measured) with fp8 matmuls and fp8 dispatch, so 2,048-wide experts are 2.4x to 3x short inside the node. The cross-node AllToAll needs $F > 19{,}800$, 9.7x short, so InfiniBand binds, consistent with DeepSeek's reported one-to-one ratio of compute to communication and the 20 SMs it reserved. Utilization well under half of peak is what that predicts, and 17 to 19% is what they got. On a v5p pod a 4x4x4 cube needs $F > 1280$ (met) and FSDP with $Z = 64$ needs `32 / 64 * 850 = 425` tokens per chip against `63e6 / 8960 = 7,000`: both rooflines hold with no overlap tricks, at the price of 4.4x more chips.

{% enddetails %}

**Question 7 [tokens per expert]:** For the configurations in Question 6, how many tokens does each expert on each GPU or chip see per step? Is the matmul unit well utilized? Now answer the same question for DeepSeek's decode deployment (144-way EP and about 88 sequences per GPU, as [Section 16](../applied-frontier) derives from the published throughput). What does this tell you about where the FLOPs go during decode?

{% details Click here for the answer. %}

In training each GPU holds 4 experts and sees `30,700 * 8 / 4 = 61,400` token-expert pairs per step, about 15,000 rows per expert matmul, far above the 296 needed to be compute-bound; the same holds on the v5p pod (`7000 * 8 / 4 = 14,000` rows). In decode each expert copy sees about 350 rows per step (see the batch-size section), also above 296 per matmul, yet the step is bound by weight and KV bytes and by the network. The lesson is that per-matmul intensity isn't the constraint in either regime: training is bound by the AllToAll, decode by everything except the matmuls.

{% enddetails %}

**Question 8 [choose a chip]:** Suppose you had to serve Qwen3.8-2.4T-A95B ($E = 512$, $k = 10$, $F = 2048$, 2.4TB of fp8 weights) and could choose between a GB200 NVL72 rack (72 GPUs, 186GB and 8e12 bytes/s each, 900GB/s NVLink egress) and a 4x4x4 TPU7x cube (64 chips, 192GB and 7.4e12 bytes/s each). Which holds the weights with more room for KV cache? On which is 64-way expert parallelism compute-bound for the AllToAll? Which gives lower per-step weight-loading time at small batch?

<h3 markdown=1 class="next-section">That's all for Section 13. For Section 14, on what happened to attention, click [here](../attention).</h3>
