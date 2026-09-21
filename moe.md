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

## What Changed Since 2024?

So you want to run one of the newest open-weight models. Chances are it's a Mixture of Experts (MoE): more or less every open-weight model above about 30B parameters now is, and [Section 4](https://jax-ml.github.io/scaling-book/transformers) only spends about two paragraphs on them (which was fine in early 2025, when our reference open model was the dense LLaMA 3). These models are also sparse in a different way than the MoEs we glanced at there: hundreds of small experts, a handful active per token, and a total parameter count larger than the active count by a factor of roughly 20 to 33, i.e. an $E/k$ sparsity of 32 to 64. **What does that do to our rooflines?** Quite a lot, it turns out, so let's work through the math of the design and find out. As usual, we care less about *why* MoEs are a good modeling idea than about what they cost us to run.

Let's start with a table. [Section 6](https://jax-ml.github.io/scaling-book/applied-training) suggested you keep a big table of open-source LLMs for yourself, so here's ours for MoEs, and we'll refer back to it throughout this section. Every row comes straight from the model's `config.json` or its technical report, so you can check each one yourself.<d-footnote>Total and active counts are the ones each model's own report gives, and you should be aware that conventions differ: DeepSeek and Kimi count the embedding and unembedding parameters in "active" while GLM and Gemma do not (GLM-5's 40B is 42B with them, and Gemma 4's 3.8B is 4.4B), and the multi-token prediction (MTP) module is usually left out of both. The active count always includes the shared experts, since they run for every token. The script that accompanies this chapter recomputes every row from the config files and lands within 2% of each technical-report total. We list the routed expert width $F$ (called <code>moe_intermediate_size</code> in most configs), since for MoEs that's the number that matters for rooflines. Kimi K3 routes a 3,584-wide latent rather than the full 7,168-wide residual, which we'll get to below.</d-footnote>

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

You'll notice a few things:

* **Total parameters grew about 20x, but active parameters only about 2.5x.** Going from Mixtral 8x22B to Kimi K3 takes us from 141B to 2.8T total parameters, but only from 39B to 104B active. In other words, almost all of the new parameters are ones that any given token never touches.
* **Sparsity went from 4 to somewhere between 32 and 64.** The ratio $E/k$ (which this book calls the sparsity) was 4 for Mixtral, but for everything released since DeepSeek-V3 it's 16 or more, and the 2026 models sit between 32 and 64.
* **How wide is an expert?** Mixtral's experts were 16,384 wide, the same as a dense MLP, but DeepSeek, Kimi, GLM and Qwen use experts between 1,024 and 3,072 wide, with hundreds of them per layer. This "fine-grained" design<d-cite key="deepseekmoe"></d-cite> is now basically the default, with Llama 4 the exception (128 experts of width 8,192, one active per token).
* **Where does dense stop?** For most labs it stops at about 30B: Qwen3.6-27B and Gemma 4 31B are the largest recent dense open models from their labs. Mistral is the exception, with dense 123B (Devstral 2) and 128B (Medium 3.5) open weights. More or less every other model above that size is sparse.

{% include figure.liquid path="assets/img/moe-total-vs-active.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> total versus active parameters for open-weight models released between 2023 and 2026. Dense models sit on the diagonal. The frontier MoEs of 2026 sit roughly 20 to 33x above it. Note that the active parameter counts of the largest 2026 models are still smaller than dense LLaMA 3 405B." %}

**Why does this matter for us?** Every roofline in this book compares something proportional to FLOPs (which scale with *active* parameters) against something proportional to bytes (which, for weights, scale with *total* parameters). An MoE with sparsity 32 pulls those two quantities apart by a factor of 32. The rules of thumb we've been carrying around, like "FSDP is compute-bound above 2,500 tokens per GPU across nodes" ([Section 12](https://jax-ml.github.io/scaling-book/gpus)) or "decode is compute-bound above batch 300" were derived for dense models, where the two coincide. The fix is actually already sketched in this book, in passing: Question 8 of [Section 4](https://jax-ml.github.io/scaling-book/transformers) and Question 5 of [Section 7](https://jax-ml.github.io/scaling-book/inference) multiply the decode roofline by $E/k$, and [Section 12](https://jax-ml.github.io/scaling-book/gpus) does the same for data parallelism and derives a GPU expert-parallel roofline. In this section let's work those results out properly: first for fine-grained experts, then for expert parallelism (EP) on the machines the models were actually trained and served on (the H800's cut-down NVLink, the GB200 NVL72 rack, and the InfiniBand between them) and on TPU tori, and finally for EP combined with FSDP.

## MoE Transformer Math

Let's fix notation. An MoE layer replaces the single MLP block of [Section 4](https://jax-ml.github.io/scaling-book/transformers) with $E$ *routed experts*, each a gated MLP with its own three matrices $W_\text{in1}[D, F]$, $W_\text{in2}[D, F]$, $W_\text{out}[F, D]$, plus (usually) a small number $E_s$ of *shared experts* that every token passes through. A *router* $W_r[D, E]$ scores each token against each expert, and we use the top $k$ scores to select which experts the token visits. We then combine the outputs of the $k$ experts with weights derived from the router scores and add them to the shared expert output.

{% include figure.liquid path="assets/img/moe-layer.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> a fine-grained MoE layer with one shared expert and $E$ routed experts, of which $k$ are active per token. The router is a single $[D, E]$ matmul. Note that the routed experts are much narrower than the shared expert or a dense MLP, and that the expensive part of the layer from a systems standpoint is not any matmul but the two AllToAlls that move tokens to their experts and back." %}

**Note [sigmoid routing and shared experts]:** The original MoE papers<d-cite key="moe"></d-cite><d-cite key="switch"></d-cite> softmax the router scores across experts and pick the top $k$. DeepSeek-V3 and most 2026 models instead apply a sigmoid to each score independently, pick the top $k$, then renormalize the selected weights to sum to one. DeepSeek-V4 uses a square-root-of-softplus. Why should we care? Mainly because these choices interact with load balancing, which we'll discuss below. The shared expert is just a plain dense MLP that every token goes through. Since it's always active, from a systems standpoint it's basically a dense layer that happens to live next to the MoE, and we can shard it like one.

### Counting parameters

Let's start with parameters. Per MoE layer, the count is

$$P_\text{MoE layer} = \underbrace{(E + E_s) \cdot 3 D F}_{\text{experts}} + \underbrace{E D}_{\text{router}}$$

and the number of parameters any single token touches is

$$P_\text{MoE layer, active} = (k + E_s) \cdot 3 D F + E D$$

You can basically ignore the router: for DeepSeek-V3 it's `256 * 7168 = 1.8M` parameters per layer against 11.3B for the experts. The total model is

$$P_\text{total} = L_\text{MoE} (E + E_s) 3DF + L_\text{dense} \cdot 3 D F_\text{dense} + L \cdot P_\text{attn} + 2 D V$$

where several models keep the first one to three layers dense (`first_k_dense_replace` in DeepSeek-style configs), since routing on barely-embedded tokens turns out to work poorly.<d-footnote>DeepSeek-V4 went a different direction and made those first layers MoE as well, but with <em>hash routing</em>: the expert is chosen by a hash of the token id rather than by a learned router<d-cite key="hashlayers"></d-cite>. This removes the router entirely from those layers, which is nice for us as systems people because the dispatch pattern is known before the layer runs.</d-footnote> Attention here is Multi-head Latent Attention (MLA), which we'll take apart in [Section 14](../attention). For now, let's just take its parameter count as given.

Let's do the accounting for DeepSeek-V3, following the same pattern [Section 6](https://jax-ml.github.io/scaling-book/applied-training) used for LLaMA 3-70B. From the [config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json): $D=7168$, $L=61$ (3 dense, 58 MoE), $E=256$, $E_s=1$, $k=8$, $F=2048$, $F_\text{dense}=18432$, $V=129280$.

| param | formula | count |
| :---- | :------ | ----: |
| Routed + shared experts | 58 layers * (256 + 1) * 3 * 7168 * 2048 | **656.5e9** |
| Dense MLPs | 3 layers * 3 * 7168 * 18432 | **1.2e9** |
| Attention (MLA) | 61 layers * 187e6 | **11.4e9** |
| Vocab | 2 * 129280 * 7168 | **1.9e9** |
| **Total** | | **671e9** |

That's the reported 671B on the nose!<d-footnote>The Hugging Face repository holds 685B parameters. The extra 14B is the multi-token prediction module (an extra Transformer block plus its own copy of the embedding and output head) that DeepSeek trains alongside the main model and that serving engines use as a speculative decoder. We'll discuss MTP in <a href="../training-2026">Section 15</a>.</d-footnote> Now let's do our active count, replacing $E + E_s = 257$ with $k + E_s = 9$:

| param | formula | count |
| :---- | :------ | ----: |
| Active experts | 58 * 9 * 3 * 7168 * 2048 | **23.0e9** |
| Dense MLPs | (as above) | **1.2e9** |
| Attention | (as above) | **11.4e9** |
| Vocab | (as above) | **1.9e9** |
| **Active total** | | **37.5e9** |

That matches too! **But notice something odd.** In the dense accounting of [Section 6](https://jax-ml.github.io/scaling-book/applied-training), the MLP was 80% of the parameters and attention only 17%. Here attention is *30% of the active parameters*! Why? Because the MLP a given token actually passes through is only 9 experts of width 2,048, i.e. an effective width of 18,432 against a 7,168-wide residual stream, so the MLP isn't nearly as dominant as it was. This means we can't wave attention away the way we mostly could for dense models, which is why we spend all of [Section 14](../attention) on it.

<p markdown=1 class="takeaway">**Takeaway:** for an MoE, total parameters are $L (E + E_s) 3DF$ to within a few percent, and active parameters are $L (k + E_s) 3DF$ plus attention. Note that the ratio of the two is *not* $E/k$, since shared experts and attention are counted in both: DeepSeek-V3 has $E/k = 32$ but a total-to-active ratio of 18. So when you see a "sparsity" quoted, check which one is meant.</p>

### Counting FLOPs

**How many FLOPs does an MoE cost?** Happily, the $6 \cdot \text{params} \cdot \text{tokens}$ rule from [Section 4](https://jax-ml.github.io/scaling-book/transformers) still holds, so long as we plug in *active* parameters. Routing adds $2BDE$ FLOPs per layer, which is basically nothing, and each token does its $k$ expert matmuls and nothing else. So DeepSeek-V3 costs `6 * 37e9 = 2.2e11` FLOPs per token to train, about one-eleventh of LLaMA 3 405B's `6 * 405e9 = 2.4e12`, while holding 65% more parameters.

This is the whole point of the design for us, and you can see it in the pretraining budgets of the models people actually use:

| Model | Active params | Tokens | Training FLOPs ($6 \cdot \text{active params} \cdot \text{tokens}$) |
| :---- | ------------: | -----: | -------------------------------------: |
| LLaMA 3 405B | 405B | 15T | 3.6e25 |
| LLaMA 3 70B | 70B | 15T | 6.3e24 |
| DeepSeek-V3 | 37B | 14.8T | 3.3e24 |
| Kimi K2 | 32B | 15.5T | 3.0e24 |
| Qwen3-235B-A22B | 22B | 36T | 4.8e24 |
| DeepSeek-V4-Pro | 49B | 33T | 9.7e24 |

So DeepSeek-V3 was trained for about half the compute of LLaMA 3 70B, and Kimi K2, which holds a trillion parameters, cost less to train than LLaMA 3 70B did. Even DeepSeek-V4-Pro, which saw 33T tokens, only used about a quarter of LLaMA 3 405B's compute.<d-footnote>Not every lab publishes token counts anymore. Kimi K3 and Qwen3.8 don't, and gpt-oss reports only "2.1M H100-hours" for the 120B model. If we assume 40% utilization of an H100's bf16 FLOPs, that's <code>2.1e6 * 3600 * 9.9e14 * 0.4 = 3e24</code> FLOPs, or about 100T tokens at 5.1B active parameters. That seems high, so either the utilization or the token count is probably lower than our guess.</d-footnote>

You'll probably also notice that the ratio of training tokens to active parameters is huge by Chinchilla<d-cite key="chinchilla"></d-cite> standards: 400 tokens per active parameter for DeepSeek-V3 and 1,600 for Qwen3-235B, against the 20 that's compute-optimal for a dense model you only train once. **Why overtrain so much?** Because inference cost scales with active parameters, so it's worth spending extra training tokens to keep the active count small, and the large total parameter count is what makes up the quality we'd otherwise lose.

<p markdown=1 class="takeaway">**Takeaway:** training FLOPs for an MoE are $6 \cdot \text{active params} \cdot \text{tokens}$, so long as we count *active* parameters. The frontier open models of 2025 and 2026 were trained for 3e24 to 1e25 FLOPs, less than LLaMA 3 405B, while carrying 2 to 7 times more parameters.</p>

### Fine-grained experts

**Why do nearly all recent models use hundreds of experts of width 1,024 to 3,072 rather than eight of width 16,384?** The modeling argument<d-cite key="deepseekmoe"></d-cite><d-cite key="finegrainedmoe"></d-cite> is basically combinatorial: if we choose 8 of 256 experts we give the model $\binom{256}{8} \approx 4 \times 10^{14}$ distinct MLP configurations per token, against $\binom{8}{2} = 28$ for Mixtral, at the same active FLOPs. DeepSeekMoE's ablations found that splitting each expert four ways and activating four times as many beat the coarse design at equal active FLOPs, and its 16B model matched LLaMA 2 7B with about 40% of the compute.

**What does this cost us on the systems side?** Every matmul got smaller. A dense LLaMA 3-70B MLP is a `[B, 8192] x [8192, 28672]` matmul, while a DeepSeek-V3 expert is `[t_e, 7168] x [7168, 2048]`, where $t_e$ is the number of tokens that happened to land on that particular expert. This has two consequences for us:

1. **The weight matrices get skinny:** a `[t_e, 7168] x [7168, 2048]` matmul with only a few hundred rows is a pretty small problem. On an H800 that's at most a few dozen Tensor Core tiles of work, which is nowhere near enough to keep 132 streaming multiprocessors busy on its own. **How do we deal with this?** DeepSeek and others wrote grouped-matmul kernels (DeepGEMM, and the ragged kernels you'll find in every MoE framework) that run all of a GPU's local experts in one launch, so the GPU sees one big ragged matmul instead of lots of tiny ones. Now imagine we tried to shard that expert eight ways with tensor parallelism: each shard would be only 256 wide (a single tile.), so any padding or imbalance would waste a big fraction of the unit. The same is true on a TPU, where $F = 2048$ is eight 256-wide MXU tiles and an eight-way shard is a single tile. So we mostly won't want to shard *within* experts, and we'll shard *across* them instead.

2. **The batch dimension becomes $t_e$ rather than $B$:** on average $t_e = Bk/E$. For the matmul unit to be efficient we need $t_e$ to fill its tile (64 rows for a Hopper Tensor Core instruction, 128 on Blackwell, 128 or 256 on a TPU systolic array), and for the matmul to be compute-bound rather than bound on loading weights we need $t_e$ to exceed the accelerator's arithmetic intensity, i.e. 296 on an H800 or H100, 206 on an H200, 281 on a B200 and 311 on TPU7x in bf16 ([Section 12](https://jax-ml.github.io/scaling-book/gpus), [Section 2](https://jax-ml.github.io/scaling-book/tpus)). With $E/k = 32$, that means $B$ has to exceed roughly `32 * 296 = 9,500` tokens per layer-step on an H800 before our experts stop being memory-bound. That's a lot! We'll come back to this number again and again, since for inference it's the single most important consequence of sparsity.

<p markdown=1 class="takeaway">**Takeaway:** fine-grained experts turn one big matmul into $E$ small ones, each with an effective batch of $Bk/E$. Every efficiency argument in this book that depended on our batch being large now has to be made with $Bk/E$ in place of $B$.</p>

## Routing Is a Systems Decision

In a dense model, we (and the compiler) know the shape of every matmul before the program runs. In an MoE we don't! The router decides at runtime how many tokens each expert gets, which means it decides the shape of every expert matmul, and the pattern of every AllToAll along with it. So even though routing sounds like a modeling question, there are three things about the router that we need to care about as systems people. Let's go through them.

**Load balance.** If expert 17 receives twice the average number of tokens, then whichever of our GPUs holds expert 17 has twice as much work to do, and every other GPU sits waiting for it. So the efficiency of an expert-parallel layer is roughly $\text{mean load} / \text{max load}$ across GPUs. **How do we keep it balanced?** The classic fixes were an auxiliary loss that pushes the router toward uniform usage<d-cite key="switch"></d-cite><d-cite key="stmoe"></d-cite>, and a *capacity factor*, i.e. giving each expert a fixed buffer of $c \cdot Bk/E$ slots and dropping any tokens that overflow it<d-cite key="gshard"></d-cite>. Neither is great for us: dropping tokens hurts quality and the auxiliary loss fights the language modeling loss, which is why DeepSeek-V3 and the 2025 models after it got rid of both. DeepSeek-V3 adds a per-expert bias to the router score that is used only for the top-$k$ selection, and after every step nudges the bias up for underloaded experts and down for overloaded ones<d-cite key="auxlossfree"></d-cite>. Kimi K3 goes further and sets each bias from a quantile of the router scores so that every expert receives exactly its target load (it computes the quantile from a histogram that is AllReduced across the cluster).<d-cite key="kimik3"></d-cite> You'll notice that none of the models in our table drop tokens; instead the expert matmuls are *ragged* (in JAX, `jax.lax.ragged_dot`) and balance is enforced by training-time bookkeeping.

**What about at inference time?** There, balance is handled differently: hot experts are simply duplicated. DeepSeek's serving system holds 32 redundant copies of routed experts per 32-GPU prefill unit and rebalances which experts get duplicated based on the load it observes<d-cite key="eplb"></d-cite>. Kimi K3's training system does the same thing *during training*, planning redundant experts per micro-batch so that every rank receives exactly $Bk/Z$ token copies, where $Z$ is the number of GPUs the experts are spread over (the expert-parallel degree, which we'll define below). This has the pleasant side effect that every matmul shape is static, so the host never has to synchronize with the device to find out how big the next matmul is.<d-footnote>Static shapes matter more than they sound like they should. In a conventional MoE implementation the host has to wait for the router to finish before it knows the size of each expert's input and can launch the expert kernels, which stalls the pipeline at every layer. With guaranteed balance the shapes are known in advance. If you've used XLA, you'll recognize this as the same reason dynamic shapes are painful on TPU.</d-footnote>

**Locality.** The router also decides how far each token's activations have to travel across our network. DeepSeek-V3 constrains each token's 8 experts to lie on at most 4 of the 8 nodes that hold the experts (`n_group = 8`, `topk_group = 4` in the config), by first picking the 4 best nodes by summed affinity and then the 8 best experts within them. That halves the number of cross-node copies of each token from 8 to 4 and is a purely systems-motivated change to the routing function. DeepSeek-V4 removed the constraint, and the same report describes a fused dispatch, expert-matmul and combine kernel that hides most of the AllToAll behind compute, which is presumably what made the constraint unnecessary.<d-cite key="deepseekv4"></d-cite>

**Latent routing.** Kimi K3 and NVIDIA's Nemotron 3 Super<d-cite key="nemotron3super"></d-cite> project each token down before dispatch: $z = W_\downarrow x \in \mathbb{R}^{\ell}$ with $\ell = D/2$ for K3 (3,584 of 7,168) and $\ell = D/4$ for Nemotron. The routed experts operate on $z$, and we use a $W_\uparrow$ to map the combined result back to $D$. The shared experts still see the full $x$. This halves (or quarters) the bytes that cross the network in both AllToAlls and shrinks each routed expert's matrices to $[\ell, F]$. Note that this doesn't change the *ratio* of expert FLOPs to dispatch bytes, since both scale with $\ell$, but it does cut the absolute cost, and if we're communication-bound the absolute cost is exactly what we care about.

<p markdown=1 class="takeaway">**Takeaway:** the router determines the shape of every expert matmul and the pattern of every AllToAll. Modern models keep it balanced with per-expert biases rather than dropped tokens, sometimes constrain where tokens can go to save network bandwidth, and sometimes shrink what gets sent, and you should think of all of this as being done for the hardware's sake.</p>

## What Sparsity Does to Memory

The obvious cost of a 2.8T-parameter model is that we have to store 2.8TB of it, even in fp8. Let's see what that does to us.

**Inference.** During inference we hold one copy of the weights. Here's the minimum number of GPUs (or TPUs) needed just to *store* the weights of a few models, before a single byte of KV cache:

| Model | Weights | H800/H100 (80GB) | H200 (141GB) | B200 (180GB) | GB200 (186GB) | TPU7x (192GB) | TPU v6e (32GB) | TPU v5e (16GB) |
| :---- | ------: | ---------------: | -----------: | -----------: | ------------: | ------------: | -------------: | -------------: |
| DeepSeek-V3, fp8 | 671GB | 9 | 5 | 4 | 4 | 4 | 21 | 42 |
| Kimi K2, fp8 | 1.04TB | 13 | 8 | 6 | 6 | 6 | 33 | 65 |
| DeepSeek-V4-Pro, MXFP4 experts (as shipped) | 0.87TB | 11 | 7 | 5 | 5 | 5 | 28 | 55 |
| Kimi K3, MXFP4 experts (as shipped) | 1.56TB | 20 | 12 | 9 | 9 | 9 | 49 | 98 |
| Kimi K3, fp8 | 2.78TB | 35 | 20 | 16 | 15 | 15 | 87 | 174 |
| Qwen3.8-2.4T, fp8 | 2.4TB | 30 | 18 | 14 | 13 | 13 | 75 | 150 |

Now let's compare those counts with the size of a fast domain. An 8-GPU H800 node has 640GB of HBM and holds none of our models. An 8-GPU H200 node (1.13TB) holds DeepSeek-V3 with room for cache, and Kimi K2 with none. An 8-GPU B200 node (1.44TB) holds V4-Pro's fp4 checkpoint (0.87TB) with room for cache, but not K3's (1.56TB), which needs an 8-GPU GB300 node (2.3TB), and that's exactly what Kimi's vLLM recipe asks for.<d-footnote>MXFP4 with a group of 32 and one E8M0 scale per group actually costs 4.25 bits per parameter rather than 4, and only the routed experts are 4-bit: Kimi K3 keeps attention, shared experts, latent projections and the output head in bf16 (about 0.11TB), and DeepSeek-V4-Pro keeps them in fp8 (about 0.03TB). The idealized 0.5 bytes per parameter would give 1.42TB and 0.83TB, but the shipped checkpoints are 1.56TB and 0.87TB, and we use the shipped sizes throughout.</d-footnote> A GB200 NVL72 rack (72 GPUs, 13.4TB) holds every model in our table with terabytes to spare, and so does a 64-chip TPU7x cube (12.3TB). On the 2024 hardware, though, these models simply don't fit in a node, so we have *no choice* but to shard the weights across nodes, and as we saw in [Section 7](https://jax-ml.github.io/scaling-book/inference), the only sharding we can afford at decode time is model parallelism (i.e. moving activations rather than weights). Expert parallelism is a form of model parallelism, so it's the natural default for MoE serving, and on H800s it has to cross InfiniBand.

Note also how much fp4 matters here. Kimi K3 and DeepSeek-V4 both ship us their expert weights in a 4-bit microscaling format (MXFP4) trained with quantization-aware training, which halves the weight footprint relative to fp8 and doubles the batch size we can fit. We'll see how that's done in [Section 15](../training-2026).

**Training.** During training we have to store weights, gradients and optimizer state. [Section 5](https://jax-ml.github.io/scaling-book/training) used 10 bytes per parameter (bf16 weights plus two fp32 Adam moments), so for DeepSeek-V3 that's 6.7TB, and for Kimi K3 it would be 28TB. (K3 actually trained with Muon, which as [Section 15](../training-2026) shows brings this down to 6 bytes and 17TB, but let's stick with Section 5's 10 bytes so we're comparing apples to apples.) That sounds terrifying! But remember that this gets sharded over the whole cluster: K3's 28TB spread across 2,048 GPUs is only 14GB per GPU, and across a 9,216-chip TPU7x pod it's just 3GB per chip. We have exactly one measured number to check this against, from the Kimi K2 report<d-cite key="kimik2"></d-cite>: bf16 weights plus fp32 gradient buffers for its 1.04T parameters come to about 6TB, which is 24GB per GPU over its 256-GPU model-parallel group, and about 30GB per GPU of total state once the sharded optimizer is added. So per-GPU weight memory isn't actually what bites us during training. **So what does?** Let's look at three things that do:

* We can't train these models on a small cluster. The "how few chips can I train on" exercise from [Section 6](https://jax-ml.github.io/scaling-book/applied-training) gives us `28e12 / 80e9 = 350` H800s for Kimi K3 before any activations, or `28e12 / 192e9 = 146` TPU7x chips. With that section's two bf16 checkpoints per layer, a 60M-token batch adds `2 * 7168 * 60e6 * 2 * 93 = 160TB` of activations, so the real floor is about `188e12 / 80e9 = 2,350` H800s or 980 TPU7x chips. Compare LLaMA 3-70B, whose weights and optimizer state needed `0.7e12 / 96e9 = 8` v5p chips (that section found 117 once activation checkpoints were included). Activations still dominate for an MoE, but only by 6x rather than 15x, since sparsity grew the weights by $E/k$ without growing the activations.
* The optimizer state is 32 to 64 times larger *relative to the compute we do per step* than for a dense model. FSDP's AllGather and ReduceScatter traffic is proportional to weight bytes, while the FLOPs we have to hide it behind are proportional to active parameters. We'll see this inflate the FSDP roofline by $E/k$ below.
* Checkpoints are enormous: a Kimi K3 checkpoint with optimizer state is 17 to 28TB depending on the optimizer, and we write one every few hours.

Our activations, by contrast, look just like those of a dense model whose MLP width is $(k + E_s) F$. There's nothing new for us there, except that the ragged expert inputs turn out to be awkward to checkpoint (Kimi K3 devotes a section of its report to a custom activation manager for exactly this reason).

<p markdown=1 class="takeaway">**Takeaway:** an MoE's weights are $E/k$ times larger than those of a dense model with the same per-token cost. At inference this forces us to shard the model across many GPUs just to hold the weights (across nodes, on anything older than an NVL72 rack), which is a big part of why 4-bit weights matter so much. During training it inflates every roofline that compares weight bytes to FLOPs by $E/k$, although per-GPU memory at cluster scale stays quite manageable.</p>

## The Batch-Size Rooflines

Now let's redo the two batch-size rooflines that dense-model intuition gets most wrong.

**Decode.** In [Section 7](https://jax-ml.github.io/scaling-book/inference) we found that a generate step is compute-bound once the batch (in tokens) exceeds the accelerator's arithmetic intensity $B_\text{crit} = C / W_\text{hbm}$, times $\text{bytes}/2$ for weights not in bf16. That calculation assumed every weight we load from HBM gets used by every token. Question 5 of [Section 7](https://jax-ml.github.io/scaling-book/inference) already noted that an MoE inflates it by $E/k$, so let's do the derivation properly. In an MoE layer we have to load *all* $E$ experts (some token in the batch is going to want each of them), but each token only does FLOPs against $k$ of them:

$$\begin{align*}
T_\text{math} &= \frac{2 \cdot B \cdot k \cdot 3DF}{C} \\
T_\text{HBM} &= \frac{E \cdot 3DF \cdot \text{bytes}}{W_\text{hbm}}
\end{align*}$$

so we are compute-bound when

$$B > \frac{E}{k} \cdot \frac{\text{bytes}}{2} \cdot \frac{C}{W_\text{hbm}} = \frac{E}{k} \cdot B_\text{crit, dense}$$

Here's what that gives us for a few models, with weights in bf16 (halve every number for fp8 weights with bf16 FLOPs, as in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference)):

| Model | $E/k$ | H800/H100 (296) | H200 (206) | B200 (281) | GB200 (312) | TPU7x (311) | TPU v5e (240) |
| :---- | ----: | --------------: | ---------: | ---------: | ----------: | ----------: | ------------: |
| DeepSeek-V3, GLM-5, gpt-oss | 32 | 9,500 | 6,600 | 9,000 | 10,000 | 10,000 | 7,700 |
| Kimi K2 | 48 | 14,200 | 9,900 | 13,500 | 15,000 | 14,900 | 11,500 |
| Qwen3.8-2.4T | 51 | 15,100 | 10,500 | 14,400 | 15,900 | 15,900 | 12,300 |
| Kimi K3 | 56 | 16,600 | 11,500 | 15,700 | 17,500 | 17,400 | 13,400 |
| DeepSeek-V4-Pro | 64 | 18,900 | 13,200 | 18,000 | 20,000 | 19,900 | 15,400 |

So to be compute-bound while decoding DeepSeek-V3, we'd need to be generating for about ten thousand sequences *at once, in one model replica*. That's a huge batch! Ten thousand sequences at 32k context with DeepSeek-V3's 35kB-per-token KV cache ([Section 14](../attention)) is 11TB of KV cache. So for any realistic deployment, **MoE decode is memory-bound on weights, on KV cache, or both**, and the arithmetic intensity of the Tensor Core basically doesn't matter. What we *can* do is spread the $E \cdot 3DF$ bytes of weights over as many GPUs as possible so that each GPU's share is small, at which point we're bound by KV cache bandwidth instead. That's what DeepSeek does with 144-way expert parallelism at decode, and it's the reason [Section 14](../attention) spends so long on shrinking the KV cache.

**What does each expert actually see in that deployment?** DeepSeek's decode step has about 88 sequences on each of 144 H800s ([Section 16](../applied-frontier)), so we have `144 * 88 * 8 = 101k` token-expert pairs per step spread over 256 routed experts plus 32 redundant copies, or about `101e3 / 288 = 350` tokens per expert copy. Each of our expert matmuls is then a `[350, 7168] x [7168, 2048]` problem, above the H800's intensity of 296, so on its own it would be compute-bound. The step as a whole is nowhere near compute-bound, though, since the weight bytes and the network are shared across all of it. The experts themselves only starve in the small-batch regime where we care about a single sequence's latency: at 8 sequences per GPU we get `8 * 8 * 144 / 288 = 32` rows per expert, half a Tensor Core tile.

**Training with FSDP.** In [Section 12](https://jax-ml.github.io/scaling-book/gpus) we found that FSDP on GPUs is compute-bound when the per-GPU batch exceeds $C / W_\text{collective}$, where $W_\text{collective}$ is a GPU's NVLink egress inside a node (2,200 tokens on an H100) or a whole node's InfiniBand egress across nodes (`990e12 / 400e9 = 2,475`). Note that Section 12's 2,475 is at the H100's bf16 peak. DeepSeek ran fp8 matmuls, which doubles every FSDP threshold in this section (to 4,950 per GPU for a dense model), so we'll quote both where it matters. [Section 5](https://jax-ml.github.io/scaling-book/training)'s TPU version is $C / (M_X W_\text{ici})$, where $M_X$ is the number of ICI axes the gather spans, i.e. 850 on v5p with three axes. Either way, we compared per-layer FLOPs $\propto B \cdot 3DF$ against per-layer weight-gather bytes $\propto 3DF$. For an MoE our FLOPs are $\propto B k \cdot 3DF$ but our gathered bytes are $\propto E \cdot 3DF$, so

$$\frac{B}{N} > \frac{E}{k} \cdot \frac{C}{W_\text{collective}}$$

Let's put in some real numbers. For DeepSeek-V3 across H800 nodes that's `32 * 2475 = 79,200` tokens per GPU at bf16 peak (or 158,400 at the fp8 peak the run actually used), and DeepSeek only had `62.9e6 / 2048 = 30,700`, so they were 2.6x (or 5.2x) short. For Kimi K3 we'd need `56 * 2475 = 139,000`. A GB200 NVL72 rack does better, since a rack of 72 GPUs pulls each weight shard in once through `72 * 50 = 3,600GB/s` of InfiniBand egress and then shares it over NVLink, so the threshold is `2.5e15 / 3.6e12 = 694` tokens per GPU for a dense model and `32 * 694 = 22,200` for DeepSeek-V3, which is still bigger than any batch anyone actually runs. On TPU v5p it's `32 * 850 = 27,200` tokens per chip, i.e. a 244M-token batch on a full pod. So pure FSDP is basically never going to work for a fine-grained MoE, on any hardware we have. We'll need to shard the experts some other way, which is exactly what expert parallelism does.

<p markdown=1 class="takeaway">**Takeaway:** both the decode batch roofline and the FSDP training roofline are inflated by $E/k$. At sparsity 32 to 64, decode needs ten to twenty thousand concurrent tokens per replica to be compute-bound (which basically never happens), and FSDP alone needs tens of thousands of tokens per GPU, which isn't practical. So every MoE deployment we'll look at ends up sharding experts across GPUs.</p>

## The Expert-Parallelism Rooflines

Expert parallelism (EP) is what it sounds like: we shard the expert dimension, $W_\text{in}[E_Z, D, F]$, $W_\text{out}[E_Z, F, D]$ over an axis $Z$ of the mesh, so that each GPU holds $E/Z$ experts. Since tokens live on whatever GPU processed the previous layer, we have to move each token's activations to the GPUs holding its $k$ experts and move the results back. That's two AllToAlls per layer ([Section 3](https://jax-ml.github.io/scaling-book/sharding)).

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

The backward pass is the same thing run in reverse: two more AllToAlls, now carrying gradients of exactly the same shapes. If we have a shared expert, we just compute it locally (it's replicated or tensor-sharded like any dense MLP), and since it doesn't depend on the dispatch we can overlap it with the AllToAlls for free.

{% enddetails %}

You'll notice that, like tensor parallelism, this layout moves *activations* rather than weights, so its communication cost is proportional to $B$ and independent of $E$. Unlike tensor parallelism, though, each token gets sent $k$ times (once per expert), so the bytes are $k$ times those of a plain activation resharding.<d-footnote>In practice implementations send one copy per <em>destination GPU</em> rather than per expert, so if two of a token's experts share a GPU only one copy travels. For $Z \gg k$ this is a small correction. DeepSeek-V3's node-limited routing exploits exactly this: by confining a token's 8 experts to 4 nodes, it guarantees at most 4 cross-node copies.</d-footnote>

### Expert parallelism on GPUs

Let's compute the roofline in this book's notation, on the machines the models were actually built on. Say we have a global batch of $B$ tokens, EP over $Z$ GPUs, and expert width $F$. Following [Section 5](https://jax-ml.github.io/scaling-book/training) we'll look at the forward pass, but this time we count all three expert matmuls rather than two, since for narrow experts the gating einsum isn't something we can ignore.<d-footnote>Section 5 dropped the gating matmul and used $4BDF$ FLOPs per layer to keep the algebra clean. For MoEs with $F = 2048$ we keep all three matrices ($6BDF$). DeepSeek's own overlap analysis in the V4 report also uses the three-matmul count, so the thresholds below match theirs. Be aware of what this choice does to the numbers, though: counting all three matrices makes every EP threshold 1.5x more permissive than Section 5's two-matmul convention, and fp8 dispatch another 1.33x, so under Section 5's $4BDF$, bf16 model you would double every $F$ in the tables below.</d-footnote> Per GPU we have:

$$T_\text{math} = \frac{6 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

Now what about $T_\text{comms}$? Each of our GPUs dispatches $Bk/Z$ token copies of $D$ elements at $b_d$ bytes each, and receives the same number back at $b_c$ bytes each. On a switched network (i.e. NVLink inside the domain, InfiniBand across it) every byte a GPU sends leaves through its own egress port at $W_\text{egress}$ bytes per second, so as in [Section 12](https://jax-ml.github.io/scaling-book/gpus)

$$T_\text{comms} = \frac{(b_d + b_c) \cdot B \cdot k \cdot D}{Z \cdot W_\text{egress}}$$

Setting $T_\text{math} > T_\text{comms}$, you'll notice the batch, $k$, $D$ and $Z$ all cancel out:

$$F > \frac{(b_d + b_c)}{6} \cdot \frac{C}{W_\text{egress}} = \frac{(b_d + b_c)}{6} \cdot \alpha$$

where $\alpha = C / W_\text{egress}$ is the network operational intensity from [Section 12](https://jax-ml.github.io/scaling-book/gpus). With bf16 activations both ways ($b_d + b_c = 4$) this is $F > 2\alpha / 3$. With fp8 dispatch and bf16 combine, which is what DeepSeek does, it's $F > \alpha / 2$, and this is exactly the condition DeepSeek gives in the V4 report for hiding the AllToAll behind expert compute.<d-cite key="deepseekv4"></d-cite>

This is a slightly surprising result, and a super useful one: **whether expert parallelism is compute-bound depends only on the expert width $F$ and on which network the AllToAll has to cross.** Notice everything that *doesn't* appear: our batch size, $k$, $E$ and the EP degree $Z$ have all cancelled out (well, $Z$ still sneaks in one way, since it decides whether we're still inside the NVLink domain). If this feels familiar, it's because [Section 12](https://jax-ml.github.io/scaling-book/gpus) found the same thing for tensor parallelism, where the condition $Y < F W / C$ also only cared about $F$ and the network. So instead of asking "how much EP can we afford?", we can just look up $F$ in a table.

Here's the minimum expert width we need for the GPUs the 2026 models run on, with fp8 dispatch and bf16 combine. We need two columns because $\alpha$ doubles when the matmuls run in fp8, so the same AllToAll suddenly has half as much compute to hide behind.

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

**Why the separate H800 rows?** Because the H800 is the GPU DeepSeek-V3 was trained and served on, and it isn't quite an H100. It has the same Tensor Cores and HBM, but NVIDIA cut its NVLink to 400GB/s bidirectional (200GB/s each way in this book's convention), against the H100's 900 and 450.<d-cite key="deepseek_isca"></d-cite><d-footnote>Section 12 quotes 300GB/s for the H800. DeepSeek's own hardware paper says "the NVLink bandwidth in H800 SXM nodes is reduced from 900 GB/s to 400 GB/s", and its V3 report measures "160GB/s, roughly 3.2 times that of IB (50GB/s)" in practice. We use 200 as the spec figure and 160 as the achieved one. Section 4's Question 7 also used an fp8 rate of 1.51e15 for the H800, which is the PCIe part. The SXM part DeepSeek used has the H100's 1.98e15.</d-footnote> Now let's compare this table against the expert widths we saw in our first table: 2,048 for DeepSeek-V3, Kimi K2 and GLM-5 (3,072 for DeepSeek-V4-Pro and Kimi K3, and 1,024 for Qwen3.5).

* **Inside an H100 or B200 node**, 2,048-wide experts clear the bf16 threshold with room to spare (1,100 on an H100, 1,250 on a B200) but fall just short of the fp8 one (2,200 on an H100 and 2,500 on a B200, so 7% and 18% short). Inside an **H800 node they miss even with bf16 matmuls** (2,475 spec, 3,090 measured, so 17% to 34% short), and with the fp8 matmuls DeepSeek actually used the threshold is 4,950 to 6,190, i.e. 2.4x to 3x above them, and that's before a single token has left the node!
* **Inside a GB200 NVL72 rack**, we find 72-way expert parallelism is compute-bound for anything wider than 1,390 with bf16 matmuls, or 2,780 with fp8. DeepSeek-V4-Pro's and Kimi K3's 3,072-wide experts clear both, which is great for us. DeepSeek-V3's 2,048 clears the first but misses the second by 26%, which is fine for serving (where the step is memory-bound anyway) but a nuisance for training.
* **Across InfiniBand**, pretty much nothing comes close. DeepSeek-V3's experts leave us short by `9900 / 2048 = 4.8x` with bf16 matmuls and 9.7x with the fp8 matmuls the run actually used. (The InfiniBand rows assume the fat tree keeps its full 50GB/s per GPU above the node, as [Section 12](https://jax-ml.github.io/scaling-book/gpus) does. DeepSeek describes a two-layer, eight-plane fat tree, and Kimi's cluster is RoCE at the same 400Gb/s per GPU, but neither says whether the upper tier is oversubscribed, as Llama 3's was at 7:1 above the pod, so you should treat these as the best case.) And we haven't yet accounted for the fact that GPUs rarely reach peak bandwidth on all-to-all traffic: DeepSeek's DeepEP kernels measure 153 to 158GB/s over NVLink and 43 to 58GB/s over InfiniBand on H800s, depending on the kernel version.<d-cite key="deepep"></d-cite> This is a big part of why the DeepSeek-V3 paper spends so many pages on communication, and we'll check the numbers against their layout below.

This is also why, for the 2025 and 2026 models, you'll see people treat the NVL72 rack (rather than the 8-GPU node) as the basic unit for serving an MoE: a trillion-parameter model's experts all fit inside one domain where the AllToAll is cheap. **What happens if we step over the rack boundary?** Our egress per GPU falls off a cliff, from 900GB/s to 50, an 18x drop. So, somewhat counterintuitively, 72-way EP that spills into a second rack is actually *worse* than 64-way EP that stays inside one.

**So what can hide an AllToAll, if not the expert matmuls?** The answer is the rest of the layer. Attention in these models is substantial (30% of active parameters for DeepSeek-V3, plus the quadratic term at long context), and with a pipeline schedule that interleaves two micro-batches we can run micro-batch 2's attention while micro-batch 1's dispatch is in flight. This is what DeepSeek's DualPipe schedule does<d-cite key="dualpipe"></d-cite>, and it's also the reason Kimi K2 kept EP at 16, as we'll see below. You should think of the roofline above as the worst case, where nothing else overlaps, and real training runs end up somewhere between it and perfect overlap.

**Why not gather instead of dispatch?** There's one alternative layout we should rule out. Instead of dispatching tokens to experts, we could **AllGather the tokens across the EP group** (as tensor parallelism does), let each of our GPUs run its local experts on whichever of the gathered tokens routed to them, and ReduceScatter the results. Then

$$T_\text{comms} = \frac{4 \cdot B \cdot D}{W_\text{egress}} \qquad T_\text{math} = \frac{6 B k D F}{Z C}$$

which is compute-bound when $Z < 1.5 \cdot k F / \alpha$. You can think of this as tensor parallelism with an effective MLP width of $kF$. For DeepSeek-V3 ($kF = 16384$) that allows $Z < 11$ over H100 NVLink, so it works inside a node, but only $Z < 1.2$ over InfiniBand, so it never works across nodes. Per local token the dispatch layout moves $(b_d + b_c) k D = 3kD$ bytes and the gather layout $4ZD$, so dispatch is cheaper for any $Z > 0.75k$, i.e. $Z > 6$ for $k = 8$. So on a switched network there's basically no regime where gathering wins for an EP group large enough to matter, and it isn't used in practice.

<p markdown=1 class="takeaway">**Takeaway:** on GPUs, expert parallelism with AllToAll dispatch is compute-bound when $F > \alpha / 2$ with fp8 dispatch and bf16 matmuls, or $F > \alpha$ with fp8 matmuls, where $\alpha = C / W_\text{egress}$ of whichever network the tokens cross. Note that this is independent of our batch, $k$, $E$ and $Z$. Inside an H100 node or a GB200 NVL72 rack, 2,048- to 3,072-wide experts clear it with bf16 matmuls. With fp8 matmuls only the 3,072-wide experts of V4-Pro and K3 do, and 2,048 misses by 7% (H100), 18% (B200) and 26% (GB200). Inside an H800 node 2,048 misses even with bf16 matmuls, and across InfiniBand they miss by 5 to 10x, so we have to overlap the AllToAll with attention or another micro-batch.</p>

### The same rooflines on a TPU torus

What changes on a TPU? The AllToAll doesn't leave through a private port anymore, it has to hop across a torus. From [Section 3](https://jax-ml.github.io/scaling-book/sharding), an AllToAll that moves $M/N$ bytes per chip on a mesh whose longest axis is $A$ costs us $(M/N) \cdot A / (4 W_\text{ici})$ per chip, so

$$T_\text{comms} = \frac{(b_d + b_c) \cdot B \cdot k \cdot D}{Z} \cdot \frac{A}{4 W_\text{ici}} \quad \Rightarrow \quad F > \frac{(b_d + b_c)}{24} \cdot A \cdot \frac{C}{W_\text{ici}}$$

With fp8 dispatch and bf16 combine this is $F > A \cdot \alpha_\text{ici} / 8$, which has the same shape as the GPU result, except that the factor of four from the bidirectional ring and the longest axis $A$ of the EP sub-mesh take the place of the switch. Here's the minimum expert width we need, with fp8 dispatch and bf16 matmuls, for the TPUs in this book (double it for fp8 matmuls, just as on GPUs):

| Chip | $\alpha = C / W_\text{ici}$ (bf16) | $A = 2$ | $A = 4$ | $A = 8$ | $A = 16$ |
| :--- | ---------------------------------: | ------: | ------: | ------: | -------: |
| TPU v5e | 2,190 | 550 | 1,100 | 2,200 | 4,400 |
| TPU v5p | 2,550 | 640 | 1,280 | 2,550 | 5,100 |
| TPU v6e | 5,110 | 1,280 | 2,560 | 5,110 | 10,200 |
| TPU7x | 12,800 | 3,200 | 6,400 | 12,800 | 25,600 |

Note that these thresholds assume we have a wraparound link (a bidirectional ring) on the longest EP axis. You'll remember from [Section 2](https://jax-ml.github.io/scaling-book/tpus) that v5e and v6e only have one on a full axis of 16, and v5p and TPU7x only on axes that are multiples of 4 inside a full cube, so for $A < 16$ on the 2D chips and for $A = 2$ on any chip we have to double the threshold: a 4x4 block on v6e needs about 5,100, rather than 2,560, and a 2x2x2 on TPU7x needs about 6,400, rather than 3,200.

* On **TPU v5p**, a 4x4x4 cube of 64-way EP ($A = 4$) needs $F > 1280$. Every model in our table clears it except Qwen3.5, whose 1,024-wide experts fall 20% short. A 4x4x8 (128-way) needs 2,550, which DeepSeek-V3's 2,048 doesn't quite reach.
* On **TPU v6e**, a 2D torus, a 4x4 (16-way) EP block has no wraparound and needs $F > 5100$, so DeepSeek-style experts leave us communication-bound by 2.5x and Qwen3.5's 1,024-wide experts by 5x. Even a 2x2 block ($A = 2$, no wraparound, and only 4-way EP) needs 2,560, which DeepSeek's experts miss by 25%.
* On **TPU7x**, even a 2x2x2 cube (no wraparound) needs $F > 6400$, and so does a 4x4x4 cube with its wraparound. So by this simple model, **none of the fine-grained MoEs in the table are compute-bound under expert parallelism on Ironwood**: DeepSeek-V3's experts fall short by a factor of 3 on a cube, and Kimi K3's by a factor of 2.

**Wait, a newer chip made the roofline worse?** Yes! This is the first time that's happened to us in this book, so it's worth understanding why. TPU7x has 5x the bf16 FLOPs/s of v5p (2.3e15 against 4.59e14) but exactly the same 9e10 bytes/s per ICI link.<d-cite key="tpu7x"></d-cite> That means its ICI operational intensity is also 5x higher, 12,800 against 2,550, so every roofline in [Section 5](https://jax-ml.github.io/scaling-book/training) that compares FLOPs to ICI bytes gets tighter by that same factor. For instance, tensor parallelism of a dense model now goes communication-bound beyond $Y > 3 F / 12800$, which is only about 6-way for LLaMA 3-70B, and FSDP now needs $12800 / 3 = 4270$ tokens per chip instead of 850. Expert parallelism was already the tightest of the three for narrow experts, so on Ironwood the AllToAll simply can't hide behind the expert matmuls anymore, and we have to find something else to hide it behind.<d-footnote>Ironwood's HBM bandwidth did grow, from 2.8e12 to 7.4e12 bytes/s, so its HBM arithmetic intensity only doubled, from 164 to 311. You can think of the chip as designed around per-chip memory bandwidth and fp8 compute (which is to say, around inference and around large-batch training), so its ICI has become the scarce resource for model-parallel training. GPUs went the other way: NVLink bandwidth doubled with each generation's FLOPs, so $\alpha_\text{nvlink}$ stayed between 2,200 and 2,800 from H100 to GB200, and the scarce resource on GPUs is the InfiniBand link out of the node.</d-footnote>

**What about the gather layout on a torus?** It behaves differently here too. Its cost is $4BD / (M_Z W_\text{ici})$, which is compute-bound when $Z < 1.5 \cdot M_Z \cdot k F / \alpha$. For DeepSeek-V3 on v5p with three axes that allows $Z < 29$, and on TPU7x $Z < 6$. Per token per axis-unit of bandwidth, the AllToAll costs us $(b_d + b_c) k A / (4 Z)$ and the gather $4 / M_Z$. On a 4x4x4 cube with $k = 8$ the AllToAll is `3 * 8 * 4 / (4 * 64) = 0.375` against `4 / 3 = 1.33`, so dispatching is 3.5x cheaper. On a single ring of 16, though, it's `3 * 8 * 16 / (4 * 16) = 6` against `4`, and gathering wins by 1.5x. So as a rule of thumb, we dispatch on a cube and gather on a ring (there's no GPU analog of this, since a switch has no rings).

<p markdown=1 class="takeaway">**Takeaway:** on TPUs the same roofline picks up the torus factor: expert parallelism is compute-bound when $F > A \cdot \alpha_\text{ici} / 8$ (fp8 dispatch, bf16 matmuls), where $A$ is the longest axis of the EP mesh. On TPU v5p a 64-way EP cube works for 2,048-wide experts. On TPU7x, whose ICI intensity is 5x higher, the same experts are 3x communication-bound and we have to overlap the AllToAll with attention or another micro-batch, exactly as across InfiniBand on GPUs.</p>

### Combining expert parallelism with FSDP

In practice, we'll run EP as just one axis of a larger layout. Say we do EP over $Z$ GPUs and data parallelism with sharded weights (FSDP, or ZeRO-3) over $X$, for $N = XZ$ total (this is [Section 5](https://jax-ml.github.io/scaling-book/training)'s $N$ for the accelerator count, while in [Section 4](https://jax-ml.github.io/scaling-book/transformers) and in Sections 14 and 16, $N$ is the number of attention heads). Each of our GPUs holds $E/Z$ experts and gathers their weights over $X$ each layer, so the FSDP traffic per layer is proportional to $(E/Z) \cdot 3DF$ rather than $E \cdot 3DF$. Let's redo the FSDP roofline, with $P$ pipeline stages also dividing the weight bytes each GPU has to gather (as in [Section 12](https://jax-ml.github.io/scaling-book/gpus)):

$$\frac{B}{N} > \frac{E}{k Z P} \cdot \frac{C}{W_\text{collective}}$$

In other words, **expert parallelism of degree $Z$ divides the FSDP inflation factor $E/k$ by $Z$.** If we choose $Z = E/k$, our MoE has exactly the same FSDP roofline as a dense model. For DeepSeek-V3 that's $Z = 32$, for Kimi K3 it's $Z = 56$, and for DeepSeek-V4-Pro it's $Z = 64$. You'll notice these are the EP sizes that get used in practice (DeepSeek-V3 trained with EP64), and that $Z = 64$ or 72 is also the largest EP that fits inside one GB200 NVL72 rack, where we just showed the AllToAll is compute-bound. It's reassuring that the two rooflines land in the same place, since it suggests our simple model is roughly right.

Let's check this against the one fully published GPU layout. DeepSeek-V3 ran EP64 across 8 H800 nodes, PP16, and 2-way data parallelism with ZeRO-1 (optimizer state sharded, weights replicated, so the per-step cost is a gradient AllReduce rather than a per-layer gather). Plugging into our formula as if it were FSDP, we need `256 / (8 * 64 * 16) * 2475 = 77` tokens per GPU (155 in fp8) against the 30,700 they had. With EP and PP carrying almost all of the model parallelism, data parallelism had basically nothing left to do, so a plain 2-way ZeRO-1 was plenty, and the hard part of the run was the AllToAll rather than the weight traffic. On a GB200 NVL72 cluster, the same arithmetic with 72-way EP inside each rack and FSDP across racks gives DeepSeek-V4-Pro a threshold of `(384 / (6 * 72)) * 1390 = 1,240` tokens per GPU (with fp8 matmuls and bf16 weight gathers, or 620 if the gathers are fp8), against the 10,200 per GPU of a 94M-token batch on 9,216 GPUs. [Section 16](../applied-frontier) works that plan out in full.

On a v5p pod the same formula reads $B/N > E/(kZ) \cdot C / (M_X W_\text{ici})$, and $Z = 64$ (a 4x4x4 cube, which we showed above is compute-bound for the AllToAll) gives us `32 / 64 * 850 = 425` tokens per chip, against the 7,000 per chip DeepSeek's batch would give on 8,960 chips. So we'd have plenty of room there.

<p markdown=1 class="takeaway">**Takeaway:** expert parallelism of degree $Z$ shrinks the MoE FSDP penalty from $E/k$ to $E/(kZ)$, and pipeline parallelism divides it by $P$ again. Setting $Z \approx E/k$ (32 to 64 for current models) gets us back the dense-model roofline, and happens to also be about the largest EP that fits inside a GB200 NVL72 rack or stays compute-bound on a TPU v5p cube. With EP and PP in place, the data-parallel axis of a 2026 training run ends up small and cheap.</p>

## What the Frontier Labs Actually Do

Let's start with the hardware. Here's what each lab itself says about where its models were trained and where it expects them to be served ("not disclosed" means the report and model card say nothing). You'll notice that no Chinese lab has published a GPU count or GPU-hours since DeepSeek-V3, and that the Western labs that do (Meta, OpenAI, Mistral, Google) publish no layout. B300 and GB300 are the Blackwell Ultra parts, the H20 is NVIDIA's export-market Hopper for China, and the MI355X is AMD's Blackwell-generation competitor.

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

**What should we notice here?** For one thing, every open frontier model outside Google and NVIDIA that tells us its training hardware trained on Hopper, and the two Chinese labs that disclose anything (DeepSeek for V3 and R1, Moonshot for Kimi K2, and MiniMax names the H800 only for M1's RL run) trained on the H800 specifically. Remember that its NVLink is cut down, which goes a long way toward explaining why the DeepSeek-V3 paper spends so much time on communication. Also notice that only NVIDIA's own Nemotron is documented on Blackwell (and only for its long-context phase), and that Gemma is the only open model we know was trained on TPUs. The *serving* side looks different, though: DeepSeek's V4 card and the day-0 vLLM recipes for Kimi K3 and Qwen3.8 all name a GB300 or B300 node as the reference deployment for the 2026 models, while the 2025 models are still specified in H200s by their own labs. (One thing you should keep in mind is that every Blackwell throughput number in this update comes from NVIDIA, SGLang, vLLM or a benchmark project, since the only *production* serving figures any lab has published are still DeepSeek's from February 2025, on H800s. That's what [Section 16](../applied-frontier) has to build on.) With that in mind, let's check our theory against the published training layouts.

**DeepSeek-V3** (2,048 H800s, 2024)<d-cite key="DeepSeek3"></d-cite>, our main reference, used 64-way EP across 8 nodes, 16-way pipeline parallelism with DualPipe, and 2-way ZeRO-1 data parallelism, with a batch of 15,360 sequences of 4,096 tokens, so `62.9M` tokens per step or `30.7k` tokens per GPU, fp8 dispatch with bf16 combine, and node-limited routing to 4 nodes per token. By our roofline the cross-node AllToAll is about 5x communication-bound with bf16 matmuls and 10x with the fp8 matmuls the run used, hidden partly by DualPipe's overlap with the attention and MLP of the other micro-batch and by the 20 SMs reserved for it. The achieved utilization, which you computed in Question 7 of [Section 4](https://jax-ml.github.io/scaling-book/transformers), was about 22% of fp8 peak by that section's H800 figure and 17% by the H800's actual dense fp8 rate ([Section 15](../training-2026) redoes it). DeepSeek's own hardware paper reports 385 TFLOP/s per GPU in steady state, which is 19% of fp8 peak once attention FLOPs are counted, or the 39% of *bf16* peak they quote.<d-cite key="deepseek_isca"></d-cite> So now we know where the rest of the FLOPs went.

**Kimi K2** (H800s with 8 x 400Gb/s RoCE per node rather than InfiniBand, 2025)<d-cite key="kimik2"></d-cite> used 16-way pipeline parallelism with virtual stages, 16-way EP, ZeRO-1 data parallelism, bf16 weights with fp32 gradient accumulation, and fp8 *storage* (but not compute) for a few insensitive activations. (RoCE is RDMA over Converged Ethernet, an Ethernet scale-out fabric with the same 50GB/s per GPU as the InfiniBand in this section's tables.) The report tells us explicitly that EP was set to "the smallest feasible" value so that the attention compute of one micro-batch covers the EP AllToAll of another under a standard one-forward-one-backward (1F1B) pipeline schedule, and that a smaller EP group "relaxes expert-balance constraints". It also tells us why they didn't use DualPipe: that schedule doubles the memory for parameters and gradients, which a 1T-parameter model on 80GB GPUs couldn't afford. Note that with $Z = 16 < E/k = 48$, their FSDP-equivalent inflation was 3x, which the pipeline stages absorbed.

**Kimi K3** (2026)<d-cite key="kimik3"></d-cite> used pipeline parallelism with virtual stages, EP, ZeRO-1, pipeline-aware ZeRO-2 gradient sharding, and context parallelism for the linear-attention layers. Most of this we've seen before, but there are two new tricks here that are worth pointing out. Their dispatch library, MoonEP, is the redundant-expert planner we described above under Routing (they also prove that at most $E/Z$ redundant expert slots per rank are ever needed, which is neat). And the shared experts are replicated across EP ranks and run on a separate stream while the AllToAll is in flight, so the AllToAll hides behind a matmul that had to happen anyway. This is exactly the "hide it behind something else" trick we keep coming back to.

**Gemma 4 26B-A4B** (TPU v6e, 2026)<d-cite key="gemma4"></d-cite> is the one TPU MoE data point we have to work with: 128 experts of width 704 with 8 active, trained on 6,144 TPU v6e chips. Its expert width is well below the 5,100 that a 4x4 EP block on v6e would need, and its sparsity of 16 is mild, so we'd expect it to have leaned on FSDP over many chips with a small EP factor and a large batch. Unfortunately Google says only that the optimizer state is ZeRO-3 (FSDP-style) sharded under GSPMD, and gives no expert-parallel degree.

**So what do all of these have in common?** You'll notice that none of them runs expert parallelism on its own: in every one of these runs, the AllToAll is hidden behind something else (attention, the shared experts, or another micro-batch). The recipe is basically the same each time: pick the biggest EP degree whose AllToAll your network can hide behind the rest of the layer, and then use pipelining and FSDP for whatever parallelism you still need. That's a pretty good rule of thumb for us too.

## What Should You Take Away from this Section?

Let's step back and summarize what we've learned:

* An MoE has $L (E + E_s) 3DF$ parameters and $L (k + E_s) 3DF$ active parameters plus attention, and training costs us $6 \cdot \text{active params} \cdot \text{tokens}$ FLOPs. The 2026 frontier open models have total-to-active ratios of roughly 20 to 33 ($E/k$ of 32 to 64) and were trained for 3e24 to 1e25 FLOPs.

* Every roofline we have that compares weight bytes to FLOPs is inflated by $E/k$. Decode needs $E/k \cdot B_\text{crit}$ tokens per step to be compute-bound (10k to 20k for current models, so in practice our decode is always memory-bound), and pure FSDP needs $E/k \cdot 2{,}475$ tokens per GPU across H800 nodes (79k for DeepSeek-V3 at bf16 peak and 158k at the fp8 peak it used, so no real run does it, and 22k on a GB200 rack or 27k on a v5p pod).

* Expert parallelism moves our activations and costs us two AllToAlls per layer. It's compute-bound when $F > \alpha / 2$ on GPUs (with fp8 dispatch and bf16 matmuls, or $F > \alpha$ with fp8 matmuls), where $\alpha = C / W_\text{egress}$ of the network we cross, or when $F > A \alpha_\text{ici} / 8$ on TPUs ($A$ = longest EP axis). Inside an H100 node or a GB200 NVL72 rack, 2,048- to 3,072-wide experts clear this with bf16 matmuls, and only the 3,072-wide ones with fp8. Inside an H800 node 2,048 misses even in bf16, and across InfiniBand or on TPU7x they miss by 3 to 10x, so we have to overlap the AllToAll with attention or another micro-batch.

* EP of degree $Z$ cuts our FSDP inflation to $E/(kZ)$, so $Z \approx E/k$ gets us dense-model behavior back, and pipeline parallelism cuts it further. This is the reason DeepSeek-V3's data-parallel axis was only 2 wide.

* We should think of the router as a systems component: it's kept balanced with per-expert biases instead of dropped tokens, it may be constrained to limit network fan-out, and its inputs may be projected to a latent to shrink what gets sent.

* You can think of the newest hardware as having been designed around these models: the GB200 NVL72 rack and the 192GB TPU7x chip both exist so that a trillion-parameter MoE fits in one fast domain. NVLink kept pace with GPU FLOPs, so $\alpha_\text{nvlink}$ stayed near 2,500 and the thing we have to watch on GPUs is the InfiniBand link out of the domain. TPU7x's ICI didn't scale with its FLOPs, so every model-parallel roofline in this book is 5x tighter on it than on v5p.

## Worked Problems

**Question 1 [Kimi K2 parameters]:** Say we want to compute the total and active parameter counts of Kimi K2 from the [Kimi K2 config](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json) ($D = 7168$, $L = 61$ with 1 dense layer, $F_\text{dense} = 18432$, $E = 384$, $E_s = 1$, $k = 8$, $F = 2048$, $V = 163840$, MLA attention with 64 heads and the same latent dimensions as DeepSeek-V3). *Attention per layer for this MLA configuration is about 101M parameters, so take that as given.*

{% details Click here for the answer, once you've thought about it! %}

| param | formula | count |
| :---- | :------ | ----: |
| Experts | 60 * 385 * 3 * 7168 * 2048 | **1017e9** |
| Dense MLP | 1 * 3 * 7168 * 18432 | **0.4e9** |
| Attention | 61 * 101e6 | **6.2e9** |
| Vocab | 2 * 163840 * 7168 | **2.3e9** |
| **Total** | | **1026e9** |

Kimi reports 1.04T, so we're within 2%. Not bad! The shipped checkpoint (1.03e12 bytes in fp8, with `num_nextn_predict_layers: 0`) holds exactly our count, so the remaining 1% is either the report's rounding or an unreleased one-layer MTP block of about 17B that the K3 report credits K2 with (the config can't tell us which). For the active count we get `60 * 9 * 3 * 7168 * 2048 = 23.8e9` for the experts, plus the same 0.4 + 6.2 + 2.3 = 8.9B, for **32.7B**, against the reported 32B. Note that K2 has half the attention heads of DeepSeek-V3 (64 vs 128), so attention is only 19% of its active parameters.

{% enddetails %}

**Question 2 [gpt-oss-120b on one GPU]:** Say we want to serve gpt-oss-120b on a single GPU. It has $E = 128$, $k = 4$, $F = D = 2880$, $L = 36$, and ships its expert weights in MXFP4 (0.5 bytes per parameter) with everything else in bf16. Its attention alternates 18 full layers with 18 sliding-window layers (128 tokens), with 8 KV heads of dimension 64.

(a) Does it fit on one 80GB H100? How much room do we have left for KV cache?

(b) What's the lower bound on decode step time at small batch, and what batch size would we need for the expert matmuls to be compute-bound (bf16 FLOPs)?

(c) At 8k context, how many sequences fit in the remaining memory with a bf16 KV cache, and what's our step time and throughput at that batch?

{% details Click here for the answer. %}

(a) Let's start with the expert weights: `36 * 128 * 3 * 2880 * 2880 = 114.7e9` parameters at 0.5 bytes is `57.3GB`. Everything else (attention, router, embeddings) is about `2.1e9` parameters in bf16, or `4.2GB`. That's `61.5GB` total, leaving us about `18.5GB` for KV cache and activations. So it fits, which was pretty clearly the design target.

(b) Loading 61.5GB from HBM at 3.35e12 bytes/s takes `18.4ms`, so at batch 1 the best we can do is about 54 tokens/s. To be compute-bound in the experts we need $B > (E/k) \cdot (\text{bytes}/2) \cdot C / W_\text{hbm} = 32 \cdot 0.25 \cdot 296 = 2,370$ tokens per step. Note that 4-bit weights cut this number by 4x relative to bf16 weights (which would need 9,500), and that's a big part of why 4-bit matters so much for MoEs.

(c) Only the 18 full-attention layers hold a KV cache that grows with context, at `2 * 8 * 64 * 18 * 2 bytes = 36.9kB` per token. The sliding layers hold at most 128 tokens each, i.e. a fixed `18 * 128 * 2048 bytes = 4.7MB` per sequence. At 8k context each sequence costs us `8192 * 36.9e3 + 4.7e6 = 307MB`, so about `18.5e9 / 307e6 = 60` sequences fit. At batch 60 the step loads `61.5GB` of weights and `18.4GB` of KV, which takes `24ms`, for a throughput of about `2,500` tokens/s on one GPU. We're nowhere near the 2,370-token batch that would make the experts compute-bound, so the GPU is loading weights and KV cache the entire time.

{% enddetails %}

**Question 3 [expert parallelism for narrow experts]:** Say we want to train Qwen3-235B-A22B ($E = 128$, $k = 8$, $F = 1536$, $D = 4096$) on 32 HGX B200 nodes (256 GPUs). (a) What's the largest EP degree we can use while staying compute-bound? What about with the gather-based layout? (b) What batch size does the FSDP roofline then require? (c) Redo (a) and (b) on a TPU v6e 16x16 slice (256 chips, 2D torus, $\alpha = 5110$). *Assume 900GB/s of NVLink per GPU, one 400Gb/s NIC per GPU, and bf16 matmuls with fp8 dispatch.*

{% details Click here for the answer. %}

(a) Let's start inside a node, where $\alpha$ = `2.25e15 / 9e11 = 2,500` and the AllToAll needs $F > \alpha / 2 = 1{,}250$, which 1,536 clears, so **8-way EP inside the node works**. Across InfiniBand $\alpha = 45{,}000$ and the threshold is 22,500, so our experts can't leave the node. For the gather layout we need $Z < 1.5 k F / \alpha = 1.5 \cdot 8 \cdot 1536 / 2500 = 7.4$, so only 4-way, and dispatch is the better layout here, as everywhere on a switched fabric.

(b) With $Z = 8$ our FSDP inflation is $E / (kZ) = 128 / 64 = 2$. The cross-node FSDP threshold on an 8-GPU B200 node is $C / W_\text{node}$ = `2.25e15 / 4e11 = 5,625` tokens per GPU, so we need `2 * 5625 = 11,250` tokens per GPU, or a global batch of `256 * 11,250 = 2.9M` tokens (twice that with fp8 matmuls). That's a perfectly normal batch for a model this size, so our plan is EP8 inside each node and FSDP across the 32 nodes, with the batch doing most of the work.

(c) On v6e, no EP block is compute-bound for 1,536-wide experts: a 2x2 block has no wraparound and needs $F > 2560$, and a 4x4 needs 5,100, both well above 1,536 (the gather layout on a 2x2 without wraparound allows only $Z < 3.6$, so it doesn't rescue 4-way either). We could run 4-way EP about 1.7x communication-bound and hide the AllToAll behind attention, or drop EP entirely and use pure FSDP over the 16x16 torus, whose full axes do have wraparound. With inflation $E/k = 16$ that needs `16 * 5110 / 2 = 40,900` tokens per chip, i.e. a 10.5M-token global batch on 256 chips, 3.6x the GPU plan's 2.9M. So either way, Qwen's 1,536-wide experts are too narrow to spread over a 2D torus of this generation, even though an NVLink node spreads them eight ways with no trouble at all.

{% enddetails %}

**Question 4 [Llama 4's odd choice]:** Llama 4 Maverick uses 128 experts of width 8,192 with $k = 1$ plus one shared expert, in every other layer, and Meta trained it on H100s. Let's compare it with DeepSeek-V3 on (a) whether expert parallelism across InfiniBand is compute-bound with bf16 dispatch and bf16 matmuls, (b) the decode batch we'd need to be compute-bound on an H100, and (c) the same two numbers on TPU v5p and TPU v5e. What tradeoff did Meta make?

{% details Answer %}

(a) With bf16 both ways our GPU condition is $F > 2\alpha / 3$, and across H100 InfiniBand $\alpha = 19{,}800$, so the threshold is 13,200. Maverick's 8,192-wide experts miss it by 1.6x (1.2x with fp8 dispatch), which is close enough to hide behind attention, while DeepSeek-V3's 2,048-wide experts miss it by 6.4x.

(b) Our decode roofline is $E/k \cdot B_\text{crit}$, i.e. `128 * 296 = 37,900` tokens per step for Maverick against `32 * 296 = 9,500` for DeepSeek-V3 on an H100.

(c) On a v5p cube our condition is $F > A \alpha / 6$, so $A < 6 \cdot 8192 / 2550 = 19$. That means any cube up to $A = 16$ works, and Maverick could be expert-parallel across all 128 experts (a 4x4x8 mesh) with margin, while DeepSeek-V3's experts allow $A < 4.8$, so a 4x4x4 cube and no more. On a v5e the decode batches are 30,700 against 7,700.

**So what was the tradeoff?** Meta picked wide experts that are easy on the network, and paid for it with a 4x higher batch requirement in decode, while DeepSeek made exactly the opposite choice. But notice that at any realistic batch, decode is memory-bound for *both* models, so DeepSeek's choice costs almost nothing at inference time and buys all the combinatorial benefit of fine-grained routing we talked about above. That's probably a big part of why most later models copied DeepSeek and not Meta.

{% enddetails %}

**Question 5 [Kimi K3 on GB200 racks, hard]:** Say we want to pretrain Kimi K3 ($E = 896$, $k = 16$, $E_s = 2$, routed expert width $F = 3072$ on a latent of $\ell = 3584$, $D = 7168$, 93 layers of which 92 are MoE) on 128 GB200 NVL72 racks with a 60M-token batch, in fp8 with fp8 dispatch and bf16 weight gathers. *Assume 9,216 GPUs at 5e15 fp8 FLOPs/s each, 900GB/s of NVLink egress inside the rack, and one 400Gb/s NIC per GPU, so 3.6TB/s of InfiniBand egress per rack.*

(a) Can we use pure FSDP?

(b) Pick an EP degree that makes FSDP compute-bound. Is the EP AllToAll itself compute-bound?

(c) Roughly what fraction of peak could we achieve if the AllToAll weren't overlapped with anything? *Routed experts are about 47% of K3's active FLOPs (the shared experts, latent projections and attention are the other 53%).*

(d) Redo (a) to (c) on a full TPU7x pod of 9,216 chips, in bf16 matmuls with fp8 dispatch, with $\alpha = 12800$ and three ICI axes.

{% details Click here for the answer. %}

(a) Let's start with the per-GPU batch, which is `60e6 / 9216 = 6,500` tokens. The FSDP threshold across racks is $C / W_\text{rack}$ = `5e15 / 3.6e12 = 1,390` tokens per GPU for a dense model in fp8, and pure FSDP for K3 needs $E/k$ times that, or `56 * 1390 = 77,800`. That's twelve times what we have, so no!

(b) With EP $Z$ inside a rack we need `(56 / Z) * 1390 < 6500`, so $Z > 12$. Let's take $Z = 64$ (a rack has 72 GPUs, and 64 divides the 896 experts evenly), which gives us a threshold of `(56 / 64) * 1390 = 1,220` tokens per GPU, five times below what we have. For the AllToAll inside the rack we need $F > \alpha_\text{fp8} / 2$ = `(5e15 / 9e11) / 2 = 2,780` with fp8 matmuls and fp8 dispatch. K3's experts are 3,072 wide, so our dispatch is **compute-bound by 10%** with nothing overlapped at all. So both of our rooflines are satisfied.

(c) Since the AllToAll is compute-bound, an unoverlapped dispatch costs us no wall-clock time by this model, so the answer is "whatever the rest of the step allows": the pipeline bubbles, the attention layers and the HBM traffic of the linear-attention state set our utilization here rather than the network.

(d) On TPU7x the per-chip batch is the same 6,500, and pure FSDP needs $E/k \cdot \alpha / 3 = 56 \cdot 4270 = 239{,}000$ tokens per chip, about 37 times what we have. With EP over a 4x4x4 cube ($Z = 64$, $A = 4$) the FSDP threshold drops to `(56 / 64) * 4270 = 3,740` tokens per chip, which is fine. But the AllToAll needs $F > A \alpha / 8 = 4 \cdot 12800 / 8 = 6{,}400$, so the dispatch is about **2.1x communication-bound** on a cube (a 2x2x2 has no wraparound and lands at the same 6,400, and $Z = 8$ would leave FSDP at 29,900 tokens per chip, which we don't have). Note that the latent projection doesn't help the ratio, since it halves the bytes and the FLOPs together. If the routed-expert time is 2.1x its compute time and routed experts are 47% of FLOPs, the step takes `0.53 + 0.47 * 2.1 = 1.5` times its compute-bound duration, so we'd get about 66% of whatever utilization we would otherwise reach. In practice we'd overlap the dispatch with the attention layers and the shared experts (which K3 runs on a separate stream for exactly this reason) and recover most of that. So with the same model, batch and chip count, the AllToAll is free on paper on the GPU cluster but 2.1x over budget on the TPU pod.

{% enddetails %}

**Question 6 [DeepSeek-V3 on its own cluster]:** DeepSeek-V3 ($E = 256$, $k = 8$, $F = 2048$) trained on 2,048 H800s with 64-way EP across 8 nodes, 16-way pipeline parallelism and 2-way ZeRO-1 data parallelism, a 63M-token batch, fp8 matmuls and fp8 dispatch. Let's check the three rooflines of this section against that layout: the data-parallel weight traffic, the in-node AllToAll (NVLink 200GB/s spec, 160 measured) and the cross-node AllToAll (50GB/s). Which one binds, by how much, and how does that square with the run's 17% utilization of fp8 peak and the 20 SMs DeepSeek reserved for communication? Then let's repeat the exercise for the same model on a full TPU v5p pod (8,960 chips, $\alpha = 2550$): is there a configuration where both the FSDP and EP rooflines are satisfied without any overlap tricks?

{% details Click here for the answer. %}

Let's start with the data-parallel weight traffic. With EP64 and PP16 the FSDP-style threshold is `256 / (8 * 64 * 16) * 4950 = 155` tokens per GPU at fp8 peak against their 30,700, so it's never the bottleneck, and that's why 2-way ZeRO-1 was plenty. Our in-node AllToAll needs $F > 4{,}950$ (spec) or 6,190 (measured) with fp8 matmuls and fp8 dispatch, so 2,048-wide experts are 2.4x to 3x short inside the node. The cross-node AllToAll needs $F > 19{,}800$, so we're 9.7x short there, and InfiniBand is our bottleneck. This agrees nicely with DeepSeek's reported one-to-one ratio of compute to communication and with the 20 SMs they reserved for it. We'd predict utilization well under half of peak, and sure enough, they got 17 to 19%. On a v5p pod, a 4x4x4 cube needs $F > 1280$ (which we meet) and FSDP with $Z = 64$ needs `32 / 64 * 850 = 425` tokens per chip against `63e6 / 8960 = 7,000`, so both rooflines hold with no overlap tricks at all, although we're paying for that with 4.4x more chips.

{% enddetails %}

**Question 7 [tokens per expert]:** For the configurations in Question 6, how many tokens does each expert on each GPU or chip see per step? Is the matmul unit well utilized? Now let's ask the same question of DeepSeek's decode deployment (144-way EP and about 88 sequences per GPU, as [Section 16](../applied-frontier) derives from the published throughput). What does this tell us about where the FLOPs go during decode?

{% details Click here for the answer, once you've thought about it! %}

In training, each GPU holds 4 experts and sees `30,700 * 8 / 4 = 61,400` token-expert pairs per step, or about 15,000 rows per expert matmul, far above the 296 we need to be compute-bound. The same holds on the v5p pod (`7000 * 8 / 4 = 14,000` rows). In decode, each of our expert copies sees about 350 rows per step (see the batch-size section above), also above 296 per matmul, and yet the step is bound by weight and KV bytes and by the network. So the lesson for us is that per-matmul intensity isn't what's constraining us in either regime: in training we're bound by the AllToAll, and in decode by everything except the matmuls.

{% enddetails %}

**Question 8 [choose a chip]:** Say we had to serve Qwen3.8-2.4T-A95B ($E = 512$, $k = 10$, $F = 2048$, 2.4TB of fp8 weights) and could choose between a GB200 NVL72 rack (72 GPUs, 186GB and 8e12 bytes/s each, 900GB/s NVLink egress) and a 4x4x4 TPU7x cube (64 chips, 192GB and 7.4e12 bytes/s each). Which one holds the weights with more room for KV cache? On which is 64-way expert parallelism compute-bound for the AllToAll? And which gives us the lower per-step weight-loading time at small batch?

<h3 markdown=1 class="next-section">That's it for Section 13! For Section 14, on what happened to attention, click [here](../attention).</h3>
