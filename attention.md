---
layout: distill
title: "All About the KV Cache"
# permalink: /main/
description: "Section 7 argued that for long sequences the KV cache, not the weights, sets the cost of generation. Here we work out the bytes, FLOPs and arithmetic intensity of the attention mechanisms that replaced grouped-query attention since then: latent compression, sliding windows, sparse selection, linear-attention layers, and token compression. The cache per token shrank 30x in two years, and attention stopped being purely memory-bound along the way."
date: 2026-09-20
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Aghyad Deeb
    url: "https://github.com/aghyad-deeb"
    affiliations:
      name: "Independent; written with Claude"

section_number: 14

previous_section_url: "../moe"
previous_section_name: "Part 13: MoE"

next_section_url: ../training-2026
next_section_name: "Part 15: Training in 2026"

giscus_comments: false

bibliography: update.bib

toc:
  - name: "Why Attention Is Back on the Critical Path"
  - name: "A Menu of Attention Mechanisms"
  - name: "Multi-Head Latent Attention"
  - subsections:
    - name: "Parameters and cache size"
    - name: "The absorption trick changes the roofline"
  - name: "Sliding Windows and Local-Global Interleaving"
  - name: "Sparse Attention: Index, Then Attend"
  - name: "Linear Attention Hybrids: State Instead of Cache"
  - name: "Token Compression: DeepSeek-V4"
  - name: "Rooflines: Decoding With a Modern Cache"
  - name: "Rooflines: Attention FLOPs at Long Context"
  - name: "Positions and Context Extension"
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

_In [Section 7](https://jax-ml.github.io/scaling-book/inference) we derived the minimum time for a generate step: (batch times KV cache bytes plus parameter bytes) divided by memory bandwidth. We then spent [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) discovering that for LLaMA 3-70B at any context over a few thousand tokens, the KV term dominates. Since then contexts went from 8k to a million tokens, and reasoning models started producing outputs tens of thousands of tokens long. The KV cache per token became arguably the most consequential number in a model's config, and most of the frontier labs redesigned attention around it. This section is about what they did and what it costs._

## Why Attention Is Back on the Critical Path

Three trends pushed attention back onto the critical path after the MLP had dominated for years.

**Sequences got long.** LLaMA 3 shipped with 8k context and was extended to 128k. Kimi K3, DeepSeek-V4 and Qwen3.8 all serve a million tokens, and agentic workloads (a coding agent reading a repository, a research agent with a hundred tool calls) routinely sit at 100k to 500k tokens of context. At 128k tokens, LLaMA 3-70B's 164kB-per-token cache is 21GB *per sequence*.<d-footnote>Section 8 wrote this as 160kB; it is `2 * 8 * 128 * 80 = 163,840` bytes, which we round to 164kB here.</d-footnote> Four of those fill an H100 with nothing left for weights.

**Outputs got long.** A chat reply is a few hundred tokens. A reasoning trace is 10k to 100k. Decode was always the expensive half of serving per token; now it's also where most of the tokens are. [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) found you need three prefill servers per decode server for 8k-in, 512-out traffic. For 8k-in, 16k-out the same calculation gives `P / 0.91 = 32 * G / (0.019 * 16384)`, or about 11 decode servers per prefill server, and the decode roofline is the one that matters.

**MoEs made attention relatively expensive.** In [Section 13](../moe) we saw that DeepSeek-V3's attention is 30% of its active parameters, against 17% for LLaMA 3-70B, because the sparse MLP per token is narrow. The FLOPs of attention didn't shrink when the MLP went sparse. In prefill and training, attention FLOPs cross MLP FLOPs at about 19k tokens for DeepSeek-V3, where a dense model of the same width with $F = 4D$ would cross at 86k by the same formula (the $8D$ rule of thumb from [Section 4](https://jax-ml.github.io/scaling-book/transformers) counts the FLOPs differently; we come back to that below).

So attention is expensive in bytes at decode, expensive in FLOPs at long context, and there's a lot more long context than there used to be. Let's look at what people did about it.

## A Menu of Attention Mechanisms

The following table is the analog of the model table in [Section 13](../moe), for attention. Grouped-query attention (GQA, what [Section 4](https://jax-ml.github.io/scaling-book/transformers) called GMQA) is the baseline. The two numbers that matter are the *stored* cache per token (which sets how many sequences fit in HBM) and the bytes *read* per decoded token at a given context (which sets step time). They differ once a mechanism reads less than it stores (sparse attention) or keeps a fixed-size state (windows, linear layers). We assume one byte per cached element and quote the read figure at 128k context, including one read and one write of any recurrent state.<d-footnote>Every figure is computed from the model's <code>config.json</code> with the formulas in this section; the script is included with the chapter. Recurrent states are fp32 where the shipped config says so (Qwen's <code>mamba_ssm_dtype</code>) and bf16 otherwise. DeepSeek-V4's stored size is derived from its config and the storage precisions in its report; DeepSeek only publishes the ratio to V3.2 (10%), and our estimate gives 11.5%.</d-footnote>

| Model | Attention | Stored per token | Fixed per sequence | Read per token at 128k |
| :---- | :-------- | ---------------: | -----------------: | ---------------------: |
| LLaMA 3 70B | GQA, 8 KV heads x 128, all 80 layers | 164kB | 0 | 21.5GB |
| Qwen3-235B-A22B | GQA, 4 x 128, all 94 layers | 96kB | 0 | 12.6GB |
| DeepSeek-V3, Kimi K2, Mistral Large 3 | MLA, 576-wide latent, 61 layers | 35kB | 0 | 4.6GB |
| Gemma 3 27B | GQA 16 x 128; 10 global + 52 local (window 1024) | 41kB | 218MB | 5.6GB |
| Llama 4 Maverick | GQA 8 x 128; 12 global (NoPE) + 36 chunked (8192) | 25kB | 604MB | 3.8GB |
| gpt-oss-120b | GQA 8 x 64; 18 global + 18 local (window 128) | 18kB | 2.4MB | 2.4GB |
| DeepSeek-V3.2 | MLA + sparse top-2048 (DSA), 61 layers | 43kB | 0 | 1.1GB |
| GLM-5.2 | MLA + DSA, 78 layers, indexer in 21 of them | 48kB | 0 | 0.44GB |
| Qwen3.8-2.4T-A95B | 23 GQA (4 x 256) + 69 Gated DeltaNet | 47kB | 579MB | 7.3GB |
| Kimi K3 | 24 MLA + 69 Kimi Delta Attention | 14kB | 217MB | 2.3GB |
| Gemma 4 26B-A4B | 5 global (2 heads x 512, K=V) + 25 local (1024) | 5kB | 105MB | 0.78GB |
| Nemotron 3 Super | 8 GQA (2 x 128) + 40 Mamba-2 | 4kB | 168MB | 0.87GB |
| DeepSeek-V4-Pro | compressed + sparse (CSA/HCA), 61 layers | 5kB | 4.5MB | 0.10GB |
| DeepSeek-V4.1-Flash | CSA2: KV shared across layers (4 source layers of 40) + fp4 main cache | 0.9kB | | |

Read from top to bottom, the stored cache shrank 30x in two years and the bytes read per token at 128k shrank 200x. LLaMA 3-70B stores 164kB per token and reads 21.5GB per decoded token at 128k; DeepSeek-V4-Pro stores 5kB and reads 0.1GB. There are five distinct ideas in the table, and most 2026 models combine at least two of them. We take them in turn.

{% include figure.liquid path="assets/img/kv-bytes-vs-context.svg" class="img-fluid" caption="<b>Figure:</b> bytes read from HBM per decoded token, per sequence, as a function of context length, for one representative of each attention family. Note the log scales. GQA and MLA are straight lines (proportional to context); sparse attention, sliding windows and recurrent state all flatten the curve, and DeepSeek-V4 flattens it twice." %}

## Multi-Head Latent Attention

Multi-head latent attention (MLA) was introduced in DeepSeek-V2<d-cite key="DeepSeek2"></d-cite> and is now used by DeepSeek-V3, Kimi K2 and K3, GLM-5 and Mistral Large 3. The idea is to cache a low-rank *latent* of the keys and values rather than the keys and values themselves.

In standard attention each layer projects the residual $x \in \mathbb{R}^D$ to $K$ key heads and $K$ value heads of dimension $H$ and caches both, for $2KH$ elements per token per layer. MLA instead projects $x$ down to a single latent $c \in \mathbb{R}^{d_c}$ with $d_c = 512$, plus a small *decoupled* key $k_R \in \mathbb{R}^{d_R}$ with $d_R = 64$ that carries the rotary position information.<d-footnote>Rotary position embeddings (RoPE) have to be applied to the key before it is dotted with the query, but a rotated key cannot be reconstructed from a shared latent by a per-head linear map. MLA's answer is to split each key into a "no position" part that is reconstructed from the latent and a small "rope" part that is computed and cached directly and shared across heads. This is why you see <code>qk_nope_head_dim</code> and <code>qk_rope_head_dim</code> in the configs.</d-footnote> Each head then reconstructs its own key and value from the latent with per-head matrices $W_{UK}^{(h)} \in \mathbb{R}^{d_c \times H}$ and $W_{UV}^{(h)}$. The cache per token per layer is $d_c + d_R = 576$ elements. For DeepSeek-V3 with 61 layers that is 35,136 elements, or **35kB in fp8**, against 164kB for LLaMA 3-70B and 258kB for 405B.

The comparison people usually make is that 576 is about what GQA with 2.25 KV heads of dimension 128 would cost. But those 2.25 heads would be shared by all 128 query heads, whereas MLA reconstructs 128 *different* keys and values from the latent. It's expressive like MHA and cheap like MQA, and the price is paid in FLOPs, as we'll see.

### Parameters and cache size

Here are the per-layer parameters for DeepSeek-V3's MLA ($D = 7168$, 128 heads, $d_c = 512$, $d_R = 64$, query latent $d_q = 1536$, per-head dimensions 128 nope, 64 rope, 128 value):

| matrix | shape | params |
| :----- | :---- | -----: |
| $W_{DQ}$ (residual to query latent) | 7168 x 1536 | 11.0M |
| $W_{UQ}$ (query latent to 128 heads x 192) | 1536 x 24576 | 37.7M |
| $W_{DKV}$ (residual to KV latent + rope key) | 7168 x 576 | 4.1M |
| $W_{UK}$ (latent to 128 heads x 128) | 512 x 16384 | 8.4M |
| $W_{UV}$ (latent to 128 heads x 128) | 512 x 16384 | 8.4M |
| $W_O$ (128 heads x 128 to residual) | 16384 x 7168 | 117.4M |
| **Total** | | **187M** |

Two things to notice. The output projection $W_O$ is 63% of attention's parameters: 128 heads of dimension 128 make a 16,384-wide concatenated output, more than twice the residual width. And the matrices that touch the latent are tiny. MLA isn't a way to save attention parameters (LLaMA 3-70B's GQA layer is 151M parameters and DeepSeek-V3's is 187M); it's a way to save cache.

### The absorption trick changes the roofline

At decode time the naive way to use the cache is to reconstruct every head's keys and values from the latent for every past token, which costs `2 * 512 * 128 * (128 + 128) = 33.5M` FLOPs per past token per layer (the latent has to be pushed through $W_{UK}$ and $W_{UV}$ for all 128 heads) and, worse, materializes $S \cdot 128 \cdot 256$ elements of keys and values that then have to be read by the attention kernel. That would throw away the whole benefit. Instead, one *absorbs* the up-projections into the query and the output:

$$q_h^\top k_{h,s} = q_h^\top W_{UK}^{(h)\top} c_s = \tilde{q}_h^\top c_s \qquad \text{where} \quad \tilde{q}_h = W_{UK}^{(h)} q_h \in \mathbb{R}^{512}$$

So each head's query is mapped once into the 512-dimensional latent space, scores are computed against the shared latent $c_s$ (plus the 64-dimensional rope term), the softmax-weighted sum is taken over the latents to give a 512-dimensional output per head, and $W_{UV}$ is applied afterwards. The attention kernel never sees a per-head key or value. It sees one 576-wide "key" and one 512-wide "value" per token, shared by all 128 heads, exactly like MQA with an unusually wide head.

Let's count. Per layer, per decoded token, per context token, the absorbed attention does

$$2 \cdot N \cdot (d_c + d_R) + 2 \cdot N \cdot d_c = 2 \cdot 128 \cdot (576 + 512) = 278{,}528 \text{ FLOPs}$$

and reads $576$ bytes (fp8). That's an arithmetic intensity of **484 FLOPs/byte with an fp8 cache, or 242 with bf16**. Compare GQA: LLaMA 3-70B does $2 \cdot 64 \cdot 128 \cdot 2 = 32{,}768$ FLOPs per context token and reads $2 \cdot 8 \cdot 128 = 2048$ bytes, an intensity of 16 (int8) or 8 (bf16), which is just the group size $G = N/K$, as Question 4 of [Section 4](https://jax-ml.github.io/scaling-book/transformers) told us.

Recall the chip intensities from [Section 2](https://jax-ml.github.io/scaling-book/tpus) and [Section 12](https://jax-ml.github.io/scaling-book/gpus), and the hardware table in the [2026 outline](..): 240 for TPU v5e, 164 for v5p, 311 for TPU7x, 296 for H100, 281 for B200 (bf16 FLOPs over HBM bandwidth). GQA attention at intensity 8 is hopelessly memory-bound on all of them, which is why [Section 7](https://jax-ml.github.io/scaling-book/inference) could treat attention as always bandwidth-bound and move on. MLA attention at 484 FLOPs/byte with an fp8 cache is above the roofline on every chip listed; with a bf16 cache (242) it's above v5e and v5p but 14 to 22% below B200, H100 and TPU7x. **With MLA and an fp8 cache, decode attention is compute-bound at long context.** One assumption is buried in that 484: it pairs an fp8 cache with bf16 score and value matmuls, which is what DeepSeek's V3 write-up describes (fp8 storage, bf16 attention core). If the matmuls ran in fp8 too, the chip intensities would double (562 for B200, 591 for H100, 622 for TPU7x) and 484 would be memory-bound on all three. The rest of this section assumes bf16 matmuls on an fp8 cache. DeepSeek-V3 does 8.5x the attention FLOPs of LLaMA 3-70B per context token and reads 3.6x fewer bytes, and the trade is worth it because bytes were the binding constraint.

This has a practical consequence you can see in DeepSeek's serving layout ([Section 16](../applied-frontier)): attention is run *data-parallel* across all 144 decode GPUs, each GPU handling the attention for its own sequences, rather than tensor-parallel across heads. Tensor parallelism over heads would require every shard to read the full shared latent for every sequence (it isn't sharded by head), so per GPU it does the same FLOPs as data parallelism but $Y$ times the bytes: it divides the arithmetic intensity by $Y$ and pushes attention back toward memory-bound. Data parallelism keeps each sequence's latent on one GPU, so the intensity stays at 484 and the attention FLOPs available grow with the number of GPUs.

It also means the head count is now a hardware knob. The intensity of absorbed MLA is proportional to $N$, so you tune the number of heads to the ratio of FLOPs to bandwidth of the chip you serve on. The GLM-5 report<d-cite key="glm5"></d-cite> says this outright: "the number of attention heads in DeepSeek-V3 is selected according to the roofline of H800", which "is inappropriate for other hardware", and GLM-5 accordingly raised the head dimension from 192 to 256 and cut the head count by a third relative to its own baseline, landing at 64 heads. Kimi K2 also chose 64 heads, and its report quotes the trade: 128 heads would have cost 83% more inference FLOPs for a 0.5 to 1.2% gain in loss.<d-cite key="kimik2"></d-cite>

Let's put numbers on it. For DeepSeek-V3 at 128k context, one decoded token's attention costs `61 * 278528 * 131072 = 2.2e12` FLOPs and reads `61 * 131072 * 576 = 4.6GB`. At batch 64 on a B200 that's `64 * 2.2e12 / 2.25e15 = 63ms` of FLOPs against `64 * 4.6e9 / 8e12 = 37ms` of bytes. That's 63ms per step for attention alone, before we do any MoE work at all, which is what DeepSeek Sparse Attention was built to fix.

<p markdown=1 class="takeaway">**Takeaway:** MLA caches a shared 576-element latent per token per layer instead of $2KH$ keys and values, cutting DeepSeek-V3's cache to 35kB per token. With the absorption trick the decode kernel behaves like MQA with a 576-wide head shared by all 128 query heads, giving an arithmetic intensity of 240 to 480 FLOPs/byte. With an fp8 cache, attention is no longer memory-bound at long context; it's compute-bound on every current chip, and scales by adding chips in data parallel.</p>

## Sliding Windows and Local-Global Interleaving

The oldest trick is to not attend to everything. A *sliding window* layer attends only to the previous $w$ tokens, so its cache holds at most $w$ entries regardless of context. Models interleave a few *global* layers that see everything with many local ones:

| Model | Pattern | Window | Global layers |
| :---- | :------ | -----: | ------------: |
| Gemma 3 27B<d-cite key="gemma3"></d-cite> | 5 local : 1 global | 1,024 | 10 of 62 |
| Gemma 4 26B-A4B<d-cite key="gemma4"></d-cite> | 5 local : 1 global | 1,024 | 5 of 30 |
| gpt-oss-120b<d-cite key="gptoss"></d-cite> | alternating | 128 | 18 of 36 |
| Llama 4 Maverick<d-cite key="llama4"></d-cite> | 3 chunked : 1 global | 8,192 (chunk) | 12 of 48 |
| Meta Muse Glimmer 30B<d-footnote>Meta's 30B dense model of August 2026, which copies Llama 4's local-global pattern with a 2,048-token window and two KV heads.</d-footnote> | 3 local : 1 global | 2,048 | 13 of 52 |
| DeepSeek-V4 (as a side branch) | every layer | 128 | see below |

The cache per token as a function of context is then

$$\text{bytes per token}(S) = 2 K H \cdot \text{bytes} \cdot \left[ L_\text{global} + L_\text{local} \cdot \min\left(1, \frac{w}{S}\right) \right]$$

Once $S \gg w$ only the global layers count, and the saving is $L / L_\text{global}$: 6x for Gemma, 4x for Llama 4, 2x for gpt-oss. Gemma 4 stacks two more tricks on its global layers: only 2 KV heads (against 8 in the local layers), and *keys reused as values* (`attention_k_eq_v` in the config), so the global cache is $1 \cdot 2 \cdot 512 = 1024$ bytes per layer per token and 5kB per token in total. That's 8x smaller than Gemma 3 27B at long context, in a model of similar size.

Windows break in two ways unless you fix them. The first is the *sink*. When the earliest tokens fall out of the window, attention has nowhere to dump the probability mass it used to park on them, and quality collapses<d-cite key="streamingllm"></d-cite>. gpt-oss adds a learned per-head bias to the softmax denominator, a slot that attends to nothing, so the excess mass has somewhere harmless to go; DeepSeek-V4 has "learnable attention-sink logits" for the same reason. The second is *positions*. Llama 4's global layers use no positional encoding at all (NoPE; Llama 4 calls the interleaving of RoPE and NoPE layers "iRoPE"), Gemma 4's use a low-frequency variant, and Muse Glimmer copies Llama 4's pattern. The local layers carry the positional signal; the global layers only have to retrieve. This is what makes the 1M and 10M context claims possible without retraining the position embedding.

For FLOPs, a local layer costs $2 N H \cdot 2 w$ per token, a constant. Only the global layers have the quadratic term, so the attention-versus-MLP crossover of [Section 4](https://jax-ml.github.io/scaling-book/transformers) moves out by $L / L_\text{global}$. For Gemma 3 27B it is over 500k tokens.

<p markdown=1 class="takeaway">**Takeaway:** Interleaving local and global layers divides the growing part of the cache by $L / L_\text{global}$ and leaves a fixed $L_\text{local} \cdot w \cdot 2KH$ bytes per sequence. With a 1024-token window and one global layer in six, plus K=V and 2 heads in the global layers, Gemma 4 gets to 5kB per token. Sinks and position-free global layers are what make it stable.</p>

## Sparse Attention: Index, Then Attend

Sliding windows decide what to attend to by position. *Sparse attention* decides by content: a cheap *indexer* scores every past token, the top $k_a$ are selected, and full attention runs over only those. DeepSeek introduced this at scale in V3.2<d-cite key="deepseekv32"></d-cite> as DeepSeek Sparse Attention (DSA), after a research version called Native Sparse Attention<d-cite key="nsa"></d-cite>; GLM-5 and MiniMax M3 adopted variants of it within months.

DSA sits on top of MLA. The *lightning indexer* has $H_I = 64$ small heads of dimension $d_I = 128$, computes one 128-dimensional indexer key per token (shared across the 64 heads), and scores past token $s$ for query $t$ as

$$I_{t,s} = \sum_{j=1}^{64} w_{t,j} \cdot \text{ReLU}(q^I_{t,j} \cdot k^I_s)$$

in fp8, with a ReLU rather than a softmax so that nothing has to be normalized across $S$. The top $k_a = 2048$ latents by score are selected and the absorbed MLA attention above runs over just those 2048. Per layer, per decoded token, this costs

| | FLOPs | bytes (fp8) |
| :-- | ----: | ----------: |
| Indexer, per context token | $2 \cdot 64 \cdot 128 = 16{,}384$ | 128 |
| Main attention, fixed | $278{,}528 \cdot 2048 = 570\text{M}$ | $2048 \cdot 576 = 1.2\text{MB}$ |

The indexer is 17x cheaper per context token in FLOPs than full MLA attention and reads 4.5x fewer bytes, and the expensive attention no longer grows with context at all. At 128k context DeepSeek-V3.2 reads 1.1GB and does 0.17 TFLOPs per decoded token, against 4.6GB and 2.2 TFLOPs for V3: 4x fewer bytes, 13x fewer FLOPs. On a B200 at batch 64 the attention step drops from 63ms (compute-bound) to about 9ms (memory-bound on the indexer keys). DeepSeek's published cost curves show decode cost per token essentially flat in context length, and its API price was cut in half on the day V3.2-Exp shipped.

Notice where the remaining cost lives: in the indexer. At 1M context the indexer is 98% of the bytes and 95% of the FLOPs of V3.2's attention. Two follow-ups attack exactly that. GLM-5.2 computes the index in only 21 of its 78 layers and reuses the selected positions in the other 57 (`indexer_types` and `index_topk_freq` in its config), cutting reads at 128k to 0.44GB. DeepSeek-V4 compresses the indexer keys 4:1 and stores them in fp4, which we come to below.

Training DSA is cheap. V3.2 was made from V3.1 by continued pretraining: 2.1B tokens with everything frozen except the indexer, trained to match the real attention distribution (a KL loss against the head-summed softmax), then 944B tokens with sparse attention turned on. That's 6% of the original pretraining budget for a 4x cut in decode bytes.

**A note on TPUs.** Top-$k$ selection over $S$ scores followed by a gather of 2048 latents at data-dependent addresses is natural on a GPU (a radix-select kernel and a scattered read). On a TPU it's the same kind of dynamic-shape, gather-heavy operation that makes MoE routing awkward, and it needs a custom kernel. This isn't a reason it can't be done, but it's a reason sparse attention arrived first on GPUs.

<p markdown=1 class="takeaway">**Takeaway:** Sparse attention replaces the 576-byte-per-context-token attention read with a 128-byte indexer read, plus a fixed attention over 2048 selected entries. For DeepSeek-V3.2 that is 4x fewer bytes and 13x fewer FLOPs than full MLA at 128k. The indexer becomes the cost at very long context, so newer models share it across layers or compress its keys.</p>

## Linear Attention Hybrids: State Instead of Cache

The most radical option is to get rid of the growing cache in most layers entirely. A *linear attention* or state-space layer maintains a fixed-size matrix state per head, updated once per token, instead of a list of past keys and values. Qwen3-Next<d-cite key="qwen3next"></d-cite>, Qwen3.5 and Qwen3.8 use Gated DeltaNet<d-cite key="gateddeltanet"></d-cite>; Kimi Linear<d-cite key="kimilinear"></d-cite> and Kimi K3 use a variant with a per-channel decay called Kimi Delta Attention (KDA); Nemotron 3 uses Mamba-2<d-cite key="mamba2"></d-cite>. All of them interleave a minority of ordinary attention layers, because pure linear attention cannot do exact retrieval.

The delta-rule recurrence, per head with key dimension $d_k$ and value dimension $d_v$, is

$$\Sigma_t = \Sigma_{t-1} \left( \gamma_t I - \beta_t k_t k_t^\top \right) + \beta_t v_t k_t^\top, \qquad o_t = \Sigma_t q_t$$

where $\Sigma_t \in \mathbb{R}^{d_v \times d_k}$ is the state (we avoid $S$, which is the context length throughout this book), $\gamma_t$ is a learned decay (not [Section 7](https://jax-ml.github.io/scaling-book/inference)'s $\alpha$) and $\beta_t$ a learned write strength. You can read it as an online least-squares update of an associative memory. What matters to us: the state has $d_k d_v$ elements per head, is read and written once per decoded token, and doesn't depend on $S$.

| Model | Linear : full | Linear heads x $d_k$ x $d_v$ | State per layer | Layers | State per sequence |
| :---- | :-----------: | :--------------------------: | --------------: | -----: | -----------------: |
| Qwen3.5-397B | 3 : 1 | 64 x 128 x 128 | 4.2MB (fp32) | 45 | 189MB |
| Qwen3.8-2.4T | 3 : 1 | 128 x 128 x 128 | 8.4MB (fp32) | 69 | 579MB |
| Kimi K3 | 3 : 1 | 96 x 128 x 128 | 3.1MB (bf16) | 69 | 217MB |
| Nemotron 3 Super | ~5 : 1 | 128 heads x 64 x 128 (Mamba-2) | 4.2MB (fp32) | 40 | 168MB |

These states aren't small. Qwen3.8 reads and writes 1.16GB of state per sequence per decode step, at *any* context length. Its 23 full-attention layers (4 KV heads of 256) read 47kB per context token, so the state traffic equals the cache traffic at `1.16e9 / 47e3 = 24,600` tokens. Below that the fixed state dominates the hybrid's step; above it the growing cache does. The comparison that matters is against an all-attention model of the same shape (92 GQA layers, 188kB per token): the hybrid reads less than that once `1.16e9 / (188e3 - 47e3) = 8,200` tokens are in context, and at 1M Qwen3.8 reads 50GB per token where the all-GQA model would read `92 * 2048 * 1048576 = 198GB`. For Kimi K3 the state stops dominating at about 31k tokens, and the hybrid beats an all-MLA 93-layer stack above about 11k.<d-footnote>Storing the state in bf16 halves these numbers. Qwen ships with an fp32 state (<code>mamba_ssm_dtype: float32</code>), presumably because the recurrence accumulates error; Kimi doesn't say and we assume bf16. It's a real design knob: the state is written every step, so its precision is more delicate than a KV cache that is written once.</d-footnote>

So hybrids are a bet on long context, and the state is a per-sequence cost that batching doesn't amortize (each sequence has its own), which makes it behave like a KV cache of fixed length rather than like weights. The FLOPs, on the other hand, are trivial: updating a $128 \times 128$ state per head is about $8 d_k d_v$ FLOPs, `8 * 128 * 128 * 128 = 16.8M` per layer for Qwen3.8, against `2 * 8.4MB = 16.8MB` of fp32 state read and written, an intensity of about 1 FLOP per byte (2 with a bf16 state), so the linear layers are memory-bound like GQA and unlike MLA.

In training, the recurrence is computed chunk-wise: within a chunk of 64 tokens everything is a matmul, and states are passed between chunks. FLOPs are $O(S d_k d_v)$ per head with no quadratic term, which is what lets Qwen3-Next claim over 10x the inference throughput of Qwen3-32B beyond 32k tokens of context. Context parallelism becomes a sequential pass of a few megabytes of state between chips rather than an AllGather of keys and values; Kimi K3's report describes a dedicated scheme for this.

One more piece of evidence about where this design sits: MiniMax M1 shipped a 7:1 linear hybrid in mid-2025, MiniMax M2 went back to full attention four months later, and MiniMax M3 went to block-sparse attention instead. Qwen and Kimi bet on hybrids, DeepSeek on sparsity plus compression, Google on windows. Nobody agrees on the mechanism, but everyone has landed within a factor of ten of each other, somewhere between 5 and 50kB stored per token.

<p markdown=1 class="takeaway">**Takeaway:** Linear-attention layers replace a growing cache with a fixed state of $d_k d_v$ elements per head, read and written every step. That is hundreds of megabytes per sequence for 2026 models, so hybrids read more than an all-attention model of the same shape below roughly 8 to 11k tokens of context, and win by 3 to 4x at 1M. They are memory-bound, per-sequence, and cheap to train at long context.</p>

## Token Compression: DeepSeek-V4

DeepSeek-V4<d-cite key="deepseekv4"></d-cite> combines most of the above and adds one new idea: **compress the KV cache along the sequence before storing it.** Its attention has three components, all sharing one cache format:

* **A single shared KV entry per token** of 512 dimensions, the last 64 of which carry RoPE, used as both key and value (`num_key_value_heads: 1`, `head_dim: 512`, `qk_rope_head_dim: 64`). It's stored as 448 fp8 plus 64 bf16 elements, 576 bytes per entry, so it matches MLA's 576 in bytes through the mixed precision rather than in element count. This is MQA taken to its conclusion, with 128 query heads (Pro) each 512 wide, read from a 1536-dimensional query latent as in MLA, and each head does `2 * (512 + 512) = 2048` FLOPs per selected entry.
* **Compressed Sparse Attention (CSA)**, in half the layers: every 4 consecutive tokens are compressed into one entry by a small learned module, an indexer (64 heads of 128, in fp4) scores the compressed entries, and attention runs over the top 1024 (Pro) or 512 (Flash) of them plus a 128-token uncompressed sliding window.
* **Heavily Compressed Attention (HCA)**, in the other half: every 128 tokens become one entry, and attention runs over all of them with no sparsity.

The layers alternate CSA, HCA, CSA, HCA. Per token, the stored cache is roughly $(576 + 64) / 4$ bytes for a CSA layer and $576 / 128$ for an HCA layer; over 30 CSA and 31 HCA layers that is about **4.9kB per token**, which DeepSeek quotes as 10% of V3.2 (our estimate is 11.5%). The bytes *read* per decoded token at context $S$ are

$$30 \cdot \left[ \frac{S}{4} \cdot 64 + 1024 \cdot 576 + 128 \cdot 576 \right] + 31 \cdot \left[ \frac{S}{128} \cdot 576 + 128 \cdot 576 \right]$$

which is 0.1GB at 128k and 0.67GB at 1M. Compare 8.3GB for V3.2 and 37GB for V3 at 1M. The FLOPs fall similarly: about 0.2 TFLOPs per token at 1M, 5x below V3.2 and 80x below V3. DeepSeek states that at 1M context V4-Pro needs 27% of V3.2's per-token FLOPs (for the whole model) and 10% of its cache, and that the cache is about 2% of what a bf16 GQA-8, head-128 model would need (about 4% of one stored in fp8, the convention we use elsewhere in this section).

Note how the compression also fixes the indexer problem from the previous section: indexer keys are stored per *compressed* entry (one per 4 tokens) in fp4 (64 bytes), so the indexer reads 16 bytes per original token per layer instead of 128. That's the 98%-of-bytes term from V3.2 divided by eight.

V4.1-Flash goes further in two ways. It shares the compressed KV entries and indexer keys across layers, so only 4 of its 40 layers produce a global cache (`kv_source_layer_ids: [2, 8, 14, 20]`, with 2:1 compression in the 20-layer encoder half of its causal encoder-decoder and 1:1 in the decoder half, and no HCA layers), and it stores that main cache in fp4 (an E2M1 element with one E4M3 scale per 16 channels; the 128-token window stays fp8). Together these reach a published 890 bytes per token, about a quarter of V4-Flash: roughly 2x from fp4 and 2x from the sharing. That's 184x smaller than LLaMA 3-70B's cache, for a model with a 552B backbone plus 196B of Engram memory tables and a million-token context.

<p markdown=1 class="takeaway">**Takeaway:** DeepSeek-V4 stores one 576-byte entry per 4 tokens (sparse-selected) or per 128 tokens (dense), with a 128-token raw window, for about 5kB per token stored and 0.7GB read per token at 1M context. Compression along the sequence multiplies with every other trick, including shrinking the indexer that limited V3.2.</p>

## Rooflines: Decoding With a Modern Cache

[Section 7](https://jax-ml.github.io/scaling-book/inference)'s step-time formula needs one amendment. With MLA-class attention the FLOPs term of attention is no longer negligible, so

$$T_\text{step} \geq \max\left( \frac{B \cdot \text{bytes read}(S) + \text{weight bytes per chip}}{W_\text{hbm}}, \; \frac{B \cdot \text{attention FLOPs}(S) + 2 B \cdot \text{active params}}{C} \right)$$

where $B$ is the per-chip batch in sequences, $\text{bytes read}(S)$ comes from the table above, and the weight bytes per chip are the model's weights divided by however many chips they are spread over (the MoE FLOPs term uses all the active parameters because every one of the chip's own tokens visits $k$ experts somewhere). Let's see what this does for a few models on a 192GB chip (TPU7x, or a GB200 at 186GB) with weights spread over a 64-chip group, at 128k context:

| Model | Weights per chip | Cache per sequence (stored) | Sequences per chip |
| :---- | ---------------: | --------------------------: | -----------------: |
| LLaMA 3 405B, int8 | 6.3GB | 33.8GB | 5 |
| DeepSeek-V3, fp8 | 10.5GB | 4.6GB | 39 |
| DeepSeek-V3.2, fp8 | 10.7GB | 5.6 GB (incl. indexer keys) | 32 |
| Qwen3.8-2.4T, fp8 | 37.5GB | 6.8GB | 23 |
| Kimi K3, fp4 experts | 22.2GB | 2.0GB | 84 |

Five 128k-token sequences per chip for a dense GQA model of 2024; 39 for an MLA model; 84 for a hybrid. That's the memory side. Now the time side, for DeepSeek-V3 versus V3.2 at batch 32 per chip on a B200 (8e12 bytes/s, 2.25e15 FLOPs/s):

| | bytes per step | FLOPs per step | $T_\text{step}$ | tokens/s per chip |
| :-- | -------------: | -------------: | --------------: | ----------------: |
| V3 (full MLA) | 32 x 4.6 + 10.5 = 158GB | 32 x 2.2e12 (attn) + 32 x 7.4e10 (MoE) = 7.3e13 | max(20ms, **32ms**) | 1,000 |
| V3.2 (DSA) | 32 x 1.1 + 10.7 = 46GB | 32 x 1.7e11 + 32 x 7.4e10 = 7.8e12 | max(**5.8ms**, 3.5ms) | 5,500 |

Full MLA is *compute*-bound on attention at this context, which no model in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) ever was. DSA drops both terms and lands memory-bound at about 5x the throughput. This is a large part of why long-context API prices fell so far between 2025 and 2026.

The same table tells you what the KV term did to the disaggregation argument. In [Section 7](https://jax-ml.github.io/scaling-book/inference) we moved the KV cache from prefill to decode servers over the network and called the cost "typically acceptable". A 128k-token DeepSeek-V3 cache is 4.6GB; at 50GB/s per GPU that is 92ms, a few decode steps. A LLaMA 3-405B cache at the same length is 34GB, 0.7 seconds. Small caches are what make disaggregated serving of long contexts work at all.

<p markdown=1 class="takeaway">**Takeaway:** Add an attention-FLOPs term to the [Section 7](https://jax-ml.github.io/scaling-book/inference) decode roofline. For MLA models at long context it is the binding term; sparse attention removes it. Cache size sets sequences per chip (5 for LLaMA 3-405B, 39 for DeepSeek-V3, 84 for Kimi K3 at 128k on a 192GB chip) and, through the KV transfer, whether prefill-decode disaggregation is even feasible at long context.</p>

## Rooflines: Attention FLOPs at Long Context

[Section 4](https://jax-ml.github.io/scaling-book/transformers) derived that dot-product attention FLOPs exceed all other matmul FLOPs, the QKVO projections included, once $T > 8D$, assuming non-causal MHA and $F = 4D$. Neither assumption holds anymore, and we'll also change the comparison: causal attention (which halves the score FLOPs) against the MLP alone. Under the book's assumptions this gives $S^* = 12D$ rather than $8D$, which is why LLaMA 3-70B appears at 86k in the table below rather than Section 4's 65k; that difference is accounting, not architecture. Write $d_{qk}$ and $d_v$ for the query-key and value head dimensions (both equal to $H$ in [Section 4](https://jax-ml.github.io/scaling-book/transformers); 192 and 128 for DeepSeek's MLA), and as in [Section 13](../moe) let $k$ be the active experts and $E_s$ the shared ones. Then per token per layer with causal attention, the score and value matmuls cost $N (d_{qk} + d_v) S$ FLOPs (the factor 2 for multiply-add and the factor $1/2$ for causality cancel) in the layers that have full attention, and the MLP costs $6 (k + E_s) D F$. Setting them equal over the whole model:

$$S^* = \frac{6 (k + E_s) D F}{N (d_{qk} + d_v)} \cdot \frac{L}{L_\text{full}}$$

| Model | $N$ | $d_{qk} + d_v$ | MLP per token per layer | $L_\text{full} / L$ | $S^*$ |
| :---- | --: | -------------: | ----------------------: | ------------------: | ----: |
| LLaMA 3 70B | 64 | 256 | 6 x 8192 x 28672 | 1 | 86k |
| LLaMA 3 405B | 128 | 256 | 6 x 16384 x 53248 | 1 | 160k |
| DeepSeek-V3 | 128 | 320 | 6 x 9 x 7168 x 2048 | 1 | **19k** |
| Kimi K2 | 64 | 320 | 6 x 9 x 7168 x 2048 | 1 | 39k |
| GLM-5 | 64 | 512 | 6 x 9 x 6144 x 2048 | 1 | 21k |
| Qwen3.8-2.4T | 64 | 512 | 6 x 11 x 8192 x 2048 | 23/92 | 135k |
| Kimi K3 | 96 | 320 | 2 x (16 x 3 x 3584 x 3072 + 2 x 3 x 7168 x 3072 + 2 x 7168 x 3584) | 24/93 | 180k |
| Gemma 3 27B | 32 | 256 | 6 x 5376 x 21504 | 10/62 | 525k |

This is the FLOPs-side reason the 2026 architectures look the way they do. Going from dense LLaMA 3 to a fine-grained MoE with 128 heads pulled the crossover in from 86k to 19k tokens: with a sparse MLP, attention is the expensive part of a 32k-token training sequence. Halving the heads (Kimi K2) buys 2x. Making three quarters of the layers linear (Qwen3.8, Kimi K3) buys 4x on top, and sliding windows in five layers of six (Gemma) buy 6x. Sparse attention changes the functional form: with DSA the quadratic term is the indexer at $H_I d_I S = 8192 S$ per token per layer, giving $S^* \approx 97\text{k}$ for V3.2, and the main attention becomes a constant.

Context parallelism ([Section 5](https://jax-ml.github.io/scaling-book/training)'s note) is what you reach for when even that is not enough, or when a single sequence's activations do not fit. For attention layers it is cheap: ring attention<d-cite key="ringattention"></d-cite> passes each chip's KV shard around the ring, `S * 576` bytes per layer for MLA, a rounding error next to the $S^2$ FLOPs it enables. For linear layers it is a sequential hand-off of the state between chunks. Most 2026 reports mention some form of it; Kimi K3 trained at 64k tokens and extended to 1M in its cooldown phase (the low-learning-rate tail of pretraining), DeepSeek-V4 trained at 64k and used YaRN to reach 1M.

<p markdown=1 class="takeaway">**Takeaway:** The attention-versus-MLP crossover is $S^* = 6(k + E_s) D F L / (N (d_{qk} + d_v) L_\text{full})$. Fine-grained MoE with many heads pulled it down to 19k tokens (DeepSeek-V3); hybrids, windows and sparsity pushed it back past 100k. That, as much as the cache, is why long-context training of 2026 models is affordable.</p>

## Positions and Context Extension

A short section on a question people ask constantly: how do models trained at 4k to 64k tokens serve a million?

Almost every model still uses RoPE, and extends it with YaRN<d-cite key="yarn"></d-cite>, which rescales the rotary frequencies so that positions beyond the training length map into the trained range. The config tells you the recipe:

| Model | Trained at | Extended to | Method |
| :---- | ---------: | ----------: | :----- |
| DeepSeek-V3 | 4k, then 32k, then 128k | 128k | YaRN factor 40 from 4k, two 1000-step phases |
| gpt-oss-120b | 4k | 128k | YaRN factor 32 |
| DeepSeek-V4 | 4k, 16k, 64k, 1M | 1M | YaRN factor 16 from 64k |
| Qwen3.8 | 262k native | 1M | YaRN factor 4 |
| Kimi K3 | 8k, 64k; cooldown 256k to 1M | 1M | none: MLA layers have no positional encoding |
| Llama 4 Scout | | 10M (claimed) | NoPE global layers; chunked local layers |

Partial RoPE and NoPE both shrink what has to be extended. *Partial RoPE* rotates only a slice of each head: 64 of 192 query dimensions in MLA, 64 of 256 in Qwen3.5 and 3.8, 64 of 512 in DeepSeek-V4. *NoPE* removes positions from the global layers entirely (Llama 4, Muse Glimmer, Kimi K3), on the theory that the local or recurrent layers carry position and the global layers only need content. Kimi K3 is the cleanest case: its linear layers encode position through their decay, its MLA layers have none, and it "extrapolates directly to 1M-token contexts without any positional-encoding modification."

For the systems person the relevant fact is the training schedule: a few percent of tokens at the final length. DeepSeek-V3 spent 2,000 of roughly 240,000 training steps on context extension; V3.2's sparse-attention conversion was 6% of pretraining. Long context is a post-hoc phase, and the architecture is chosen so that the phase is short.

## What Should You Take Away from this Section?

* Stored cache per token, in fp8: GQA-8 with 128-dim heads is $256 L$ bytes (164kB for LLaMA 3-70B); MLA is $576 L$ (35kB for 61 layers); interleaved windows divide the growing part by $L / L_\text{global}$; hybrids keep it only in the full-attention layers plus a fixed state of $d_k d_v$ per head; DeepSeek-V4's compression gets to about 5kB and V4.1's fp4 cache to 0.9kB.

* Bytes *read* per decoded token can be far below bytes stored: sparse attention reads 128 bytes of indexer key per context token plus a fixed 2048 entries, cutting DeepSeek-V3.2's reads 4x at 128k relative to V3 and holding them nearly flat in context.

* MLA's absorbed decode has an arithmetic intensity of 240 to 480 FLOPs/byte, so attention is compute-bound at long context for the first time. Sparse attention removes 13x of those FLOPs. Data-parallel attention across the expert-parallel (EP) group is the layout that follows.

* Linear-attention states are hundreds of megabytes per sequence and are read and written every step; hybrids beat full attention only past about 8 to 11k tokens, and win 3 to 4x at 1M.

* The attention-versus-MLP FLOPs crossover is $6 (k + E_s) D F L / (N (d_{qk} + d_v) L_\text{full})$: 19k tokens for DeepSeek-V3, over 100k for the 2026 hybrids.

* Long context is a short training phase enabled by YaRN, partial RoPE, and position-free global layers, not a property of the whole pretraining run.

## Worked Problems

**Question 1 [cache sizes]:** GLM-5.2 has 78 layers of MLA (latent 512 + 64) with a DSA indexer (128-dimensional keys) computed in 21 layers and reused in the rest. Kimi K3 has 24 MLA layers and 69 KDA layers with 96 heads of $128 \times 128$ state in bf16. For each, compute the stored cache per token, the fixed per-sequence state, and the bytes read per decoded token at 128k context (one byte per element; count one read and one write of recurrent state).

{% details Click here for the answer. %}

**GLM-5.2:** stored per token is `78 * 576 + 21 * 128 = 44.9kB + 2.7kB = 47.6kB`; no fixed state. Reads at 128k: the 21 indexers read `21 * 131072 * 128 = 352MB` and all 78 layers read 2048 selected latents, `78 * 2048 * 576 = 92MB`, for **444MB**. Storing a 128k sequence costs `47.6e3 * 131072 = 6.2GB`, of which only 7% is touched per step.

**Kimi K3:** stored per token is `24 * 576 = 13.8kB` (only the MLA layers grow). The KDA state is `69 * 96 * 128 * 128 * 2 bytes = 217MB` per sequence. Reads at 128k: `24 * 131072 * 576 = 1.81GB` of latents plus `2 * 217MB = 434MB` of state traffic, **2.25GB**. At 8k the same calculation gives 113MB + 434MB = 547MB, of which 80% is the state: at short context the hybrid's fixed cost dominates.

{% enddetails %}

**Question 2 [Gemma 3 versus Gemma 4]:** Gemma 3 27B has 62 layers, 16 KV heads of 128, with 10 global layers and 52 local layers of window 1024. Gemma 4 26B-A4B has 30 layers, 5 global layers with 2 KV heads of 512 that reuse keys as values, and 25 local layers with 8 KV heads of 256 and window 1024. Compare their bytes read per decoded token at 8k and 256k context, and explain the ratio.

{% details Click here for the answer. %}

Gemma 3: global layers read `10 * 2 * 16 * 128 = 41kB` per context token; local layers hold `52 * 1024 * 4096 = 218MB` fixed. At 8k: `41e3 * 8192 + 218e6 = 554MB`. At 256k: `10.7GB + 0.2GB = 10.9GB`.

Gemma 4: global layers read `5 * 1 * 2 * 512 = 5.1kB` per context token (the factor 1 rather than 2 is K=V); local layers hold `25 * 1024 * 4096 = 105MB`. At 8k: `42MB + 105MB = 147MB`. At 256k: `1.34GB + 0.1GB = 1.44GB`.

The long-context ratio is 7.6x. Per global layer, Gemma 4 stores 1024 bytes per token against Gemma 3's 4096 (2 heads instead of 16 is 8x fewer, 512 dims instead of 128 is 4x more, K=V is 2x fewer), and it has half as many global layers. At 8k the fixed local caches are a large fraction for both, and the ratio is only 3.8x.

{% enddetails %}

**Question 3 [MLA intensity]:** Kimi K2 and GLM-5 use MLA with 64 query heads rather than DeepSeek-V3's 128. What is the arithmetic intensity of their absorbed decode attention with an fp8 cache and with a bf16 cache? On which of TPU v5e, v5p, TPU7x, H100 and B200 is it compute-bound?

{% details Click here for the answer. %}

Per context token per layer the FLOPs are `2 * 64 * (576 + 512) = 139,264` and the bytes are 576 (fp8) or 1152 (bf16), so the intensity is **242 (fp8)** or **121 (bf16)**. Chip intensities in bf16 FLOPs are 240 (v5e), 164 (v5p), 311 (TPU7x), 296 (H100), 281 (B200). With an fp8 cache, 64-head MLA is compute-bound on v5p and just at the line on v5e, and memory-bound (by 15 to 25%) on the 2026 chips. With a bf16 cache it is memory-bound everywhere. So halving the heads relative to DeepSeek-V3 puts these models back on the memory-bound side of the line, which is probably the right side to be on given how expensive the FLOPs turned out to be for V3. Note that the intensity of MLA is proportional to $N$: it's a design knob.

{% enddetails %}

**Question 4 [where DSA's cost goes]:** For DeepSeek-V3.2 at 1M context, compute the bytes and FLOPs per decoded token and the fraction due to the indexer. How much does GLM-5.2's scheme of indexing in 21 of 78 layers save at that length? What does DeepSeek-V4 do to the indexer term?

{% details Click here for the answer. %}

Per layer at $S = 2^{20}$: indexer bytes `1048576 * 128 = 134MB`, main attention bytes `2048 * 576 = 1.2MB`; indexer FLOPs `16384 * 1048576 = 17.2G`, main `0.57G`. Over 61 layers: **8.3GB and 1.08 TFLOPs per token, 98% and 95% of it indexer.** GLM-5.2's approach (scaled to V3.2's 61 layers, indexing in 16 of them) would read `16 * 134MB + 61 * 1.2MB = 2.2GB`, 3.7x less. DeepSeek-V4 stores one 64-byte fp4 indexer key per 4 tokens, so the indexer reads 16 bytes per original token per CSA layer: `30 * 1048576 * 16 = 0.5GB` at 1M across its 30 CSA layers, 16x less than V3.2's indexer, with the main attention over 1024 compressed entries plus a 128-token window adding another 20MB.

{% enddetails %}

**Question 5 [hybrid break-even]:** Qwen3.8 has 23 full-attention layers with 4 KV heads of 256 and 69 Gated DeltaNet layers with 128 heads of $128 \times 128$ state. At what context length does the per-step state traffic (one read, one write) equal the full-attention cache traffic, with the state in fp32? In bf16? How does this compare with a DeepSeek-V3 sequence?

{% details Click here for the answer. %}

State per layer: `128 * 128 * 128 * 4 = 8.4MB` in fp32; over 69 layers, 579MB; read plus write, **1.16GB per step**. Full layers read `23 * 2 * 4 * 256 = 47kB` per context token. Equal at `1.16e9 / 47.1e3 = 24,600` tokens. In bf16 the state traffic halves and the break-even is 12,300 tokens. A DeepSeek-V3 sequence reads `61 * 576 = 35kB` per context token, so 1.16GB is what V3 reads at 33k tokens: a Qwen3.8 sequence costs as much state traffic per step as a 33k-token DeepSeek-V3 sequence, whether it is 1k or 1M tokens long.

{% enddetails %}

**Question 6 [V4 versus its predecessors]:** Using the formula in the token-compression section, compute DeepSeek-V4-Pro's bytes read per decoded token at 128k and 1M, and compare with V3.2 (DSA) and V3 (full MLA) at 1M.

{% details Click here for the answer. %}

At 128k: CSA layers read `30 * (32768 * 64 + 1024 * 576 + 128 * 576) = 30 * (2.1MB + 0.59MB + 0.07MB) = 83MB`; HCA layers read `31 * (1024 * 576 + 128 * 576) = 20MB`; total **0.10GB**. At 1M: CSA `30 * (16.8MB + 0.66MB) = 524MB`, HCA `31 * (8192 + 128) * 576 = 149MB`, total **0.67GB**. V3.2 at 1M reads 8.3GB (12x more) and V3 reads `61 * 1048576 * 576 = 37GB` (55x more). The sliding-window branch, present in all 61 layers, is `61 * 128 * 576 = 4.5MB` of fixed state per sequence, negligible.

{% enddetails %}

**Question 7 [attention crossover]:** Derive the crossover $S^*$ in the table above for Kimi K3 (18 active experts of width 3072 on a 3584-wide latent, 96 MLA heads with $d_{qk} + d_v = 320$, 24 full-attention layers of 93). Then work out what happens to $S^*$ if K3 had used MLA in every layer.

{% details Click here for the answer. %}

MLP FLOPs per token per layer: the 16 routed experts operate on the 3584-wide latent (`16 * 3 * 3584 * 3072 = 528M` parameters), the two shared experts run at full width (`2 * 3 * 7168 * 3072 = 132M`), and the latent down- and up-projections add `2 * 7168 * 3584 = 51M`, for 712M active MLP parameters or `1.42G` FLOPs. Attention per context token per full layer: `96 * 320 = 30,720`. Then `S* = 1.42e9 * 93 / (30720 * 24) = 180k` tokens. With MLA in all 93 layers, $L / L_\text{full} = 1$ and `S* = 46k`. The 3:1 hybrid buys 3.9x in the training crossover, on top of the decode savings.

{% enddetails %}

**Question 8 [serving a million tokens, open-ended]:** Kimi K3 serves 1M-token contexts. Using its cache and state sizes, how many concurrent 1M-token sequences fit on a 64-chip TPU7x group after the weights (1.42TB with fp4 experts)? What is the lower bound on step time at that batch? Now consider a coding agent that keeps a 400k-token repository in context and makes 200 tool calls, each appending 2k tokens. How much prefix-cache memory does one such session pin, and for how long? What does this say about where the cache should live between turns?

<h3 markdown=1 class="next-section">That's all for Section 14. Section 15, on how training changed (fp8 and fp4, Muon, multi-token prediction and reinforcement learning), is [here](../training-2026).</h3>
