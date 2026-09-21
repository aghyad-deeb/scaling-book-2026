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
    subsections:
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

## Why Attention Is Back on the Critical Path

In [Section 7](https://jax-ml.github.io/scaling-book/inference) we found that a generate step takes at least (batch times KV cache bytes plus parameter bytes) divided by memory bandwidth, and in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) we saw that for LLaMA 3-70B the KV term dominates at any context beyond a few thousand tokens. So why does attention now need a whole section of its own, when [Section 4](https://jax-ml.github.io/scaling-book/transformers) told us the MLP is where the FLOPs are? There are basically three things that changed, so let's go through them.

**Contexts got a lot longer:** when Section 8 was written LLaMA 3 had an 8k context (later extended to 128k). Kimi K3, DeepSeek-V4 and Qwen3.8 all serve a million tokens now, and if you look at an agentic workload (say, a coding agent reading a repository, or a research agent making a hundred tool calls) you'll typically find 100k to 500k tokens of context. At 128k tokens, LLaMA 3-70B's 164kB-per-token cache works out to 21GB *per sequence*.<d-footnote>Section 8 wrote this as 160kB, but it's actually <code>2 * 8 * 128 * 80 = 163,840</code> bytes, which we round to 164kB here.</d-footnote> Three of those would nearly fill an H100 (64GB), leaving us very little room for the weights!

**Longer outputs:** A chat reply is a few hundred tokens, but a reasoning trace is 10k to 100k. Decode has always been the expensive half of serving per token, so if most of our tokens are now decoded tokens rather than prefilled ones, that's where we should expect the cost to go. In [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) we found that you need three prefill servers per decode server for 8k-in, 512-out traffic. What happens with 8k-in, 16k-out? The same calculation (with $P$ prefill and $G$ decode servers and Section 8's per-server throughputs) gives `P / 0.91 = 32 * G / (0.019 * 16384)`, or about 11 decode servers per prefill server, so the decode roofline is the one we care about.

**Sparse MLPs:** In [Section 13](../moe) we saw that DeepSeek-V3's attention is 30% of its active parameters, against 17% for LLaMA 3-70B, since the sparse MLP each token sees is narrow while attention is just as big as it would be in a dense model (making the MLP sparse does nothing to the attention FLOPs). In prefill and training, attention FLOPs cross MLP FLOPs at about 19k tokens for DeepSeek-V3, where a dense model of the same width with $F = 4D$ would cross at 86k by the same formula (the $8D$ rule of thumb from [Section 4](https://jax-ml.github.io/scaling-book/transformers) counts the FLOPs a bit differently, but we'll come back to that below).

So attention is expensive in bytes at decode, expensive in FLOPs at long context, and there's a lot more long context around than there used to be. Let's look at what we can do about it, and what the recent models actually do.

## A Menu of Attention Mechanisms

Let's start with a table, the attention analog of the model table in [Section 13](../moe). Grouped-query attention (GQA, which [Section 4](https://jax-ml.github.io/scaling-book/transformers) called GMQA) is our baseline. The two numbers we care about are the *stored* cache per token (which sets how many sequences we can fit in HBM) and the bytes *read* per decoded token at a given context (which sets our step time). These differ once a mechanism reads less than it stores (sparse attention) or keeps a fixed-size state (windows, linear layers). We assume one byte per cached element and quote the read figure at 128k context, including one read and one write of any recurrent state.<d-footnote>Every figure here is computed from the model's <code>config.json</code> with the formulas in this section (the script is included with the chapter). Recurrent states are fp32 where the shipped config says so (e.g. Qwen's <code>mamba_ssm_dtype</code>) and bf16 otherwise. DeepSeek-V4's stored size is derived from its config and the storage precisions in its report. We don't derive a read figure for V4.1-Flash (its layer-sharing scheme is described below).</d-footnote>

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
| DeepSeek-V4.1-Flash | CSA2: KV shared across layers (4 source layers of 40) + fp4 main cache | 0.9kB | — | — |

Reading down the table, you'll notice the stored cache shrinks 30x and the bytes read per token at 128k shrink 200x: LLaMA 3-70B stores 164kB per token and reads 21.5GB per decoded token at 128k, while DeepSeek-V4-Pro stores 5kB and reads 0.1GB. That's a huge change! There are five distinct ideas hiding in this table (most 2026 models combine at least two of them), so let's take them one at a time.

{% include figure.liquid path="assets/img/kv-bytes-vs-context.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> bytes read from HBM per decoded token, per sequence, as a function of context length, for one representative of each attention family. Note the log scales. GQA and MLA are straight lines (proportional to context); sparse attention, sliding windows and recurrent state all flatten the curve, and DeepSeek-V4 flattens it twice." %}

## Multi-Head Latent Attention

Let's start with multi-head latent attention (MLA), which first appeared in DeepSeek-V2<d-cite key="DeepSeek2"></d-cite> and which DeepSeek-V3, Kimi K2 and K3, GLM-5 and Mistral Large 3 all use. The basic idea is to cache a low-rank *latent* of the keys and values rather than the keys and values themselves.

You'll remember that in standard attention each layer projects the residual $x \in \mathbb{R}^D$ to $N$ query heads, $K$ key heads and $K$ value heads of dimension $H$, and caches the keys and values, i.e. $2KH$ elements per token per layer. MLA instead projects $x$ down to a single latent $c \in \mathbb{R}^{d_c}$ with $d_c = 512$, plus a small *decoupled* key $k_R \in \mathbb{R}^{d_R}$ with $d_R = 64$ that carries the rotary position information.<d-footnote>We have to apply rotary position embeddings (RoPE) to the key before dotting it with the query, but we can't reconstruct a rotated key from a shared latent with a per-head linear map. MLA's answer is to split each key into a "no position" part (reconstructed from the latent) and a small "rope" part that is computed and cached directly and shared across heads. This is why you'll see <code>qk_nope_head_dim</code> and <code>qk_rope_head_dim</code> in the configs.</d-footnote> Each head then reconstructs its own key and value from the latent with per-head matrices $W_{UK}^{(h)} \in \mathbb{R}^{d_c \times H}$ and $W_{UV}^{(h)}$. So our cache per token per layer is $d_c + d_R = 576$ elements. For DeepSeek-V3 with 61 layers that's 35,136 elements, or **35kB in fp8**, against 164kB for LLaMA 3-70B and 258kB for 405B. Not bad!

**Isn't this just GQA with fewer heads?** You might notice that 576 is about what GQA with 2.25 KV heads of dimension 128 would cost. But those 2.25 heads would be shared by all 128 query heads, whereas MLA reconstructs 128 *different* keys and values from the latent. So we get something as expressive as MHA and as cheap in bytes as MQA, and (as we'll see) we pay for it in FLOPs.

### Parameters and cache size

Let's count the per-layer parameters for DeepSeek-V3's MLA ($D = 7168$, 128 heads, $d_c = 512$, $d_R = 64$, query latent $d_q = 1536$, per-head dimensions 128 nope, 64 rope, 128 value):

| matrix | shape | params |
| :----- | :---- | -----: |
| $W_{DQ}$ (residual to query latent) | 7168 x 1536 | 11.0M |
| $W_{UQ}$ (query latent to 128 heads x 192) | 1536 x 24576 | 37.7M |
| $W_{DKV}$ (residual to KV latent + rope key) | 7168 x 576 | 4.1M |
| $W_{UK}$ (latent to 128 heads x 128) | 512 x 16384 | 8.4M |
| $W_{UV}$ (latent to 128 heads x 128) | 512 x 16384 | 8.4M |
| $W_O$ (128 heads x 128 to residual) | 16384 x 7168 | 117.4M |
| **Total** | | **187M** |

You'll notice two things here. First, the output projection $W_O$ is 63% of attention's parameters, since 128 heads of dimension 128 give us a 16,384-wide concatenated output, more than twice the residual width. Second, the matrices that actually touch the latent are tiny. So, perhaps surprisingly, MLA doesn't buy us anything in attention parameters (LLaMA 3-70B's GQA layer is 151M parameters and DeepSeek-V3's is 187M). All of the savings are in the cache, which is fine, since bytes read at decode were what we were worried about in the first place.

### The absorption trick changes the roofline

**How do we actually use this cache at decode time?** The naive way is to reconstruct every head's keys and values from the latent for every past token, which costs `2 * 512 * 128 * (128 + 128) = 33.5M` FLOPs per past token per layer (we have to push the latent through $W_{UK}$ and $W_{UV}$ for all 128 heads) and, worse, materializes $S \cdot 128 \cdot 256$ elements of keys and values that the attention kernel then has to read. That would throw away the whole benefit! Instead, we *absorb* the up-projections into the query and the output:

$$q_h^\top k_{h,s} = q_h^\top W_{UK}^{(h)\top} c_s = \tilde{q}_h^\top c_s \qquad \text{where} \quad \tilde{q}_h = W_{UK}^{(h)} q_h \in \mathbb{R}^{512}$$

So we map each head's query once into the 512-dimensional latent space, compute scores against the shared latent $c_s$ (plus the 64-dimensional rope term), take the softmax-weighted sum over the latents to get a 512-dimensional output per head, and apply $W_{UV}$ afterwards. Notice that the attention kernel never sees a per-head key or value at all: it sees one 576-wide "key" and one 512-wide "value" per token, shared by all 128 heads, which is basically MQA with an unusually wide head.

Let's count FLOPs. Per layer, per decoded token, per context token, the absorbed attention does

$$2 \cdot N \cdot (d_c + d_R) + 2 \cdot N \cdot d_c = 2 \cdot 128 \cdot (576 + 512) = 278{,}528 \text{ FLOPs}$$

and reads $576$ bytes (fp8). That's an arithmetic intensity of **484 FLOPs/byte with an fp8 cache, or 242 with bf16**. How does that compare to GQA? LLaMA 3-70B does $2 \cdot 64 \cdot 128 \cdot 2 = 32{,}768$ FLOPs per context token and reads $2 \cdot 8 \cdot 128 = 2048$ bytes, an intensity of 16 (int8) or 8 (bf16), which is just the group size $G = N/K$, exactly as Question 4 of [Section 4](https://jax-ml.github.io/scaling-book/transformers) told us.

You'll remember the accelerator intensities from [Section 12](https://jax-ml.github.io/scaling-book/gpus), [Section 2](https://jax-ml.github.io/scaling-book/tpus) and the hardware table in the [2026 outline](..) (i.e. bf16 FLOPs over HBM bandwidth): 296 for an H800 or H100, 206 for an H200, 281 for a B200, 312 for a GB200, and 311 for TPU7x, 240 for v5e and 164 for v5p. GQA attention at intensity 8 is hopelessly memory-bound on all of them, so [Section 7](https://jax-ml.github.io/scaling-book/inference) was quite happy to treat attention as always bandwidth-bound and move on. MLA attention at 484 FLOPs/byte with an fp8 cache is above the roofline on every one of them. With a bf16 cache (242) it's above the H200 and the two older TPUs, but 14 to 22% below the H800, B200, GB200 and TPU7x. **So with MLA and an fp8 cache, decode attention is compute-bound at long context.**

Put another way, DeepSeek-V3 does 8.5x the attention FLOPs of LLaMA 3-70B per context token but reads 3.6x fewer bytes. Since bytes were what was binding us, that's a trade worth making.

**Note [precision]:** there's one assumption buried in that 484 figure, which is that we're pairing an fp8 cache with bf16 score and value matmuls. DeepSeek's V3 write-up tells us the attention core runs in bf16 but doesn't tell us how the cache is stored, so we'll assume fp8 here (as [Section 16](../applied-frontier) does), which seems safe: DeepSeek's FlashMLA kernels support it, and DeepSeek-V4 says it uses an fp8 cache with bf16 RoPE dimensions.<d-footnote>If the attention matmuls themselves ran in fp8, the accelerator intensities would double (591 for H800/H100, 412 for H200, 562 for B200, 625 for GB200, 622 for TPU7x) and 484 would be memory-bound on the H800, B200, GB200 and TPU7x, and compute-bound only on the H200. The rest of this section assumes bf16 matmuls on an fp8 cache.</d-footnote>

**Why does this matter in practice?** You can see one consequence in DeepSeek's serving layout ([Section 16](../applied-frontier)): attention runs *data-parallel* across all 144 decode GPUs, with each GPU handling the attention for its own sequences, rather than tensor-parallel across heads. Why not shard over heads? Tensor parallelism over heads would require every shard to read the full shared latent for every sequence (the latent isn't sharded by head), so per GPU we'd do the same FLOPs as with data parallelism but $Y$ times the bytes. That divides our arithmetic intensity by $Y$ and pushes attention back toward memory-bound. Data parallelism keeps each sequence's latent on one GPU, so the intensity stays at 484 and the attention FLOPs available to us grow with the number of GPUs.

This also means we can treat the number of heads as a hardware knob. Since the intensity of absorbed MLA is proportional to $N$, we can tune the head count to the FLOPs-to-bandwidth ratio of whatever chip we serve on. The GLM-5 report<d-cite key="glm5"></d-cite> says as much: "the number of attention heads in DeepSeek-V3 is selected according to the roofline of H800", which "is inappropriate for other hardware", so GLM-5 raised the head dimension from 192 to 256 and cut the head count by a third relative to its own baseline, landing at 64 heads. Kimi K2 also went with 64 heads, and its report tells us what the trade was: 128 heads would have cost 83% more inference FLOPs for a 0.5 to 1.2% gain in loss.<d-cite key="kimik2"></d-cite>

Let's put in some real numbers. For DeepSeek-V3 at 128k context, one decoded token's attention costs `61 * 278528 * 131072 = 2.2e12` FLOPs and reads `61 * 131072 * 576 = 4.6GB`. At batch 64 on a B200 that's `64 * 2.2e12 / 2.25e15 = 63ms` of FLOPs against `64 * 4.6e9 / 8e12 = 37ms` of bytes. That's 63ms per step for attention alone, before we've done any MoE work at all. As we'll see shortly, this is exactly the problem DeepSeek Sparse Attention is there to fix.

<p markdown=1 class="takeaway">**Takeaway:** MLA caches a shared 576-element latent per token per layer instead of $2KH$ keys and values, which cuts DeepSeek-V3's cache to 35kB per token. With the absorption trick, our decode kernel behaves like MQA with a 576-wide head shared by all 128 query heads, giving an arithmetic intensity of 240 to 480 FLOPs/byte. With an fp8 cache our attention is compute-bound (rather than memory-bound) at long context on every current GPU, so we scale it by adding GPUs in data parallel.</p>

## Sliding Windows and Local-Global Interleaving

Let's turn to the oldest trick, which is simply not to attend to everything. A *sliding window* layer attends only to the previous $w$ tokens, so its cache holds at most $w$ entries no matter how long the context gets. Models interleave a few *global* layers that see everything with many local ones:

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

Once $S \gg w$ only the global layers count, and we save a factor of $L / L_\text{global}$: 6x for Gemma, 4x for Llama 4, 2x for gpt-oss. Gemma 4 stacks two more tricks on top of its global layers: only 2 KV heads (against 8 in the local layers), and *keys reused as values* (`attention_k_eq_v` in the config), so the global cache is $1 \cdot 2 \cdot 512 = 1024$ bytes per layer per token, or 5kB per token in total. That's 8x smaller than Gemma 3 27B at long context, in a model of about the same size. Not bad!

**What goes wrong with windows?** Two things, it turns out, and both need a fix. The first is the attention *sink*: when the earliest tokens fall out of the window, attention has nowhere to dump the probability mass it used to park on them, and quality collapses<d-cite key="streamingllm"></d-cite>. gpt-oss adds a learned per-head bias to the softmax denominator (a slot that attends to nothing) so the excess mass has somewhere harmless to go, and DeepSeek-V4 has "learnable attention-sink logits" for the same reason. The second problem is *positions*. Llama 4's global layers use no positional encoding at all (NoPE, and Llama 4 calls the interleaving of RoPE and NoPE layers "iRoPE"), Gemma 4's use a low-frequency variant, and Muse Glimmer copies Llama 4's pattern. The idea is that the local layers carry the positional signal, and the global layers only have to retrieve. This is what makes the 1M and 10M context claims possible without retraining the position embedding.

What about FLOPs? A local layer costs $2 N H \cdot 2 w$ per token, which is a constant. Only the global layers have the quadratic term, so the attention-versus-MLP crossover of [Section 4](https://jax-ml.github.io/scaling-book/transformers) moves out by $L / L_\text{global}$: for Gemma 3 27B it's over 500k tokens.

<p markdown=1 class="takeaway">**Takeaway:** interleaving local and global layers divides the growing part of our cache by $L / L_\text{global}$ and leaves a fixed $L_\text{local} \cdot w \cdot 2KH$ bytes per sequence. With a 1024-token window and one global layer in six, plus K=V and 2 heads in the global layers, Gemma 4 gets down to 5kB per token. Sinks and position-free global layers are what keep it stable.</p>

## Sparse Attention: Index, Then Attend

Sliding windows pick what we attend to by position. What if we picked by content instead? That's the idea of *sparse attention*: a cheap *indexer* scores every past token, we select the top $k_a$, and full attention runs over only those. The version we'll work through is DeepSeek Sparse Attention (DSA), which DeepSeek shipped in V3.2<d-cite key="deepseekv32"></d-cite> (it grew out of a research version called Native Sparse Attention<d-cite key="nsa"></d-cite>), and GLM-5 and MiniMax M3 picked up variants of it within months.

DSA sits on top of MLA, so everything we worked out in the previous section carries over. Its *lightning indexer* has $H_I = 64$ small heads of dimension $d_I = 128$, computes one 128-dimensional indexer key per token (shared across the 64 heads), and scores past token $s$ for query $t$ as

$$I_{t,s} = \sum_{j=1}^{64} w_{t,j} \cdot \text{ReLU}(q^I_{t,j} \cdot k^I_s)$$

in fp8, with a ReLU rather than a softmax so that nothing has to be normalized across $S$. We then select the top $k_a = 2048$ latents by score and run the absorbed MLA attention from above over just those 2048. Per layer, per decoded token, this costs us

| | FLOPs | bytes (fp8) |
| :-- | ----: | ----------: |
| Indexer, per context token | $2 \cdot 64 \cdot 128 = 16{,}384$ | 128 |
| Main attention, fixed | $278{,}528 \cdot 2048 = 570\text{M}$ | $2048 \cdot 576 = 1.2\text{MB}$ |

So the indexer is 17x cheaper per context token in FLOPs than full MLA attention and reads 4.5x fewer bytes, and the expensive attention no longer grows with context at all. At 128k context DeepSeek-V3.2 reads 1.1GB and does 0.17 TFLOPs per decoded token, against 4.6GB and 2.2 TFLOPs for V3, i.e. 4x fewer bytes and 13x fewer FLOPs. On a B200 at batch 64 our attention step drops from 63ms (compute-bound) to about 9ms (now memory-bound on the indexer keys). You can actually see this in DeepSeek's published cost curves, which show decode cost per token nearly flat in context length (and, less scientifically, in the fact that DeepSeek halved its API price on the day V3.2-Exp, the experimental release that preceded V3.2, shipped).

**Where does the remaining cost live now?** Almost entirely in the indexer: at 1M context it is 99% of the bytes and 97% of the FLOPs of V3.2's attention. **How do we shrink the indexer?** There are two things we could do, and both show up in shipped models. We could compute the index in fewer layers: GLM-5.2 does it in only 21 of its 78 layers and reuses the selected positions in the other 57 (`indexer_types` and `index_topk_freq` in its config), which cuts its reads at 128k to 0.44GB. Or we could make each indexer key smaller: DeepSeek-V4 compresses the indexer keys 4:1 and stores them in fp4, which we'll come to below.

**How expensive is DSA to train?** Less than you might expect, since V3.2 was made from V3.1 by continued pretraining: 2.1B tokens with everything frozen except the indexer (trained to match the real attention distribution, i.e. a KL loss against the head-summed softmax), then 944B tokens with sparse attention turned on. That's 6% of the original pretraining budget for a 4x cut in decode bytes, which is a pretty good deal.

**Note [TPUs]:** top-$k$ selection over $S$ scores followed by a gather of 2048 latents at data-dependent addresses is quite natural on a GPU (a radix-select kernel and a scattered read). On a TPU it's the same kind of dynamic-shape, gather-heavy operation that makes MoE routing awkward, and it needs a custom kernel. None of this means it can't be done on a TPU, but it does help explain why we saw sparse attention on GPUs first.

<p markdown=1 class="takeaway">**Takeaway:** sparse attention replaces the 576-byte-per-context-token attention read with a 128-byte indexer read, plus a fixed attention over 2048 selected entries. For DeepSeek-V3.2 that gives us 4x fewer bytes and 13x fewer FLOPs than full MLA at 128k. The indexer becomes our cost at very long context, so newer models share it across layers or compress its keys.</p>

## Linear Attention Hybrids: State Instead of Cache

The most radical option we have is to get rid of the growing cache in most layers entirely. A *linear attention* or state-space layer maintains a fixed-size matrix state per head, updated once per token, instead of a list of past keys and values. Qwen3-Next<d-cite key="qwen3next"></d-cite>, Qwen3.5 and Qwen3.8 use Gated DeltaNet<d-cite key="gateddeltanet"></d-cite>, Kimi Linear<d-cite key="kimilinear"></d-cite> and Kimi K3 use a variant with a per-channel decay called Kimi Delta Attention (KDA), and Nemotron 3 uses Mamba-2<d-cite key="mamba2"></d-cite>. All of them interleave a minority of ordinary attention layers, since pure linear attention can't do exact retrieval.

The delta-rule recurrence, per head with key dimension $d_k$ and value dimension $d_v$, is

$$\Sigma_t = \Sigma_{t-1} \left( \gamma_t I - \beta_t k_t k_t^\top \right) + \beta_t v_t k_t^\top, \qquad o_t = \Sigma_t q_t$$

where $\Sigma_t \in \mathbb{R}^{d_v \times d_k}$ is the state (we avoid $S$, since that's the context length throughout this book), $\gamma_t$ is a learned decay (the literature writes $\alpha_t$, but we keep $\alpha$ for arithmetic intensity as the rest of this book does) and $\beta_t$ is a learned write strength. You can think of it as an online least-squares update of an associative memory. What matters to us is that the state has $d_k d_v$ elements per head, is read and written once per decoded token, and doesn't depend on $S$ at all.

| Model | Linear : full | Linear heads x $d_k$ x $d_v$ | State per layer | Layers | State per sequence |
| :---- | :-----------: | :--------------------------: | --------------: | -----: | -----------------: |
| Qwen3.5-397B | 3 : 1 | 64 x 128 x 128 | 4.2MB (fp32) | 45 | 189MB |
| Qwen3.8-2.4T | 3 : 1 | 128 x 128 x 128 | 8.4MB (fp32) | 69 | 579MB |
| Kimi K3 | 3 : 1 | 96 x 128 x 128 | 3.1MB (bf16) | 69 | 217MB |
| Nemotron 3 Super | ~5 : 1 | 128 x 64 x 128 (Mamba-2) | 4.2MB (fp32) | 40 | 168MB |

You'll notice these states aren't small! Qwen3.8 reads and writes 1.16GB of state per sequence per decode step, at *any* context length. Its 23 full-attention layers (4 KV heads of 256) read 47kB per context token, so the state traffic equals the cache traffic at `1.16e9 / 47e3 = 24,600` tokens. Below that the fixed state dominates the hybrid's step, and above it the growing cache does. The comparison we actually care about is against an all-attention model of the same shape (92 GQA layers, 188kB per token): the hybrid reads less than that once `1.16e9 / (188e3 - 47e3) = 8,200` tokens are in context, and at 1M Qwen3.8 reads 50GB per token where the all-GQA model would read `92 * 2048 * 1048576 = 198GB`. For Kimi K3 the state stops dominating at about 31k tokens, and the hybrid beats an all-MLA 93-layer stack above about 11k.<d-footnote>Storing the state in bf16 halves these numbers. Qwen ships with an fp32 state (<code>mamba_ssm_dtype: float32</code>), presumably because the recurrence accumulates error, while Kimi doesn't say and we assume bf16. This is a real design knob: the state is written every step, so its precision is more delicate than that of a KV cache, which is written once.</d-footnote>

So a hybrid is basically a design that only pays off at long context. Note that the state is a per-sequence cost that batching doesn't amortize (each sequence has its own), so it behaves like a KV cache of fixed length rather than like weights. The FLOPs, on the other hand, are trivial: updating a $128 \times 128$ state per head costs about $8 d_k d_v$ FLOPs, or `8 * 128 * 128 * 128 = 16.8M` per layer for Qwen3.8, against `2 * 8.4MB = 16.8MB` of fp32 state read and written. That's an intensity of about 1 FLOP per byte (2 with a bf16 state), so the linear layers are memory-bound like GQA and unlike MLA.

**What about training?** There the recurrence is computed chunk-wise: within a chunk of 64 tokens everything is a matmul, and states are passed between chunks. FLOPs are $O(S d_k d_v)$ per head with no quadratic term, which is what lets Qwen3-Next claim over 10x the inference throughput of Qwen3-32B beyond 32k tokens of context. Context parallelism becomes a sequential pass of a few megabytes of state between GPUs rather than an AllGather of keys and values (Kimi K3's report describes a dedicated scheme for this).

**Should we expect every model to end up a hybrid, then?** Not necessarily. MiniMax is a nice example of how unsettled this is: MiniMax M1 shipped a 7:1 linear hybrid in mid-2025, MiniMax M2 went back to full attention four months later, and MiniMax M3 went to block-sparse attention instead<d-cite key="minimaxm3"></d-cite>. Qwen and Kimi use hybrids, DeepSeek uses sparsity plus compression, and Google uses windows. If you look back at the table at the start of this chapter, though, you'll notice that the 2025 and 2026 flagships have all landed within a factor of ten of each other, between about 5 and 50kB stored per token (DeepSeek-V4.1 is the outlier, below 1kB). So while nobody agrees on which mechanism to use, everyone seems to agree on roughly how many bytes per token they can afford.

<p markdown=1 class="takeaway">**Takeaway:** linear-attention layers replace a growing cache with a fixed state of $d_k d_v$ elements per head, read and written every step. That's hundreds of megabytes per sequence for 2026 models, so hybrids actually read *more* than an all-attention model of the same shape below roughly 8 to 11k tokens of context, and win by 3 to 4x at 1M. You can think of them as memory-bound, per-sequence, and cheap to train at long context.</p>

## Token Compression: DeepSeek-V4

Finally, let's look at DeepSeek-V4<d-cite key="deepseekv4"></d-cite>, which combines most of the above and adds one new idea: **compress the KV cache along the sequence before we store it.** Its attention has three components, all sharing one cache format:

* **A single shared KV entry per token** of 512 dimensions, the last 64 of which carry RoPE, used as both key and value (`num_key_value_heads: 1`, `head_dim: 512`, `qk_rope_head_dim: 64`). It's stored as 448 fp8 plus 64 bf16 elements, i.e. 576 bytes per entry, so it matches MLA's 576 in bytes through the mixed precision rather than in element count. You can think of this as MQA taken to its logical conclusion, with 128 query heads (Pro) each 512 wide, read from a 1536-dimensional query latent as in MLA, and each head doing `2 * (512 + 512) = 2048` FLOPs per selected entry.
* **Compressed Sparse Attention (CSA)**, in half the layers: every 4 consecutive tokens are compressed into one entry by a small learned module, an indexer (64 heads of 128, in fp4) scores the compressed entries, and attention runs over the top 1024 (Pro) or 512 (Flash) of them plus a 128-token uncompressed sliding window.
* **Heavily Compressed Attention (HCA)**, in the other half: every 128 tokens become one entry, and attention runs over all of them with no sparsity.

The layers alternate CSA, HCA, CSA, HCA. Per token, our stored cache is roughly $(576 + 64) / 4$ bytes for a CSA layer and $576 / 128$ for an HCA layer, so over 30 CSA and 31 HCA layers that comes to about **4.9kB per token**, which DeepSeek quotes as 10% of V3.2 (our estimate is 11.5%). The bytes *read* per decoded token at context $S$ are

$$30 \cdot \left[ \frac{S}{4} \cdot 64 + 1024 \cdot 576 + 128 \cdot 576 \right] + 31 \cdot \left[ \frac{S}{128} \cdot 576 + 128 \cdot 576 \right]$$

which works out to 0.1GB at 128k and 0.67GB at 1M. That's against 8.3GB for V3.2 and 37GB for V3 at 1M! The FLOPs fall similarly, to about 0.2 TFLOPs per token at 1M, 5x below V3.2 and 80x below V3. DeepSeek itself says that at 1M context V4-Pro needs 27% of V3.2's per-token FLOPs (for the whole model) and 10% of its cache, and that the cache is about 2% of what a bf16 GQA-8, head-128 model would need (about 4% of one stored in fp8, which is the convention we use elsewhere in this section).

**What happened to the indexer problem from the previous section?** Compression fixes that too! Indexer keys are stored per *compressed* entry (one per 4 tokens) in fp4 (64 bytes), so the indexer reads 16 bytes per original token per layer instead of 128. This means the term that was 99% of V3.2's bytes is now eight times smaller.

V4.1-Flash goes further still, in two ways. It shares the compressed KV entries and indexer keys across layers, so only 4 of its 40 layers produce a cache at all (`kv_source_layer_ids: [2, 8, 14, 20]`), and it stores that cache in fp4 (the 128-token window stays fp8). Together these get it to a published 890 bytes per token, roughly 2x from fp4 and 2x from the sharing, which is 184x smaller than LLaMA 3-70B's cache.<d-footnote>V4.1 is a causal encoder-decoder: the first 20 layers compress 2:1 and the last 20 run 1:1, with no HCA layers. Its 196B of Engram memory tables (hashed n-gram embeddings fetched from host memory) sit outside the 552B backbone.</d-footnote>

<p markdown=1 class="takeaway">**Takeaway:** DeepSeek-V4 stores one 576-byte entry per 4 tokens (sparse-selected) or per 128 tokens (dense), plus a 128-token raw window, for about 5kB per token stored and 0.7GB read per token at 1M context. Compressing along the sequence multiplies with every other trick we've seen, including shrinking the indexer that limited V3.2.</p>

## Rooflines: Decoding With a Modern Cache

Now let's return to [Section 7](https://jax-ml.github.io/scaling-book/inference)'s step-time formula, which needs one amendment. With MLA-class attention the FLOPs term of attention is no longer negligible, so we have

$$\begin{aligned} T_\text{step} \geq \max\Big( &\frac{B \cdot \text{bytes read}(S) + \text{weight bytes per GPU}}{W_\text{hbm}}, \\ &\frac{B \cdot \text{attention FLOPs}(S) + 2 B \cdot \text{active params}}{C} \Big) \end{aligned}$$

where $B$ is the per-GPU batch in sequences, $\text{bytes read}(S)$ comes from the table above, and the weight bytes per GPU are the model's weights divided by however many GPUs we spread them over (the MoE FLOPs term uses all the active parameters, since every one of the GPU's own tokens visits $k$ experts somewhere). Let's see what this does for a few models on a GB200 NVL72 rack (72 GPUs in one NVLink domain with 186GB each, see [Section 12](https://jax-ml.github.io/scaling-book/gpus)) with every weight, attention included, spread over 64 of them (a power of two, so the experts divide evenly), at 128k context:

| Model | Weights per GPU | Cache per sequence (stored) | Sequences per GPU |
| :---- | --------------: | --------------------------: | ----------------: |
| LLaMA 3 405B, int8 | 6.3GB | 33.8GB | 5 |
| DeepSeek-V3, fp8 | 10.5GB | 4.6GB | 38 |
| DeepSeek-V3.2, fp8 | 10.7GB | 5.6GB (with indexer keys) | 31 |
| Qwen3.8-2.4T, fp8 | 37.5GB | 6.8GB | 22 |
| Kimi K3, fp4 experts (as shipped) | 24.4GB | 2.0GB | 80 |

So with everything sharded we can only fit five 128k-token sequences per GPU for a dense GQA model from 2024 (the LLaMA 3 row), against 38 for an MLA model and 80 for a hybrid. That's a big difference! One thing to be careful about is that those rows assume the attention weights are sharded too. Under the data-parallel-attention layout of the previous section, the 17GB of attention, shared-expert, dense and embedding weights are replicated on every GPU ([Section 16](../applied-frontier) gets 27GB per GPU for V3), and the counts fall to `(186 - 27) / 4.6 = 34` for V3, 28 for V3.2 and about 54 for Kimi K3, whose 56GB of non-expert weights are the largest replicated block.

**What about the hardware DeepSeek actually serves on?** The same arithmetic is tighter there: under its 144-way expert parallelism (EP, [Section 13](../moe)) each H800 holds 22GB of weights ([Section 16](../applied-frontier)) and has about 58GB free, which is `58 / 4.6 = 12` full-length sequences at 128k, or 25 on an H200. The TPU picture is more or less the same: a 64-chip TPU7x cube at 192GB per chip lands within 4% of every row above, with one extra sequence for the MLA and GQA models and three for Kimi K3.

That's the memory side. Now let's look at the time side, for DeepSeek-V3 versus V3.2 at batch 32 per GPU on a GB200 (8e12 bytes/s, 2.5e15 FLOPs/s), with the weights spread over the rack as in the table above:

| | bytes per step | FLOPs per step | $T_\text{step}$ | tokens/s per GPU |
| :-- | -------------: | -------------: | --------------: | ----------------: |
| V3 (full MLA) | 32 x 4.6 + 10.5 = 158GB | 32 x 2.2e12 (attn) + 32 x 7.4e10 (MoE) = 7.3e13 | max(20ms, **29ms**) | 1,100 |
| V3.2 (DSA) | 32 x 1.1 + 10.7 = 46GB | 32 x 1.7e11 + 32 x 7.4e10 = 7.8e12 | max(**5.8ms**, 3.1ms) | 5,500 |

You'll notice that full MLA is *compute*-bound on attention at this context, which no model in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) ever was! DSA drops both terms and lands us memory-bound at about 5x the throughput. (If you've wondered why long-context API prices fell so far between 2025 and 2026 then a good part of the answer is sitting in this table.)

**What does this do to disaggregation?** In [Section 7](https://jax-ml.github.io/scaling-book/inference) we moved the KV cache from prefill to decode servers over the network and called the cost "typically acceptable". A 128k-token DeepSeek-V3 cache is 4.6GB, so at the 50GB/s one-way that each GPU's 400Gb/s InfiniBand NIC provides ([Section 12](https://jax-ml.github.io/scaling-book/gpus)) that's 92ms, i.e. a few decode steps. A LLaMA 3-405B cache at the same length is 34GB, or 0.7 seconds! So it's really the small caches that let us disaggregate prefill and decode at long context at all.

<p markdown=1 class="takeaway">**Takeaway:** we need to add an attention-FLOPs term to the [Section 7](https://jax-ml.github.io/scaling-book/inference) decode roofline. For MLA models at long context it's the binding term, and sparse attention removes it. Cache size sets how many sequences we fit per GPU (5 for LLaMA 3-405B, 38 for DeepSeek-V3 and 80 for Kimi K3 at 128k on a GB200 with everything sharded, 34 for V3 with attention replicated, and 12 for DeepSeek-V3 on the H800s it's actually served on) and, through the KV transfer, whether prefill-decode disaggregation is even feasible at long context.</p>

## Rooflines: Attention FLOPs at Long Context

You'll remember that [Section 4](https://jax-ml.github.io/scaling-book/transformers) derived that dot-product attention FLOPs exceed all other matmul FLOPs (QKVO projections included) once $T > 8D$, assuming non-causal MHA and $F = 4D$. Neither assumption holds anymore, and we'll also change the comparison a bit: causal attention (which halves the score FLOPs) against the MLP alone. Under Section 4's original assumptions this gives $S^\ast = 12D$ rather than $8D$, so LLaMA 3-70B appears at 86k in the table below rather than Section 4's 65k (that difference is purely a matter of accounting, since the architecture is the same). Let's write $d_{qk}$ and $d_v$ for the query-key and value head dimensions (both equal to $H$ in [Section 4](https://jax-ml.github.io/scaling-book/transformers), and 192 and 128 for DeepSeek's MLA), and as in [Section 13](../moe) let $k$ be the active experts and $E_s$ the shared ones. Then per token per layer with causal attention, the score and value matmuls cost $N (d_{qk} + d_v) S$ FLOPs (the factor of 2 for multiply-add and the factor of $1/2$ for causality cancel) in the layers that have full attention, and the MLP costs $6 (k + E_s) D F$. Setting them equal over the whole model, we get:

$$S^\ast = \frac{6 (k + E_s) D F}{N (d_{qk} + d_v)} \cdot \frac{L}{L_\text{full}}$$

| Model | $N$ | $d_{qk} + d_v$ | MLP per token per layer | $L_\text{full} / L$ | $S^\ast$ |
| :---- | --: | -------------: | ----------------------: | ------------------: | ----: |
| LLaMA 3 70B | 64 | 256 | 6 x 8192 x 28672 | 1 | 86k |
| LLaMA 3 405B | 128 | 256 | 6 x 16384 x 53248 | 1 | 160k |
| DeepSeek-V3 | 128 | 320 | 6 x 9 x 7168 x 2048 | 1 | **19k** |
| Kimi K2 | 64 | 320 | 6 x 9 x 7168 x 2048 | 1 | 39k |
| GLM-5 (dense phase)<d-footnote>GLM-5 trains with its DSA indexer (32 heads of 128, top-2048) in every layer after the dense phase, so its quadratic term drops to $4096 S$ per token per layer and $S^\ast$ moves to about 166k.</d-footnote> | 64 | 512 | 6 x 9 x 6144 x 2048 | 1 | 21k |
| Qwen3.8-2.4T | 64 | 512 | 6 x 11 x 8192 x 2048 | 23/92 | 135k |
| Kimi K3 | 96 | 320 | 2 x (16 x 3 x 3584 x 3072 + 2 x 3 x 7168 x 3072 + 2 x 7168 x 3584) | 24/93 | 180k |
| Gemma 3 27B | 32 | 256 | 6 x 5376 x 21504 | 10/62 | 525k |

**What does this table tell us?** Look at the DeepSeek-V3 row first: going from dense LLaMA 3 to a fine-grained MoE with 128 heads pulls the crossover in from 86k to 19k tokens, so once our MLP is sparse, attention is the expensive part of a 32k-token training sequence. **How do the 2026 models deal with this?** You can read it off the other rows, since each of them pulls one lever in the formula. If we halve the number of heads (Kimi K2), $N$ halves and we get 2x. If we make three quarters of the layers linear (Qwen3.8, Kimi K3), $L / L_\text{full}$ gives us another 4x on top of that, and sliding windows in five layers of six (Gemma) give us 6x the same way. Sparse attention is a different kind of fix: with DSA the quadratic term is the indexer at $H_I d_I S = 8192 S$ per token per layer, which gives $S^* \approx 97\text{k}$ for V3.2, and the main attention becomes a constant.

Context parallelism (see [Section 5](https://jax-ml.github.io/scaling-book/training)'s note on it) is what we reach for when even that isn't enough, or when a single sequence's activations don't fit. For attention layers it's cheap: ring attention<d-cite key="ringattention"></d-cite> passes each GPU's KV shard around the ring, `S * 576` bytes per layer for MLA, which is a rounding error next to the $S^2$ FLOPs it enables. For linear layers it's a sequential hand-off of the state between chunks. Most 2026 reports mention some form of it. Kimi K3 trained at 64k tokens and extended to 1M in its cooldown phase (the low-learning-rate tail of pretraining), while DeepSeek-V4 trained at 64k and reached 1M with the YaRN rescaling we describe in the next section.

<p markdown=1 class="takeaway">**Takeaway:** the attention-versus-MLP crossover is $S^\ast = 6(k + E_s) D F L / (N (d_{qk} + d_v) L_\text{full})$. Fine-grained MoE with many heads pulled it down to 19k tokens (DeepSeek-V3), and hybrids, windows and sparsity pushed it back past 100k. This, as much as the cache, is what makes it affordable for us to train 2026 models at long context.</p>

## Positions and Context Extension

Let's briefly answer a question that comes up constantly: how can a model trained at 4k to 64k tokens serve a million?

Almost every model still uses RoPE, and extends it with YaRN<d-cite key="yarn"></d-cite>, which rescales the rotary frequencies so that positions beyond the training length map into the trained range. The config tells us the recipe:

| Model | Trained at | Extended to | Method |
| :---- | ---------: | ----------: | :----- |
| DeepSeek-V3 | 4k, then 32k, then 128k | 128k | YaRN factor 40 from 4k, two 1000-step phases |
| gpt-oss-120b | 4k | 128k | YaRN factor 32 |
| DeepSeek-V4 | 4k, 16k, 64k, 1M | 1M | YaRN factor 16 from 64k |
| Qwen3.8 | 262k native | 1M | YaRN factor 4 |
| Kimi K3 | 8k, 64k; cooldown 256k to 1M | 1M | none: MLA layers have no positional encoding |
| Llama 4 Scout | | 10M (claimed) | NoPE global layers; chunked local layers |

**Can we shrink the part that needs extending?** Two tricks do exactly that. With *partial RoPE* we only rotate a slice of each head (64 of 192 query dimensions in MLA, 64 of 256 in Qwen3.5 and 3.8, and 64 of 512 in DeepSeek-V4). With *NoPE* we remove positions from the global layers entirely (Llama 4, Muse Glimmer, Kimi K3), the idea being that the local or recurrent layers carry position and the global layers only need content. Kimi K3 takes this the furthest: its linear layers encode position through their decay, its MLA layers have none, and so (as its report puts it) it "extrapolates directly to 1M-token contexts without any positional-encoding modification."<d-cite key="kimik3"></d-cite>

**Why should we care about this as systems people?** Mostly because of what it means for the training schedule: only a few percent of our tokens ever get spent at the final length. DeepSeek-V3, for example, spent 2,000 of roughly 240,000 training steps on context extension, and V3.2's sparse-attention conversion was 6% of pretraining. So when we're budgeting a training run, we can think of the long-context phase as a short one that we tack on at the end, and (as we've seen throughout this section) a lot of the architectural choices above are made so that we can get away with doing it that way.

## What Should You Take Away from this Section?

* Our stored cache per token, in fp8, depends on the mechanism. GQA-8 with 128-dim heads is $2048 L$ bytes (164kB for LLaMA 3-70B's 80 layers) and MLA is $576 L$ (35kB for 61 layers). Interleaved windows divide the growing part by $L / L_\text{global}$, hybrids keep it only in the full-attention layers plus a fixed state of $d_k d_v$ per head, and DeepSeek-V4's compression gets down to about 5kB (V4.1's fp4 cache to 0.9kB).

* The bytes we *read* per decoded token can be far below the bytes we store. Sparse attention reads 128 bytes of indexer key per context token plus a fixed 2048 entries, which cuts DeepSeek-V3.2's reads 4x at 128k relative to V3 and holds them nearly flat in context.

* MLA's absorbed decode has an arithmetic intensity of 240 to 480 FLOPs/byte, so our attention is now compute-bound at long context (which was never the case with GQA). Sparse attention cuts those FLOPs 13x. Data-parallel attention across the expert-parallel (EP) group is the layout that follows from this.

* Linear-attention states are hundreds of megabytes per sequence and we read and write them every step, so hybrids only beat full attention past about 8 to 11k tokens, and win 3 to 4x at 1M.

* The attention-versus-MLP FLOPs crossover we derived is $6 (k + E_s) D F L / (N (d_{qk} + d_v) L_\text{full})$, which comes to 19k tokens for DeepSeek-V3 and over 100k for the 2026 hybrids.

* We should think of long context as a short training phase (enabled by YaRN, partial RoPE, and position-free global layers), since only a few percent of a pretraining run is spent at the final length.

## Worked Problems

**Question 1 [cache sizes]:** Say we want to compare two different cache designs. GLM-5.2 has 78 layers of MLA (latent 512 + 64) with a DSA indexer (128-dimensional keys) computed in 21 layers and reused in the rest. Kimi K3 has 24 MLA layers and 69 KDA layers with 96 heads of $128 \times 128$ state in bf16. For each, what is the stored cache per token, the fixed per-sequence state, and the bytes read per decoded token at 128k context? *Assume one byte per element, and count one read and one write of recurrent state.*

{% details Click here for the answer, once you've thought about it! %}

**GLM-5.2:** Let's start with the stored cache per token, which is `78 * 576 + 21 * 128 = 44.9kB + 2.7kB = 47.6kB`, with no fixed state. For reads at 128k, the 21 indexers read `21 * 131072 * 128 = 352MB` and all 78 layers read 2048 selected latents, `78 * 2048 * 576 = 92MB`, for **444MB** in total. Storing a 128k sequence costs us `47.6e3 * 131072 = 6.2GB`, of which only 7% is touched per step!

**Kimi K3:** Here the stored cache per token is `24 * 576 = 13.8kB` (only the MLA layers grow). The KDA state is `69 * 96 * 128 * 128 * 2 bytes = 217MB` per sequence. At 128k we read `24 * 131072 * 576 = 1.81GB` of latents plus `2 * 217MB = 434MB` of state traffic, or **2.25GB**. At 8k the same calculation gives 113MB + 434MB = 547MB, of which 80% is the state, so at short context the hybrid's fixed cost dominates.

{% enddetails %}

**Question 2 [Gemma 3 versus Gemma 4]:** Let's say we want to compare the two Gemma generations. Gemma 3 27B has 62 layers, 16 KV heads of 128, with 10 global layers and 52 local layers of window 1024. Gemma 4 26B-A4B has 30 layers, 5 global layers with 2 KV heads of 512 that reuse keys as values, and 25 local layers with 8 KV heads of 256 and window 1024. How many bytes does each read per decoded token at 8k and 256k context, and why is the ratio what it is?

{% details Click here for the answer. %}

Let's start with Gemma 3. Its global layers read `10 * 2 * 16 * 128 = 41kB` per context token, and its local layers hold a fixed `52 * 1024 * 4096 = 218MB`. At 8k that's `41e3 * 8192 + 218e6 = 554MB`, and at 256k it's `10.7GB + 0.2GB = 10.9GB`.

Gemma 4's global layers read `5 * 1 * 2 * 512 = 5.1kB` per context token (the factor of 1 rather than 2 is the K=V trick) and the local layers hold `25 * 1024 * 4096 = 105MB`. At 8k that's `42MB + 105MB = 147MB`, and at 256k it's `1.34GB + 0.1GB = 1.44GB`.

So the long-context ratio is 7.6x. Where does it come from? Per global layer, Gemma 4 stores 1024 bytes per token against Gemma 3's 4096 (2 heads instead of 16 is 8x fewer, 512 dims instead of 128 is 4x more, and K=V is 2x fewer), and it has half as many global layers. At 8k the fixed local caches are a large fraction of the total for both models, so the ratio is only 3.8x.

{% enddetails %}

**Question 3 [MLA intensity]:** Kimi K2 and GLM-5 use MLA with 64 query heads rather than DeepSeek-V3's 128. What is the arithmetic intensity of their absorbed decode attention with an fp8 cache and with a bf16 cache? On which of H800/H100, H200, B200, GB200, TPU v5e, v5p and TPU7x would we be compute-bound?

{% details Click here for the answer. %}

Per context token per layer, we do `2 * 64 * (576 + 512) = 139,264` FLOPs and read 576 bytes (fp8) or 1152 (bf16), so our intensity is **242 (fp8)** or **121 (bf16)**. Recall the accelerator intensities in bf16 FLOPs: 296 (H800/H100), 206 (H200), 281 (B200), 312 (GB200), 240 (v5e), 164 (v5p) and 311 (TPU7x). With an fp8 cache, 64-head MLA is compute-bound on the H200 and on v5p, just at the line on v5e, and memory-bound on the H800 and H100, B200, GB200 and TPU7x, with an intensity 14 to 22% below their rooflines. With a bf16 cache we're memory-bound everywhere. So halving the heads relative to DeepSeek-V3 puts these models back on the memory-bound side of the line, which is probably where we'd want to be given how expensive the FLOPs turned out to be for V3. Note that the intensity of MLA is proportional to $N$, so the head count really is a design knob.

{% enddetails %}

**Question 4 [where DSA's cost goes]:** Say we run DeepSeek-V3.2 at 1M context. How many bytes and FLOPs does it cost per decoded token, and what fraction of each is due to the indexer? How much does GLM-5.2's scheme of indexing in 21 of 78 layers save at that length? What does DeepSeek-V4 do to the indexer term?

{% details Click here for the answer. %}

Let's work per layer at $S = 2^{20}$. The indexer reads `1048576 * 128 = 134MB` and the main attention reads `2048 * 576 = 1.2MB`, while the indexer does `16384 * 1048576 = 17.2G` FLOPs and the main attention `0.57G`. Over 61 layers that's **8.3GB and 1.08 TFLOPs per token, 98% and 95% of it indexer.** That's a lot of indexing! GLM-5.2's approach (scaled to V3.2's 61 layers, indexing in 16 of them) would read `16 * 134MB + 61 * 1.2MB = 2.2GB`, or 3.7x less. DeepSeek-V4 stores one 64-byte fp4 indexer key per 4 tokens, so its indexer reads 16 bytes per original token per CSA layer, i.e. `30 * 1048576 * 16 = 0.5GB` at 1M across its 30 CSA layers, 16x less than V3.2's indexer, with the main attention over 1024 compressed entries plus a 128-token window adding another 20MB.

{% enddetails %}

**Question 5 [hybrid break-even]:** Qwen3.8 has 23 full-attention layers with 4 KV heads of 256 and 69 Gated DeltaNet layers with 128 heads of $128 \times 128$ state. At what context length does the per-step state traffic (one read, one write) equal the full-attention cache traffic if we keep the state in fp32? What about in bf16? How does this compare with a DeepSeek-V3 sequence?

{% details Click here for the answer. %}

The state per layer is `128 * 128 * 128 * 4 = 8.4MB` in fp32, so over 69 layers we have 579MB, and with one read plus one write that's **1.16GB per step**. The full layers read `23 * 2 * 4 * 256 = 47kB` per context token, so the two are equal at `1.16e9 / 47.1e3 = 24,600` tokens. In bf16 the state traffic halves and the break-even drops to 12,300 tokens. For comparison, a DeepSeek-V3 sequence reads `61 * 576 = 35kB` per context token, so 1.16GB is what V3 reads at 33k tokens. In other words, a Qwen3.8 sequence costs us as much state traffic per step as a 33k-token DeepSeek-V3 sequence, whether it's 1k or 1M tokens long!

{% enddetails %}

**Question 6 [V4 versus its predecessors]:** Using the formula from the token-compression section, how many bytes does DeepSeek-V4-Pro read per decoded token at 128k and at 1M? How does that compare with V3.2 (DSA) and V3 (full MLA) at 1M?

{% details Click here for the answer. %}

Let's do 128k first. The CSA layers read `30 * (32768 * 64 + 1024 * 576 + 128 * 576) = 30 * (2.1MB + 0.59MB + 0.07MB) = 83MB` and the HCA layers read `31 * (1024 * 576 + 128 * 576) = 20MB`, for a total of **0.10GB**. At 1M the CSA layers read `30 * (16.8MB + 0.66MB) = 524MB` and the HCA layers `31 * (8192 + 128) * 576 = 149MB`, for a total of **0.67GB**. V3.2 at 1M reads 8.3GB (12x more) and V3 reads `61 * 1048576 * 576 = 37GB` (55x more)! The sliding-window branch, present in all 61 layers, is only `61 * 128 * 576 = 4.5MB` of fixed state per sequence, which is basically negligible.

{% enddetails %}

**Question 7 [attention crossover]:** Let's derive the crossover $S^\ast$ in the table above for Kimi K3 (18 active experts of width 3072 on a 3584-wide MoE bottleneck, i.e. the projected-down residual of Section 13, which is unrelated to the MLA latent, plus 96 MLA heads with $d_{qk} + d_v = 320$ and 24 full-attention layers of 93). Then, what would happen to $S^\ast$ if K3 had used MLA in every layer?

{% details Click here for the answer. %}

Let's start with the MLP FLOPs per token per layer. The 16 routed experts operate on the 3584-wide latent (`16 * 3 * 3584 * 3072 = 528M` parameters), the two shared experts run at full width (`2 * 3 * 7168 * 3072 = 132M`), and the latent down- and up-projections add `2 * 7168 * 3584 = 51M`, for 712M active MLP parameters or `1.42G` FLOPs. Attention costs `96 * 320 = 30,720` per context token per full layer. So `S* = 1.42e9 * 93 / (30720 * 24) = 180k` tokens. If we used MLA in all 93 layers instead, $L / L_\text{full} = 1$ and `S* = 46k`. So the 3:1 hybrid buys us 3.9x in the training crossover, on top of the decode savings we saw earlier.

{% enddetails %}

**Question 8 [serving a million tokens, open-ended]:** Say we want to serve Kimi K3 at 1M-token contexts. Using its cache and state sizes, how many concurrent 1M-token sequences can we fit in a GB200 NVL72 rack (72 GPUs, 186GB each) after the weights (1.56TB as shipped)? How many in the 8-GPU GB300 node (8 x 288GB) that the vLLM K3 recipe calls the minimum, or in the 32 H200s (TP8 x PP4) of NVIDIA's Dynamo K3 recipe? What's our lower bound on step time at that batch on each? (Kimi's 16-GPU H200 unit is the K2 deploy guide's, for a 1T fp8 model. You can redo it for a 64-chip TPU7x group if you like, and it lands about 10% below the rack.) Now consider a coding agent that keeps a 400k-token repository in context and makes 200 tool calls, each appending 2k tokens. How much prefix-cache memory does one such session pin, and for how long? What does this tell us about where the cache should live between turns?

<h3 markdown=1 class="next-section">That's it for Section 14! For Section 15, on how training changed (fp8 and fp4, Muon, multi-token prediction and reinforcement learning), click [here](../training-2026).</h3>
