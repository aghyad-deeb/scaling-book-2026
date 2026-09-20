---
layout: distill
title: "Serving DeepSeek-V3 and Training DeepSeek-V4"
# permalink: /main/
description: "In the spirit of Sections 6 and 8, we take a real model and real hardware and work the numbers. DeepSeek published the configuration, throughput and cost of its production serving system for V3 and R1 on H800s, which makes it the only frontier deployment whose numbers we can check against the rooflines of Sections 13 and 14. We do that, then ask what changes on a GB200 NVL72, and finally what it would take to pretrain a DeepSeek-V4-Pro-class model on a TPU7x pod. As in the earlier applied sections, try each question with a pen before opening the answer."
date: 2026-09-20
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Aghyad Deeb
    url: "https://github.com/aghyad-deeb"
    affiliations:
      name: "Independent; written with Claude"

section_number: 16

previous_section_url: "../training-2026"
previous_section_name: "Part 15: Training in 2026"

next_section_url: ..
next_section_name: "2026 Update: Outline"

giscus_comments: false

bibliography: update.bib

toc:
  - name: "What Does DeepSeek Serve, and How?"
  - name: "Reconstructing the Decode Step"
  - name: "Prefill, Cost, and the Shape of the Traffic"
  - name: "The Same Model on a GB200 NVL72"
  - name: "Training a V4-Class Model on a TPU7x Pod"
  - name: "Worked Problems"

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
---

_[Section 8](https://jax-ml.github.io/scaling-book/applied-inference) worked out how to serve LLaMA 3-70B on TPU v5e from first principles. In March 2025 DeepSeek published a day's worth of statistics from its production system for V3 and R1<d-cite key="openinfra"></d-cite>: the parallelism, the node counts, the tokens per second per node, the daily token volumes and the daily cost. That gives us something the book has not had before: a frontier-scale deployment to check our arithmetic against. Let's see how close the rooflines get, and where they miss._

## What Does DeepSeek Serve, and How?

Here's DeepSeek-V3 again, the same model as in [Section 13](../moe):

| **hyperparam** | **value** |
| :------------- | --------: |
| $L$ (layers; 3 dense, 58 MoE) | 61 |
| $D$ | 7,168 |
| $E$ routed experts, $k$ active, shared | 256, 8, 1 |
| $F$ (expert width) | 2,048 |
| $N$ heads (MLA) | 128 |
| KV latent per token per layer | 512 + 64 |
| $V$ | 129,280 |
| Total / active parameters | 671B / 37B |

And here is what DeepSeek said about serving it on H800 nodes (8 GPUs, 80GB each, 400GB/s NVLink, 400Gb/s or 50GB/s of InfiniBand per GPU) in the 24 hours ending noon on February 28, 2025:

* **Prefill:** routed experts 32-way expert parallel, attention and shared expert 32-way data parallel, over 4 nodes. 32 redundant routed experts, so each GPU holds 9 routed experts and the shared expert. Two micro-batches overlapped.
* **Decode:** routed experts 144-way expert parallel, attention and shared expert 144-way data parallel, over 18 nodes. 32 redundant experts; each GPU holds 2 routed experts and the shared expert. Attention split into two steps and a five-stage pipeline to overlap communication with compute.
* **Precision:** fp8 for the matmuls and the dispatch, bf16 for the attention core and the combine.
* **Load:** 278 nodes at peak, 226.75 on average. 608B input tokens, of which 342B (56.3%) hit the prefix cache (which DeepSeek keeps on disk); 168B output tokens. Average output speed 20 to 22 tokens per second per user, average context length per output token 4,989.
* **Throughput:** about 73.7k input tokens per second per node in prefill (counting cache hits), about 14.8k output tokens per second per node in decode.
* **Cost:** at $2 per GPU-hour, $87,072 per day. Billed at R1's prices ($0.14, $0.55 and $2.19 per million cache-hit, cache-miss and output tokens) the day's tokens would have been worth $562,027.

Several of these numbers are consequences of things we derived in the last three sections. Let's check.

**Question:** Why data-parallel attention and expert-parallel experts, rather than tensor parallelism as in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference)?

{% details Click here for the answer. %}

Two reasons from [Sections 13](../moe) and [14](../attention). MLA caches a single 576-element latent per token per layer that every one of the 128 heads reads; tensor parallelism over heads would have to replicate that latent on every shard, so it saves no KV memory and no KV bandwidth. Data-parallel attention keeps one copy per sequence. And the model is 671GB of fp8 weights, 656GB of it experts, which can't fit on a node, so the experts must be sharded, and expert parallelism is the sharding that moves activations rather than weights. The shared expert runs for every token, so it is replicated alongside attention. The GLM-5 report<d-cite key="glm5"></d-cite> states the attention half of this directly: "DP-attention is primarily introduced to prevent copying KV across different ranks."

{% enddetails %}

**Question:** How large is the KV cache per token, and per sequence at the average context of 4,989 tokens?

{% details Click here for the answer. %}

`61 * 576 = 35,136` elements, **35kB per token in fp8**. DeepSeek's write-up says the attention core runs in bf16 but doesn't say how the cache is stored; we assume fp8, which its FlashMLA kernels support, and note where the bf16 alternative would matter. At 4,989 tokens that's **175MB per sequence**. For comparison LLaMA 3-70B, at Section 8's 160kB per token, would be `160e3 * 4989 = 800MB` at the same context, and 405B (126 layers, 8 KV heads) 1.3GB.

{% enddetails %}

**Question:** Under 144-way expert parallelism with 2 routed experts per GPU, how many bytes of weights does each decode GPU hold, and how much HBM is left for KV cache?

{% details Click here for the answer. %}

Routed experts: `2 * 3 * 7168 * 2048 * 58 layers = 5.1GB`. Replicated on every GPU: attention (11.4B parameters), the shared experts (`58 * 3 * 7168 * 2048 = 2.6B`), the dense MLPs (1.2B) and the embeddings (1.9B), about 17GB in fp8. **About 22GB per GPU**, leaving 58GB of the 80GB for KV cache, activations and communication buffers. Note that three quarters of the per-GPU weight bytes are the *replicated* part: expert parallelism has done its job so well that the replicated attention stack, not the experts, is what takes the memory.

{% enddetails %}

**Question:** How many sequences is each decode GPU actually serving at once? Is it limited by KV memory?

{% details Click here for the answer. %}

14.8k tokens per second per node is `14.8e3 / 8 = 1,850` per GPU. At 20 to 22 tokens per second per user, that's **about 88 concurrent sequences per GPU**. At 175MB each they occupy 15GB of KV cache. Allowing 10GB for activations and buffers there is room for about 270 sequences, so **memory isn't the limit**, at least at the average context length. Something else caps the batch at 88.

{% enddetails %}

## Reconstructing the Decode Step

**Question:** What is the decode step time, and how does the [Section 7](https://jax-ml.github.io/scaling-book/inference) roofline account for it? Take 88 sequences per GPU at 4,989 tokens of context, fp8 weights, 3.35e12 bytes/s of HBM and 1.98e15 fp8 FLOPs/s. *Assume one token per sequence per step: DeepSeek's write-up does not say whether production decode used multi-token prediction. If it did, at the V3 paper's 1.8 tokens per step, the step would be about 86ms with 160 tokens through the MoE, and everything below scales accordingly.*

{% details Click here for the answer. %}

Twenty-one tokens per second per user means a step every **48ms**. The roofline pieces:

| Component | Calculation | Time |
| :-------- | :---------- | ---: |
| Weight bytes | `22e9 / 3.35e12` | 6.6ms |
| KV bytes | `88 * 175e6 / 3.35e12` | 4.6ms (9.2ms if the cache is bf16) |
| MoE FLOPs | `88 * 2 * 37e9 / 1.98e15` | 3.3ms |
| Attention FLOPs (absorbed MLA, bf16 core) | `88 * 61 * 278528 * 4989 / 0.99e15` | 7.5ms |

Even summing everything with no overlap, the roofline says **about 22ms**. The measured step is 48ms. We are missing more than half of it, and none of the terms in [Section 7](https://jax-ml.github.io/scaling-book/inference)'s formula can supply it.

{% enddetails %}

**Question:** What is missing? *Hint: count the MoE layers, and look at what crosses the network in each one.*

{% details Click here for the answer. %}

The AllToAlls. Each of the 58 MoE layers does a dispatch and a combine across 144 GPUs on 18 nodes over InfiniBand, and [Section 7](https://jax-ml.github.io/scaling-book/inference)'s formula has no term for the network. Per GPU per layer, 88 tokens times 8 experts times 7,168 elements is `88 * 8 * 7168 = 5.0MB` of fp8 dispatch and, because the combine sums in bf16, `10.1MB` of combine. That has to leave the GPU through its 50GB/s InfiniBand port, and DeepSeek's own DeepEP benchmarks<d-cite key="deepep"></d-cite> show its decode kernels reaching about 39 to 40GB/s of that at 128- to 256-way EP. So each MoE layer spends `15.1e6 / 40e9 = 0.38ms` on the network, and the step spends `58 * 0.38 = 22ms`.<d-footnote>DeepEP's low-latency table gives 192µs for dispatch and 365µs for combine with a 128-token batch, which is 7.3MB and 14.7MB at 38 to 40GB/s: the kernels are running at the NIC's bandwidth, so we scale the times to our 88 tokens. At these message sizes the fixed latency of a hop is a few tens of microseconds and is not the dominant term.</d-footnote> Add the 22ms of roofline work and we're at 44ms against the measured 48; DeepSeek's five-stage pipeline overlaps the two micro-batches' attention and MoE with each other's communication, which is why the sum is a little more than the observed step rather than a little less.

So DeepSeek-V3's production decode step is **bound by the network bandwidth of the expert AllToAll**, about half of it, and the other half by HBM and FLOPs. [Section 13](../moe)'s expert-parallel roofline predicts exactly this: at 50GB/s per GPU the cross-node AllToAll is communication-bound for 2,048-wide experts by about 5x, so the experts cannot hide it, and it shows up as wall-clock. It also explains three of DeepSeek's design choices at once. The batch per GPU is 88 rather than 270 because the 20 to 22 tokens per second per user target fixes the step at about 48ms, 22ms of which is network; at 270 sequences the bytes and FLOPs terms would triple and per-user speed would halve. The deployment uses 144-way rather than the 320-way EP the V3 paper originally designed because 2 experts per GPU already makes the weight bytes small, and a smaller unit is easier to keep balanced and healthy. And multi-token prediction is valuable here because a second token per step adds to the bytes and FLOPs terms but not to the AllToAll's fixed per-token cost per layer, so it comes at a discount.

{% enddetails %}

{% include figure.liquid path="assets/img/decode-step-breakdown.svg" class="img-fluid" caption="<b>Figure:</b> the decode step of DeepSeek's production system, as the Section 7 roofline accounts for it (top), as the expert AllToAll over InfiniBand accounts for it (middle), and as measured from the reported 20 to 22 tokens per second per user (bottom). The five-stage pipeline overlaps some of the top two rows, which is why the measured step is a little less than their sum." %}

<p markdown=1 class="takeaway">**Takeaway:** For a fine-grained MoE served with wide expert parallelism over InfiniBand, about half the decode step is the expert AllToAll: 58 layers times 15MB per GPU at 40GB/s is 22ms before any HBM bytes or FLOPs. [Section 7](https://jax-ml.github.io/scaling-book/inference)'s decode roofline has no network term and misses it; [Section 13](../moe)'s expert-parallel roofline predicts it. The batch, the EP degree and the value of speculative decoding all follow from that number.</p>

## Prefill, Cost, and the Shape of the Traffic

**Question:** What utilization does the prefill throughput imply? *Be careful about cache hits.*

{% details Click here for the answer. %}

73.7k tokens per second per node is `73.7e3 / 8 = 9,200` per GPU. If every one of those tokens were computed, that's `9200 * 2 * 37e9 = 6.8e14` FLOPs/s, **34%** of fp8 peak, plus attention FLOPs at 5k context: prefill can't absorb the latent the way decode does, so each token's causal attention over an average of 2,500 past tokens costs `61 * 2 * 128 * (192 + 128) * 2500 = 1.2e10` FLOPs, about 17% on top of the `7.4e10` of matmuls. But 56.3% of input tokens hit the cache and cost almost nothing, so the tokens actually computed are 44% of the total, and the matmul utilization on those is closer to **15%**. The truth is in between, because the 73.7k is an average over prefill nodes that includes cache-hit tokens at a rate DeepSeek doesn't break out. Either way it's far below the 40% [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) assumed for prefill, which was based on dense models with no AllToAll.

{% enddetails %}

**Question:** What does a million output tokens cost DeepSeek to produce, and what did they charge?

{% details Click here for the answer. %}

A node costs `8 * $2 = $16` per hour and produces `14.8e3 * 3600 = 53M` output tokens per hour in decode, so **$0.30 per million output tokens** in raw hardware cost. R1 was priced at $2.19, seven times that, which is the origin of DeepSeek's reported "545% cost profit margin" (the theoretical daily revenue of $562K against $87K of cost). The actual margin was lower: V3 was priced below R1, the app was free, and nighttime traffic was discounted. Still, at $0.30 of hardware per million tokens the economics of a 671B-parameter model were not what most people assumed in early 2025.

For reference, DeepSeek-V4-Pro's list price is $3.96 per million output tokens at peak and half that off-peak (since mid-September 2026 DeepSeek has been routing V4-Pro traffic to V4.1-Flash at $1.20 while V4.1-Pro is pending), while the GB200 deployment worked out below produces a V3-class token for about $0.17 of hardware at 2026 rental rates. The ratio of price to hardware cost hasn't moved much in eighteen months; what moved is the hardware cost.

{% enddetails %}

**Question:** [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) found that a chat workload of 8k input and 512 output tokens needs three prefill servers per decode server. Using DeepSeek's daily volumes, what is the ratio for this workload?

{% details Click here for the answer. %}

DeepSeek's 73.7k input tokens per node-second is quoted *including* cache hits, so all 608B input tokens go through it: `608e9 / 73.7e3 = 8.2M` node-seconds, **95 node-days**. The 168B output tokens at 14.8k per node-second are `168e9 / 14.8e3 = 11.4M` node-seconds, **131 node-days**. Decode needs **1.4 times** the node-time of prefill: the ratio from [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) has flipped. Two things did it. Reasoning models produce long outputs (168B output against 608B input is 0.28 output tokens per input token, against 0.06 for the chat workload in Section 8), and prefix caching made more than half of the prefill nearly free, which is what lets a node clear 73.7k tokens per second. As a check on the method, `95 + 131 = 226` node-days of work against an average of 226.75 nodes in service: the two throughput figures are simply DeepSeek's daily totals divided by node-time, and they reproduce the fleet size to within a fraction of a percent.

This is the biggest change to the serving picture since the book was written. The machines are in decode, and most of the attention redesigns in [Section 14](../attention) are attempts to make decode cheaper.

{% enddetails %}

**Question:** DeepSeek-V3.2-Exp replaced V3's attention with DeepSeek Sparse Attention (DSA), top-2,048 selection over an indexer ([Section 14](../attention)), and cut API prices by half the day it shipped. At the *average* context of 4,989 tokens, how much does DSA save in the decode step above? Why might the price cut still make sense?

{% details Click here for the answer. %}

At 4,989 tokens of context, selecting the top 2,048 saves only `1 - 2048 / 4989 = 59%` of the core attention reads and FLOPs, and the indexer adds its own read of 128 bytes per token, so the net saving in bytes is 37%. Per sequence the KV bytes read fall from 175MB to about `61 * (4989 * 128 + 2048 * 576) = 111MB`, and the attention FLOPs from 7.5ms to about 4ms per step in the table above. Against a step that is 22ms of AllToAll and 11ms of weight reads, that's a minor saving.

The saving isn't in the average request. It's in the tail: a 128k-token request reads 4.6GB of KV per token with full MLA and 1.1GB with DSA, and does 13x fewer attention FLOPs. Long-context requests are a minority of requests but a large fraction of the *cost*, because cost per token grows with context for full attention. Flattening that curve, which is what DeepSeek's published cost-versus-position plots show<d-cite key="deepseekv32"></d-cite>, is what let them cut a flat per-token price. It's also the workload that has been growing: coding agents that keep hundreds of thousands of tokens of repository in context and append a few hundred tokens per step.<d-footnote>A 2026 study of coding-agent traces<d-cite key="tracelab"></d-cite> found that about 96% of prompt tokens were served from the prefix cache and that a median step read back 126k cached tokens while appending 857 and generating 252. That step is almost entirely attention over cached context, which is exactly the cost DSA and V4's compression attack, and almost none of it is prefill or MoE FLOPs.</d-footnote>

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway:** For DeepSeek's measured 2025 traffic, decode needs about 1.4 times the node-time of prefill, the inverse of Section 8's chat-era ratio, and the two published throughput figures reproduce the average fleet size exactly. Long reasoning outputs (0.28 output tokens per input token) and a 56% prefix-cache hit rate did it. Sparse attention barely helps the average 5k-token request; it flattens the cost of the long tail, which is where the money was going.</p>

## The Same Model on a GB200 NVL72

Everything above was done on 2023 hardware with an 8-GPU NVLink domain. The GB200 NVL72 puts 72 GPUs, 13.4TB of HBM and 130TB/s of NVLink in one rack. Let's see what it changes.

**Question:** With 64-way expert parallelism inside a single NVL72 rack, how many bytes of weights does each GPU hold with fp8 experts? With fp4 experts? Is the AllToAll compute-bound?

{% details Click here for the answer. %}

Routed experts: `656e9 / 64 = 10.3GB` in fp8 or 5.1GB in fp4, plus the same 17GB of replicated attention, shared expert, dense and embedding weights in fp8. Call it **27GB with fp8 experts or 22GB with fp4 experts**, out of 186GB per GPU, leaving about 160GB for KV cache. From [Section 13](../moe), expert parallelism is compute-bound when $F > \alpha / 2$ with fp8 dispatch, where $\alpha = C / W_\text{NVLink}$; for the GB200 that's `2.5e15 / 9e11 = 2,800` in bf16 and a threshold of 1,400, which V3's 2,048-wide experts clear. But SGLang's deployment below runs fp8 attention and fp4 experts, and at those rates $\alpha$ is 5,600 or 11,100 and the thresholds 2,800 and 5,600: the AllToAll is 1.4x to 2.7x communication-bound even inside the rack. What the rack changes is not whether the AllToAll is hidden but how long it takes: the same 15MB per GPU per layer that took 0.38ms over InfiniBand takes `15.1e6 / 726e9 = 21µs` at the 726GB/s DeepEP measures on Blackwell NVLink (at 8-way EP inside a node), so the 22ms per step becomes about `58 * 21e-6 = 1.2ms`.

{% enddetails %}

**Question:** SGLang reported 13,386 output tokens per second per GPU for DeepSeek-V3 on a GB200 NVL72 with fp8 attention and NVFP4 experts at 2k-token inputs<d-cite key="sglang_gb200"></d-cite>. How does that compare with DeepSeek's H800 production number, and what explains the gap?

{% details Click here for the answer. %}

DeepSeek's 14.8k per node is 1,850 per GPU, so **7.2x per GPU**. Some of that is the benchmark rather than the hardware: 2k tokens of context instead of 5k, and no 20-tokens-per-second-per-user constraint, so SGLang could run a much larger batch; SGLang's own EP72 deployment on H100s reached 2,800 tokens per second per GPU on the same 2k-input workload, which suggests about 1.5x of the 7.2x is workload. Of the remaining 4.8x, the GB200 GPU's 2.4x HBM bandwidth (8e12 against 3.35e12), 2.5x fp8 FLOPs and fp4 experts account for perhaps half. The other half is the domain: the AllToAll stays on NVLink instead of crossing 18 nodes of InfiniBand, so the 22ms per step becomes about 1ms and the batch per GPU can grow into the 186GB. The cleanest evidence is a measurement that holds per-user speed fixed and changes only the domain<d-cite key="inferencex"></d-cite>: at 125 tokens per second per user, a GB200 NVL72 running 32-way EP delivered 4,130 tokens per second per GPU against 941 for B200s in 8-GPU nodes with EP confined to the node (DeepSeek-R1 in fp4 in both cases), a 4.4x difference between two chips of nearly identical compute. In dollars: at about $8 per GB200-hour, a mid-2026 rental rate, 13,386 tokens per second is `8 / (13386 * 3600) * 1e6 = $0.17` of hardware per million output tokens, against the $0.30 we found for the H800 at $2, and that is before the per-user speed target that cost DeepSeek a factor of 1.5 in batch.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway:** Moving a fine-grained MoE from 8-GPU nodes over InfiniBand to a 72-GPU NVLink domain is worth about 4x in decode throughput per GPU at equal silicon (4.4x GB200 NVL72 against B200 nodes), and about 7x once the chip upgrade from H800 is included. This is what Section 13 meant by calling the rack, rather than the node, the unit of MoE serving.</p>

## Training a V4-Class Model on a TPU7x Pod

Now the other direction. DeepSeek-V4-Pro is 1.6T total parameters, 49B active, 384 experts of width 3,072 with 6 active per token, 61 layers, trained on 33T tokens with a batch that ramps to 94.4M tokens, in fp8, with Muon. DeepSeek doesn't say what hardware it trained on. Suppose we wanted to train something like it on a full TPU7x pod: 9,216 chips, 192GB and 7.4e12 bytes/s of HBM each, 4.61e15 fp8 FLOPs/s, a 3D torus with 1.8e11 bytes/s per axis, so $\alpha = C / W_\text{ici}$ is `2.3e15 / 1.8e11 = 12,800` per axis in bf16 and 25,600 in fp8.

**Question:** How many FLOPs is the run, and how long does it take at 40% utilization of fp8 peak?

{% details Click here for the answer. %}

`6 * 49e9 * 33e12 = 9.7e24` FLOPs. The pod delivers `9216 * 4.61e15 = 4.2e19` FLOPs/s at peak, so at 40% the run takes `9.7e24 / (4.2e19 * 0.4) = 5.7e5 s`, **6.6 days**. At the 17 to 22% DeepSeek achieved for V3 on H800s (17% at the H800's dense fp8 rate of 1.98e15; the book's Section 4 used 1.51e15 and got 22%) it would be 13 to 16 days. For comparison, DeepSeek-V3's 3.3e24 FLOPs took about 54 days on 2,048 H800s (2.664M GPU-hours). A 9,216-chip Ironwood pod is roughly a 10x larger machine than that cluster in fp8 FLOPs, so the frontier open models of 2026 are, by pod standards, a week's work.

{% enddetails %}

**Question:** What parallelism plan does [Section 13](../moe) recommend, and is it compute-bound?

{% details Click here for the answer. %}

The batch is 94.4M tokens over 9,216 chips, **10,200 tokens per chip**. Pure FSDP would need `(384 / 6) * 25600 / 3 = 546,000` tokens per chip in fp8, fifty times what we have. So that's out.

Take 64-way expert parallelism on a 4x4x4 cube (so the longest EP axis is $A = 4$) and FSDP over the remaining 144. The FSDP threshold becomes `(384 / (6 * 64)) * 25600 / 3 = 8,500` tokens per chip in fp8 (4,300 if the gathered tensors are fp8), and we have 10,200. Compute-bound, with a little margin, and only if the FSDP gathers get all three axes to themselves while the AllToAll is running elsewhere in the step; if the two share links, the margin is gone. The expert AllToAll needs $F > A \alpha / 8 = 4 \cdot 25600 / 8 = 12{,}800$ with fp8 FLOPs and fp8 dispatch; V4's experts are 3,072 wide, so the AllToAll is **4x communication-bound** if nothing overlaps it. DeepSeek's own condition for hiding the AllToAll behind expert compute, $C / W \leq 2F = 6{,}144$ FLOPs per byte, isn't met by TPU7x's per-axis 25,600 in fp8 either. So the plan is 64-way expert parallelism and 144-way FSDP, and the AllToAll must be overlapped with attention and the shared expert, as every lab in [Section 13](../moe) does. If it were not overlapped at all, the MoE layers (about 60% of active FLOPs) would run 4x slow and the step would take `0.4 + 0.6 * 4 = 2.8x` its compute-bound time, taking utilization from 40% to about 14%. That range, 14 to 40%, brackets the 17 to 22% DeepSeek-V3 achieved on H800s ([Section 15](../training-2026)).

{% enddetails %}

**Question:** How much memory does the run need per chip for weights and optimizer state, and for activations?

{% details Click here for the answer. %}

Weights and Muon state: 1.6T parameters at 6 bytes each ([Section 15](../training-2026)) is 9.6TB, sharded over the whole pod: **1GB per chip**. With fp32 master weights and fp32 gradient accumulators as in DeepSeek's recipe, add 8 bytes per parameter, another 1.4GB per chip, 2.4GB in total. Activations: 10,200 tokens per chip times 61 layers times a few checkpoints of width 7,168 in bf16 is `10200 * 61 * 7168 * 2 * 4 = 36GB` with four checkpoints per layer, and the per-token MLP activations are of width $(6 + 1) \cdot 3072 = 21{,}504$, comparable to a dense model's. Memory is a non-issue at pod scale, exactly as [Section 6](https://jax-ml.github.io/scaling-book/applied-training) found for LLaMA 3-70B and as [Section 13](../moe) argued it would be; ICI is the constraint.

{% enddetails %}

**Question:** DeepSeek-V3.2 says its RL budget exceeded 10% of pretraining. If a V4-class run spends 10% of its FLOPs on RL, how many tokens are generated, and how long does it take on the same pod if decode runs at 10,000 tokens per second per chip? *Every generated token is also trained on.*

{% details Click here for the answer. %}

10% of 9.7e24 is 9.7e23 FLOPs. Each generated token costs `2 * 49e9` FLOPs to generate and `6 * 49e9` to train on, `8 * 49e9 = 3.9e11` in all, so the budget buys **about 2.5 trillion generated tokens**, about 8% of the pretraining corpus, produced one token at a time. At 10,000 tokens per second per chip (a generous figure for a 49B-active model at long context) the pod generates 9.2e7 tokens per second and needs `2.5e12 / 9.2e7 = 27,000 s`, **about 7.5 hours** if every chip decoded at full rate for the whole time. The FLOPs aren't the problem. The problem, per [Section 15](../training-2026), is that generation runs at a fraction of the pod's efficiency and can't finish before its longest response does, so the calendar time is several times the FLOPs time.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway:** A 2026 frontier MoE is about a week of pretraining on an Ironwood pod at 40% utilization, and the plan that gets there is expert parallelism of about $E/k$ plus FSDP over the rest, with the expert AllToAll hidden behind attention because TPU7x's ICI can't carry it unoverlapped. Memory is irrelevant at this scale. The RL phase that follows generates a few trillion tokens at decode speed.</p>

## Worked Problems

**Question 1 [Kimi K3 on H20s]:** Kimi K3 (2.8T parameters, 104B active, 24 multi-head latent attention layers with a 576-element latent, 69 Kimi Delta Attention layers with 217MB of bf16 state per sequence, fp4 expert weights totaling 1.42TB) is served on H20 nodes (8 GPUs, 96GB each, 900GB/s NVLink, 400Gb/s InfiniBand). Using 128-way expert parallelism over 16 nodes, how many bytes of weights does each GPU hold? How many 128k-token sequences fit per GPU? At 100 sequences per GPU, how many bytes does each of the 92 MoE layers' AllToAlls move per GPU (16 active experts, a 3,584-wide latent, fp8 dispatch and bf16 combine), how long does that take at 40GB/s, and what does that imply for tokens per second per user without speculation?

**Question 2 [prefill-decode ratio for an agent]:** A coding agent keeps 200k tokens of repository in context, appends 1k tokens per step, generates 300 tokens per step, and takes 50 steps per task, with a 96% prefix-cache hit rate. For DeepSeek-V3 (full MLA) and V3.2 (DSA), compute the bytes read from HBM per step for attention, the prefill FLOPs per step, and the ratio of decode to prefill node-time. Compare with the 1.4:1 measured for DeepSeek's overall traffic.

**Question 3 [DSA at the tail]:** Using [Section 14](../attention)'s formulas, compute the decode step time for DeepSeek-V3 and V3.2 at 88 sequences per GPU when the context is 128k rather than 5k, on an H800, including the 22ms of AllToAll time. Which term dominates in each case?

**Question 4 [Ironwood versus NVL72 for serving]:** A 4x4x4 TPU7x cube (64 chips, 192GB each, 7.4e12 bytes/s, 6 ICI links at 9e10 bytes/s one-way) and a GB200 NVL72 (72 GPUs, 186GB, 8e12 bytes/s, 900GB/s NVLink egress) both hold DeepSeek-V4-Pro's 0.83TB of fp4-expert weights. For a decode step of 128 sequences per chip at 128k context, compute the KV bytes per step using [Section 14](../attention)'s figure of about 100MB read per token for V4-Pro at that context, the weight bytes, and the AllToAll bytes per layer. Which system is bandwidth-bound and on what, and which has the shorter latency floor?

**Question 5 [the fp8 gather question]:** In the TPU7x training plan above, the FSDP threshold was 8,500 tokens per chip if weights are gathered in bf16 and 4,300 if gathered in fp8. DeepSeek's recipe keeps fp32 master weights and quantizes to fp8 per 128x128 block before each matmul. Design where in the FSDP gather the quantization should happen so that ICI carries fp8, and say what extra memory or compute it costs.

**Question 6 [what the book got right]:** Go back to [Section 8](https://jax-ml.github.io/scaling-book/applied-inference)'s LLaMA 3-70B serving analysis and redo the "prefill to generate server ratio" question for a reasoning workload with 8k input and 16k output tokens and a 50% prefix-cache hit rate. Then redo it for LLaMA 3-70B's KV cache at a 128k context. Which of the book's conclusions survive, and which were artifacts of the 2024 workload?

<h3 markdown=1 class="next-section">That's the end of the 2026 update. The outline, the model tables and the scripts that produced every number are back at the [start](..).</h3>
