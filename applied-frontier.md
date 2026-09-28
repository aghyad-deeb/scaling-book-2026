---
layout: distill
title: "Serving DeepSeek-V3 and Training DeepSeek-V4"
# permalink: /main/
description: "In the spirit of Sections 6 and 8, we take a real model and real hardware and work the numbers. DeepSeek published the configuration, throughput and cost of its production serving system for V3 and R1 on H800s, which makes it the only frontier deployment whose numbers we can check against the rooflines of Sections 13 and 14. We do that, ask what changes on a GB200 NVL72, and then work out what it would take to pretrain a DeepSeek-V4-Pro-class model, first on GB200 NVL72 racks and then on a TPU7x pod."
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
  - name: "Training a V4-Class Model on GB200 NVL72 Racks"
  - name: "The Same Run on a TPU7x Pod"
  - name: "Worked Problems"

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
---

_This section applies the rooflines of the last three sections to a real production deployment, the way [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) did for LLaMA 3-70B on TPU v5e. We're lucky here: in March 2025 DeepSeek published a day's worth of statistics from its production system for V3 and R1<d-cite key="openinfra"></d-cite> (parallelism, node counts, tokens per second per node, daily token volumes and daily cost), so for once we can check our arithmetic against a frontier-scale system. As in the earlier applied sections, try to work out each answer with a pen and paper before opening it!_

## What Does DeepSeek Serve, and How?

Let's remind ourselves what DeepSeek-V3 looks like (it's the model we worked through in [Section 13](../moe)):

| **hyperparam** | **value** |
| :------------- | --------: |
| $L$ (layers; 3 dense, 58 MoE) | 61 |
| $D$ | 7,168 |
| $E$ routed experts, $k$ active, shared | 256, 8, 1 |
| $F$ (expert width) | 2,048 |
| $N$ heads (MLA) | 128 |
| KV latent per token per layer (elements) | 512 + 64 |
| $V$ | 129,280 |
| Total / active parameters | 671B / 37B |

DeepSeek serves it on H800 nodes of 8 GPUs, each with 80GB of HBM. Each GPU has 200GB/s of one-way NVLink egress<d-footnote>This is half the 400GB/s bidirectional figure NVIDIA quotes. DeepSeek measures about 160GB/s of it in practice.</d-footnote> and one 400Gb/s (50GB/s) InfiniBand NIC. Here's what DeepSeek reported for the 24 hours ending at noon on February 28, 2025:

* **Prefill:** the routed experts are 32-way expert parallel, and attention and the shared expert are 32-way data parallel, over 4 nodes. There are 32 redundant routed experts, so each GPU holds 9 routed experts plus the shared expert, and two micro-batches are overlapped.
* **Decode:** the routed experts are 144-way expert parallel, and attention and the shared expert are 144-way data parallel, over 18 nodes. Again there are 32 redundant experts, so each GPU holds 2 routed experts and the shared expert. Attention is split into two steps and run in a five-stage pipeline so that communication overlaps with compute.
* **Precision:** fp8 for the matmuls and the dispatch, bf16 for the attention core and the combine.
* **Load:** 278 nodes at peak and 226.75 on average. 608B input tokens, of which 342B (56.3%) hit the prefix cache (which DeepSeek keeps on disk), and 168B output tokens. The average output speed was 20 to 22 tokens per second per user, and the average context length per output token was 4,989.
* **Throughput:** about 73.7k input tokens per second per node in prefill (counting cache hits) and about 14.8k output tokens per second per node in decode.
* **Cost:** at <span>$2</span> per GPU-hour, <span>$87,072</span> per day. Billed at R1's prices (<span>$0.14</span>, <span>$0.55</span> and <span>$2.19</span> per million cache-hit, cache-miss and output tokens), the day's tokens would have been worth <span>$562,027</span>.

Quite a few of these numbers follow from the last three sections. Let's check.

**Question:** Why does DeepSeek use data-parallel attention and expert-parallel experts, rather than the tensor parallelism [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) used?

{% details Click here for the answer. %}

There are two reasons, one from [Section 14](../attention) and one from [Section 13](../moe). First, MLA caches a single 576-element latent per token per layer, which all 128 heads read. Tensor parallelism over heads would have to replicate that latent on every shard, saving no KV memory or bandwidth at all, whereas data-parallel attention keeps one copy per sequence. Second, the model is 671GB of fp8 weights, 656GB of it experts, which can't fit on a node. The experts have to be sharded somehow, and expert parallelism is the sharding that moves activations rather than weights. The shared expert runs for every token, so it's replicated alongside attention. The GLM-5 report<d-cite key="glm5"></d-cite> puts the attention half of this well: "DP-attention is primarily introduced to prevent copying KV across different ranks."

{% enddetails %}

**Question:** How large is the KV cache per token, and per sequence at the average context of 4,989 tokens?

{% details Click here for the answer. %}

Per token, it's `61 * 576 = 35,136` elements, or **35kB per token in fp8**.<d-footnote>DeepSeek says the attention core runs in bf16 but not how the cache is stored. We'll assume fp8, which its FlashMLA kernels support, and note where bf16 would matter.</d-footnote> At 4,989 tokens that's **175MB per sequence**. For comparison, LLaMA 3-70B, at 164kB per token,<d-footnote>Section 8 rounds this to 160kB.</d-footnote> would need `164e3 * 4989 = 818MB` at the same context, and LLaMA 3-405B (126 layers, 8 KV heads) would need 1.3GB. So MLA saves a great deal of memory here.

{% enddetails %}

**Question:** Under 144-way expert parallelism with 2 routed experts per GPU, how many bytes of weights does each decode GPU hold, and how much HBM is left for KV cache?

{% details Answer %}

The routed experts are `2 * 3 * 7168 * 2048 * 58 = 5.1e9` parameters, or 5.1GB in fp8. Everything else is replicated on every GPU: attention (11.4B parameters), the shared experts (`58 * 3 * 7168 * 2048 = 2.6B`), the dense MLPs (1.2B) and the embeddings (1.9B), about 17GB in fp8. That's **about 22GB per GPU**, leaving 58GB of the 80GB for KV cache, activations and communication buffers. Notice that about three quarters of the per-GPU weight bytes are the *replicated* part. Expert parallelism shards the experts so well that the attention stack copied onto every GPU now takes more memory than the experts do.

{% enddetails %}

**Question:** How many sequences is each decode GPU serving at once? Is it limited by KV memory?

{% details Click here for the answer, once you've thought about it! %}

14.8k tokens per second per node is `14.8e3 / 8 = 1,850` per GPU. At 20 to 22 tokens per second per user, that's **about 88 concurrent sequences per GPU**. At 175MB each they use 15GB of KV cache. Allowing 10GB for activations and buffers, there's room for about 270 sequences, so **the batch isn't limited by memory**, at least at the average context length. Something else is capping it at 88. *Any guesses?*

{% enddetails %}

## Reconstructing the Decode Step

Now let's rebuild the step time from the roofline and see how close we get.

**Question:** What is the decode step time, and how does the [Section 7](https://jax-ml.github.io/scaling-book/inference) roofline account for it? Assume 88 sequences per GPU at 4,989 tokens of context, fp8 weights, 3.35e12 bytes/s of HBM and 1.98e15 fp8 FLOPs/s. *Assume one token per sequence per step.*<d-footnote>DeepSeek doesn't say whether production decode used multi-token prediction. If it did, at the V3 paper's 1.8 tokens per step, the step would be about 86ms with 160 tokens through the MoE, and everything below scales accordingly.</d-footnote>

{% details Click here for the answer. %}

Twenty-one tokens per second per user means a step every **48ms**. Here's the roofline, piece by piece:

| Component | Calculation | Time |
| :-------- | :---------- | ---: |
| Weight bytes | `22e9 / 3.35e12` | 6.6ms |
| KV bytes | `88 * 175e6 / 3.35e12` | 4.6ms (9.2ms if the cache is bf16) |
| MoE FLOPs | `88 * 2 * 37e9 / 1.98e15` | 3.3ms |
| Attention FLOPs (absorbed MLA, bf16 core; [Section 14](../attention)'s $2N(d_c + d_v)$ per token per position) | `88 * 61 * (2 * 128 * (576 + 512)) * 4989 / 0.99e15` | 7.5ms |

Even adding everything up with no overlap at all, the roofline gives only **about 22ms**, against a measured 48ms. More than half the step is missing, and none of the terms in [Section 7](https://jax-ml.github.io/scaling-book/inference)'s formula can account for it.

{% enddetails %}

**Question:** What is missing? *Hint: count the MoE layers, and look at what crosses the network in each one.*

{% details Click here for the answer. %}

It's the AllToAlls! Each of the 58 MoE layers does a dispatch and a combine across 144 GPUs on 18 nodes over InfiniBand, and [Section 7](https://jax-ml.github.io/scaling-book/inference)'s formula has no network term at all. Per GPU per layer, the dispatch sends each of 88 tokens, a 7,168-element vector, to 8 experts, which is `88 * 8 * 7168 = 5.0MB` in fp8. The combine sums in bf16, so it moves twice that, `10.1MB`. All of it has to leave through the GPU's 50GB/s InfiniBand port. DeepSeek's own DeepEP benchmarks<d-cite key="deepep"></d-cite> show its decode kernels reaching about 39 to 40GB/s of that at 128- to 256-way EP. So each MoE layer should spend `15.1e6 / 40e9 = 0.38ms` on the network, and the step `58 * 0.38 = 22ms`.<d-footnote>DeepEP's low-latency table gives 192µs for dispatch and 365µs for combine with a 128-token batch, which works out to 7.3MB and 14.7MB at 38 to 40GB/s. In other words the kernels run at the NIC's bandwidth, so we can scale the times to our 88 tokens. At these message sizes the fixed latency of a hop is a few tens of microseconds, so it isn't the dominant term.</d-footnote> Adding back the 22ms of roofline work puts us at 44ms against the measured 48. DeepSeek's five-stage pipeline overlaps each micro-batch's attention and MoE with the other's communication, so if anything our un-overlapped sum should *exceed* the measured step. Falling 4ms short means the roofline is still missing something: kernel launches and synchronization, the router and gating, load imbalance across the experts, or a bf16 rather than fp8 KV cache, which on its own would add 4.6ms.

So roughly half of DeepSeek-V3's production decode step is spent **waiting on the expert AllToAll over InfiniBand**, and only the other half on HBM bytes and FLOPs. That's what [Section 13](../moe)'s expert-parallel roofline predicts: at 50GB/s per GPU, the cross-node AllToAll is about 10x communication-bound for 2,048-wide experts with the fp8 matmuls this deployment runs (and 5x even with bf16 matmuls). The expert matmuls have no hope of hiding that much communication, so it shows up as wall-clock time.

Knowing that the AllToAll is half the step explains several of DeepSeek's choices. Take the batch first. Memory would allow 270 sequences per GPU, but DeepSeek runs 88. The target of 20 to 22 tokens per second per user fixes the step at about 48ms, and 22ms of that is network. At 270 sequences, the bytes and FLOPs terms would triple and the per-user speed would halve, so 88 is about as many as the latency target allows.

It also explains why the deployment uses 144-way EP rather than the 320-way the V3 paper originally designed for. With 2 experts per GPU the weight bytes are already small, and a smaller unit is easier to keep balanced and healthy.

Finally, multi-token prediction is worth less here than you might expect. Only the 6.6ms of weight reads and the 4.6ms of KV reads are paid once per step, however many tokens come out. The AllToAll, MoE and attention terms all scale with tokens, so a second token per sequence costs about `22 + 3.3 + 7.5 = 33ms` more. It comes at only about a 30% discount.

{% enddetails %}

{% include figure.liquid path="assets/img/decode-step-breakdown.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> the decode step of DeepSeek's production system, as the Section 7 roofline accounts for it (top), as the expert AllToAll over InfiniBand accounts for it (middle), and as measured from the reported 20 to 22 tokens per second per user (bottom). The five-stage pipeline overlaps some of the top two rows, yet the measured step is still a little more than their sum, so the two rows are a lower bound on where the time goes." %}

<p markdown=1 class="takeaway">**Takeaway:** for a fine-grained MoE served with wide expert parallelism over InfiniBand, about half the decode step is the expert AllToAll: 58 layers times 15MB per GPU at 40GB/s is 22ms before any HBM bytes or FLOPs. [Section 7](https://jax-ml.github.io/scaling-book/inference)'s decode roofline has no network term and misses it entirely, while [Section 13](../moe)'s expert-parallel roofline predicts it. The batch and the EP degree both follow from that number, and since the AllToAll scales with tokens per step, it also caps what speculative decoding can buy.</p>

## Prefill, Cost, and the Shape of the Traffic

Now for prefill, and for what the day's traffic cost.

**Question:** What utilization does the prefill throughput imply? *Be careful about cache hits.*

{% details Click here for the answer. %}

73.7k tokens per second per node is `73.7e3 / 8 = 9,200` per GPU. If all of those tokens were actually computed, that's `9200 * 2 * 37e9 = 6.8e14` FLOPs/s, or **34%** of fp8 peak, counting only the matmuls.<d-footnote>Prefill can't absorb the latent the way decode does, so attention adds to this. Each token's causal attention over an average of 2,500 past tokens costs <code>61 * 2 * 128 * (192 + 128) * 2500 = 1.2e10</code> FLOPs, about 17% on top of the <code>7.4e10</code> of matmuls.</d-footnote> But 56.3% of input tokens hit the cache and cost almost nothing. Only the other 44% are actually computed, and the matmul utilization on those is closer to **15%**. The 73.7k is an average over prefill nodes, and DeepSeek doesn't say at what rate it includes cache-hit tokens, so the truth is somewhere between these two figures. Either way it's well below the 40% that [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) assumed for prefill, based on dense models with no AllToAll.

{% enddetails %}

**Question:** What does a million output tokens cost DeepSeek to produce, and what did they charge for it?

{% details Click here for the answer. %}

A node costs `8 * $2 = $16` per hour and produces `14.8e3 * 3600 = 53M` output tokens per hour in decode, so the raw hardware cost is **<span>$0.30</span> per million output tokens**. R1 was priced at <span>$2.19</span>, about seven times that, which is where DeepSeek's reported "545% cost profit margin" comes from (<span>$562K</span> of theoretical daily revenue against <span>$87K</span> of cost). The actual margin was lower, since V3 was priced below R1, the app was free, and nighttime traffic was discounted. Still, <span>$0.30</span> of hardware per million tokens is far cheaper than most people guessed a 671B-parameter model could be served for in early 2025.

For reference, DeepSeek-V4-Pro's list price is <span>$3.96</span> per million output tokens at peak and half that off-peak.<d-footnote>As of mid-September 2026, DeepSeek has been routing V4-Pro traffic to V4.1-Flash at <span>$1.20</span> while V4.1-Pro is pending.</d-footnote> The GB200 deployment we'll work out below produces a V3-class token for about <span>$0.22</span> of hardware at 2026 on-demand rates. Against that machine, R1's 2025 price of <span>$2.19</span> would be about a 10x markup. V4-Pro's <span>$3.96</span> would be about 18x, although V4-Pro is a heavier model with 49B active parameters. On its own H800s at <span>$2</span> per GPU-hour, DeepSeek's markup was roughly 7x. So the hardware cost per token went down, but the markup went up faster. Keep this in mind when you compare API prices across years.

{% enddetails %}

**Question:** [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) found that a chat workload of 8k input and 512 output tokens needs three prefill servers per decode server. Using DeepSeek's daily volumes, what is the ratio for this workload?

{% details Click here for the answer, once you've thought about it! %}

DeepSeek's 73.7k input tokens per node-second is quoted *including* cache hits, so all 608B input tokens go through it: `608e9 / 73.7e3 = 8.2M` node-seconds, or **95 node-days**. The 168B output tokens at 14.8k per node-second are `168e9 / 14.8e3 = 11.4M` node-seconds, or **131 node-days**. So decode needs **1.4 times** the node-time of prefill, and the ratio from [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) has reversed. Two things changed. First, reasoning models produce long outputs. DeepSeek's traffic has `168e9 / 608e9 = 0.28` output tokens per input token, against 0.06 for the chat workload in Section 8. Second, prefix caching made more than half of the prefill nearly free, letting a node clear 73.7k tokens per second. As a check on the bookkeeping, `95 + 131 = 226` node-days of work against an average of 226.75 nodes in service. That isn't an independent prediction (DeepSeek's two throughput figures are just daily totals divided by node-time), but it confirms they were computed the same way and that we've split the fleet correctly.

This is probably the biggest change to the serving picture since [Section 8](https://jax-ml.github.io/scaling-book/applied-inference): most node-time now goes to decode, and most of the attention redesigns in [Section 14](../attention) are about making decode cheaper.

{% enddetails %}

**Question:** DeepSeek-V3.2-Exp swapped V3's attention for DeepSeek Sparse Attention (DSA), top-2,048 selection over an indexer ([Section 14](../attention)). When it shipped, DeepSeek also cut its API prices in half. At the *average* context of 4,989 tokens, how much does DSA save in the decode step above? Can you see why the price cut might still make sense?

{% details Click here for the answer. %}

At the average context of 4,989 tokens, selecting the top 2,048 cuts the core attention reads and FLOPs by `1 - 2048 / 4989 = 59%`, but the indexer adds its own read of 128 bytes per token per layer, so the net saving in KV bytes is 37%. Per sequence, the KV bytes read fall from 175MB to about `61 * (4989 * 128 + 2048 * 576) = 111MB`, and the attention FLOPs from 7.5ms to about 4ms per step in the table above. Against a step with 22ms of AllToAll and 11ms of weight and KV reads, that's a minor saving.

The saving comes from long requests. Take one with 128k tokens of context. With full MLA it reads 4.6GB of KV per decoded token, while with DSA it reads only 1.1GB and does 13x fewer attention FLOPs. There aren't many requests like that, but under full attention the cost per token grows with the context, so they account for a surprisingly large share of the *cost*. DeepSeek's published cost-versus-position plots<d-cite key="deepseekv32"></d-cite> show exactly this curve being flattened, which is what lets them charge one flat per-token price. Long requests are also becoming more common, for instance coding agents that keep hundreds of thousands of tokens of repository in context and append a few hundred tokens per step.<d-footnote>A 2026 study of coding-agent traces<d-cite key="tracelab"></d-cite> found that about 96% of prompt tokens were served from the prefix cache, and that a median step read back 126k cached tokens while appending 857 and generating 252. A step like that is almost entirely attention over cached context (exactly the cost DSA and V4's compression go after), and almost none of it is prefill or MoE FLOPs.</d-footnote>

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway:** for DeepSeek's measured 2025 traffic, decode needs about 1.4 times the node-time of prefill, the reverse of Section 8's chat-era ratio, and the two published throughput figures reproduce the average fleet size exactly. Long reasoning outputs (0.28 output tokens per input token) and a 56% prefix-cache hit rate caused the reversal. Sparse attention barely helps the average 5k-token request, but it flattens the cost curve for long-context requests, which is where a large share of the cost is.</p>

## The Same Model on a GB200 NVL72

So far we've been on the H800, 2023 hardware with an 8-GPU NVLink domain. The GB200 NVL72 puts 72 GPUs, 13.4TB of HBM and 65TB/s of NVLink egress (130TB/s bidirectional, as NVIDIA quotes it) into a single rack. Let's see what that changes.

**Question:** Say we run 64-way expert parallelism inside a single NVL72 rack. How many bytes of weights does each GPU hold with fp8 experts? With fp4 experts? Is the AllToAll compute-bound?

{% details Click here for the answer. %}

The routed experts are `656e9 / 64 = 10.3GB` per GPU in fp8 or 5.1GB in fp4, plus the same 17GB of replicated attention, shared-expert, dense and embedding weights in fp8. That's roughly **27GB with fp8 experts or 22GB with fp4 experts**, out of 186GB per GPU, leaving about 160GB for KV cache. Plenty of room!

**Is the AllToAll compute-bound?** From [Section 13](../moe), expert parallelism is compute-bound when $F > \alpha / 2$ with fp8 dispatch, where $\alpha = C / W_\text{NVLink}$. On the GB200 that's `2.5e15 / 9e11 = 2,780` in bf16, so the threshold is 1,390, which V3's 2,048-wide experts clear. But SGLang's deployment below runs fp8 attention and fp4 experts. At the fp8 rate $\alpha$ is 5,560 and the threshold 2,780, and at the fp4 rate $\alpha$ is 11,100 and the threshold 5,560. So even inside the rack, the AllToAll is 1.4x to 2.7x communication-bound. The rack shortens the AllToAll, as we'll see next, but it still isn't hidden.

**How long does it take?** At the bandwidth NVIDIA's own kernels reach on an NVL72 with large batches (about 80% of 900GB/s), the 15MB per GPU per layer that took 0.38ms over InfiniBand would take `15.1e6 / 700e9 = 22µs`. At DeepSeek's 88 tokens per GPU, though, the kernels are latency-bound. TensorRT-LLM's measurements on a GB200 NVL72 at 64-way EP show that the dispatch and the combine each have a floor that doesn't shrink with fewer bytes.<d-footnote>At 64 to 128 tokens per GPU, the dispatch takes about 29 to 42µs and the combine 38 to 45µs. The dispatch's floor is 18µs at 8-way EP and 20 to 29µs at 64-way, and the combine's is 31µs.<d-cite key="trtllm_blog18"></d-cite></d-footnote> Call it 65 to 90µs per layer, so the 22ms per step becomes **4 to 5ms**. That's a 5x saving rather than the 18x the bandwidth ratio promises, because at decode batch sizes the NVLink AllToAll is limited by *latency*, while the InfiniBand one was limited by bandwidth.

{% enddetails %}

**Question:** SGLang reported 13,386 output tokens per second per GPU for DeepSeek-V3 on a GB200 NVL72 with fp8 attention and fp4 experts (NVIDIA's NVFP4 format) at 2k-token inputs<d-cite key="sglang_gb200"></d-cite>. How does that compare with DeepSeek's H800 production number, and what explains the gap?

{% details Click here for the answer. %}

DeepSeek's 14.8k per node is 1,850 per GPU, so this is **7.2x per GPU**. Some of that is the benchmark rather than the hardware: 2k tokens of context instead of 5k, and no 20-tokens-per-second-per-user constraint, so SGLang could run a much larger batch. We can size the workload effect with SGLang's own EP72 deployment on H100s.<d-cite key="sglang_ep"></d-cite> On the same 2k-input workload it reached 2,800 tokens per second per GPU, running 256 sequences per GPU at about 11 tokens per second per user. Since `2800 / 1850 = 1.5`, about 1.5x of the 7.2x is workload. That leaves 4.8x to explain. Roughly half of it is the better GPU: the GB200 has 2.4x the HBM bandwidth (8e12 against 3.35e12) and 2.5x the fp8 FLOPs, and the experts run in fp4. The other half is the NVLink domain. The AllToAll stays on NVLink instead of crossing 18 nodes of InfiniBand, so the 22ms per step becomes the 4 to 5ms we just found, and the batch per GPU can grow into the 186GB.

We can check this with a measurement that holds the per-user speed fixed and changes only the domain.<d-cite key="inferencex"></d-cite> Both systems served DeepSeek-R1 in fp4 at 125 tokens per second per user. A GB200 NVL72 running 32-way EP delivered 4,130 tokens per second per GPU, against 941 for B200s in 8-GPU nodes with EP confined to the node. That's a 4.4x difference between two GPUs with nearly identical compute, so the NVLink domain really is doing the work.

To put this in dollars, take <span>$10.50</span> per GB200-hour, CoreWeave's on-demand rate in September 2026.<d-footnote>Other clouds quote <span>$16</span> and up, and reserved deals start around <span>$8</span>.</d-footnote> Then 13,386 tokens per second is `10.5 / (13386 * 3600) * 1e6 = $0.22` of hardware per million output tokens, or anywhere from <span>$0.17</span> to <span>$0.33</span> depending on the rate. That's a bit cheaper than the <span>$0.30</span> we found for the H800 at DeepSeek's assumed <span>$2</span>, although the benchmark isn't paying for a per-user speed target, which we saw cost DeepSeek a factor of 1.5 in batch.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway:** moving a fine-grained MoE from 8-GPU InfiniBand nodes to a 72-GPU NVLink domain buys about 2x to 4x in decode throughput per GPU between GPUs with the same compute, with most of the gain at tight per-user latency targets. The GB200-against-B200 measurement above found 4.4x at 125 tokens per second per user. With the GPU upgrade and a looser latency target included, it's about 7x against DeepSeek's H800 production figure. This is what Section 13 meant by treating the rack, rather than the node, as the unit of MoE serving.</p>

## Training a V4-Class Model on GB200 NVL72 Racks

Now say we wanted to *train* a model of this class instead of serving one. Here's what DeepSeek-V4-Pro looks like:

| **hyperparam** | **value** |
| :------------- | --------: |
| $L$ (layers) | 61 |
| $E$ routed experts, $k$ active | 384, 6 |
| $F$ (expert width) | 3,072 |
| Total / active parameters | 1.6T / 49B |
| Training tokens | 33T |
| Batch (after ramp) | 94.4M tokens |

It was trained in fp8 with Muon, [Section 15](../training-2026)'s optimizer, which needs 6 rather than 10 bytes of state per parameter.

**What hardware should we plan for?** DeepSeek doesn't say what it trained on, and almost no 2026 frontier run names its machine, so we'll have to guess.<d-footnote>Neither do Kimi, Zhipu or Qwen for their 2026 models. About the only exceptions are NVIDIA's own Nemotron 3 Super and Ultra, whose long-context phases ran on GB200 GPUs. These were presumably NVL72 racks, since Ultra's 128-way expert parallelism doesn't fit in a smaller domain, although the main phases' hardware isn't stated. The most recent non-NVIDIA run we know of that names its hardware is Mistral Large 3, in December 2025, on 3,000 H200s.</d-footnote> Let's plan the run on the machine a well-funded GPU lab would most likely use: 128 GB200 NVL72 racks, or 9,216 GPUs. Each GPU has 186GB of HBM at 8e12 bytes/s and does 5e15 fp8 FLOPs/s. Inside its rack it has 900GB/s of NVLink egress to every other GPU, and out of the rack it has one 400Gb/s (50GB/s) InfiniBand NIC, so each rack has 3.6TB/s of egress.

[Section 12](https://jax-ml.github.io/scaling-book/gpus)'s intensities in fp8 are then `5e15 / 900e9 = 5,560` over NVLink and `5e15 / 3.6e12 = 1,390` for fully sharded data parallelism (FSDP, [Section 5](https://jax-ml.github.io/scaling-book/training)). For anything gathered across racks, we take the rack's 3.6TB/s of InfiniBand egress as Section 12's $W_\text{collective}$.

**Question:** How many FLOPs is the run, and how long does it take at 40% utilization of fp8 peak?

{% details Click here for the answer. %}

The run is `6 * 49e9 * 33e12 = 9.7e24` FLOPs. The cluster delivers `9216 * 5e15 = 4.6e19` FLOPs/s at peak, so at 40% the run takes `9.7e24 / (4.6e19 * 0.4) = 5.3e5 s`, or **6.1 days**. At the 17% that DeepSeek achieved for V3 on H800s ([Section 15](../training-2026)), it would take 14 days.<d-footnote>Counting attention FLOPs, DeepSeek-V3's utilization was 19%.</d-footnote> For comparison, DeepSeek-V3's 3.3e24 FLOPs took about 54 days on 2,048 H800s, or 2.664M GPU-hours. Our run on that cluster at that efficiency would take `9.7e24 / (2048 * 1.98e15 * 0.17) = 1.4e7 s`, or **163 days**. There's no way DeepSeek could have trained a V4-class model on its V3 cluster in any reasonable time. On 2,048 GB200s at 40% it would take 27 days, which is more plausible. Our 128-rack cluster is roughly 10x DeepSeek's V3 cluster in fp8 FLOPs, so a 2026 frontier open model is only about a week of compute on it. That assumes we can keep the chips busy, so let's work out the sharding.

{% enddetails %}

**Question:** What parallelism plan does [Section 13](../moe) recommend, and are we compute-bound?

{% details Click here for the answer, once you've thought about it! %}

The batch is 94.4M tokens over 9,216 GPUs, or **10,200 tokens per GPU**. Pure FSDP across racks would need `(384 / 6) * 1390 = 89,000` tokens per GPU with bf16 weight gathers, about nine times what we have, so it's ruled out, as [Section 13](../moe) predicted.

Instead, let's put the expert parallelism inside the rack, 72-way with one group per rack<d-footnote>384 experts over 72 GPUs is five or six per GPU. That's fine for a grouped-matmul kernel, and Kimi K3's training system would make it six each by adding redundant copies of the hottest experts. You could also run 64-way EP on 64 of the rack's GPUs and give the other 8 the redundant copies, as DeepSeek's serving system does. The arithmetic below barely changes either way.</d-footnote>, and FSDP across the 128 racks.

The FSDP threshold then becomes `(384 / (6 * 72)) * 1390 = 1,240` tokens per GPU with bf16 gathers, or 620 with fp8 gathers, against our 10,200: **compute-bound with an 8x margin**. The expert AllToAll inside the rack needs $F > \alpha_\text{fp8} / 2 = 5560 / 2 = 2{,}780$ with fp8 matmuls, fp8 dispatch and bf16 combine. V4-Pro's experts are 3,072 wide, so at the spec 900GB/s the AllToAll is **compute-bound by 10%** without any overlap at all. In practice, though, NVIDIA's and DeepSeek's kernels reach 700 to 730GB/s on Blackwell NVLink with large batches. That puts $\alpha$ at about 7,000 and the threshold at about 3,500, so V4-Pro's experts miss by 10 to 15%.<d-footnote>DeepSeek's own condition for hiding the AllToAll behind expert compute, $C / W \leq 2F = 6{,}144$ FLOPs per byte, is met at spec bandwidth and needs 90% of it in practice.</d-footnote> So there's a little margin at spec bandwidth and a small shortfall at the bandwidth we actually get. To be safe we'd overlap the AllToAll with attention or with a second micro-batch, which is exactly what DualPipe-style schedules provide.

On paper, then, nothing in this plan is communication-bound by more than about 15%, and a modest overlap covers that, so we don't need the heavy overlap machinery of [Section 13](../moe). Other things set the utilization. The first is the pipeline: we don't need one for memory, but nearly every published run uses one to keep the data-parallel axis small and to interleave micro-batches. The others are the attention layers, the HBM traffic of the sparse-attention indexer, and how close the grouped expert matmuls get to peak with 10,200 tokens spread over five or six experts per GPU.

**Note [keep the EP group inside the rack]:** this plan only works because the AllToAll never leaves NVLink. If an EP group spills into a second rack, its bandwidth drops from 900GB/s to 50GB/s per GPU, an 18x loss. On a cluster of 8-GPU HGX B200 nodes instead of racks, the AllToAll would run over InfiniBand at $\alpha_\text{fp8} = 90{,}000$, where essentially nothing is compute-bound.

{% enddetails %}

**Question:** How much memory does the run need per GPU for weights and optimizer state, and for activations?

{% details Answer %}

Weights and Muon state first: 1.6T parameters at 6 bytes each ([Section 15](../training-2026)) is 9.6TB, sharded over the whole cluster, so **1GB per GPU**. fp32 master weights and fp32 gradient accumulators, as in DeepSeek's recipe, add 8 bytes per parameter, another 1.4GB per GPU, for 2.4GB in total. For activations, 10,200 tokens per GPU times 61 layers times a few checkpoints of width 7,168 in bf16 is `10200 * 61 * 7168 * 2 * 4 = 36GB` with four checkpoints per layer, out of 186GB. Memory is a non-issue at this scale, just as [Section 6](https://jax-ml.github.io/scaling-book/applied-training) found for LLaMA 3-70B and [Section 13](../moe) predicted. (On 2,048 GPUs the same run would have 46,000 tokens per GPU and the activations would grow to 160GB, so a smaller cluster needs pipeline parallelism or more checkpointing to fit, as DeepSeek-V3's did.)

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway:** a 2026 frontier MoE is about a week of pretraining on 128 GB200 NVL72 racks at 40% utilization. The plan is expert parallelism of about $E/k$ inside each rack, where NVLink keeps the AllToAll within about 10 to 15% of compute-bound for 3,072-wide experts even in fp8, and FSDP or data parallelism across racks, where the rack's 3.6TB/s of InfiniBand egress makes the weight gathers cheap. Memory is irrelevant at this scale, and by the rooflines nothing is communication-bound by more than about 15%. For comparison, the same model on 2,048 H800s at DeepSeek-V3's efficiency would take about five months.</p>

## The Same Run on a TPU7x Pod

Now say we had a full TPU7x (Ironwood) pod instead. It has 9,216 chips in a 3D torus. Each chip has 192GB of HBM at 7.4e12 bytes/s, does 4.61e15 fp8 FLOPs/s, and has 1.8e11 bytes/s of ICI bandwidth per axis. That makes $\alpha = C / W_\text{ici}$ equal to `2.3e15 / 1.8e11 = 12,800` per axis in bf16, and 25,600 in fp8. The pod delivers `9216 * 4.61e15 = 4.2e19` FLOPs/s at peak, so at 40% the run takes `9.7e24 / (4.2e19 * 0.4) = 5.7e5 s`, or **6.6 days**, within 10% of the GB200 cluster. So the two machines have about the same FLOPs, and the difference between them is in the network.

**Question:** What parallelism plan does [Section 13](../moe) recommend on the pod, and are we compute-bound?

{% details Click here for the answer. %}

The batch is 94.4M tokens over 9,216 chips, so again **10,200 tokens per chip**. Pure FSDP would need `(384 / 6) * 25600 / 3 = 546,000` tokens per chip with fp8 FLOPs and bf16 gathers, about fifty times what we have, so we can rule it out straight away.

Instead, let's take 64-way expert parallelism on a 4x4x4 cube (so the longest EP axis is $A = 4$) and FSDP over the remaining 144. The FSDP threshold becomes `(384 / (6 * 64)) * 25600 / 3 = 8,500` tokens per chip with bf16 gathers (4,270 with fp8 gathers), and we have 10,200. That's compute-bound with a little margin, but only if the FSDP gathers get all three axes to themselves while the AllToAll runs at another point in the step. If the two have to share links, the margin is gone.

The expert AllToAll needs $F > A \alpha / 8 = 4 \cdot 25600 / 8 = 12{,}800$ with fp8 FLOPs and fp8 dispatch, and V4's experts are only 3,072 wide, so the AllToAll is **4x communication-bound** if nothing overlaps it. Unlike the GB200 cluster, where nothing was more than about 15% communication-bound, here the ICI is the limit. DeepSeek's own condition for hiding the AllToAll behind expert compute, $C / W \leq 2F = 6{,}144$ FLOPs per byte, isn't met by TPU7x's per-axis 25,600 in fp8 either. So the plan is 64-way expert parallelism and 144-way FSDP, with the AllToAll overlapped with attention and the shared expert, as in every training layout in [Section 13](../moe). Without any overlap, the MoE layers, about 60% of active FLOPs, would run 4x slower. The step would take `0.4 + 0.6 * 4 = 2.8x` its compute-bound time, and utilization would fall from 40% to about 14%. The 17 to 19% DeepSeek-V3 achieved on H800s ([Section 15](../training-2026)) falls inside that range of 14 to 40%. Memory is the same non-issue as on the GPU cluster: 1GB of weights and optimizer state per chip, and 36GB of activations.

{% enddetails %}

**Question:** DeepSeek-V3.2 says its RL budget exceeded 10% of pretraining. If a V4-class run spends 10% of its FLOPs on RL, how many tokens are generated, and how long does that take on either machine if decode runs at 10,000 tokens per second per accelerator? *Every generated token is also trained on.*<d-footnote>DeepSeek's production H800s manage 1,850 per GPU under a per-user latency target, and SGLang's GB200 benchmark hits 13,400 without one. A training cluster has no users to serve, so 10,000 is on the generous side but not crazy.</d-footnote>

{% details Click here for the answer. %}

10% of our 9.7e24 is 9.7e23 FLOPs. Each generated token costs `2 * 49e9` FLOPs to generate and `6 * 49e9` to train on, so `8 * 49e9 = 3.9e11` in all, and the budget buys **about 2.5 trillion generated tokens**, roughly 8% of the pretraining corpus, produced one token at a time. At 10,000 tokens per second per accelerator, the 9,216-way machine generates 9.2e7 tokens per second and needs `2.5e12 / 9.2e7 = 27,000 s`, or **about 7.5 hours**, if every accelerator decoded at full rate the whole time. So the FLOPs aren't the worry. The problem, as we saw in [Section 15](../training-2026), is that generation runs at a fraction of the pod's efficiency and can't finish before its longest response does, so the calendar time ends up several times the FLOPs time.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway:** an Ironwood pod runs the same plan, expert parallelism of about $E/k$ plus FSDP over the rest, and at the same 40% utilization it also takes about a week. The difference is that on the pod the expert AllToAll is 4x communication-bound and has to be hidden behind attention, since TPU7x's ICI didn't grow with its FLOPs the way a GB200's NVLink did. The RL phase that follows generates a few trillion tokens at decode speed on either machine.</p>

## Worked Problems

Let's finish with a few problems to try yourself.

**Question 1 [Kimi K3 on H20s]:** Say we want to serve Kimi K3 on H20s. K3 has 2.8T parameters, 104B active, with 24 MLA layers (a 576-element latent) and 69 Kimi Delta Attention layers (217MB of bf16 state per sequence), and its shipped weights are 1.56TB with fp4 experts. The H20 is the Hopper GPU NVIDIA sells in export markets, with 96GB of HBM, about 148 bf16 and 296 fp8 TFLOPs/s, 450GB/s of one-way NVLink (900 bidirectional) and one 400Gb/s InfiniBand NIC per GPU. Its arithmetic intensity is only about 37, so it's memory-bound on nearly everything, which is actually what you want for decode. Assume 128-way expert parallelism over 16 nodes. (a) How many bytes of weights does each GPU hold? (b) How many 128k-token sequences fit per GPU? (c) At 100 sequences per GPU, how many bytes does each of the 92 MoE layers' AllToAlls move per GPU? Each token goes to 16 active experts with a 3,584-wide latent, with fp8 dispatch and bf16 combine. (d) How long does that take at 40GB/s, and what does it imply for tokens per second per user without speculation?

**Question 2 [prefill-decode ratio for an agent]:** Say a coding agent keeps 200k tokens of repository in context, appends 1k tokens per step, generates 300 tokens per step, and takes 50 steps per task, with a 96% prefix-cache hit rate. For DeepSeek-V3 (full MLA) and V3.2 (DSA), compute the bytes read from HBM per step for attention, the prefill FLOPs per step, and the ratio of decode to prefill node-time. How does it compare with the 1.4:1 we measured for DeepSeek's overall traffic?

**Question 3 [DSA at the tail]:** Using [Section 14](../attention)'s formulas, compute the decode step time we'd expect for DeepSeek-V3 and V3.2 at 88 sequences per GPU when the context is 128k rather than 5k, on an H800, including the 22ms of AllToAll time. Which term dominates in each case?

**Question 4 [Ironwood versus NVL72 for serving]:** Say we have a 4x4x4 TPU7x cube (64 chips, 192GB each, 7.4e12 bytes/s of HBM, 6 ICI links at 9e10 bytes/s one-way, so 1.8e11 per axis as above) and a GB200 NVL72 (72 GPUs, 186GB, 8e12 bytes/s, 900GB/s NVLink egress), and both hold DeepSeek-V4-Pro's 0.87TB of fp4-expert weights. For a decode step of 128 sequences per chip at 128k context, compute the KV bytes per step using [Section 14](../attention)'s figure of about 100MB read per token for V4-Pro at that context, the weight bytes, and the AllToAll bytes per layer. Which system ends up bandwidth-bound, and on which resource (HBM, ICI or NVLink)? Which one has the lower latency floor, and why?

**Question 5 [the fp8 gather question]:** In the GB200 training plan above, the FSDP threshold across racks was 1,240 tokens per GPU if we gather the weights in bf16 and 620 if we gather them in fp8, and in the TPU7x plan it was 8,500 and 4,270 per chip. DeepSeek's recipe keeps fp32 master weights and quantizes to fp8 per 128x128 block before each matmul. Where in the FSDP gather should the quantization happen so that the network carries fp8, and what extra memory or compute does that cost?

**Question 6 [what Section 8 got right]:** Go back to [Section 8](https://jax-ml.github.io/scaling-book/applied-inference)'s LLaMA 3-70B serving analysis and redo the "prefill to generate server ratio" question for a reasoning workload with 8k input and 16k output tokens and a 50% prefix-cache hit rate. Then redo it for LLaMA 3-70B's KV cache at a 128k context. Which of that section's conclusions survive, and which were artifacts of the 2024 workload?

<h3 markdown=1 class="next-section">That's it for the 2026 update! For the outline, the model tables and the scripts that produced every number, head back to the [start](..).</h3>
