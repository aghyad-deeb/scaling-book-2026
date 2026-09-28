---
layout: distill
title: "How to Scale Your Model: The 2026 Update"
subtitle: "What changed between LLaMA 3 and DeepSeek-V4"
# permalink: /main/
description: "How to Scale Your Model was written around LLaMA 3: a dense model, trained in bf16 with AdamW, served with grouped-query attention on 8-GPU nodes and 16GB TPUs. The frontier open models of 2026 are trillion-parameter mixtures of experts with KV caches of a few kilobytes per token. They're trained in fp8 with Muon, finished with reinforcement learning, and run on chips with 180 to 288GB of HBM in 72-GPU NVLink domains. These four sections redo the book's arithmetic for these models and this hardware."
date: 2026-09-20
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Aghyad Deeb
    url: "https://github.com/aghyad-deeb"
    affiliations:
      name: "Independent; written with Claude"

giscus_comments: false

previous_section_url: "https://jax-ml.github.io/scaling-book/gpus"
previous_section_name: "Part 12: GPUs"

next_section_url: moe
next_section_name: "Part 13: MoE"

bibliography: update.bib

toc:
  - name: "What Changed, on One Page"
  - name: "The Hardware That Runs 2026 Models"
  - name: "The Four New Sections"
  - name: "Where the Original Book Needs a Footnote"
  - name: "How the Numbers Were Checked"
  - name: "What We Left Out"
---

{% include figure.liquid path="assets/img/hero-hardware.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> the three scale-up domains the 2026 models run in, drawn to the same scale of HBM: an 8-GPU H800 node (640GB; DeepSeek-V3 and Kimi K2 trained on these), a GB200 NVL72 rack (13.4TB; Nemotron 3's long-context training, SGLang and Dynamo serving) and a 4x4x4 TPU7x cube (12.3TB, one of 144 in a pod). Bandwidths are one-way." %}

_This is an unofficial supplement to [How to Scale Your Model](https://jax-ml.github.io/scaling-book/). The book's authors didn't write or review it. It applies the book's method to the models of 2026 instead of LLaMA 3: take a handful of hardware constants, count FLOPs and bytes, and predict how a model will run before you run it. The four new sections are numbered 13 to 16 so they follow the GPU section, and they assume you've read [Sections 4](https://jax-ml.github.io/scaling-book/transformers), [5](https://jax-ml.github.io/scaling-book/training) and [7](https://jax-ml.github.io/scaling-book/inference) of the original._

## What Changed, on One Page

Let's start with the model the book keeps coming back to, LLaMA 3-70B. It's dense, with 70B parameters and 8 KV heads. It was trained on 15T tokens in bf16 with AdamW, which took 6.4M H100-hours, and its 405B sibling ran on 16,000 H100s. The book serves it on TPU v5e with tensor parallelism. Almost none of that describes a frontier open model in September 2026. Here's what changed, and where we work each change out.

* **Nearly every frontier open model is a Mixture of Experts, and a very sparse one.** Kimi K3 has 2.8T parameters and activates 104B per token, DeepSeek-V4-Pro has 1.6T and activates 49B, and Qwen3.8 has 2.4T and activates 95B. Each has hundreds of narrow experts, of which 6 to 16 run for any given token. Since Mixtral, total parameters have grown about 20x while active parameters have grown about 2.5x. Every roofline that compares weight bytes to FLOPs picks up a factor of the sparsity $E/k$ (total experts over active experts), and [Section 13](moe) redoes the Transformer math with that in mind.

* **The KV cache per token is about 30x smaller.** Most of the architectural change since 2024 is aimed at the cache: latent attention (MLA), sliding windows, top-$k$ sparse attention, linear-attention layers with a fixed-size state, and compression along the sequence. DeepSeek-V4 stores about 5kB per token where LLaMA 3-70B stored 164kB, both at one byte per element (LLaMA's is 328kB in bf16). V4.1-Flash, with an fp4 cache, keeps only 890 bytes per token in HBM. One side effect is that decode attention with MLA can be compute-bound, which essentially never happened before. [Section 14](attention) works out the bytes and FLOPs of each mechanism.

* **Matmuls run in fp8, and the weights you download are often 4-bit.** DeepSeek's fp8 recipe is now the standard.<d-footnote>E4M3 elements, one scale per 1x128 activation tile and per 128x128 weight block, and fp32 accumulation.</d-footnote> Pretraining in fp4 works but is rare. The common pattern is fp8 pretraining followed by quantization-aware post-training to 4-bit experts. fp8 matmuls run twice as fast as bf16, so any roofline whose bytes didn't also halve is now twice as hard to satisfy. [Section 15](training-2026) says which ones.

* **Muon has replaced Adam** for most of the open MoE trainers. It orthogonalizes each weight matrix's momentum before applying it. That costs 0.2 to 0.5% of FLOPs and a whole-matrix gather once per step, and it halves the optimizer memory. [Section 15](training-2026) counts the cost.

* **Models are trained to predict two tokens at a time.** An extra head that predicts the token after next costs about 4% more training FLOPs, and at inference it doubles as a speculative drafter worth 1.6 to 1.8x in tokens per second at small batch. This is also in [Section 15](training-2026).

* **Post-training is about a tenth of pretraining compute, and it's bound by generation.** DeepSeek says its post-training budget, most of it reinforcement learning, passed 10% of pretraining compute by V3.2, and that most of the wall clock goes to generation. Generation is only about a quarter of the FLOPs, but it takes most of the time because responses run 10k to 65k tokens with a long tail, and the whole batch waits for the longest one. RL adds two more constraints: a trillion-parameter policy has to be copied to the samplers in seconds, and the training and inference engines have to agree numerically. [Section 15](training-2026) shows how these constraints feed back into the architecture.

* **Serving spends more time decoding than prefilling.** DeepSeek's published production system spends about 1.4 node-days decoding for every node-day of prefill. The chat workload in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference) needed about three prefill servers per decode server, so the balance has flipped. Half of DeepSeek's decode step is the expert AllToAll over InfiniBand, a term Section 7's decode roofline doesn't have. [Section 16](applied-frontier) checks the arithmetic.

* **The hardware grew, but unevenly.** B200-class GPUs and TPU7x have 180 to 192GB of HBM per chip, and GB300 and Rubin have 288GB. The GB200 NVL72 puts 72 GPUs in one NVLink domain, so a whole trillion-parameter model fits inside it. NVLink grew along with GPU FLOPs. The InfiniBand NIC and TPU ICI didn't, so every roofline that depends on them is harder to satisfy than it used to be. The next section collects the constants.

## The Hardware That Runs 2026 Models

Every roofline in this book comes down to four numbers per accelerator: FLOPs/s, HBM bandwidth, scale-up bandwidth (NVLink or ICI) and scale-out bandwidth (InfiniBand or DCN). Here they are for the hardware the 2026 models run on, GPUs first, since that's where nearly all of these models were trained and are served. The last two columns are the intensities that show up in every derivation: $C / W_\text{hbm}$, the critical batch for decode, and $\alpha = C / W_\text{net}$ over the scale-up network.

**A note on units:** FLOPs are dense PFLOPs/s (no structured sparsity) in bf16 unless the column says otherwise, and HBM is in GB and TB/s. Scale-up and scale-out are one-way GB/s per GPU (or per TPU chip, with six or four ICI links), as elsewhere in this book, so NVIDIA's bidirectional NVLink figures are halved. NVL72 figures are per GPU, and $\alpha$ is per GPU on the GPU rows and per ICI axis on the TPU rows.<d-footnote>Sources: Google Cloud TPU documentation for v5p, v6e and TPU7x; NVIDIA product pages for H100/H200, HGX B200, GB200 NVL72, GB300 NVL72 and Vera Rubin NVL72, which quote sparse figures by default, so we halve them; and AMD product pages for MI355X and MI455X. TPU ICI follows the book's original tables (9e10 bytes/s one-way per link, Google's 200 GB/s bidirectional per axis rounded down), and GB200 per-GPU figures are rack totals divided by 72. Rubin is "in full production" per NVIDIA's August 2026 earnings but not yet broadly available, and its product page and launch blog disagree on NVLink 6 (3.0 versus 3.6 TB/s) and HBM4 bandwidth (19.2 versus 22 TB/s); we use the product page. AMD's MI455X page and its Helios page also disagree on HBM bandwidth (23.3 versus 19.6 TB/s); we use the product page's 23.3, which gives an intensity of 215 rather than the 255 we'd get at 19.6. AMD gives the MI455X's UALink scale-out as 600GB/s bidirectional per GPU (300GB/s one-way) and names 800Gb/s NICs for the Helios rack without saying how many per GPU. MI355X systems ship with OEM-chosen 400 or 800Gb/s NICs, and we assume 400Gb/s (50GB/s) per GPU, as in NVIDIA's 8-GPU nodes. Rubin's product page quotes 0.45TB/s bidirectional scale-out against a 1.6Tb/s ConnectX-9 line rate, and we use the line rate, 200GB/s one-way. AMD's MI355X page gives 153 GB/s per Infinity Fabric link without a direction; we assume bidirectional, as AMD states for the MI455X, and halve the seven-link total. GB300 and Rubin HBM use the per-GPU specification (288 GB) rather than the rounded rack total, and where a vendor page labels a figure dense (as Rubin's does for its training columns) we use it as given. TPU 8t and 8i were announced in April 2026 with FP4 figures only (12.6 and 10.1 PFLOPs, 216 and 288 GB, 6.5 and 8.6 TB/s, and twice TPU7x's ICI). Google hasn't published their bf16 or fp8 rates, so we leave them out.</d-footnote>

<div class="l-body-outset" markdown="1">

| Accelerator | bf16 | fp8 | fp4 | HBM | HBM TB/s | Scale-up | Domain | Scale-out | $C/W_\text{hbm}$ | $\alpha$ |
| :--- | ------------: | --: | --: | --: | -------: | :-------------- | -----: | :--------------- | ---------------: | ---------------: |
| H800 (2023) | 0.99 | 1.98 | | 80 | 3.35 | 200 (160 measured) | 8 | 50 | 296 | 4,950 |
| H100 (2022) | 0.99 | 1.98 | | 80 | 3.35 | 450 | 8 | 50 | 296 | 2,200 |
| H200 (2024) | 0.99 | 1.98 | | 141 | 4.8 | 450 | 8 | 50 | 206 | 2,200 |
| HGX B200 (2025) | 2.25 | 4.5 | 9 | 180 | 8 | 900 | 8 | 50 | 281 | 2,500 |
| GB200 NVL72 (2025) | 2.5 | 5 | 10 | 186 | 8 | 900 | 72 | 50 (3,600/rack) | 312 | 2,780 |
| GB300 NVL72 (2026) | 2.5 | 5 | 15 | 288 | 8 | 900 | 72 | 100 (7,200/rack) | 312 | 2,780 |
| Vera Rubin NVL72 (2026) | 4 | 17.5 | 35 | 288 | 19.2 | 1,500 | 72 | 200 (14,400/rack) | 208 | 2,700 |
| AMD MI355X (2025) | 2.5 | 5 | 10.1 | 288 | 8 | about 535 | 8 | 50 | 312 | 4,700 |
| AMD MI455X Helios (2026) | 5 | 20.1 | 40.3 | 432 | 19.6 to 23.3 | 1,800 | 72 | 300 (21,600/rack) | 215 | 2,800 |
| TPU7x, Ironwood (2025) | 2.3 | 4.6 | | 192 | 7.4 | 6 x 90 | 9,216 | 12.5 | 311 | 12,800 |
| TPU v6e (2024) | 0.92 | 1.84 (int8) | | 32 | 1.6 | 4 x 90 | 256 | 12.5 | 575 | 5,110 |
| TPU v5p (2023) | 0.46 | 0.92 (int8) | | 96 | 2.8 | 6 x 90 | 8,960 | 6.25 | 164 | 2,550 |

</div>

{% include figure.liquid path="assets/img/chip-intensity.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> the HBM arithmetic intensity $C / W_\text{hbm}$ of each accelerator in the table, one bar per precision it publishes a dense rate for, GPUs then TPUs. The dashed line is the book's 240 to 300 bf16 critical batch on TPU v5e and H100. TPU v6e is the outlier at 575. TPUs publish no fp4 rate." %}

**Domain** is the number of accelerators that share one scale-up network (an NVLink domain or an ICI pod). **Scale-out** is the one-way bandwidth from one GPU into the InfiniBand or data-center network; for a TPU it's one chip's share of its host's NICs. For the rack-scale systems we also give the rack total. A few rows are here because of particular models: DeepSeek-V3 and Kimi K2 trained on the H800, Nemotron 3 Super and Ultra name the GB200 for their long-context phase,<d-footnote>Their main pretraining phase names no GPU.</d-footnote> and Gemma 4 trained on TPU v6e.

Let's look at what the last few columns tell us.

**The HBM intensity has barely moved in bf16.** From TPU v5p to TPU7x, and from H100 to Rubin, bf16 $C / W_\text{hbm}$ stays between about 160 and 320, because FLOPs and HBM bandwidth grew at about the same rate. (TPU v6e is the exception, at `0.92e15 / 1.6e12 = 575`.) So [Section 7](https://jax-ml.github.io/scaling-book/inference)'s rule that decode needs a batch of about 240 to be compute-bound still holds in bf16. Lower precision changes it. In fp8 the chip's intensity doubles, and in fp4 it quadruples or more: 1,250 on GB200 and 1,875 on GB300, whose fp4 rate is six times its bf16 rate. Section 7's $B_\text{crit}$ grows by the same factor unless the weights shrink by that factor too. With fp8 weights and fp8 matmuls we're back at a batch of 240 to 300, and bf16 weights with fp8 matmuls need twice that.

**Inside an NVLink domain the network intensity has barely changed, but across domains it got much worse.** NVLink bandwidth doubled with each generation's FLOPs, so $\alpha$ over NVLink stays between 2,200 and 2,800 from H100 to GB200, and [Section 12](https://jax-ml.github.io/scaling-book/gpus)'s tensor-parallel and in-node rooflines carry over unchanged in bf16 ([Section 15](training-2026) shows what fp8 does to them). The InfiniBand NIC, on the other hand, stayed at 400Gb/s per GPU from H100 through B200 and GB200 (GB300 doubles it to 800Gb/s). The cross-node intensity went from `990e12 / 50e9 = 19,800` on an H100 to 45,000 on a B200. In Section 12, fully sharded data parallelism (FSDP) across H100 nodes needs 2,475 tokens per GPU, and on 8-GPU HGX B200 nodes that threshold becomes 5,625. The GB200 NVL72 gets around this with its size. A rack pulls each weight shard in once, through 3.6TB/s of egress, and shares it over NVLink, which brings the threshold down to 694.

**Note [the H800]:** NVIDIA cut the H800's NVLink to 200GB/s each way, so DeepSeek's in-node intensity was 4,950, more than twice an H100's 2,200. As [Section 13](moe) shows, this is a real constraint on the expert AllToAll.

**On TPUs, the ICI intensity grew fivefold.** TPU7x has five times the FLOPs of v5p but exactly the same ICI bandwidth per link, so its per-axis intensity is `2.3e15 / 1.8e11 = 12,800` against v5p's 2,550, and every ICI roofline got five times harder. [Section 5](https://jax-ml.github.io/scaling-book/training)'s constants (FSDP compute-bound above 850 tokens per chip, tensor parallelism up to $3F/2550$) are v5p constants. On TPU7x they become 4,270 tokens and $3F/12800$.

**Most of the new FLOPs are low-precision FLOPs.** GB300 does 6x its bf16 rate in fp4, Rubin nearly 9x and MI455X 8x, while nothing in the memory or network columns grew anywhere near that much. The vendors expect these chips to be used mostly for fp4 serving, with bf16 training fitted around that. So when you see a headline FLOPs number, check which precision it's quoted in.

## The Four New Sections

* [**Section 13: How to Think About Mixture of Experts**](moe). How many parameters, FLOPs and bytes does a 2.8T-parameter MoE really have? Why do sparse models need thousands of tokens per GPU to be compute-bound? How much expert parallelism can we do before NVLink, InfiniBand or ICI runs out, how does it combine with FSDP, and what do DeepSeek, Kimi, NVIDIA and Google actually do?

* [**Section 14: All About the KV Cache**](attention). How many bytes does each attention mechanism in a 2026 model store per token, and how many does it read per decoded token? How does latent attention make decode compute-bound? When do a sliding window, a top-$k$ indexer, a linear-attention state or DeepSeek-V4's token compression pay off, and what does each do to the decode and long-context training rooflines?

* [**Section 15: How Training Changed**](training-2026). What exactly runs in fp8, and which rooflines get harder because of it? Is anyone really pretraining in fp4? What does Muon cost, and why does it need a whole-matrix gather? How much does multi-token prediction buy? Why is reinforcement learning bound by generation rather than by gradient steps, and how do runs reach a million tokens of context?

* [**Section 16: Serving DeepSeek-V3 and Training DeepSeek-V4**](applied-frontier). Do our rooflines reproduce DeepSeek's published production numbers, and which term is missing? What changes on a GB200 NVL72? How long would a V4-class pretraining run take on 128 GB200 racks, or on a TPU7x pod, and how would we shard it on each?

Sections 13 to 15 end with takeaways and worked problems, most with the answer hidden behind a click and the last few left for you. Section 16, like Sections 6 and 8, works its questions inline and leaves its closing problems for you. Try each one with a pen before opening the answer.

## Where the Original Book Needs a Footnote

Two of the items below fix errors in the original: the H800's fp8 rate in Section 4's Question 7, and the H800's NVLink bandwidth and DeepSeek-V3's batch size in Section 12. The rest are places where the 2026 models or hardware change a number, or where a claim now needs a caveat it didn't need before.

* **Section 4, Question 7** computes DeepSeek-V3's utilization as 22%, using an H800 rate of 1.51e15 fp8 FLOPs/s from a vendor sheet. That's the PCIe H800. The SXM part DeepSeek used has H100 tensor cores and 1.98e15 dense fp8 FLOPs/s, which gives about 17%.<d-footnote>16.5% on the question's 2.79M total hours, or 17.3% on the 2.66M pretraining hours.</d-footnote> [Section 15](training-2026) redoes the calculation, and [Section 13](moe) explains why it's so low.

* **Section 4's** rule that attention FLOPs dominate above $T = 8D$ assumes multi-head attention (MHA) and $F = 4D$. With a sparse MLP and 128 heads, DeepSeek-V3 crosses over at 19k tokens, where a dense MHA model of the same width with $F = 4D$ crosses at 86k.<d-footnote>Both numbers use Section 14's accounting, which compares causal attention against the MLP alone. That moves Section 4's $8D$ to $12D$, or 86k with $D = 7168$.</d-footnote> [Section 14](attention) gives the general formula.

* **Section 5's** constants 2,550 and 850 are TPU v5p numbers, and **Section 12's** 2,200 and 2,475 are H100 numbers. The hardware table above gives the TPU7x, HGX B200 and GB200 NVL72 values, and [Section 13](moe) uses them.

* **Section 12** gives the H800's NVLink as 300GB/s, and its FSDP example gives DeepSeek-V3's batch as 4M tokens. DeepSeek's hardware paper says the H800's NVLink was cut from 900 to 400GB/s bidirectional, which is 200GB/s in this book's one-way convention. In practice DeepSeek measures about 160GB/s, against 50GB/s of InfiniBand. The V3 batch was 15,360 sequences of 4,096 tokens, or 62.9M tokens (30,700 per GPU), as Section 12's own DeepSeek example later says. [Section 13](moe) uses 200 and 160.

* **Section 7** treats decode attention as always bandwidth-bound. That's right for grouped-query attention, whose intensity is the group size (about 8). Multi-head latent attention is different. With 128 heads, an fp8 cache and the absorption trick from Section 14, its intensity is 480. If the score and value matmuls run in bf16, as in DeepSeek's V3 recipe, decode attention is then compute-bound at long context on every current GPU, and on every TPU in the table except v6e. With a bf16 cache, or with the 64 heads of Kimi K2 and GLM-5, it's memory-bound on the H800, the Blackwell parts and TPU7x, 14 to 22% below their rooflines, though it stays compute-bound on the H200. If the attention matmuls themselves run in fp8, as in SGLang's GB200 deployment in [Section 16](applied-frontier), the accelerator intensities double and only the H200 stays compute-bound. [Section 14](attention) works through all of this.

* **Section 7's** Appendix D already points at embedded drafter heads (EAGLE<d-cite key="eagle"></d-cite>, Medusa<d-cite key="medusa"></d-cite>, and DeepSeek-V3's own). Multi-token prediction is that same head, now trained into most 2026 MoEs (DeepSeek-V4.1-Flash trains its drafter afterwards instead). [Section 15](training-2026) has its training cost and measured acceptance lengths.

* **Section 8** finds that we need three prefill servers per decode server. DeepSeek's measured 2025 traffic needs about 1.4 decode nodes per prefill node instead, because of long reasoning outputs and a 56% prefix-cache hit rate. [Section 16](applied-frontier) works out why.

* **Section 12's** expert-parallel roofline counts two expert matmuls, as Section 5 does to keep the algebra clean. [Section 13](moe) counts all three, as DeepSeek's own overlap analysis in the V4 report does, which makes the minimum expert width 1.5x smaller.

* **Section 12's** takeaway that expert parallelism "can span 1-2 nodes" is still right on InfiniBand. On a GB200 NVL72, though, the whole expert-parallel group can stay inside one NVLink domain and never touch InfiniBand. [Section 16](applied-frontier) finds this is worth about 2x to 4x in decode throughput per GPU between GPUs with the same compute.<d-footnote>One measurement, at 32-way EP under a tight per-user latency target, found 4.4x.</d-footnote> Against DeepSeek's H800 production figure the gain is about 7x, but that also counts the GPU upgrade and an easier workload.

## How the Numbers Were Checked

**Where do the numbers come from?** Every model number in these sections comes from the model's `config.json` or its technical report, as of September 2026. A few short scripts ship with these pages (`scripts/numbers.py` and its companions in `scripts/`); they recompute every number quoted in the text and draw every figure, so you can check anything yourself. The parameter totals they produce match each technical report to within 2%. Hardware numbers follow the book's original tables where they exist and vendor documentation where they don't. Published systems numbers (DeepSeek's serving throughput, the latencies of its DeepEP communication kernels, SGLang and vLLM benchmarks, the RL system papers) are cited to their primary write-ups in each section. If you find a number that doesn't check out, please tell us.

**What don't we know?** Quite a lot, and where a report doesn't give a number we leave a blank rather than guess. Nobody has published pretraining token counts for Kimi K3, Qwen3.8 or Gemma 4. No 2026 DeepSeek, Kimi, Qwen or GLM model comes with its training hardware, GPU-hours or cost.<d-footnote>Nemotron names GB200s only for the long-context phase of Super and Ultra, and Llama 4 gave GPU-hours, but neither gives a breakdown.</d-footnote> Nor do we have production acceptance rates for speculative decoding, or mean production output lengths for any thinking mode. When a number is derived rather than stated (DeepSeek-V4's KV bytes, for instance), the text says so.

## What We Left Out

We've stuck to text models on hardware you can rent. That leaves out the vision encoders and audio paths most 2026 models carry, a 400M to 2.5B-parameter tower with a fairly small systems footprint. It also leaves out diffusion and image generation, and on-device models below about 10B parameters such as Gemma 4 E2B and E4B and Ministral. Nor do we cover the domestic Chinese accelerators GLM-5 is also deployed on (Huawei Ascend, Cambricon and others), which have published serving recipes but few numbers precise enough for a roofline. We also don't work out DeepSeek-V4.1-Flash's Engram, 196B parameters of hashed n-gram embedding tables fetched from host memory at every step. It adds a byte term to the decode roofline that we'd like to work through, but it isn't documented well enough yet to pin down. Finally, we don't cover training data at all, which is where a large and growing share of the work of building these models goes (the original book doesn't claim to cover it either).

<h3 markdown=1 class="next-section">That's it for the overview! For [Section 13](moe), on Mixture of Experts, [click here](moe).</h3>
