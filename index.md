---
layout: distill
title: "How to Scale Your Model: The 2026 Update"
subtitle: "What changed between LLaMA 3 and DeepSeek-V4"
# permalink: /main/
description: "How to Scale Your Model was written around LLaMA 3, a dense model trained in bf16 with Adam and served with grouped-query attention on 8-GPU nodes and 16GB TPUs. Eighteen months later the frontier open models are trillion-parameter mixtures of experts with KV caches of a few kilobytes per token, trained in fp8 with Muon and finished with reinforcement learning, and the hardware has 180 to 288GB of HBM per GPU and 72-GPU NVLink domains. These four sections extend the book's arithmetic to that world. Nothing in the original is wrong, but a lot of it now needs a footnote, and these sections are it."
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

{% include figure.liquid path="assets/img/hero-hardware.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> the three fast domains the 2026 models run in, drawn to the same scale of HBM: an 8-GPU H800 node (DeepSeek-V3 and Kimi K2 trained on these; 640GB), a GB200 NVL72 rack (Nemotron 3's long-context training, SGLang and Dynamo serving; 13.4TB) and a 4x4x4 TPU7x cube (12.3TB, one of 144 in a pod). Bandwidths are one-way." %}

_This is an independent supplement to How to Scale Your Model; the book's authors didn't write or review it. The book's method is to take a handful of hardware constants, count FLOPs and bytes, and predict how a model will run before you run it. These four sections do that for the models of 2026. They are numbered 13 to 16 so that they slot in after the GPU section, and they assume you've read [Sections 4](https://jax-ml.github.io/scaling-book/transformers), [5](https://jax-ml.github.io/scaling-book/training) and [7](https://jax-ml.github.io/scaling-book/inference) of the original._

## What Changed, on One Page

The reference model in the book is LLaMA 3-70B: 70B dense parameters, 8 KV heads, trained on 15T tokens in bf16 with AdamW for 6.4M H100-hours (its 405B sibling ran on 16,000 H100s), served on TPU v5e with tensor parallelism. Here's what each of those looks like for the frontier open models of September 2026, and where it is worked out.

* **Every frontier open model is a Mixture of Experts, and a sparse one.** Kimi K3 has 2.8T parameters and activates 104B; DeepSeek-V4-Pro has 1.6T and activates 49B; Qwen3.8 has 2.4T and activates 95B. Every one of them has hundreds of narrow experts and activates 6 to 16 of them per token. Total parameters grew 20x since Mixtral while active parameters grew 2.5x. [Section 13](moe) redoes the Transformer math and every roofline that compares weight bytes to FLOPs, which inflate by the sparsity $E/k$, the ratio of total to active experts.

* **The KV cache per token shrank 30x, and it was the thing being optimized.** Latent attention (MLA), sliding windows, top-$k$ sparse attention, linear-attention layers with a fixed state, and outright token compression each cut it; DeepSeek-V4 stores about 5kB per token where LLaMA 3-70B stored 164kB at the same one byte per element (328kB in bf16), and V4.1-Flash, with an fp4 cache, keeps 890 bytes per token of HBM-resident cache. Along the way MLA made decode attention compute-bound for the first time. [Section 14](attention) computes bytes and FLOPs for each mechanism.

* **Matmuls run in fp8; the weights you download are often 4-bit.** DeepSeek's fp8 recipe (E4M3, 1x128 and 128x128 scales, fp32 accumulation) is now standard. fp4 pretraining works but is rare; fp8 pretraining followed by quantization-aware post-training to 4-bit experts is common. Every roofline whose byte term didn't halve got 2x harder. [Section 15](training-2026).

* **Muon replaced Adam** for most open MoE trainers. It orthogonalizes each weight matrix's momentum before applying it, at a cost of 0.2 to 0.5% of FLOPs and half the optimizer memory, with a whole-matrix gather once per step. [Section 15](training-2026).

* **Models predict two tokens at a time, and the second head is a speculative drafter.** It costs about 4% more training FLOPs and buys 1.6 to 1.8x in tokens per second at small batch. [Section 15](training-2026).

* **Post-training is a tenth of pretraining compute, and it is bound by generation.** DeepSeek reports that its post-training budget, most of it reinforcement learning, passed 10% of pretraining compute by V3.2, and most of its wall-clock, though only about a quarter of its FLOPs, goes to generation. Responses run 10k to 65k tokens with a long straggler tail, a trillion-parameter policy has to be re-synchronized to the samplers in seconds, and the training and inference engines have to agree numerically, which ends up shaping the architecture. [Section 15](training-2026).

* **The serving workload flipped.** DeepSeek's published production system spends about 1.4 node-days decoding for every node-day of prefill, the inverse of the chat-era ratio in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference), and half of its decode step goes to the expert AllToAll over InfiniBand, a term Section 7's decode roofline doesn't have. [Section 16](applied-frontier).

* **Hardware moved to meet the models, unevenly.** B200-class GPUs and TPU7x put 180 to 192GB on a chip, GB300 and Rubin 288GB, and the GB200 NVL72 puts 72 GPUs in one NVLink domain, so a trillion-parameter model fits in one fast fabric. NVLink kept pace with GPU FLOPs; the InfiniBand NIC and TPU ICI didn't, and the rooflines that depend on them tightened. The next section has the constants.

## The Hardware That Runs 2026 Models

The book's rooflines need four numbers per accelerator: FLOPs/s, HBM bandwidth, scale-up network bandwidth (NVLink or ICI) and scale-out network bandwidth (InfiniBand or DCN). Here they are for what the 2026 models run on, GPUs first since that is where nearly all of them were trained and are served, with the operational intensities that appear in every derivation: $C / W_\text{hbm}$ (the decode critical batch) and $\alpha = C / W_\text{net}$ over the scale-up network (per GPU, or per ICI axis). FLOPs are dense PFLOPs/s, no structured sparsity, in bf16 unless the column says otherwise; HBM is in GB and TB/s; the scale-up and scale-out columns are one-way GB/s per GPU (or per TPU chip, with six or four ICI links), as in the book, so NVIDIA's bidirectional NVLink figures have been halved; rack-level figures are per GPU for the NVL72 systems. $\alpha$ is per GPU on the GPU rows and per ICI axis on the TPU rows.<d-footnote>Sources: Google Cloud TPU documentation for v5p, v6e and TPU7x; NVIDIA product pages for H100/H200, HGX B200, GB200 NVL72, GB300 NVL72 and Vera Rubin NVL72 (which quote sparse figures by default; we halve them); AMD product pages for MI355X and MI455X. TPU ICI figures follow the original book's tables (9e10 bytes/s one-way per link, which is Google's 200 GB/s bidirectional per axis rounded down). GB200 per-GPU figures are the rack totals divided by 72. Rubin is "in full production" per NVIDIA's August 2026 earnings but not broadly available; its product page and its launch blog disagree on NVLink 6 (3.0 versus 3.6 TB/s) and HBM4 bandwidth (19.2 versus 22 TB/s); we use the product page. AMD's MI455X page and its Helios page disagree on HBM bandwidth (23.3 versus 19.6 TB/s); we use the product page's 23.3 (intensity 215; 255 at 19.6). AMD gives the MI455X's UALink scale-out as 600GB/s bidirectional per GPU, 300GB/s one-way, and names 800Gb/s NICs for the Helios rack without a count per GPU. MI355X systems ship with OEM-chosen 400 or 800Gb/s NICs; we assume 400Gb/s (50GB/s) per GPU as in the NVIDIA 8-GPU nodes. Rubin's scale-out is quoted as 0.45TB/s bidirectional on the product page against a 1.6Tb/s ConnectX-9 line rate; we use the line rate, 200GB/s one-way. AMD's MI355X page gives 153 GB/s per Infinity Fabric link without stating a direction; we assume bidirectional, as AMD states for the MI455X, and halve the seven-link total. GB300 and Rubin HBM use the per-GPU specification (288 GB) rather than the rounded rack total. Where a vendor page labels a figure dense, as Rubin's product page does for its training columns, we use it as given. TPU 8t and 8i were announced in April 2026 with FP4 figures only (12.6 and 10.1 PFLOPs, 216 and 288 GB, 6.5 and 8.6 TB/s, twice TPU7x's ICI); Google has not published their bf16 or fp8 rates, so they are omitted.</d-footnote>

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

{% include figure.liquid path="assets/img/chip-intensity.svg" class="img-fluid" zoomable=true caption="<b>Figure:</b> the HBM arithmetic intensity $C / W_\text{hbm}$ of each accelerator in the table, one bar per precision it publishes a dense rate for, GPUs then TPUs. The dashed line is the book's 240 to 300 bf16 critical batch on TPU v5e and H100. TPU v6e is the outlier at 575. TPUs publish no fp4 rate." %}

Domain is the number of accelerators in one scale-up fabric, an NVLink domain or an ICI pod; scale-out is what one GPU (or one TPU chip's share of its host NIC) can push into the InfiniBand or data-center network, one way, with the rack total where a whole rack shares one fabric. The H800 is the GPU DeepSeek-V3 and Kimi K2 trained on, the GB200 is what Nemotron 3 Super and Ultra name for their long-context phase (their main phase names no GPU), and v6e is Gemma 4's. A few things jump out of the last three columns.

**The HBM intensity barely moved.** From v5p to TPU7x, from H100 to Rubin, the bf16 number stays between 160 and 320 (TPU v6e, with 1.6TB/s of HBM bandwidth for 0.92 PFLOPs/s, is the exception at `0.92e15 / 1.6e12 = 575`). Chips got faster and their memory got faster in step, so [Section 7](https://jax-ml.github.io/scaling-book/inference)'s "batch 240 to be compute-bound" rule still holds in bf16. It moves once the matmuls run in fp8 or fp4: the chip's intensity doubles in fp8 and quadruples or more in fp4 (1,250 on GB200, 1,875 on GB300, whose fp4 rate is six times its bf16 rate), and Section 7's $B_\text{crit}$ doubles with it unless the weights shrink in step. fp8 weights with fp8 FLOPs land back at 240 to 300; bf16 weights with fp8 FLOPs need twice the batch.

**On GPUs the scale-up intensity held and the scale-out intensity didn't.** NVLink doubled with each generation's FLOPs, so $\alpha$ inside the domain sits between 2,200 and 2,800 from H100 to GB200 and [Section 12](https://jax-ml.github.io/scaling-book/gpus)'s tensor-parallel and in-node rooflines carry over unchanged in bf16 ([Section 15](training-2026) shows what fp8 does to them). The InfiniBand NIC stayed at 400Gb/s per GPU from H100 through B200 and GB200 (GB300 doubles it to 800Gb/s), so the cross-node intensity went from `990e12 / 50e9 = 19,800` on an H100 to 45,000 on a B200, and [Section 12](https://jax-ml.github.io/scaling-book/gpus)'s cross-node threshold for fully sharded data parallelism (FSDP, 2,475 tokens per GPU on H100 nodes) becomes 5,625 on 8-GPU HGX B200 nodes. The GB200 NVL72's answer is the domain: 72 GPUs share one NVLink fabric, and a rack pulls each weight shard in once through 3.6TB/s of egress, which brings that threshold down to 694. The H800 is the odd one out: NVIDIA cut its NVLink to 200GB/s each way, so DeepSeek's in-node intensity is 4,950, worse than an H100's 2,200 and a real constraint on the expert AllToAll ([Section 13](moe)).

**On TPUs the scale-up intensity moved.** TPU7x has five times the FLOPs of v5p and the same ICI per link, so its per-axis intensity is 12,800 against 2,550 (`2.3e15 / 1.8e11 = 12,800`). [Section 5](https://jax-ml.github.io/scaling-book/training)'s constants (FSDP compute-bound above 850 tokens per chip, tensor parallelism up to $3F/2550$) are v5p constants; on Ironwood they are 4,270 tokens and $3F/12800$.

**fp4 is where the compute is going.** GB300 has 6x its bf16 rate in fp4, Rubin nearly 9x, MI455X 8x. Nothing in the memory or network columns grew like that, which tells you what the vendors think these chips are for: fp4 serving, with bf16 training as the workload that has to fit around it.

## The Four New Sections

* [**Section 13: How to Think About Mixture of Experts**](moe). How many parameters, FLOPs and bytes does a 2.8T-parameter MoE really have? Why do sparse models want thousands of tokens per GPU to be compute-bound? How much expert parallelism can you do before NVLink, InfiniBand or ICI runs out, how does it combine with FSDP, and what do DeepSeek, Kimi, NVIDIA and Google actually do?

* [**Section 14: All About the KV Cache**](attention). How many bytes does each attention mechanism in a 2026 model store per token, and how many does it read per decoded token? How does latent attention make decode compute-bound? When does a sliding window, a top-$k$ indexer, a linear-attention state or DeepSeek-V4's token compression pay off, and what does that do to the decode and long-context training rooflines?

* [**Section 15: How Training Changed**](training-2026). What exactly runs in fp8, and which rooflines get harder because of it? Is anyone really pretraining in fp4? What does Muon cost and why does it need a whole-matrix gather? How much does multi-token prediction buy? Why is reinforcement learning bound by generation rather than gradient steps, and how do runs reach a million tokens of context?

* [**Section 16: Serving DeepSeek-V3 and Training DeepSeek-V4**](applied-frontier). Do the rooflines reproduce DeepSeek's published production numbers, and which term was missing? What changes on a GB200 NVL72? How long would a V4-class pretraining run take on 128 GB200 racks, or on a TPU7x pod, and how would you shard it on each?

Sections 13 to 15 end with takeaways and worked problems, most answered behind a click and the last few left for you. Section 16, like Sections 6 and 8, works its questions inline and leaves its closing problems unanswered.

## Where the Original Book Needs a Footnote

With two exceptions (the H800's fp8 rate in Section 4's Question 7 and its NVLink bandwidth in Section 12), nothing below is a mistake in the original; each is a place where the 2026 models or hardware move a number or qualify a claim.

* **Section 4, Question 7** computes DeepSeek-V3's utilization at 22% using an H800 figure of 1.51e15 fp8 FLOPs/s from a vendor sheet. That is the PCIe H800's rate; the SXM part DeepSeek used has H100 tensor cores at 1.98e15 dense fp8, which gives about 17% (16.5% on the book's 2.79M total hours, 17.3% on the 2.66M pretraining hours). [Section 15](training-2026) redoes it either way, and [Section 13](moe) explains why it is low.

* **Section 4's** "attention FLOPs dominate above $T = 8D$" assumes ordinary multi-head attention (MHA) and $F = 4D$. For DeepSeek-V3 the crossover is 19k tokens, against 86k ($12D$, with $D = 7168$) for a dense MHA model of the same width and $F = 4D$ by the same formula, because the MLP is sparse and there are 128 heads. [Section 14](attention) has the general formula.

* **Section 5's** constants 2,550 and 850 are TPU v5p numbers, and **Section 12's** 2,200 and 2,475 are H100 numbers; the hardware section above gives the TPU7x, HGX B200 and GB200 NVL72 values, and [Section 13](moe) uses them.

* **Section 12** gives the H800's NVLink as 300GB/s and DeepSeek-V3's batch as 4M tokens. DeepSeek's own hardware paper says the H800's NVLink was cut from 900 to 400GB/s bidirectional, 200GB/s each way in the book's convention, and DeepSeek measures about 160GB/s in practice against 50GB/s of InfiniBand; the V3 batch was 15,360 sequences of 4,096 tokens, 62.9M in all, or 30,700 per GPU. [Section 13](moe) uses 200 and 160.

* **Section 7** treats attention as always bandwidth-bound during generation. That's true for grouped-query attention (intensity equal to the group size, about 8). Multi-head latent attention with the absorption trick of Section 14, 128 heads and an fp8 cache has an intensity of 480 and, with bf16 score and value matmuls on that cache (DeepSeek's V3 recipe), is compute-bound at long context on every current GPU and TPU; with a bf16 cache, or with the 64 heads of Kimi K2 and GLM-5, it sits 14 to 22% below the roofline of the H800, the Blackwell parts and TPU7x, and above the H200's. If the attention matmuls themselves run in fp8, as SGLang's GB200 deployment in [Section 16](applied-frontier) does, the accelerator intensities double and only the H200 stays compute-bound. [Section 14](attention).

* **Section 7's** Appendix D already points at embedded drafter heads (EAGLE<d-cite key="eagle"></d-cite>, Medusa<d-cite key="medusa"></d-cite>, DeepSeek-V3's own). Multi-token prediction is that head, now trained into most 2026 MoEs (DeepSeek-V4.1-Flash instead trains its drafter afterwards), and its training cost and measured acceptance lengths are in [Section 15](training-2026).

* **Section 8** finds you need three prefill servers per decode server. DeepSeek's measured 2025 traffic needs about 1.4 decode nodes per prefill node instead, because of long reasoning outputs and a 56% prefix-cache hit rate. [Section 16](applied-frontier).

* **Section 12's** expert-parallelism roofline counts two expert matmuls, as Section 5 does to keep the algebra clean; [Section 13](moe) keeps all three, as DeepSeek's own overlap analysis in the V4 report does, which lowers the minimum expert width by 1.5x (the threshold is 1.5x more permissive than under the two-matmul count).

* **Section 12's** takeaway that expert parallelism "can span 1-2 nodes" over InfiniBand is right; what happened since is the GB200 NVL72, where the whole expert-parallel group stays inside one NVLink domain, and [Section 16](applied-frontier) shows that is worth about 2x to 4x in decode throughput per GPU at equal silicon (4.4x measured at 32-way EP and a tight per-user latency target), and about 7x against DeepSeek's H800 production figure once the GPU upgrade and an easier workload are included.

## How the Numbers Were Checked

Every model number in these sections comes from the model's `config.json` or its technical report, fetched in September 2026, and a few short scripts that ship alongside these pages (`scripts/numbers.py` and its companions in `scripts/`) recompute every number quoted in the text and draw every figure; the parameter totals they produce match the technical-report totals to within 2% for every model in the tables. Hardware numbers follow the original book's tables where they exist and the vendors' documentation where they don't. Published systems numbers (DeepSeek's serving throughput, the latencies of its DeepEP communication kernels, SGLang and vLLM benchmarks, the RL system papers) are cited to the primary write-up in each section. If you find a number that doesn't check out, say so; corrections are welcome.

A few things the labs don't publish, and we don't guess at: Kimi K3's, Qwen3.8's and Gemma 4's pretraining token counts; the training hardware, GPU-hours or cost of any 2026 DeepSeek, Kimi, Qwen or GLM model (Nemotron names GB200s only for the long-context phase of Super and Ultra, and Llama 4 gave GPU-hours, neither with a breakdown); production acceptance rates for speculative decoding; mean output lengths in production for any thinking mode. Where a number is derived rather than stated (the KV bytes of DeepSeek-V4, for instance), the text says so.

## What We Left Out

This update is about text models on the hardware people can rent. It doesn't cover the vision encoders and audio paths that most 2026 models carry (a 400M to 2.5B-parameter tower whose systems footprint is small), diffusion or image generation, on-device models below about 10B parameters (Gemma 4 E2B and E4B, Ministral), or the domestic Chinese accelerators (Huawei Ascend, Cambricon and others) on which GLM-5 is also deployed and for which the labs publish serving recipes but few roofline-grade numbers. Nor does it work out DeepSeek-V4.1-Flash's Engram, 196B parameters of hashed n-gram embedding tables fetched from host memory at every step, which adds a byte term to the decode roofline that the labs have not yet published enough about to pin down. It also doesn't cover training data, which is where a large and growing part of the actual work of building these models happens, and which the book never claimed to cover either.

<h3 markdown=1 class="next-section">Start with [Section 13](moe), on Mixture of Experts.</h3>
