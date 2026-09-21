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

_A quick note before we start: this is an independent supplement to the book, so the original authors didn't write or review it. We're trying to do the same thing they do, though: take a handful of hardware constants, count FLOPs and bytes, and predict how a model will run before you run it, just for the models of 2026 instead of LLaMA 3. We've numbered these four sections 13 to 16 so they slot in after the GPU section, and we'll assume you've read [Sections 4](https://jax-ml.github.io/scaling-book/transformers), [5](https://jax-ml.github.io/scaling-book/training) and [7](https://jax-ml.github.io/scaling-book/inference) of the original (if you haven't, go do that first!)._

## What Changed, on One Page

Let's start with the model the rest of this book keeps coming back to, LLaMA 3-70B. It's a dense model with 70B parameters and 8 KV heads. It was trained on 15T tokens in bf16 with AdamW, which took 6.4M H100-hours (its 405B sibling ran on 16,000 H100s), and it was served on TPU v5e with tensor parallelism. How many of those facts still hold for a frontier open model in September 2026, then? Almost none of them, as it turns out! Let's go through what changed, one item at a time, and say where we work each one out.

* **More or less every frontier open model is now a Mixture of Experts, and a very sparse one.** To give you a sense of how sparse: Kimi K3 has 2.8T parameters but only activates 104B of them per token, DeepSeek-V4-Pro has 1.6T and activates 49B, and Qwen3.8 has 2.4T and activates 95B. Each has hundreds of narrow experts and activates only 6 to 16 of them per token. Put another way, total parameters have grown about 20x since Mixtral while active parameters have only grown about 2.5x. Why does this matter for us? Every roofline that compares weight bytes to FLOPs inflates by the sparsity $E/k$ (the ratio of total to active experts), and [Section 13](moe) redoes the Transformer math with that in mind.

* **The KV cache per token is about 30x smaller.** If you look at what the new architectures actually changed, most of it is aimed at the KV cache: latent attention (MLA), sliding windows, top-$k$ sparse attention, linear-attention layers with a fixed state, and outright token compression each make it smaller. How much smaller? DeepSeek-V4 stores about 5kB per token where LLaMA 3-70B stored 164kB at the same one byte per element (328kB in bf16), and V4.1-Flash, with an fp4 cache, keeps only 890 bytes per token of HBM-resident cache. One side effect worth knowing about: with MLA, decode attention can now be compute-bound rather than bandwidth-bound, which (more or less) never used to happen. [Section 14](attention) works out the bytes and FLOPs for each mechanism.

* **Matmuls now run in fp8, and the weights you download are often 4-bit.** DeepSeek's fp8 recipe (E4M3, 1x128 and 128x128 scales, fp32 accumulation) is now standard. fp4 pretraining works but is rare, while fp8 pretraining followed by quantization-aware post-training to 4-bit experts is common. Every roofline whose byte term didn't halve just got 2x harder. [Section 15](training-2026) says which ones.

* **Muon has replaced Adam** for most of the open MoE trainers. It orthogonalizes each weight matrix's momentum before applying it, at a cost of 0.2 to 0.5% of FLOPs and half the optimizer memory, plus a whole-matrix gather once per step. [Section 15](training-2026) counts the cost.

* **Models now predict two tokens at a time, and the second head doubles as a speculative drafter.** This costs about 4% more training FLOPs and buys us 1.6 to 1.8x in tokens per second at small batch. Also in [Section 15](training-2026).

* **Post-training is now about a tenth of pretraining compute, and it's bound by generation.** DeepSeek reports that its post-training budget (most of it reinforcement learning) passed 10% of pretraining compute by V3.2, and that most of its wall-clock goes to generation. Why would generation take most of the time when it's only about a quarter of the FLOPs? A few reasons: responses run 10k to 65k tokens with a long straggler tail (so the whole batch sits around waiting for the longest one), a trillion-parameter policy has to be re-synchronized to the samplers in seconds, and the training and inference engines have to agree numerically. It turns out these constraints reach all the way back into the architecture, which we'll see in [Section 15](training-2026).

* **Serving now spends more time decoding than prefilling.** DeepSeek's published production system spends about 1.4 node-days decoding for every node-day of prefill, the inverse of the chat-era ratio we saw in [Section 8](https://jax-ml.github.io/scaling-book/applied-inference). Half of its decode step goes to the expert AllToAll over InfiniBand, a term that Section 7's decode roofline simply doesn't have. [Section 16](applied-frontier) checks the arithmetic.

* **The hardware has changed too, but some parts of it more than others.** B200-class GPUs and TPU7x put 180 to 192GB of HBM on a chip, GB300 and Rubin put 288GB, and the GB200 NVL72 puts 72 GPUs in one NVLink domain, which means a trillion-parameter model now fits in a single fast fabric (something you couldn't say about an 8-GPU node!). The network between nodes is a different story, though. NVLink grew along with GPU FLOPs, but the InfiniBand NIC and TPU ICI didn't, so any roofline that depends on them is harder to satisfy than it used to be. We'll go through the actual constants in the next section.

## The Hardware That Runs 2026 Models

You'll remember that every roofline in this book comes down to just four numbers per accelerator: how many FLOPs/s it can do, how fast its HBM is, how fast its scale-up network is (NVLink or ICI) and how fast its scale-out network is (InfiniBand or DCN). Let's collect them for the hardware the 2026 models actually run on. We'll do GPUs first, since that is where nearly all of these models were trained and are served. While we're at it, we'll also write down the two operational intensities that show up in every derivation, $C / W_\text{hbm}$ (i.e. the decode critical batch) and $\alpha = C / W_\text{net}$ over the scale-up network (per GPU, or per ICI axis), so you don't have to keep working them out yourself.

**A note on units:** FLOPs are dense PFLOPs/s (no structured sparsity) in bf16 unless the column says otherwise, and HBM is in GB and TB/s. The scale-up and scale-out columns are one-way GB/s per GPU (or per TPU chip, with six or four ICI links), as elsewhere in this book, so we've halved NVIDIA's bidirectional NVLink figures. Rack-level figures are per GPU for the NVL72 systems, and $\alpha$ is per GPU on the GPU rows and per ICI axis on the TPU rows.<d-footnote>Sources: Google Cloud TPU documentation for v5p, v6e and TPU7x, NVIDIA product pages for H100/H200, HGX B200, GB200 NVL72, GB300 NVL72 and Vera Rubin NVL72 (which quote sparse figures by default, so we halve them), and AMD product pages for MI355X and MI455X. TPU ICI figures follow this book's original tables (9e10 bytes/s one-way per link, i.e. Google's 200 GB/s bidirectional per axis rounded down), and GB200 per-GPU figures are the rack totals divided by 72. Rubin is "in full production" per NVIDIA's August 2026 earnings but not broadly available yet, and its product page and launch blog disagree on NVLink 6 (3.0 versus 3.6 TB/s) and HBM4 bandwidth (19.2 versus 22 TB/s), so we use the product page. AMD's MI455X page and its Helios page disagree on HBM bandwidth (23.3 versus 19.6 TB/s), so we use the product page's 23.3 (intensity 215 instead of the 255 we'd get at 19.6). AMD gives the MI455X's UALink scale-out as 600GB/s bidirectional per GPU, i.e. 300GB/s one-way, and names 800Gb/s NICs for the Helios rack without saying how many per GPU. MI355X systems ship with OEM-chosen 400 or 800Gb/s NICs, so we assume 400Gb/s (50GB/s) per GPU, as in the NVIDIA 8-GPU nodes. Rubin's scale-out is quoted as 0.45TB/s bidirectional on the product page against a 1.6Tb/s ConnectX-9 line rate, and we use the line rate, 200GB/s one-way. AMD's MI355X page gives 153 GB/s per Infinity Fabric link without stating a direction, so we assume bidirectional (as AMD states for the MI455X) and halve the seven-link total. GB300 and Rubin HBM use the per-GPU specification (288 GB) rather than the rounded rack total, and where a vendor page labels a figure dense (as Rubin's does for its training columns) we use it as given. Finally, TPU 8t and 8i were announced in April 2026 with FP4 figures only (12.6 and 10.1 PFLOPs, 216 and 288 GB, 6.5 and 8.6 TB/s, and twice TPU7x's ICI), and since Google hasn't published their bf16 or fp8 rates we leave them out.</d-footnote>

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

Two of the columns need a word of explanation. **Domain** is the number of accelerators in one scale-up fabric (an NVLink domain or an ICI pod). **Scale-out** is what one GPU (or one TPU chip's share of its host NIC) can push into the InfiniBand or data-center network, one way, with the rack total where a whole rack shares one fabric. You might also wonder why a few of these rows are here at all. The H800 is the GPU that DeepSeek-V3 and Kimi K2 trained on, the GB200 is what Nemotron 3 Super and Ultra name for their long-context phase (their main phase doesn't name a GPU), and v6e is Gemma 4's chip.

Let's look at the last three columns, since a few things stand out.

**Has the HBM arithmetic intensity changed?** In bf16 it has barely moved. From TPU v5p to TPU7x, and from H100 to Rubin, the bf16 number stays roughly between 160 and 320 (TPU v6e, with 1.6TB/s of HBM bandwidth for 0.92 PFLOPs/s, is the exception at `0.92e15 / 1.6e12 = 575`). Each generation's FLOPs and its HBM bandwidth grew at about the same rate, so [Section 7](https://jax-ml.github.io/scaling-book/inference)'s "batch 240 to be compute-bound" rule still holds in bf16. It does move once our matmuls run in fp8 or fp4, though: the chip's intensity doubles in fp8 and quadruples or more in fp4 (1,250 on GB200, and 1,875 on GB300, whose fp4 rate is six times its bf16 rate), so Section 7's $B_\text{crit}$ doubles with it unless our weights shrink in step. Put another way, fp8 weights with fp8 FLOPs land us back at a batch of 240 to 300. With bf16 weights and fp8 FLOPs we need twice the batch.

**What about the network intensities on GPUs?** Inside the node, you'll find that not much has changed, since NVLink doubled along with each generation's FLOPs. That leaves $\alpha$ inside the domain between 2,200 and 2,800 from H100 to GB200, so [Section 12](https://jax-ml.github.io/scaling-book/gpus)'s tensor-parallel and in-node rooflines carry over unchanged in bf16 ([Section 15](training-2026) shows what fp8 does to them). Across nodes, though, we're in worse shape, since the InfiniBand NIC stayed at 400Gb/s per GPU from H100 through B200 and GB200 (GB300 doubles it to 800Gb/s). So the cross-node intensity went from `990e12 / 50e9 = 19,800` on an H100 to 45,000 on a B200, and [Section 12](https://jax-ml.github.io/scaling-book/gpus)'s cross-node threshold for fully sharded data parallelism (FSDP, 2,475 tokens per GPU on H100 nodes) becomes 5,625 on 8-GPU HGX B200 nodes. That's a lot of tokens per GPU! How does the GB200 NVL72 get around this? With its bigger domain: 72 GPUs share one NVLink fabric, so a rack only has to pull each weight shard in once, through 3.6TB/s of egress, and that brings the threshold down to 694.

**Note [the H800]:** the H800 is a bit unusual, since NVIDIA cut its NVLink to 200GB/s each way. That makes DeepSeek's in-node intensity 4,950, worse than an H100's 2,200 and (as we'll see in [Section 13](moe)) a real constraint on the expert AllToAll.

**What about TPUs?** Here the scale-up intensity really did move, because TPU7x has five times the FLOPs of v5p but exactly the same ICI per link. That puts its per-axis intensity at 12,800 against v5p's 2,550 (`2.3e15 / 1.8e11 = 12,800`), so every ICI roofline got five times harder. You'll want to remember that [Section 5](https://jax-ml.github.io/scaling-book/training)'s constants (FSDP compute-bound above 850 tokens per chip, tensor parallelism up to $3F/2550$) are v5p constants, and that on Ironwood they become 4,270 tokens and $3F/12800$.

**Where did all the new FLOPs go?** Look down the fp4 column: GB300 can do 6x its bf16 rate in fp4, Rubin nearly 9x and MI455X 8x, while nothing in the memory or network columns grew anything like that. What does that tell us? That the vendors expect these chips to spend most of their lives serving in fp4, and that bf16 training is the workload that has to fit around it. (This is worth keeping in mind whenever a headline FLOPs number looks too good to be true: check which column it came from.)

## The Four New Sections

* [**Section 13: How to Think About Mixture of Experts**](moe). How many parameters, FLOPs and bytes does a 2.8T-parameter MoE really have? Why do sparse models want thousands of tokens per GPU to be compute-bound? How much expert parallelism can we do before NVLink, InfiniBand or ICI runs out, how does it combine with FSDP, and what do DeepSeek, Kimi, NVIDIA and Google actually do?

* [**Section 14: All About the KV Cache**](attention). How many bytes does each attention mechanism in a 2026 model store per token, and how many does it read per decoded token? How does latent attention make decode compute-bound? When does a sliding window, a top-$k$ indexer, a linear-attention state or DeepSeek-V4's token compression pay off, and what does each of them do to the decode and long-context training rooflines?

* [**Section 15: How Training Changed**](training-2026). What exactly runs in fp8, and which rooflines get harder because of it? Is anyone really pretraining in fp4? What does Muon cost us, and why does it need a whole-matrix gather? How much does multi-token prediction buy? Why is reinforcement learning bound by generation rather than gradient steps, and how do runs reach a million tokens of context?

* [**Section 16: Serving DeepSeek-V3 and Training DeepSeek-V4**](applied-frontier). Do our rooflines reproduce DeepSeek's published production numbers, and which term was missing? What changes on a GB200 NVL72? How long would a V4-class pretraining run take on 128 GB200 racks, or on a TPU7x pod, and how would we shard it on each?

Sections 13 to 15 end with takeaways and worked problems, most answered behind a click and the last few left for you. Section 16, like Sections 6 and 8, works its questions inline and leaves its closing problems for you as well. Try grabbing a pen before you open an answer.

## Where the Original Book Needs a Footnote

**Is anything in the original actually wrong?** Not really! With two exceptions (the H800's fp8 rate in Section 4's Question 7 and its NVLink bandwidth in Section 12), nothing below is a mistake. Each item is just a place where the 2026 models or hardware move a number, or where a claim now needs a caveat that it didn't use to need.

* **Section 4, Question 7** computes DeepSeek-V3's utilization at 22% using an H800 figure of 1.51e15 fp8 FLOPs/s from a vendor sheet. It turns out that is the PCIe H800's rate. The SXM part DeepSeek actually used has H100 tensor cores at 1.98e15 dense fp8, which gives about 17% instead (16.5% on the 2.79M total hours the original question uses, or 17.3% on the 2.66M pretraining hours). [Section 15](training-2026) redoes the calculation either way, and [Section 13](moe) explains why it's so low.

* **Section 4's** rule that "attention FLOPs dominate above $T = 8D$" assumes ordinary multi-head attention (MHA) and $F = 4D$. For DeepSeek-V3, the crossover is at 19k tokens instead of the 86k ($12D$, with $D = 7168$) that the same formula gives for a dense MHA model of the same width with $F = 4D$, since the MLP is sparse and there are 128 heads. [Section 14](attention) gives the general formula.

* **Section 5's** constants 2,550 and 850 are TPU v5p numbers, and **Section 12's** 2,200 and 2,475 are H100 numbers. The hardware table above gives you the TPU7x, HGX B200 and GB200 NVL72 values, and [Section 13](moe) uses them.

* **Section 12** gives the H800's NVLink as 300GB/s and DeepSeek-V3's batch as 4M tokens. DeepSeek's own hardware paper says the H800's NVLink was cut from 900 to 400GB/s bidirectional (i.e. 200GB/s each way in the one-way convention we use throughout this book), and DeepSeek measures about 160GB/s in practice, against 50GB/s of InfiniBand. The V3 batch was actually 15,360 sequences of 4,096 tokens, or 62.9M tokens in all (30,700 per GPU). [Section 13](moe) uses 200 and 160.

* **Section 7** treats attention as always being bandwidth-bound during generation. That's true for grouped-query attention (its intensity is the group size, about 8), but multi-head latent attention with the absorption trick of Section 14, 128 heads and an fp8 cache has an intensity of 480 and, with bf16 score and value matmuls on that cache (DeepSeek's V3 recipe), is compute-bound at long context on every current GPU and TPU. With a bf16 cache, or with the 64 heads of Kimi K2 and GLM-5, it sits 14 to 22% below the roofline of the H800, the Blackwell parts and TPU7x (and above the H200's). If the attention matmuls themselves run in fp8, as SGLang's GB200 deployment in [Section 16](applied-frontier) does, the accelerator intensities double and only the H200 stays compute-bound. [Section 14](attention) works it all through.

* **Section 7's** Appendix D already points at embedded drafter heads (EAGLE<d-cite key="eagle"></d-cite>, Medusa<d-cite key="medusa"></d-cite>, and DeepSeek-V3's own). Multi-token prediction is basically that same head, now trained into most 2026 MoEs (DeepSeek-V4.1-Flash instead trains its drafter afterwards). You'll find its training cost and measured acceptance lengths in [Section 15](training-2026).

* **Section 8** finds that you need three prefill servers per decode server. DeepSeek's measured 2025 traffic needs about 1.4 decode nodes per prefill node instead, thanks to long reasoning outputs and a 56% prefix-cache hit rate. [Section 16](applied-frontier) works out why.

* **Section 12's** expert-parallelism roofline counts two expert matmuls, as Section 5 does to keep the algebra clean. [Section 13](moe) keeps all three (as DeepSeek's own overlap analysis in the V4 report does), which lowers the minimum expert width by 1.5x, i.e. the threshold is 1.5x more permissive than under the two-matmul count.

* **Section 12's** takeaway that expert parallelism "can span 1-2 nodes" over InfiniBand is still right, as long as you're on InfiniBand. On a GB200 NVL72, though, the whole expert-parallel group can stay inside one NVLink domain and never touch InfiniBand at all. [Section 16](applied-frontier) shows that this is worth about 2x to 4x in decode throughput per GPU at equal silicon (4.4x was measured at 32-way EP with a tight per-user latency target), and about 7x against DeepSeek's H800 production figure once we include the GPU upgrade and an easier workload.

## How the Numbers Were Checked

**Where do the numbers come from?** Every model number in these sections comes from the model's `config.json` or its technical report, fetched in September 2026, and we've also written a few short scripts that ship alongside these pages (`scripts/numbers.py` and its companions in `scripts/`) which recompute every number quoted in the text and draw every figure, so if you want to check something you can run them yourself. As a sanity check, the parameter totals they produce match the technical-report totals to within 2% for every model in the tables. Hardware numbers follow this book's original tables where they exist and the vendors' documentation where they don't. Published systems numbers (e.g. DeepSeek's serving throughput, the latencies of its DeepEP communication kernels, SGLang and vLLM benchmarks, and the RL system papers) are cited to the primary write-up in each section. If you find a number that doesn't check out, please say so. Corrections are very welcome.

**What don't we know?** Quite a bit, unfortunately, and where a report doesn't give a number we'd rather leave a blank than guess. Nobody has published pretraining token counts for Kimi K3, Qwen3.8 or Gemma 4. Nor do we have the training hardware, GPU-hours or cost of any 2026 DeepSeek, Kimi, Qwen or GLM model (Nemotron names GB200s only for the long-context phase of Super and Ultra, and Llama 4 gave GPU-hours, but neither gives a breakdown). Production acceptance rates for speculative decoding and mean output lengths in production for any thinking mode are missing too. Whenever a number is derived rather than stated (the KV bytes of DeepSeek-V4, for instance), we'll say so in the text so you can check the derivation yourself.

## What We Left Out

**What's not in here?** We've stuck to text models running on hardware you can rent, which leaves a few things out. We don't cover the vision encoders and audio paths that most 2026 models carry (a 400M to 2.5B-parameter tower whose systems footprint is fairly small), diffusion or image generation, on-device models below about 10B parameters (e.g. Gemma 4 E2B and E4B, Ministral), or the domestic Chinese accelerators (Huawei Ascend, Cambricon and others) that GLM-5 is also deployed on, for which there are published serving recipes but few roofline-grade numbers. We also don't work out DeepSeek-V4.1-Flash's Engram, 196B parameters of hashed n-gram embedding tables fetched from host memory at every step. That adds a byte term to the decode roofline that we'd love to work through, but it hasn't been documented well enough for us to pin down yet. And we don't cover training data at all, which is where a large and growing part of the work of building these models happens (and which this book never claimed to cover either).

<h3 markdown=1 class="next-section">That's it for the overview! For [Section 13](moe), on Mixture of Experts, [click here](moe).</h3>
