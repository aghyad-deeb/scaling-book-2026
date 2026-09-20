#!/usr/bin/env python3
"""Numbers for Section 15 (training) and Section 16 (applied)."""

GB, TB = 1e9, 1e12

print("=" * 90)
print("DeepSeek-V3 utilization, redone with both H800 FP8 figures")
print("=" * 90)
flops_total = 6 * 37e9 * 14.8e12
for name, c in [("H800 = H100 SXM fp8 dense 1.98e15", 1.98e15), ("book's Lenovo sheet 1.513e15", 1.513e15)]:
    for hours, label in [(2.664e6, "pretraining 2.664M h"), (2.788e6, "total 2.788M h")]:
        print(f"  {name:<38} {label:<22} MFU = {flops_total / (hours * 3600 * c) * 100:5.1f}%")
print(f"  per trillion tokens: 180K GPU-h -> useful/available = {(6 * 37e9 * 1e12) / (180e3 * 3600 * 1.98e15) * 100:.0f}% of 1.98e15 peak")

print()
print("=" * 90)
print("Muon: Newton-Schulz FLOPs per step for DeepSeek-V3 (5 iterations, 4mn^2 + 2n^3 each)")
print("=" * 90)
def ns_flops(m, n, iters=5):
    m, n = max(m, n), min(m, n)
    return iters * (4 * m * n * n + 2 * n ** 3)
expert = ns_flops(7168, 2048)
n_expert_mats = 58 * 257 * 3
attn = 61 * (ns_flops(7168, 1536) + ns_flops(1536, 24576) + ns_flops(7168, 576) + 2 * ns_flops(512, 16384) + ns_flops(16384, 7168))
dense = 3 * 3 * ns_flops(7168, 18432)
total_ns = expert * n_expert_mats + attn + dense
step_flops = 6 * 37e9 * 62.9e6
print(f"  expert matrix NS FLOPs {expert:.3g} x {n_expert_mats} matrices = {expert * n_expert_mats:.3g}; attention {attn:.3g}; dense {dense:.3g}")
print(f"  total NS per step {total_ns:.3g} vs training FLOPs per step {step_flops:.3g} -> {total_ns / step_flops * 100:.2f}%")
print(f"  with 10 iterations (V4 hybrid): {2 * total_ns / step_flops * 100:.2f}%")
print("  optimizer bytes/param: AdamW fp32 m,v = 8; Muon fp32 momentum = 4; bf16 weights add 2 -> 10 vs 6")
print(f"  Kimi K2 states: 1.04e12 params x (2 bf16 + 4 fp32 grad-accum) = {1.04e12 * 6 / TB:.1f} TB (report: ~6 TB over a 256-GPU model-parallel group -> {1.04e12 * 6 / 256 / GB:.0f} GB/GPU)")

print()
print("=" * 90)
print("MTP overhead for DeepSeek-V3 (one extra block + extra output head)")
print("=" * 90)
fwd_per_token = 2 * 36.5e9
block = fwd_per_token / 61
head = 2 * 7168 * 129280
print(f"  fwd FLOPs/token {fwd_per_token:.3g}; one block {block:.3g} ({block / fwd_per_token * 100:.1f}%); extra head {head:.3g} ({head / fwd_per_token * 100:.1f}%) -> total {(block + head) / fwd_per_token * 100:.1f}%")

print()
print("=" * 90)
print("RL rollout arithmetic (DeepSeek-R1-Zero style step on a V3-class model)")
print("=" * 90)
seqs = 8192           # outputs per rollout (512 questions x 16)
avg_len, max_len = 10_000, 65_536
tokens = seqs * avg_len
print(f"  {seqs} sequences x {avg_len:,} avg tokens = {tokens / 1e6:.0f}M generated tokens per rollout")
per_gpu_tps = 14.8e3 / 8
gpus = 512
print(f"  at DeepSeek's production {per_gpu_tps:.0f} tok/s/GPU on {gpus} GPUs: {tokens / (per_gpu_tps * gpus):.0f} s if perfectly batched")
train_flops = 6 * 37e9 * tokens
print(f"  training FLOPs on those tokens {train_flops:.3g} -> at 512 x 1.98e15 x 30% = {train_flops / (512 * 1.98e15 * 0.3):.0f} s")
step_ms = 50
print(f"  longest sample: {max_len:,} tokens at {step_ms} ms/step = {max_len * step_ms / 1e3 / 60:.0f} min of wall clock")
kv = tokens * 35_136
print(f"  KV cache for all rollouts at once: {kv / TB:.1f} TB fp8 = {kv / gpus / GB:.0f} GB per GPU on {gpus} GPUs")
print(f"  R1 cost: 147K H800-h vs V3 pretraining 2,664K -> {147 / 2664 * 100:.1f}%; V3.2 says RL budget > 10% of pretraining")
print(f"  MiniMax-M1 RL: 512 H800 x 3 weeks = {512 * 21 * 24 / 1e3:.0f}K GPU-h")

print()
print("=" * 90)
print("Weight sync for a 1T-parameter policy (Kimi K2, fp8 = 1.04 TB)")
print("=" * 90)
print(f"  K2 report: < 30 s; checkpoint-engine measured 16.0 s on 256 H20 -> {1.04e12 / 16 / GB:.0f} GB/s effective into every replica")
print(f"  training step budget at 1 step/min: sync = {16 / 60 * 100:.0f}% of the step; async hides it entirely")

print()
print("=" * 90)
print("FP4 rooflines on 2026 chips (dense FLOPs)")
print("=" * 90)
chips = [("GB200 NVL72", 2.5e15, 5e15, 10e15, 8e12, 900e9), ("GB300 NVL72", 2.5e15, 5e15, 15e15, 8e12, 900e9),
         ("Rubin NVL72", 4e15, 17.5e15, 35e15, 19.2e12, 1.5e12), ("MI455X Helios", 5e15, 20.1e15, 40.3e15, 21.5e12, 1.8e12), ("TPU7x", 2.3e15, 4.61e15, None, 7.4e12, 5.4e11)]
for name, bf16, fp8, fp4, hbm, egress in chips:
    parts = [f"{name:<14} a_hbm bf16 {bf16 / hbm:5.0f} fp8 {fp8 / hbm:5.0f}" + (f" fp4 {fp4 / hbm:5.0f}" if fp4 else "        ")]
    parts.append(f" | a_net bf16 {bf16 / egress:6.0f} fp8 {fp8 / egress:6.0f}" + (f" fp4 {fp4 / egress:6.0f}" if fp4 else ""))
    if fp4:
        parts.append(f" | EP min F (fp8 dispatch, fp8 FLOPs) {fp8 / egress / 2:6.0f}; (fp4 weights, fp4 FLOPs, fp8 dispatch) {fp4 / egress / 2:6.0f}")
    print("".join(parts))
print("  (TPU7x egress = 6 links x 9e10 one-way; per-axis alpha used in the chapters is C / 1.8e11)")

print()
print("=" * 90)
print("Section 16: DeepSeek-V3/R1 production serving, Feb 2025 (H800, EP144 decode, EP32 prefill)")
print("=" * 90)
W = 671e9
print(f"  weights per GPU under EP144: routed experts {58 * 256 * 3 * 7168 * 2048 / 144 / GB:.1f} GB (2 routed experts/GPU x 58 layers = {2 * 3 * 7168 * 2048 * 58 / GB:.1f} GB) + attention+shared+dense replicated {(11.4e9 + 58 * 3 * 7168 * 2048 + 1.2e9 + 1.85e9) / GB:.1f} GB")
per_gpu_w = 2 * 3 * 7168 * 2048 * 58 + (11.4e9 + 58 * 3 * 7168 * 2048 + 1.2e9 + 1.85e9)
print(f"  -> about {per_gpu_w / GB:.0f} GB fp8 per GPU, leaving {(80e9 - per_gpu_w) / GB:.0f} GB of 80 GB for KV cache and activations")
kv_tok = 35_136
avg_ctx = 4989
free = 80e9 - per_gpu_w - 10e9
print(f"  average KV length per output token 4,989 -> {avg_ctx * kv_tok / 1e6:.0f} MB per sequence -> ~{free / (avg_ctx * kv_tok):.0f} sequences per GPU fit in {free / GB:.0f} GB")
tps_gpu = 14.8e3 / 8
print(f"  14.8k out tok/s per node = {tps_gpu:.0f} tok/s/GPU; at 20-22 tok/s per user -> {tps_gpu / 21:.0f} concurrent sequences per GPU (before MTP)")
step = 1 / 21
print(f"  step time ~{step * 1e3:.0f} ms. Bytes per step at 88 seqs: weights {per_gpu_w / GB:.1f} GB + KV {88 * avg_ctx * kv_tok / GB:.1f} GB = {(per_gpu_w + 88 * avg_ctx * kv_tok) / GB:.1f} GB -> {(per_gpu_w + 88 * avg_ctx * kv_tok) / 3.35e12 * 1e3:.1f} ms at 3.35 TB/s")
disp, comb = 88 * 8 * 7168 * 1, 88 * 8 * 7168 * 2
print(f"  AllToAll bytes per GPU per layer at 88 tokens: dispatch {disp / 1e6:.1f} MB fp8 + combine {comb / 1e6:.1f} MB bf16 = {(disp + comb) / 1e6:.1f} MB; at DeepEP's ~40 GB/s: {(disp + comb) / 40e9 * 1e3:.2f} ms/layer x 58 = {(disp + comb) / 40e9 * 58 * 1e3:.1f} ms")
print(f"  (DeepEP low-latency table: 128 tokens -> 7.3 MB / 192 us = {128 * 8 * 7168 / 192e-6 / 1e9:.0f} GB/s dispatch, 14.7 MB / 365 us = {128 * 8 * 7168 * 2 / 365e-6 / 1e9:.0f} GB/s combine: NIC-bandwidth-bound)")
print(f"  prefill attention (unabsorbed, causal, S=4989): {61 * 2 * 128 * 320 * 4989 / 2:.3g} FLOPs/token vs matmuls {2 * 37e9:.3g} -> {61 * 2 * 128 * 320 * 4989 / 2 / (2 * 37e9) * 100:.0f}%")
print(f"  attention FLOPs at 5k ctx, 88 seqs: {88 * 61 * 278528 * avg_ctx / 1e12:.2f} TFLOPs -> {88 * 61 * 278528 * avg_ctx / 1.98e15 * 1e3:.1f} ms at fp8 peak (bf16 core: {88 * 61 * 278528 * avg_ctx / 0.99e15 * 1e3:.1f} ms)")
print(f"  MoE FLOPs: 88 x 2 x 37e9 = {88 * 2 * 37e9 / 1e12:.1f} TFLOPs -> {88 * 2 * 37e9 / 1.98e15 * 1e3:.1f} ms")
print(f"  cost: 8 GPUs x $2/h / 14.8k tok/s = ${8 * 2 / (14.8e3 * 3600) * 1e6:.2f} per 1M output tokens; R1 price $2.19; day: $87,072 cost vs $562,027 revenue")
print(f"  prefill: 73.7k tok/s/node -> {73.7e3 / 8:.0f} tok/s/GPU; FLOPs per token 2 x 37e9 -> {73.7e3 / 8 * 2 * 37e9 / 1.98e15 * 100:.0f}% of fp8 peak (incl. cache hits, so real MFU on misses is higher)")
inp, hit, out = 608e9, 342e9, 168e9
print(f"  daily: {inp / 1e9:.0f}B in ({hit / inp * 100:.1f}% cache hits -> {(inp - hit) / 1e9:.0f}B prefilled), {out / 1e9:.0f}B out; out/in = {out / inp:.2f}")
pre_node_s = inp / 73.7e3      # the 73.7k figure already includes cache hits
dec_node_s = out / 14.8e3
print(f"  node-time: prefill {pre_node_s / 86400:.0f} node-days, decode {dec_node_s / 86400:.0f} node-days -> decode:prefill = {dec_node_s / pre_node_s:.1f}:1 (Section 8 found 1:3 for 8k-in/512-out)")
print(f"  check: {(pre_node_s + dec_node_s) / 86400:.1f} node-days of work vs 226.75 average nodes in service: the throughput figures reproduce the fleet size")

print()
print("=" * 90)
print("Section 16: GB200 NVL72 vs H800 for DeepSeek-class decode")
print("=" * 90)
print(f"  SGLang GB200: 13,386 out tok/s/GPU (fp8 attn + nvfp4 MoE, 2k in) vs DeepSeek H800 production {tps_gpu:.0f} -> {13386 / tps_gpu:.1f}x per GPU")
print(f"  SGLang 96xH100 (EP72 decode): 22.3k/node = {22.3e3 / 8:.0f} tok/s/GPU; InferenceX GB200 at 125 tok/s/user: 4,130 tok/s/GPU vs B200 8-GPU node 941")
rep = 11.4e9 + 58 * 3 * 7168 * 2048 + 1.2e9 + 1.85e9
print(f"  weights per GPU on NVL72 with EP64: experts {656e9 / 64 / GB:.1f} GB fp8 or {656e9 * 0.5 / 64 / GB:.1f} GB fp4, plus replicated {rep / GB:.0f} GB fp8 -> {(656e9 / 64 + rep) / GB:.0f} / {(656e9 * 0.5 / 64 + rep) / GB:.0f} GB; AllToAll 15.1 MB at 726 GB/s = {15.1e6 / 726e9 * 1e6:.0f} us/layer -> {15.1e6 / 726e9 * 58 * 1e3:.1f} ms/step")

print()
print("=" * 90)
print("Section 16: pretraining DeepSeek-V4-Pro-class model on a TPU7x pod")
print("=" * 90)
F = 6 * 49e9 * 33e12
for mfu in (0.3, 0.4, 0.5):
    print(f"  {F:.2g} FLOPs on 9216 x 4.61e15 fp8 at {mfu:.0%} MFU: {F / (9216 * 4.61e15 * mfu) / 86400:.1f} days; bf16 peak: {F / (9216 * 2.3e15 * mfu) / 86400:.1f} days")
print(f"  batch 94.4M tokens / 9216 chips = {94.4e6 / 9216:,.0f} tokens/chip; FSDP needs (E/(kZ)) x alpha/3: Z=64, fp8 FLOPs -> {(384 / 6) / 64 * 25600 / 3:,.0f} (bf16 gather) / {(384 / 6) / 64 * 12800 / 3:,.0f} (fp8 gather)")
print(f"  DeepSeek-V3 for comparison on 8960 v5p: {3.29e24 / (8960 * 9.18e14 * 0.4) / 86400:.0f} days at 40% int8/fp8 MFU; they took 2.664M GPU-h / 2048 = {2.664e6 / 2048 / 24:.0f} days on 2048 H800")
print(f"  RL budget at 10% of pretraining = {0.1 * F:.2g} FLOPs; each generated token costs 2N (generate) + 6N (train) = 8N -> {0.1 * F / (8 * 49e9) / 1e12:.1f}T generated tokens = {0.1 * F / (8 * 49e9) / 33e12 * 100:.0f}% of the corpus; at 1e4 tok/s/chip: {0.1 * F / (8 * 49e9) / (1e4 * 9216) / 3600:.1f} hours")
