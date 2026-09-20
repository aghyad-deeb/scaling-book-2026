#!/usr/bin/env python3
"""Numbers for Section 14 (attention). Bytes read per decoded token per
sequence as a function of context length S, for each attention family, plus
the MLA intensity and the attention-vs-MLP crossover. All KV bytes assume
1 byte per element (fp8/int8) unless noted; recurrent states are fp32 where
the shipped config says so (Qwen: mamba_ssm_dtype float32) and bf16 otherwise.
"""
import math

KB, MB, GB = 1e3, 1e6, 1e9


def gqa(S, L_full, K, H, L_local=0, window=0, b=1):
    """Bytes read per decoded token for a GQA model with L_full unbounded
    layers and L_local sliding-window layers."""
    per_layer_token = 2 * K * H * b
    return L_full * S * per_layer_token + L_local * min(S, window) * per_layer_token


def mla(S, L, latent=576, b=1):
    return L * S * latent * b


def dsa(S, L, latent=576, topk=2048, idx_dim=128, idx_layers=None, b=1):
    idx_layers = L if idx_layers is None else idx_layers
    return idx_layers * S * idx_dim * b + L * topk * latent * b


def hybrid(S, L_full, K, H, L_lin, state_elems, state_bytes=4, rw=2, b=1):
    """Full-attention layers with a growing cache plus linear layers whose
    state is read and written every step (rw=2)."""
    return gqa(S, L_full, K, H, b=b) + L_lin * state_elems * state_bytes * rw


def v4_pro(S, b=1):
    """DeepSeek-V4-Pro: 30 CSA layers (4:1 compression, top-1024, 128-token
    sliding branch, fp4 indexer keys of 128 dims = 64 B per compressed entry)
    and 31 HCA layers (128:1 compression), both with the 128-token sliding
    branch. Single 512-d shared KV entry whose last 64 dims are RoPE;
    stored as 448 fp8 + 64 bf16 = 576 B per entry."""
    entry = 576 * b
    csa = 30 * ((S / 4) * 64 + 1024 * entry + 128 * entry)
    hca = 31 * ((S / 128) * entry + 128 * entry)
    return csa + hca


MODELS = {
    # name: (fn, stored_bytes_per_token, fixed_bytes_per_seq)
    "LLaMA 3 70B (GQA 8x128, 80 layers)": (lambda S: gqa(S, 80, 8, 128), 2 * 8 * 128 * 80, 0),
    "Qwen3-235B (GQA 4x128, 94 layers)": (lambda S: gqa(S, 94, 4, 128), 2 * 4 * 128 * 94, 0),
    "DeepSeek-V3 / Kimi K2 (MLA, 61 layers)": (lambda S: mla(S, 61), 61 * 576, 0),
    "DeepSeek-V3.2 (MLA + DSA top-2048)": (lambda S: dsa(S, 61), 61 * (576 + 128), 0),
    "GLM-5.2 (MLA + DSA, 78 layers, 21 indexers)": (lambda S: dsa(S, 78, idx_layers=21), 78 * 576 + 21 * 128, 0),
    "gpt-oss-120b (18 full + 18 window-128, GQA 8x64)": (lambda S: gqa(S, 18, 8, 64, 18, 128), 18 * 2 * 8 * 64, 18 * 128 * 2 * 8 * 64),
    "Gemma 3 27B (10 global + 52 window-1024, GQA 16x128)": (lambda S: gqa(S, 10, 16, 128, 52, 1024), 10 * 2 * 16 * 128, 52 * 1024 * 2 * 16 * 128),
    "Gemma 4 26B-A4B (5 global K=V 2x512 + 25 window-1024 8x256)": (lambda S: 5 * S * 2 * 512 + 25 * min(S, 1024) * 2 * 8 * 256, 5 * 2 * 512, 25 * 1024 * 2 * 8 * 256),
    "Llama 4 Maverick (12 NoPE global + 36 chunked-8192, GQA 8x128)": (lambda S: gqa(S, 12, 8, 128, 36, 8192), 12 * 2 * 8 * 128, 36 * 8192 * 2 * 8 * 128),
    "Qwen3.8-2.4T (23 GQA 4x256 + 69 GDN, fp32 state)": (lambda S: hybrid(S, 23, 4, 256, 69, 128 * 128 * 128, 4), 23 * 2 * 4 * 256, 69 * 128 * 128 * 128 * 4),
    "Qwen3.5-397B (15 GQA 2x256 + 45 GDN, fp32 state)": (lambda S: hybrid(S, 15, 2, 256, 45, 64 * 128 * 128, 4), 15 * 2 * 2 * 256, 45 * 64 * 128 * 128 * 4),
    "Kimi K3 (24 MLA + 69 KDA, bf16 state)": (lambda S: mla(S, 24) + 69 * 96 * 128 * 128 * 2 * 2, 24 * 576, 69 * 96 * 128 * 128 * 2),
    "Nemotron 3 Super (8 GQA 2x128 + 40 Mamba-2, fp32 state)": (lambda S: hybrid(S, 8, 2, 128, 40, 128 * 64 * 128, 4), 8 * 2 * 2 * 128, 40 * 128 * 64 * 128 * 4),
    "DeepSeek-V4-Pro (CSA/HCA, 61 layers)": (v4_pro, None, 61 * 128 * 576),
}


def fmt(x):
    if x >= GB:
        return f"{x / GB:6.2f} GB"
    if x >= MB:
        return f"{x / MB:6.1f} MB"
    return f"{x / KB:6.1f} kB"


def main():
    print("=" * 100)
    print("BYTES READ PER DECODED TOKEN PER SEQUENCE (1 byte/elem KV; states as noted)")
    print("=" * 100)
    print(f"{'model':<62}{'stored/token':>13}{'fixed/seq':>11}{'S=8k':>11}{'S=32k':>11}{'S=128k':>11}{'S=1M':>11}")
    for name, (fn, stored, fixed) in MODELS.items():
        st = fmt(stored) if stored else "  (~4.9 kB)"
        print(f"{name:<62}{st:>13}{fmt(fixed):>11}" + "".join(f"{fmt(fn(S)):>11}" for S in (8192, 32768, 131072, 1048576)))

    print()
    print("=" * 100)
    print("MLA DECODE (absorbed): FLOPs and bytes per token per layer per context token")
    print("=" * 100)
    for name, N, dc, dr in [("DeepSeek-V3 / V3.2", 128, 512, 64), ("Kimi K2 / GLM-5 / Mistral L3", 64, 512, 64), ("Kimi K3 MLA layers", 96, 512, 64)]:
        flops = 2 * N * ((dc + dr) + dc)
        byts = dc + dr
        print(f"{name:<32} FLOPs/ctx-token = {flops:>8,}   bytes/ctx-token (fp8) = {byts}   intensity fp8 {flops / byts:5.0f}  bf16 {flops / (2 * byts):5.0f}")
    print(f"{'LLaMA 3 70B GQA (64 q, 8 kv, 128)':<32} FLOPs/ctx-token = {2 * 64 * 128 * 2:>8,}   bytes/ctx-token (int8) = {2 * 8 * 128}   intensity int8 {2 * 64 * 128 * 2 / (2 * 8 * 128):5.0f}  bf16 {2 * 64 * 128 * 2 / (2 * 2 * 8 * 128):5.0f}")

    print()
    print("=" * 100)
    print("DeepSeek-V3 decode at S=128k, per token: full MLA vs DSA")
    print("=" * 100)
    S = 131072
    f_full = 61 * 2 * 128 * 1088 * S
    b_full = 61 * S * 576
    f_dsa = 61 * (2 * 64 * 128 * S + 2 * 128 * 1088 * 2048)
    b_dsa = 61 * (S * 128 + 2048 * 576)
    print(f"full MLA : {f_full:.3g} FLOPs, {b_full / GB:.2f} GB  -> intensity {f_full / b_full:.0f}")
    print(f"DSA      : {f_dsa:.3g} FLOPs, {b_dsa / GB:.2f} GB  -> intensity {f_dsa / b_dsa:.0f}; FLOPs x{f_full / f_dsa:.1f} fewer, bytes x{b_full / b_dsa:.1f} fewer")
    for chip, C, W in [("B200", 2.25e15, 8e12), ("TPU7x", 2.3e15, 7.4e12), ("H200", 9.9e14, 4.8e12)]:
        B = 64
        print(f"  {chip}: batch 64 full-MLA attention: max(FLOPs {B * f_full / C * 1e3:.0f} ms, bytes {B * b_full / W * 1e3:.0f} ms); DSA: max({B * f_dsa / C * 1e3:.1f} ms, {B * b_dsa / W * 1e3:.1f} ms)")

    print()
    print("=" * 100)
    print("ATTENTION vs MLP FLOPs crossover in training/prefill (causal): S* = 6 (k+Es) D F L / (N (dqk+dv) L_full)")
    print("=" * 100)
    rows = [
        ("LLaMA 3 70B", 80, 80, 64, 128 + 128, 1, 0, 8192, 28672),
        ("LLaMA 3 405B", 126, 126, 128, 256, 1, 0, 16384, 53248),
        ("DeepSeek-V3", 61, 61, 128, 192 + 128, 8, 1, 7168, 2048),
        ("Kimi K2", 61, 61, 64, 320, 8, 1, 7168, 2048),
        ("GLM-5", 78, 78, 64, 256 + 256, 8, 1, 6144, 2048),
        ("Qwen3.8-2.4T (23 full of 92)", 92, 23, 64, 512, 10, 1, 8192, 2048),
        ("Kimi K3 (24 MLA of 93)", 93, 24, 96, 320, 16, 2, 3584, 3072, 16 * 3 * 3584 * 3072 + 2 * 3 * 7168 * 3072 + 2 * 7168 * 3584),
        ("Gemma 3 27B (10 global of 62)", 62, 10, 32, 256, 1, 0, 5376, 21504),
    ]
    for name, L, Lf, N, d, k, Es, D, F, *mlp_params in rows:
        # K3's shared experts run at full width and its routed path has latent projections, so pass MLP params explicitly
        mlp = (2 * mlp_params[0] if mlp_params else 6 * (k + Es) * D * F) * L
        attn_per_S = N * d * Lf
        print(f"{name:<32} S* = {mlp / attn_per_S:>9,.0f} tokens")
    print("(book's dense rule of thumb T = 8D gives", 8 * 8192, "for LLaMA 3 70B with F=4D)")

    print()
    print("=" * 100)
    print("HYBRID CROSSOVER: context S where full-layer KV traffic equals linear-state read+write traffic")
    print("=" * 100)
    for name, Lf, K, H, Ll, st, sb in [("Qwen3.8 fp32 state", 23, 4, 256, 69, 128 * 128 * 128, 4), ("Qwen3.8 bf16 state", 23, 4, 256, 69, 128 * 128 * 128, 2),
                                        ("Qwen3.5 fp32 state", 15, 2, 256, 45, 64 * 128 * 128, 4), ("Kimi K3 bf16 state", 24, 0, 0, 69, 96 * 128 * 128, 2)]:
        state = Ll * st * sb * 2
        per_tok = 2 * K * H * Lf if K else 576 * Lf
        print(f"{name:<24} state r+w/step = {state / GB:.2f} GB; per-token KV = {per_tok / KB:.1f} kB; equal at S = {state / per_tok:,.0f}")
    print("DeepSeek-V3 KV at 32k:", fmt(mla(32768, 61)))

    print()
    print("=" * 100)
    print("V4-Pro at 1M vs V3.2 DSA at 1M vs V3 MLA at 1M (bytes read per token)")
    print("=" * 100)
    S = 1048576
    print("V4-Pro", fmt(v4_pro(S)), " V3.2 DSA", fmt(dsa(S, 61)), " V3 MLA", fmt(mla(S, 61)), " ratio DSA/V4", dsa(S, 61) / v4_pro(S), " ratio MLA/V4", mla(S, 61) / v4_pro(S))
    print("V4-Pro stored per token (derived): CSA 30*(576+64)/4 + HCA 31*576/128 =", (30 * 640 / 4 + 31 * 576 / 128) / KB, "kB; V3.2 stored", 61 * 704 / KB, "kB; ratio", (30 * 640 / 4 + 31 * 576 / 128) / (61 * 704))

    print()
    print("=" * 100)
    print("SEQUENCES PER GPU AT 128k CONTEXT, after weights are spread over the group")
    print("=" * 100)
    models = [("DeepSeek-V3 fp8 (MLA)", 671e9, lambda S: mla(S, 61)), ("DeepSeek-V3.2 fp8 (stored incl. indexer)", 685e9, lambda S: 61 * 704 * S),
              ("Kimi K3 fp4 experts", 1.42e12, lambda S: 24 * 576 * S + 69 * 96 * 128 * 128 * 2), ("Qwen3.8 fp8", 2.4e12, lambda S: 23 * 2048 * S + 69 * 128 ** 3 * 4),
              ("LLaMA 3 405B int8 (GQA)", 405e9, lambda S: gqa(S, 126, 8, 128))]
    for hw, hbm, group in [("GB200 NVL72, 64 GPUs of the rack", 186e9, 64), ("TPU7x 4x4x4 cube", 192e9, 64), ("H200 node group, 144 GPUs (DeepSeek EP144)", 141e9, 144), ("H800, 144 GPUs (DeepSeek EP144, ~22GB weights incl. replicated attention)", 80e9, 144)]:
        print(f"-- {hw}")
        for name, w_total, fn in models:
            per_w = w_total / group if "H800" not in hw or "V3" not in name else 22e9   # Section 16: 22GB per H800 for V3 under EP144
            free = hbm - per_w
            per_seq = fn(131072)
            if free <= 0:
                print(f"   {name:<44} weights/GPU {per_w / GB:5.1f} GB > HBM"); continue
            print(f"   {name:<44} weights/GPU {per_w / GB:5.1f} GB, per-seq state {per_seq / GB:5.2f} GB -> {free / per_seq:6.0f} sequences per GPU, {group * free / per_seq:6.0f} per group")

if __name__ == "__main__":
    main()
