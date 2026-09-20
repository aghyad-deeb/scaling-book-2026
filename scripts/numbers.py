#!/usr/bin/env python3
"""Every number quoted in the update chapters is computed here.

Run `python3 scripts/numbers.py` to print the full report. Model configs are
transcribed from each model's config.json / tech report (see the URL next to
each entry). Hardware numbers follow the tables in the original book
(tpus.md, gpus.md) so that the two sets of chapters agree with each other.

Notation follows the book: B is batch in tokens, D = d_model, F = expert (or
dense) hidden width, L layers, N query heads, K KV heads, H head dim,
E routed experts, k experts per token, C = FLOPs/s, W = bandwidth.
"""
from dataclasses import dataclass, field
from typing import Optional
import math

# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------

@dataclass
class Chip:
    name: str
    hbm: float                 # bytes
    hbm_bw: float              # bytes/s
    c_bf16: float              # FLOPs/s
    c_fp8: float               # FLOPs/s (int8 on older TPUs)
    ici_bidi: float            # one-way bytes/s per ICI link (TPU, the book's 9e10/1.8e11) or NVLink egress per GPU (GPU)
    axes: int                  # number of torus axes (TPU); 1 for GPU
    node_egress: Optional[float] = None   # scale-out egress per chip (GPU: IB per GPU, one-way)
    c_fp4: Optional[float] = None
    domain: int = 8            # accelerators sharing the scale-up fabric (8-GPU node, 72-GPU NVL72 rack)
    measured_egress: Optional[float] = None  # achieved NVLink egress where a lab published it (H800: DeepSeek's 160 GB/s)

    @property
    def collective_bw_cross_node(self):
        """W_collective for FSDP across nodes/racks: the whole domain's scale-out egress (Section 12)."""
        return self.node_egress * self.domain

    @property
    def alpha_hbm_bf16(self):
        return self.c_bf16 / self.hbm_bw

    @property
    def alpha_hbm_fp8(self):
        return self.c_fp8 / self.hbm_bw

    @property
    def alpha_ici(self):
        """C / W_ici per axis in bf16, the book's 'ICI operational intensity'."""
        return self.c_bf16 / self.ici_bidi


# Numbers from tpus.md / gpus.md in the book.
V5E = Chip("TPU v5e", 16e9, 8.2e11, 1.97e14, 3.94e14, 9e10, 2)
V5P = Chip("TPU v5p", 96e9, 2.8e12, 4.59e14, 9.18e14, 1.8e11, 3)
V6E = Chip("TPU v6e", 32e9, 1.6e12, 9.20e14, 1.84e15, 1.8e11, 2)
TPU7X = Chip("TPU7x (Ironwood)", 192e9, 7.4e12, 2.30e15, 4.61e15, 1.8e11, 3)
# GPUs: NVLink egress is one-way (NVIDIA quotes bidirectional; halved). H800 = H100 tensor cores and HBM with NVLink cut to 400 GB/s bidirectional.
H800 = Chip("H800", 80e9, 3.35e12, 9.9e14, 1.98e15, 2.0e11, 1, node_egress=5e10, measured_egress=1.6e11)
H100 = Chip("H100", 80e9, 3.35e12, 9.9e14, 1.98e15, 4.5e11, 1, node_egress=5e10)
H200 = Chip("H200", 141e9, 4.8e12, 9.9e14, 1.98e15, 4.5e11, 1, node_egress=5e10)
B200 = Chip("B200 (HGX)", 180e9, 8.0e12, 2.25e15, 4.5e15, 9e11, 1, node_egress=5e10, c_fp4=9e15)
GB200 = Chip("GB200 NVL72", 186e9, 8.0e12, 2.5e15, 5e15, 9e11, 1, node_egress=5e10, c_fp4=10e15, domain=72)
GB300 = Chip("GB300 NVL72", 288e9, 8.0e12, 2.5e15, 5e15, 9e11, 1, node_egress=1e11, c_fp4=15e15, domain=72)
CHIPS = [H800, H100, H200, B200, GB200, GB300, TPU7X, V5P, V6E, V5E]
GPUS = [H800, H100, H200, B200, GB200, GB300]
TPUS = [TPU7X, V5P, V6E, V5E]

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

@dataclass
class Attention:
    kind: str                  # "gqa", "mla", "hybrid", "sliding-gqa"
    n_heads: int
    kv_heads: int = 0
    head_dim: int = 128
    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope: int = 0
    qk_rope: int = 0
    v_head: int = 0
    # sliding window mixes: fraction of layers that are local, and the window
    local_frac: float = 0.0
    window: int = 0
    # hybrid linear attention: fraction of layers that are linear, state per layer in bytes(bf16)
    linear_frac: float = 0.0
    linear_state_elems: int = 0

@dataclass
class Model:
    name: str
    src: str
    D: int
    L: int
    V: int
    attn: Attention
    F_dense: int = 0           # dense MLP width (used on dense layers)
    dense_layers: int = 0      # number of dense layers (rest are MoE); L if fully dense
    E: int = 0                 # routed experts
    k: int = 0                 # experts per token
    shared: int = 0            # shared experts (always on)
    F_e: int = 0               # expert width
    tied_embeddings: bool = False
    mtp_depth: int = 0
    tokens_trained: float = 0.0
    reported_total: float = 0.0
    reported_active: float = 0.0
    notes: str = ""

    # --- parameter counting -------------------------------------------------
    def attn_params_per_layer(self):
        a = self.attn
        D = self.D
        if a.kind == "mla":
            qk = a.qk_nope + a.qk_rope
            wq = D * a.q_lora_rank + a.q_lora_rank * a.n_heads * qk if a.q_lora_rank else D * a.n_heads * qk
            wdkv = D * (a.kv_lora_rank + a.qk_rope)
            wuk = a.kv_lora_rank * a.n_heads * a.qk_nope
            wuv = a.kv_lora_rank * a.n_heads * a.v_head
            wo = a.n_heads * a.v_head * D
            return wq + wdkv + wuk + wuv + wo
        # GQA / MHA (also used for the full-attention layers of hybrids)
        return D * a.n_heads * a.head_dim * 2 + D * a.kv_heads * a.head_dim * 2

    def moe_layers(self):
        return self.L - self.dense_layers if self.E else 0

    def mlp_params_total(self):
        dense = self.dense_layers * 3 * self.D * self.F_dense if self.E else self.L * 3 * self.D * self.F_dense
        moe = self.moe_layers() * (self.E + self.shared) * 3 * self.D * self.F_e
        return dense + moe

    def mlp_params_active(self):
        dense = self.dense_layers * 3 * self.D * self.F_dense if self.E else self.L * 3 * self.D * self.F_dense
        moe = self.moe_layers() * (self.k + self.shared) * 3 * self.D * self.F_e
        return dense + moe

    def embed_params(self):
        return (1 if self.tied_embeddings else 2) * self.V * self.D

    def total_params(self):
        return self.L * self.attn_params_per_layer() + self.mlp_params_total() + self.embed_params()

    def active_params(self):
        return self.L * self.attn_params_per_layer() + self.mlp_params_active() + self.embed_params()

    # --- KV cache ------------------------------------------------------------
    def kv_elems_per_token(self, S=None):
        """Elements of KV cache per token, averaged over layers. For sliding
        window models this depends on S (the sequence length) because the
        local layers store at most `window` tokens."""
        a = self.attn
        if a.kind == "mla":
            return self.L * (a.kv_lora_rank + a.qk_rope)
        per_full_layer = 2 * a.kv_heads * a.head_dim
        n_full = self.L * (1 - a.local_frac - a.linear_frac)
        n_local = self.L * a.local_frac
        elems = n_full * per_full_layer
        if n_local and S:
            elems += n_local * per_full_layer * min(1.0, a.window / S)
        elif n_local:
            elems += n_local * per_full_layer
        return elems

    def kv_bytes_per_token(self, bytes_per=1, S=None):
        return self.kv_elems_per_token(S) * bytes_per

    def linear_state_bytes(self, bytes_per=2):
        a = self.attn
        return self.L * a.linear_frac * a.linear_state_elems * bytes_per

    # --- FLOPs ---------------------------------------------------------------
    def flops_per_token_fwd(self):
        # 2 * active params, excluding the input embedding lookup
        return 2 * (self.active_params() - self.V * self.D)

    def attn_score_flops_per_token_fwd(self, S, decode=False):
        """S-dependent attention FLOPs per token per layer (all layers summed).
        Training/prefill: causal, so average over S/2 keys. Decode with MLA uses
        weight absorption (latent dims), which is what changes the intensity."""
        a = self.attn
        if a.kind == "mla":
            if decode:
                per_key = 2 * a.n_heads * ((a.kv_lora_rank + a.qk_rope) + a.kv_lora_rank)
                return self.L * per_key * S
            per_key = 2 * a.n_heads * ((a.qk_nope + a.qk_rope) + a.v_head)
            return self.L * per_key * S / 2
        per_key = 2 * a.n_heads * a.head_dim * 2
        n_full = self.L * (1 - a.local_frac - a.linear_frac)
        n_local = self.L * a.local_frac
        if decode:
            return n_full * per_key * S + n_local * per_key * min(S, a.window)
        return n_full * per_key * S / 2 + n_local * per_key * min(S, a.window) / 2


# --- configs ------------------------------------------------------------------
LLAMA3_70B = Model("LLaMA 3 70B", "https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json",
    D=8192, L=80, V=128256, F_dense=28672,
    attn=Attention("gqa", 64, 8, 128), tokens_trained=15e12, reported_total=70e9)

LLAMA3_405B = Model("LLaMA 3 405B", "https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json",
    D=16384, L=126, V=128256, F_dense=53248,
    attn=Attention("gqa", 128, 8, 128), tokens_trained=15e12, reported_total=405e9)

DEEPSEEK_V3 = Model("DeepSeek-V3", "https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json",
    D=7168, L=61, V=129280, F_dense=18432, dense_layers=3, E=256, k=8, shared=1, F_e=2048,
    attn=Attention("mla", 128, q_lora_rank=1536, kv_lora_rank=512, qk_nope=128, qk_rope=64, v_head=128),
    mtp_depth=1, tokens_trained=14.8e12, reported_total=671e9, reported_active=37e9,
    notes="sigmoid routing, aux-loss-free bias, node-limited routing to 4 nodes, fp8 training")

KIMI_K2 = Model("Kimi K2", "https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json",
    D=7168, L=61, V=163840, F_dense=18432, dense_layers=1, E=384, k=8, shared=1, F_e=2048,
    attn=Attention("mla", 64, q_lora_rank=1536, kv_lora_rank=512, qk_nope=128, qk_rope=64, v_head=128),
    tokens_trained=15.5e12, reported_total=1.04e12, reported_active=32e9, notes="MuonClip optimizer")

QWEN3_235B = Model("Qwen3-235B-A22B", "https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json",
    D=4096, L=94, V=151936, E=128, k=8, shared=0, F_e=1536,
    attn=Attention("gqa", 64, 4, 128), tokens_trained=36e12, reported_total=235e9, reported_active=22e9)

GPT_OSS_120B = Model("gpt-oss-120b", "https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json",
    D=2880, L=36, V=201088, E=128, k=4, shared=0, F_e=2880,
    attn=Attention("sliding-gqa", 64, 8, 64, local_frac=0.5, window=128),
    reported_total=116.8e9, reported_active=5.1e9, notes="alternating sliding window 128 / full; MXFP4 expert weights")

LLAMA4_MAVERICK = Model("Llama 4 Maverick", "https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E/blob/main/config.json",
    D=5120, L=48, V=202048, F_dense=16384, dense_layers=24, E=128, k=1, shared=1, F_e=8192,
    attn=Attention("gqa", 40, 8, 128), reported_total=400e9, reported_active=17e9,
    notes="every other layer is MoE (interleave_moe_layer_step=2); iRoPE: every 4th layer NoPE with global attention, others chunked to 8192")

GEMMA3_27B = Model("Gemma 3 27B", "https://huggingface.co/google/gemma-3-27b-it/blob/main/config.json",
    D=5376, L=62, V=262144, F_dense=21504, tied_embeddings=True,
    attn=Attention("sliding-gqa", 32, 16, 128, local_frac=5/6, window=1024), reported_total=27e9)

QWEN3_NEXT_80B = Model("Qwen3-Next-80B-A3B", "https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json",
    D=2048, L=48, V=151936, E=512, k=10, shared=1, F_e=512,
    attn=Attention("hybrid", 16, 2, 256, linear_frac=0.75, linear_state_elems=32 * 128 * 128),
    mtp_depth=1, tokens_trained=15e12, reported_total=80e9, reported_active=3e9,
    notes="3 Gated DeltaNet layers : 1 gated full-attention layer; linear layers have 32 V heads of dim 128 (state 32x128x128 per layer)")

MODELS = [LLAMA3_70B, LLAMA3_405B, DEEPSEEK_V3, KIMI_K2, QWEN3_235B, GPT_OSS_120B, LLAMA4_MAVERICK, GEMMA3_27B, QWEN3_NEXT_80B]

# ----------------------------------------------------------------------------
# Rooflines
# ----------------------------------------------------------------------------

def moe_decode_critical_batch(m: Model, chip: Chip, weight_bytes=2, flops_fp8=False):
    """Tokens per decode step needed for the expert weights to be compute-bound.
    Dense rule: B > (bytes/2) * C / W_hbm. MoE: multiply by E/k since only k of E
    experts see each token but all E must be loaded."""
    c = chip.c_fp8 if flops_fp8 else chip.c_bf16
    dense = (weight_bytes / 2) * c / chip.hbm_bw
    return dense * (m.E / m.k) if m.E else dense

def ep_critical_expert_width(chip: Chip, longest_axis: int, dispatch_bytes=2, combine_bytes=2):
    """Expert width F needed for expert parallelism to be compute-bound on a TPU
    torus whose EP mesh has longest axis A (book's AllToAll model). Forward
    pass, counting all three expert matmuls (6 B k D F FLOPs per layer), as
    Section 13 does and as DeepSeek's V4 overlap condition does:
       T_math  = 6 B k D F / (Z C)
       T_comms = (dispatch+combine bytes) * B k D / Z * A / (4 W_ici)
    compute-bound iff F > (bytes_total/24) * A * C / W_ici."""
    total = dispatch_bytes + combine_bytes
    return (total / 24) * longest_axis * chip.alpha_ici

def ep_critical_expert_width_gpu(chip: Chip, cross_node=False, dispatch_bytes=2, combine_bytes=2):
    """GPU version, all-to-all over a switch: every byte leaves the GPU once.
       T_math  = 6 B k D F / (Z C);  T_comms = total_bytes * B k D / (Z W)
    compute-bound iff F > (total_bytes/6) * C / W."""
    W = chip.node_egress if cross_node else chip.ici_bidi
    total = dispatch_bytes + combine_bytes
    return (total / 6) * chip.c_bf16 / W

def mla_decode_intensity(m: Model, kv_bytes=2):
    a = m.attn
    flops = 2 * a.n_heads * ((a.kv_lora_rank + a.qk_rope) + a.kv_lora_rank)
    byts = (a.kv_lora_rank + a.qk_rope) * kv_bytes
    return flops / byts

def gqa_decode_intensity(m: Model, kv_bytes=2):
    a = m.attn
    flops = 2 * a.n_heads * a.head_dim * 2
    byts = 2 * a.kv_heads * a.head_dim * kv_bytes
    return flops / byts


def fmt(x, unit=""):
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    if -3 < e < 4:
        return f"{x:,.3g}{unit}"
    return f"{x:.3g}{unit}"


def fsdp_critical_tokens_gpu(chip: Chip, E=1, k=1, Z=1, P=1, fp8_flops=False, gather_bytes=2):
    """Per-GPU tokens for FSDP across nodes/racks to be compute-bound (Section 12 convention):
    B/N > (E/(kZP)) * C / W_collective, with W_collective the domain's total scale-out egress.
    gather_bytes=1 for fp8 weight gathers (halves the threshold)."""
    C = chip.c_fp8 if fp8_flops else chip.c_bf16
    return (E / (k * Z * P)) * (C / chip.collective_bw_cross_node) * (gather_bytes / 2)


def gather_layout_max_Z_gpu(chip: Chip, k, F, cross_node=False):
    """Largest EP degree for which the AllGather+ReduceScatter expert layout is compute-bound: Z < 1.5 k F / alpha."""
    W = chip.node_egress if cross_node else chip.ici_bidi
    return 1.5 * k * F / (chip.c_bf16 / W)


def report():
    print("=" * 78)
    print("HARDWARE (from the book's tables)")
    print("=" * 78)
    print(f"{'chip':<18}{'HBM':>8}{'HBM BW':>10}{'bf16':>10}{'fp8':>10}{'a_hbm bf16':>12}{'a_hbm fp8':>11}{'a_ici/axis':>12}")
    for c in CHIPS:
        print(f"{c.name:<18}{fmt(c.hbm):>8}{fmt(c.hbm_bw):>10}{fmt(c.c_bf16):>10}{fmt(c.c_fp8):>10}"
              f"{c.alpha_hbm_bf16:>12.0f}{c.alpha_hbm_fp8:>11.0f}{c.alpha_ici:>12.0f}")

    print()
    print("=" * 78)
    print("MODELS: parameters, KV cache, FLOPs")
    print("=" * 78)
    for m in MODELS:
        tot, act = m.total_params(), m.active_params()
        print(f"\n{m.name}  ({m.src})")
        print(f"  D={m.D} L={m.L} V={m.V} E={m.E} k={m.k} shared={m.shared} F_e={m.F_e} F_dense={m.F_dense} dense_layers={m.dense_layers}")
        print(f"  attn params/layer = {fmt(m.attn_params_per_layer())}   (x{m.L} = {fmt(m.L*m.attn_params_per_layer())})")
        print(f"  MLP total = {fmt(m.mlp_params_total())}  MLP active = {fmt(m.mlp_params_active())}  embed = {fmt(m.embed_params())}")
        print(f"  TOTAL = {fmt(tot)}  (reported {fmt(m.reported_total)})   ACTIVE = {fmt(act)}  (reported {fmt(m.reported_active)})")
        if m.E:
            print(f"  sparsity E/k = {m.E/m.k:.1f}   total/active = {tot/act:.1f}")
        print(f"  fwd FLOPs/token = {fmt(m.flops_per_token_fwd())}   train FLOPs/token = {fmt(3*m.flops_per_token_fwd())}")
        if m.tokens_trained:
            print(f"  pretraining FLOPs @ {fmt(m.tokens_trained)} tokens = {fmt(3*m.flops_per_token_fwd()*m.tokens_trained)}")
        for S in (8192, 32768, 131072):
            print(f"  KV bytes/token (1 byte/elem) at S={S}: {fmt(m.kv_bytes_per_token(1, S))}"
                  + (f"   + linear state per sequence {fmt(m.linear_state_bytes(2))} (bf16)" if m.attn.linear_frac else ""))
        if m.attn.kind == "mla":
            print(f"  MLA decode intensity: bf16 KV {mla_decode_intensity(m,2):.0f} FLOPs/byte, fp8 KV {mla_decode_intensity(m,1):.0f}")
        elif m.attn.kv_heads:
            print(f"  GQA decode intensity: bf16 KV {gqa_decode_intensity(m,2):.0f} FLOPs/byte")
        # attention vs MLP crossover
        mlp_fwd = 2 * m.mlp_params_active() / m.L * m.L  # all layers
        for decode in (False, True):
            f1 = m.attn_score_flops_per_token_fwd(1, decode)
            if f1 > 0:
                S_cross = (2 * m.mlp_params_active()) / f1
                print(f"  attention-score FLOPs = active-MLP FLOPs at S ~ {S_cross:,.0f} ({'decode, absorbed' if decode else 'prefill/train, causal'})")

    print()
    print("=" * 78)
    print("MoE DECODE: tokens per step for expert weights to be compute-bound")
    print("=" * 78)
    for m in (DEEPSEEK_V3, KIMI_K2, QWEN3_235B, GPT_OSS_120B, LLAMA4_MAVERICK):
        row = [f"{m.name:<22}"]
        for c in (H800, H200, B200, GB200, TPU7X, V5E):
            row.append(f"{c.name.split(' (')[0]}: bf16 {moe_decode_critical_batch(m, c, 2):>7,.0f} | fp8w+fp8 {moe_decode_critical_batch(m, c, 1, True):>7,.0f}")
        print("  ".join(row))

    print()
    print("=" * 78)
    print("EXPERT PARALLELISM on GPUs: expert width F needed to be compute-bound (three matmuls; fp8 dispatch + bf16 combine)")
    print("=" * 78)
    for c in GPUS:
        nv_bf = ep_critical_expert_width_gpu(c, False, 1, 2); ib_bf = ep_critical_expert_width_gpu(c, True, 1, 2)
        extra = f"   measured NVLink {c.measured_egress / 1e9:.0f} GB/s: F>{nv_bf * c.ici_bidi / c.measured_egress:,.0f} / {2 * nv_bf * c.ici_bidi / c.measured_egress:,.0f}" if c.measured_egress else ""
        print(f"{c.name:<14} NVLink alpha={c.c_bf16 / c.ici_bidi:,.0f}: F>{nv_bf:,.0f} (bf16 matmuls) / {2 * nv_bf:,.0f} (fp8 matmuls)   IB alpha={c.c_bf16 / c.node_egress:,.0f}: F>{ib_bf:,.0f} / {2 * ib_bf:,.0f}{extra}")
    print()
    print("=" * 78)
    print("EXPERT PARALLELISM on TPUs: F needed (bf16 dispatch+combine; three matmuls; wraparound on the EP axis assumed)")
    print("=" * 78)
    for c in TPUS:
        print(f"{c.name:<18} alpha_ici/axis={c.alpha_ici:,.0f}  " +
              "  ".join(f"A={A}: F>{ep_critical_expert_width(c, A):,.0f}" for A in (2, 4, 8, 16)) +
              f"   [fp8 dispatch: x0.75]")
    print()
    print("=" * 78)
    print("FSDP across nodes/racks on GPUs: tokens per GPU to be compute-bound (bf16 matmuls, bf16 gathers)")
    print("=" * 78)
    for c in GPUS:
        dense = fsdp_critical_tokens_gpu(c)
        print(f"{c.name:<14} W_collective = {c.domain} x {c.node_egress / 1e9:.0f} GB/s = {c.collective_bw_cross_node / 1e12:.1f} TB/s -> dense {dense:,.0f};  DeepSeek-V3 pure FSDP (E/k=32) {fsdp_critical_tokens_gpu(c, 256, 8):,.0f};  with EP64 {fsdp_critical_tokens_gpu(c, 256, 8, 64):,.0f};  EP64+PP16 {fsdp_critical_tokens_gpu(c, 256, 8, 64, 16):,.0f};  V4-Pro EP64 fp8 FLOPs {fsdp_critical_tokens_gpu(c, 384, 6, 64, 1, True):,.0f}")
    print(f"DeepSeek-V3 had 62.9e6 / 2048 = {62.9e6 / 2048:,.0f} tokens per GPU; decode EP144 at 88 seq/GPU: {144 * 88 * 8 / 288:,.0f} tokens per expert copy per step")
    print(f"gather layout max Z for DeepSeek-V3 (kF=16384): H100 NVLink {gather_layout_max_Z_gpu(H100, 8, 2048):.0f}, H100 IB {gather_layout_max_Z_gpu(H100, 8, 2048, True):.1f}")

if __name__ == "__main__":
    report()


# ----------------------------------------------------------------------------
# 2026 flagships: explicit parameter arithmetic from config.json fields.
# These architectures (latent MoE, hybrid linear attention, grouped output
# projections) do not fit the simple Model class above, so each is written out.
# ----------------------------------------------------------------------------

def gated_mlp(d_in, f, d_out=None):
    d_out = d_in if d_out is None else d_out
    return 2 * d_in * f + f * d_out


def report_2026():
    print()
    print("=" * 78)
    print("2026 FLAGSHIPS: parameter counts from config.json (reported totals in brackets)")
    print("=" * 78)

    # --- Kimi K3: https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json (text_config)
    D, L, V = 7168, 93, 163840
    E, k, Es, latent, Fe = 896, 16, 2, 3584, 3072
    moe_layers = L - 1
    routed = moe_layers * E * gated_mlp(latent, Fe)
    routed_active = moe_layers * k * gated_mlp(latent, Fe)
    shared = moe_layers * gated_mlp(D, Es * Fe)          # shared experts: full width, intermediate Es*Fe (modeling code)
    latent_proj = moe_layers * 2 * D * latent            # W_down and W_up around the routed path
    dense = gated_mlp(D, 33792)
    mla = 24 * (D * 1536 + 1536 * 96 * 192 + D * 576 + 2 * 512 * 96 * 128 + 2 * 96 * 128 * D)   # incl. output gate
    kda = 69 * (5 * D * 96 * 128)                                                                  # q, k, v, gate, out: five full-rank D x 12288 matrices
    emb = 2 * V * D
    vit = 0.4e9
    total = routed + shared + latent_proj + dense + mla + kda + emb + vit
    active = routed_active + shared + latent_proj + dense + mla + kda + emb + vit
    print(f"Kimi K3          total {total/1e12:.2f}T [2.78T]   active {active/1e9:.0f}B [104B]   routed experts alone {routed/1e12:.2f}T; attention {(mla+kda)/1e9:.0f}B")
    flop_active = active - emb / 2 - vit   # input embedding is a lookup; the vision tower is not in the text path
    print(f"                 routed experts are {routed_active / flop_active:.0%} of K3's FLOP-bearing active params (shared {shared / flop_active:.0%}, latent proj {latent_proj / flop_active:.0%}, attention {(mla + kda) / flop_active:.0%})")

    # --- DeepSeek-V4-Pro: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json
    D, L, V = 7168, 61, 129280
    E, k, Es, Fe = 384, 6, 1, 3072
    experts = L * (E + Es) * gated_mlp(D, Fe)
    experts_active = L * (k + Es) * gated_mlp(D, Fe)
    attn = L * (D * 1536 + 1536 * 128 * 512 + D * 576 + 16 * 4096 * 1024 + 16 * 1024 * D)   # q latent+up, shared KV, grouped out-proj
    indexer = 30 * (D * 64 * 128 + D * 128)
    emb = 2 * V * D
    total = experts + attn + indexer + emb
    active = experts_active + attn + indexer + emb
    print(f"DeepSeek-V4-Pro  total {total/1e12:.2f}T [1.6T]    active {active/1e9:.0f}B [49B]    experts {experts/1e12:.2f}T; attention+indexer {(attn+indexer)/1e9:.0f}B")

    # --- GLM-5 / 5.x: https://huggingface.co/zai-org/GLM-5/raw/main/config.json
    D, L, V = 6144, 78, 154880
    E, k, Es, Fe = 256, 8, 1, 2048
    moe_layers = L - 3
    experts = moe_layers * (E + Es) * gated_mlp(D, Fe)
    experts_active = moe_layers * (k + Es) * gated_mlp(D, Fe)
    dense = 3 * gated_mlp(D, 12288)
    mla = L * (D * 2048 + 2048 * 64 * 256 + D * 576 + 512 * 64 * 192 + 512 * 64 * 256 + 64 * 256 * D)
    indexer = L * (D * 128 + 2048 * 32 * 128 + D * 32)
    emb = 2 * V * D
    total = experts + dense + mla + indexer + emb
    active = experts_active + dense + mla + indexer + emb
    print(f"GLM-5            total {total/1e9:.0f}B [744B]    active {active/1e9:.0f}B [40B]    attention {mla/1e9:.1f}B")

    # --- Qwen3.5-397B-A17B: https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json
    D, L, V = 4096, 60, 248320
    E, k, Fe = 512, 10, 1024
    experts = L * (E + 1) * gated_mlp(D, Fe)
    experts_active = L * (k + 1) * gated_mlp(D, Fe)
    gdn = 45 * (2 * D * 16 * 128 + 2 * D * 64 * 128 + 64 * 128 * D)      # q,k (16 heads) | v, gate (64 heads) | out
    full = 15 * (2 * D * 32 * 256 + 2 * D * 2 * 256 + 32 * 256 * D)      # q + output gate | k, v | out
    emb = 2 * V * D
    vit = 0.4e9
    total = experts + gdn + full + emb + vit
    active = experts_active + gdn + full + emb + vit
    print(f"Qwen3.5-397B     total {total/1e9:.0f}B [397B]    active {active/1e9:.0f}B [17B]    linear-attention layers {gdn/1e9:.1f}B, full {full/1e9:.1f}B")

    # --- Qwen3.8-2.4T-A95B: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/config.json
    D, L, V = 8192, 92, 248320
    E, k, Fe = 512, 10, 2048
    experts = L * (E + 1) * gated_mlp(D, Fe)
    experts_active = L * (k + 1) * gated_mlp(D, Fe)
    gdn = 69 * (2 * D * 16 * 128 + 2 * D * 128 * 128 + 128 * 128 * D)
    full = 23 * (2 * D * 64 * 256 + 2 * D * 4 * 256 + 64 * 256 * D)
    emb = 2 * V * D
    total = experts + gdn + full + emb
    active = experts_active + gdn + full + emb
    print(f"Qwen3.8-2.4T     total {total/1e12:.2f}T [2.4T]    active {active/1e9:.0f}B [95B]    linear-attention layers {gdn/1e9:.0f}B, full {full/1e9:.1f}B")

    # --- Gemma 4 26B-A4B: https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/config.json
    D, L, V = 2816, 30, 262144
    E, k, Fe = 128, 8, 704
    experts = L * E * gated_mlp(D, Fe)
    experts_active = L * k * gated_mlp(D, Fe)
    shared = L * gated_mlp(D, 2112)
    local = 25 * (D * 16 * 256 + 2 * D * 8 * 256 + 16 * 256 * D)
    glob = 5 * (D * 16 * 512 + D * 2 * 512 + 16 * 512 * D)          # K reused as V
    emb = V * D                                                      # tied
    vit = 0.55e9
    total = experts + shared + local + glob + emb + vit
    active = experts_active + shared + local + glob + emb + vit
    print(f"Gemma 4 26B-A4B  total {total/1e9:.1f}B [25.2B card / 26B report]   active {active/1e9:.1f}B [3.8B]")

    # --- Mistral Large 3: https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/raw/main/params.json
    D, L, V = 7168, 61, 131072
    E, k, Es, Fe = 128, 4, 1, 4096
    moe_layers = L - 3
    experts = moe_layers * (E + Es) * gated_mlp(D, Fe)
    experts_active = moe_layers * (k + Es) * gated_mlp(D, Fe)
    dense = 3 * gated_mlp(D, 16384)
    mla = L * (D * 1536 + 1536 * 128 * 192 + D * 576 + 2 * 512 * 128 * 128 + 128 * 128 * D)
    emb = 2 * V * D
    vit = 2.5e9
    total = experts + dense + mla + emb + vit
    active = experts_active + dense + mla + emb + vit
    print(f"Mistral Large 3  total {total/1e9:.0f}B [675B]    active {active/1e9:.0f}B [41B]")


if __name__ == "__main__":
    report_2026()
