#!/usr/bin/env python3
"""Generates the SVG figures for the update chapters. No dependencies beyond
the standard library; run `python3 scripts/figures.py` and the files land in
assets/img/. The charts are deliberately plain (thin lines, small labels) to
sit next to the original book's figures without looking like a dashboard."""
import math
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "img")
os.makedirs(OUT, exist_ok=True)

FONT = "font-family='Helvetica, Arial, sans-serif'"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ----------------------------------------------------------------------------
# generic log-log scatter / line chart
# ----------------------------------------------------------------------------
class LogChart:
    def __init__(self, w=760, h=520, ml=70, mr=30, mt=30, mb=60,
                 xlim=(1e10, 5e12), ylim=(1e9, 5e11), xlabel="", ylabel="", title=""):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, ml, mr, mt, mb
        self.xlim, self.ylim = xlim, ylim
        self.parts = []
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title

    def x(self, v):
        lo, hi = map(math.log10, self.xlim)
        return self.ml + (math.log10(v) - lo) / (hi - lo) * (self.w - self.ml - self.mr)

    def y(self, v):
        lo, hi = map(math.log10, self.ylim)
        return self.h - self.mb - (math.log10(v) - lo) / (hi - lo) * (self.h - self.mt - self.mb)

    def axes(self, xticks, yticks, xfmt, yfmt):
        p = self.parts
        x0, x1 = self.ml, self.w - self.mr
        y0, y1 = self.mt, self.h - self.mb
        # grid
        for v in xticks:
            p.append(f"<line x1='{self.x(v):.1f}' y1='{y0}' x2='{self.x(v):.1f}' y2='{y1}' stroke='#e6e6e6' stroke-width='1'/>")
            p.append(f"<text x='{self.x(v):.1f}' y='{y1 + 18}' text-anchor='middle' font-size='12' fill='#444' {FONT}>{xfmt(v)}</text>")
        for v in yticks:
            p.append(f"<line x1='{x0}' y1='{self.y(v):.1f}' x2='{x1}' y2='{self.y(v):.1f}' stroke='#e6e6e6' stroke-width='1'/>")
            p.append(f"<text x='{x0 - 8}' y='{self.y(v) + 4:.1f}' text-anchor='end' font-size='12' fill='#444' {FONT}>{yfmt(v)}</text>")
        p.append(f"<rect x='{x0}' y='{y0}' width='{x1 - x0}' height='{y1 - y0}' fill='none' stroke='#999' stroke-width='1'/>")
        p.append(f"<text x='{(x0 + x1) / 2:.1f}' y='{self.h - 14}' text-anchor='middle' font-size='13' fill='#222' {FONT}>{esc(self.xlabel)}</text>")
        p.append(f"<text x='16' y='{(y0 + y1) / 2:.1f}' text-anchor='middle' font-size='13' fill='#222' transform='rotate(-90 16 {(y0 + y1) / 2:.1f})' {FONT}>{esc(self.ylabel)}</text>")
        if self.title:
            p.append(f"<text x='{(x0 + x1) / 2:.1f}' y='{y0 - 10}' text-anchor='middle' font-size='14' font-weight='bold' fill='#222' {FONT}>{esc(self.title)}</text>")

    def line(self, pts, color, width=2, dash=None, label=None, label_dy=-6):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.x(x):.1f},{self.y(y):.1f}" for i, (x, y) in enumerate(pts))
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        self.parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{width}'{dash_attr}/>")
        if label:
            x, y = pts[-1]
            self.parts.append(f"<text x='{self.x(x) + 6:.1f}' y='{self.y(y) + label_dy:.1f}' font-size='12' fill='{color}' {FONT}>{esc(label)}</text>")

    def point(self, x, y, color, label=None, dx=7, dy=4, r=4.5, anchor="start"):
        self.parts.append(f"<circle cx='{self.x(x):.1f}' cy='{self.y(y):.1f}' r='{r}' fill='{color}' stroke='#fff' stroke-width='1'/>")
        if label:
            self.parts.append(f"<text x='{self.x(x) + dx:.1f}' y='{self.y(y) + dy:.1f}' font-size='11.5' fill='#222' text-anchor='{anchor}' {FONT}>{esc(label)}</text>")

    def text(self, x, y, s, color="#444", size=12, anchor="start", italic=False):
        st = " font-style='italic'" if italic else ""
        self.parts.append(f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' text-anchor='{anchor}'{st} {FONT}>{esc(s)}</text>")

    def legend(self, items, x, y):
        for i, (label, color) in enumerate(items):
            yy = y + i * 18
            self.parts.append(f"<circle cx='{x}' cy='{yy - 4}' r='4.5' fill='{color}'/>")
            self.parts.append(f"<text x='{x + 10}' y='{yy}' font-size='12' fill='#222' {FONT}>{esc(label)}</text>")

    def save(self, name):
        svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {self.w} {self.h}' width='{self.w}' height='{self.h}'>"
               f"<rect width='100%' height='100%' fill='white'/>" + "".join(self.parts) + "</svg>")
        with open(os.path.join(OUT, name), "w") as f:
            f.write(svg)
        print("wrote", name)


def fmt_params(v):
    if v >= 1e12:
        return f"{v / 1e12:g}T"
    return f"{v / 1e9:g}B"


# ----------------------------------------------------------------------------
# Figure: total vs active parameters
# ----------------------------------------------------------------------------
def fig_total_vs_active():
    c = LogChart(xlim=(2e10, 4e12), ylim=(3e9, 6e11),
                 xlabel="total parameters", ylabel="active parameters per token",
                 title="Open-weight models, 2023 to 2026: total versus active parameters")
    c.axes([3e10, 1e11, 3e11, 1e12, 3e12], [5e9, 1e10, 3e10, 1e11, 3e11], fmt_params, fmt_params)
    # dense diagonal
    c.line([(3e9, 3e9), (6e11, 6e11)], "#bbb", 1.5, dash="6,4")
    c.text(c.x(4.5e11) - 60, c.y(4.5e11) - 10, "dense (total = active)", "#888", 12, italic=True)
    for ratio, lab in [(10, "10x"), (30, "30x"), (60, "60x")]:
        c.line([(3e9 * ratio, 3e9), (6e11 * ratio, 6e11)], "#ddd", 1, dash="2,4")
        xx = min(6e11 * ratio, 3.6e12)
        c.text(c.x(xx) - 4, c.y(xx / ratio) - 6, lab, "#999", 11, anchor="end")
    Y = {2023: "#9e9e9e", 2024: "#6baed6", 2025: "#3182bd", 2026: "#b509ac"}
    pts = [
        ("Mixtral 8x7B", 47e9, 13e9, 2023, 7, 4, "start"),
        ("Mixtral 8x22B", 141e9, 39e9, 2024, 7, 12, "start"),
        ("DBRX", 132e9, 36e9, 2024, -8, -4, "end"),
        ("Grok-1", 314e9, 86e9, 2024, 7, -4, "start"),
        ("DeepSeek-V2", 236e9, 21e9, 2024, 7, 12, "start"),
        ("LLaMA 3 405B", 405e9, 405e9, 2024, -8, -6, "end"),
        ("LLaMA 2/3 70B", 70e9, 70e9, 2024, 8, 4, "start"),
        ("DeepSeek-V3", 671e9, 37e9, 2024, -8, 14, "end"),
        ("Llama 4 Maverick", 400e9, 17e9, 2025, -8, 12, "end"),
        ("Qwen3-235B", 235e9, 22e9, 2025, -8, -6, "end"),
        ("Kimi K2", 1.04e12, 32e9, 2025, 7, 14, "start"),
        ("gpt-oss-120b", 117e9, 5.1e9, 2025, 7, 4, "start"),
        ("GLM-4.5", 355e9, 32e9, 2025, -8, 4, "end"),
        ("Mistral Large 3", 675e9, 41e9, 2025, -8, -6, "end"),
        ("Qwen3.5-397B", 397e9, 17e9, 2026, 7, 4, "start"),
        ("GLM-5", 744e9, 40e9, 2026, 7, -6, "start"),
        ("DeepSeek-V4-Flash", 284e9, 13e9, 2026, 7, 12, "start"),
        ("DeepSeek-V4-Pro", 1.6e12, 49e9, 2026, 7, -6, "start"),
        ("MiniMax M3", 428e9, 23e9, 2026, 7, -6, "start"),
        ("Nemotron 3 Ultra", 550e9, 55e9, 2026, -8, -6, "end"),
        ("Qwen3.8-2.4T", 2.4e12, 95e9, 2026, -8, -6, "end"),
        ("Kimi K3", 2.8e12, 104e9, 2026, 7, 12, "start"),
        ("Gemma 4 31B", 31e9, 31e9, 2026, 7, 12, "start"),
        ("Qwen3.6-27B", 27e9, 27e9, 2026, 8, -4, "start"),
    ]
    for name, tot, act, yr, dx, dy, anc in pts:
        c.point(tot, act, Y[yr], name, dx, dy, anchor=anc)
    c.legend([(str(y), col) for y, col in Y.items()], c.ml + 14, c.mt + 22)
    c.save("moe-total-vs-active.svg")


# ----------------------------------------------------------------------------
# Figure: MoE layer diagram
# ----------------------------------------------------------------------------
def fig_moe_layer():
    W, H = 880, 500
    p = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
         "<rect width='100%' height='100%' fill='white'/>",
         "<defs><marker id='arr' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0,0 L6,3 L0,6 z' fill='#555'/></marker></defs>"]

    def box(x, y, w, h, label, fill="#f4f4f6", stroke="#777", sub=None, bold=False):
        p.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='6' fill='{fill}' stroke='{stroke}' stroke-width='1.2'/>")
        fw = " font-weight='bold'" if bold else ""
        if sub:
            p.append(f"<text x='{x + w / 2}' y='{y + h / 2 - 4}' text-anchor='middle' font-size='13'{fw} fill='#222' {FONT}>{esc(label)}</text>")
            p.append(f"<text x='{x + w / 2}' y='{y + h / 2 + 13}' text-anchor='middle' font-size='11.5' fill='#555' {FONT}>{esc(sub)}</text>")
        else:
            p.append(f"<text x='{x + w / 2}' y='{y + h / 2 + 5}' text-anchor='middle' font-size='13'{fw} fill='#222' {FONT}>{esc(label)}</text>")

    def arrow(x1, y1, x2, y2, label=None, color="#555", dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        p.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='1.4' marker-end='url(#arr)'{d}/>")
        if label:
            p.append(f"<text x='{(x1 + x2) / 2 + 6}' y='{(y1 + y2) / 2 - 6}' font-size='11.5' fill='#444' {FONT}>{esc(label)}</text>")

    # input
    box(20, 200, 120, 60, "x[B, D]", sub="tokens")
    # router
    box(190, 60, 130, 56, "Router", sub="x . W_r[D, E]", fill="#fdf8d2")
    arrow(140, 215, 190, 95)
    box(360, 60, 130, 56, "top-k", sub="indices, gates", fill="#fdf8d2")
    arrow(320, 88, 360, 88)
    # dispatch all-to-all
    box(190, 190, 130, 80, "AllToAll", sub="dispatch (k copies)", fill="#d2e7fd", bold=True)
    arrow(140, 230, 190, 230)
    arrow(425, 116, 425, 150, color="#999", dash="3,3")
    p.append(f"<text x='432' y='140' font-size='11' fill='#777' {FONT}>routing pattern</text>")
    arrow(425, 150, 320, 205, color="#999", dash="3,3")
    # experts
    ex_x, ex_y0 = 370, 150
    for i in range(6):
        y = ex_y0 + i * 34
        active = i in (1, 4)
        fill = "#e2f5ec" if active else "#f4f4f6"
        stroke = "#359469" if active else "#aaa"
        label = f"expert {i + 1}" if i < 5 else f"expert E"
        if i == 3:
            p.append(f"<text x='{ex_x + 55}' y='{y + 22}' text-anchor='middle' font-size='16' fill='#888' {FONT}>...</text>")
            continue
        box(ex_x, y, 110, 26, label, fill=fill, stroke=stroke)
        if active:
            arrow(320, 230, ex_x, y + 13, color="#359469")
    p.append(f"<text x='{ex_x + 55}' y='{ex_y0 + 6 * 34 + 12}' text-anchor='middle' font-size='11.5' fill='#444' {FONT}>each: 3 matmuls, [t_e, D] x [D, F]</text>")
    p.append(f"<text x='{ex_x + 55}' y='{ex_y0 + 6 * 34 + 27}' text-anchor='middle' font-size='11.5' fill='#444' {FONT}>t_e = B k / E tokens on average</text>")
    # combine
    box(530, 190, 130, 80, "AllToAll", sub="combine", fill="#d2e7fd", bold=True)
    for i in (1, 4):
        y = ex_y0 + i * 34
        arrow(ex_x + 110, y + 13, 530, 230, color="#359469")
    # shared expert
    box(360, 415, 130, 50, "Shared expert", sub="dense, width F_s", fill="#f4f4f6")
    arrow(80, 260, 80, 440)
    arrow(80, 440, 360, 440)
    # sum
    p.append("<circle cx='720' cy='230' r='18' fill='#fff' stroke='#555' stroke-width='1.4'/>")
    p.append(f"<text x='720' y='236' text-anchor='middle' font-size='18' fill='#222' {FONT}>+</text>")
    arrow(660, 230, 702, 230)
    p.append(f"<text x='681' y='214' text-anchor='middle' font-size='11' fill='#444' {FONT}>sum of k</text>")
    p.append(f"<text x='681' y='258' text-anchor='middle' font-size='11' fill='#444' {FONT}>gated outputs</text>")
    arrow(490, 440, 720, 440)
    arrow(720, 440, 720, 248)
    arrow(738, 230, 800, 230)
    p.append(f"<text x='806' y='235' font-size='13' fill='#222' {FONT}>y[B, D]</text>")
    # caption-ish notes
    p.append(f"<text x='20' y='30' font-size='13' font-weight='bold' fill='#222' {FONT}>One MoE layer with expert parallelism: experts live on different chips, tokens travel</text>")
    p.append(f"<text x='20' y='488' font-size='11.5' fill='#555' {FONT}>Blue: communication (2 AllToAlls of B k D elements). Green: the k active experts for one token. The router and gates are a rounding error in FLOPs.</text>")
    p.append("</svg>")
    with open(os.path.join(OUT, "moe-layer.svg"), "w") as f:
        f.write("".join(p))
    print("wrote moe-layer.svg")


# ----------------------------------------------------------------------------
# Figure: bytes read per decoded token vs context length
# ----------------------------------------------------------------------------
def fig_kv_vs_context():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import attention_numbers as an
    c = LogChart(w=780, h=520, xlim=(4096, 1.2e6), ylim=(1e7, 4e11),
                 xlabel="context length (tokens)", ylabel="bytes read per decoded token, per sequence",
                 title="What a decode step has to read, by attention family (fp8 cache)")

    def fx(v):
        return f"{v // 1024}k" if v < 1e6 else "1M"

    def fy(v):
        return f"{v / 1e9:g} GB" if v >= 1e9 else f"{v / 1e6:g} MB"
    c.axes([4096, 16384, 65536, 262144, 1048576], [1e7, 1e8, 1e9, 1e10, 1e11], fx, fy)
    S = [4096 * 2 ** (i / 2) for i in range(0, 17)]
    series = [
        ("LLaMA 3 70B (GQA 8x128)", lambda s: an.gqa(s, 80, 8, 128), "#9e9e9e", -8),
        ("DeepSeek-V3 (MLA)", lambda s: an.mla(s, 61), "#6baed6", -8),
        ("Qwen3.8 (3:1 linear hybrid)", lambda s: an.hybrid(s, 23, 4, 256, 69, 128 ** 3, 4), "#3182bd", 14),
        ("Gemma 3 27B (5:1 windows)", lambda s: an.gqa(s, 10, 16, 128, 52, 1024), "#e6550d", -8),
        ("Kimi K3 (3:1 hybrid, MLA)", lambda s: an.mla(s, 24) + 69 * 96 * 128 * 128 * 2 * 2, "#31a354", 14),
        ("DeepSeek-V3.2 (MLA + DSA)", lambda s: an.dsa(s, 61), "#756bb1", -8),
        ("DeepSeek-V4-Pro (compression + DSA)", an.v4_pro, "#b509ac", 14),
    ]
    for label, fn, color, dy in series:
        c.line([(s, fn(s)) for s in S], color, 2)
    # legend inside the plot, top-left
    lx, ly = c.ml + 14, c.mt + 22
    c.parts.append(f"<rect x='{lx - 8}' y='{ly - 16}' width='262' height='{18 * len(series) + 10}' fill='white' fill-opacity='0.92' stroke='#ddd'/>")
    for i, (label, fn, color, dy) in enumerate(series):
        yy = ly + i * 18
        c.parts.append(f"<line x1='{lx}' y1='{yy - 4}' x2='{lx + 22}' y2='{yy - 4}' stroke='{color}' stroke-width='2.5'/>")
        c.parts.append(f"<text x='{lx + 30}' y='{yy}' font-size='12' fill='#222' {FONT}>{esc(label)}</text>")
    c.save("kv-bytes-vs-context.svg")


if __name__ == "__main__":
    fig_total_vs_active()
    fig_moe_layer()
    fig_kv_vs_context()


# ----------------------------------------------------------------------------
# Figure: DeepSeek decode step, roofline terms vs measured (Section 16)
# ----------------------------------------------------------------------------
def fig_decode_breakdown():
    W, H = 900, 350
    p = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>", "<rect width='100%' height='100%' fill='white'/>"]
    x0, x1 = 210, 740
    scale = (x1 - x0) / 55.0  # ms -> px
    rows = [
        ("Roofline terms, summed", [("weights 6.6", 6.6, "#9e9e9e"), ("KV 4.6", 4.6, "#6baed6"), ("MoE FLOPs 3.3", 3.3, "#31a354"), ("attention FLOPs 7.5", 7.5, "#3182bd")]),
        ("AllToAll over InfiniBand, 58 layers", [("dispatch 7.3", 58 * 5.05e6 / 40e9 * 1e3, "#e6550d"), ("combine 14.6", 58 * 10.1e6 / 40e9 * 1e3, "#fd8d3c")]),
        ("Measured (21 tok/s per user)", [("48 ms", 48.0, "#b509ac")]),
    ]
    y = 60
    for label, segs in rows:
        p.append(f"<text x='{x0 - 12}' y='{y + 22}' text-anchor='end' font-size='13' fill='#222' {FONT}>{esc(label)}</text>")
        x = x0
        for seg, ms, color in segs:
            w = ms * scale
            p.append(f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='34' fill='{color}' stroke='white' stroke-width='1'/>")
            if w > 46:
                p.append(f"<text x='{x + w / 2:.1f}' y='{y + 22}' text-anchor='middle' font-size='11.5' fill='white' {FONT}>{esc(seg)}</text>")
            x += w
        p.append(f"<text x='{x + 8:.1f}' y='{y + 22}' font-size='12' fill='#222' {FONT}>{sum(m for _, m, _ in segs):.0f} ms</text>")
        y += 62
    for ms in range(0, 56, 10):
        xx = x0 + ms * scale
        p.append(f"<line x1='{xx:.1f}' y1='{y}' x2='{xx:.1f}' y2='{y + 6}' stroke='#888'/>")
        p.append(f"<text x='{xx:.1f}' y='{y + 20}' text-anchor='middle' font-size='11.5' fill='#444' {FONT}>{ms}</text>")
    p.append(f"<line x1='{x0}' y1='{y}' x2='{x1}' y2='{y}' stroke='#888'/>")
    p.append(f"<text x='{(x0 + x1) / 2:.1f}' y='{y + 40}' text-anchor='middle' font-size='12.5' fill='#222' {FONT}>milliseconds per decode step (DeepSeek-V3, H800, 88 sequences per GPU at 4,989 tokens, EP144)</text>")
    p.append(f"<text x='20' y='30' font-size='14' font-weight='bold' fill='#222' {FONT}>Where DeepSeek's production decode step goes</text>")
    items = [("weights", "#9e9e9e"), ("KV cache", "#6baed6"), ("MoE FLOPs", "#31a354"), ("attention FLOPs", "#3182bd"), ("dispatch, 5MB fp8 at 40GB/s", "#e6550d"), ("combine, 10MB bf16", "#fd8d3c")]
    for row in (0, 1):
        lx = 210
        for lab, col in items[3 * row:3 * row + 3]:
            yy = H - 48 + 18 * row
            p.append(f"<rect x='{lx}' y='{yy}' width='11' height='11' fill='{col}'/>")
            p.append(f"<text x='{lx + 15}' y='{yy + 9}' font-size='11' fill='#333' {FONT}>{lab}</text>")
            lx += 16 + 7 * len(lab) + 24
    p.append("</svg>")
    with open(os.path.join(OUT, "decode-step-breakdown.svg"), "w") as f:
        f.write("".join(p))
    print("wrote decode-step-breakdown.svg")


# ----------------------------------------------------------------------------
# Figure: the RL loop, colocated vs disaggregated (Section 15)
# ----------------------------------------------------------------------------
def fig_rl_loop():
    W, H = 900, 450
    p = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>", "<rect width='100%' height='100%' fill='white'/>",
         "<defs><marker id='ar' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0,0 L6,3 L0,6 z' fill='#555'/></marker></defs>"]

    def box(x, y, w, h, label, sub=None, fill="#f4f4f6", stroke="#777"):
        p.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='6' fill='{fill}' stroke='{stroke}' stroke-width='1.2'/>")
        if sub:
            p.append(f"<text x='{x + w / 2}' y='{y + h / 2 - 5}' text-anchor='middle' font-size='13' font-weight='bold' fill='#222' {FONT}>{esc(label)}</text>")
            for i, line in enumerate(sub.split("|")):
                p.append(f"<text x='{x + w / 2}' y='{y + h / 2 + 12 + 14 * i}' text-anchor='middle' font-size='11' fill='#444' {FONT}>{esc(line)}</text>")
        else:
            p.append(f"<text x='{x + w / 2}' y='{y + h / 2 + 5}' text-anchor='middle' font-size='13' fill='#222' {FONT}>{esc(label)}</text>")

    def arrow(x1, y1, x2, y2, label=None, dy=-8, color="#555", dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        p.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='1.4' marker-end='url(#ar)'{d}/>")
        if label:
            p.append(f"<text x='{(x1 + x2) / 2}' y='{(y1 + y2) / 2 + dy}' text-anchor='middle' font-size='11' fill='#444' {FONT}>{esc(label)}</text>")

    p.append(f"<text x='20' y='28' font-size='14' font-weight='bold' fill='#222' {FONT}>One reinforcement-learning step, and where the time goes</text>")
    # main loop
    box(30, 70, 190, 90, "Generate", "G samples per prompt|decode, 10k to 65k tokens each|60 to 90% of wall clock", fill="#fdf8d2")
    box(300, 70, 160, 90, "Score", "verifier / tests / judge|group-normalized advantage", fill="#f4f4f6")
    box(540, 70, 190, 90, "Train", "clipped policy gradient|6 N_active FLOPs per token|a few gradient steps", fill="#d2e7fd")
    arrow(220, 115, 300, 115, "responses + logprobs")
    arrow(460, 115, 540, 115, "advantages")
    # weight sync arrow back
    p.append("<path d='M635,160 L635,192 L125,192 L125,160' fill='none' stroke='#b509ac' stroke-width='1.6' marker-end='url(#ar)'/>")
    p.append(f"<text x='380' y='212' text-anchor='middle' font-size='11.5' fill='#b509ac' {FONT}>weight sync: 1 TB in 16 to 30 s (colocated broadcast) or every K steps over the network (disaggregated)</text>")
    # two placements
    p.append(f"<text x='30' y='258' font-size='13' font-weight='bold' fill='#222' {FONT}>Colocated</text>")
    p.append(f"<text x='30' y='276' font-size='11.5' fill='#444' {FONT}>same GPUs alternate; trainer state offloaded to DRAM/NVMe during generation</text>")
    p.append(f"<text x='30' y='292' font-size='11.5' fill='#444' {FONT}>(Kimi K2/K3, DeepSeek-V4.1, slime for reasoning RL)</text>")
    p.append(f"<text x='470' y='258' font-size='13' font-weight='bold' fill='#222' {FONT}>Disaggregated</text>")
    p.append(f"<text x='470' y='276' font-size='11.5' fill='#444' {FONT}>separate pools, ~3:1 generation:training GPUs; async, a few steps of staleness</text>")
    p.append(f"<text x='470' y='292' font-size='11.5' fill='#444' {FONT}>(AReaL, PipelineRL, Magistral, slime for agentic RL)</text>")
    # straggler strip (rows stacked vertically)
    lengths = [3, 5, 6, 8, 9, 11, 12, 14, 18, 24, 34, 65]
    y0 = 322
    p.append(f"<text x='30' y='{y0 + 8}' font-size='12.5' font-weight='bold' fill='#222' {FONT}>The long tail:</text>")
    for i, L in enumerate(lengths):
        yy = y0 + i * 7
        w = L * 9
        p.append(f"<rect x='140' y='{yy}' width='{w}' height='5' fill='{'#e6550d' if L == 65 else '#9ecae1'}'/>")
    p.append(f"<text x='{140 + 65 * 9 + 8}' y='{y0 + 7 * 11 + 5}' font-size='11' fill='#e6550d' {FONT}>65k: 55 min at 20 tok/s</text>")
    p.append(f"<text x='140' y='{y0 + 7 * 12 + 16}' font-size='11' fill='#444' {FONT}>response lengths in one batch (std 4k to 4.5k tokens); a synchronous step waits for the longest one.</text>")
    p.append(f"<text x='140' y='{y0 + 7 * 12 + 31}' font-size='11' fill='#444' {FONT}>Fixes: partial rollouts (pause the tail, resume next step) and asynchronous training (do not wait).</text>")
    p.append("</svg>")
    with open(os.path.join(OUT, "rl-loop.svg"), "w") as f:
        f.write("".join(p))
    print("wrote rl-loop.svg")


# ----------------------------------------------------------------------------
# Figure: operational intensities by chip (landing page)
# ----------------------------------------------------------------------------
def fig_chip_intensity():
    chips = [
        ("H800/H100", 0.99e15, 1.98e15, None, 3.35e12, 4.5e11),
        ("H200", 0.99e15, 1.98e15, None, 4.8e12, 4.5e11),
        ("B200", 2.25e15, 4.5e15, 9e15, 8e12, 9e11),
        ("GB300", 2.5e15, 5e15, 15e15, 8e12, 9e11),
        ("Rubin", 4e15, 17.5e15, 35e15, 19.2e12, 1.5e12),
        ("MI455X", 5e15, 20.1e15, 40.3e15, 23.3e12, 1.8e12),
        ("TPU7x", 2.3e15, 4.61e15, None, 7.4e12, 5.4e11),
        ("TPU v5p", 0.46e15, 0.92e15, None, 2.8e12, 5.4e11),
        ("TPU v6e", 0.92e15, 1.84e15, None, 1.6e12, 3.6e11),
    ]
    W, H = 780, 420
    p = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>", "<rect width='100%' height='100%' fill='white'/>"]
    p.append(f"<text x='390' y='26' text-anchor='middle' font-size='14' font-weight='bold' fill='#222' {FONT}>Arithmetic intensity C / W_hbm by precision (dense FLOPs), GPUs then TPUs</text>")
    x0, y0, x1, y1 = 70, 50, 750, 340
    ymax = 2000
    for v in (0, 500, 1000, 1500, 2000):
        yy = y1 - v / ymax * (y1 - y0)
        p.append(f"<line x1='{x0}' y1='{yy:.1f}' x2='{x1}' y2='{yy:.1f}' stroke='#e6e6e6'/>")
        p.append(f"<text x='{x0 - 8}' y='{yy + 4:.1f}' text-anchor='end' font-size='11.5' fill='#444' {FONT}>{v}</text>")
    p.append(f"<line x1='{x0}' y1='{y1 - 240 / ymax * (y1 - y0):.1f}' x2='{x1}' y2='{y1 - 240 / ymax * (y1 - y0):.1f}' stroke='#b509ac' stroke-dasharray='5,4'/>")
    legend_line = (f"<rect x='{x0 + 6}' y='{y0 + 24}' width='300' height='20' fill='white' fill-opacity='0.9'/>"
                   f"<line x1='{x0 + 10}' y1='{y0 + 34}' x2='{x0 + 42}' y2='{y0 + 34}' stroke='#b509ac' stroke-dasharray='5,4'/>"
                   f"<text x='{x0 + 48}' y='{y0 + 38}' font-size='12' fill='#b509ac' {FONT}>240 to 300, the book's bf16 critical batch</text>")
    gw = (x1 - x0) / len(chips)
    colors = {"bf16": "#9e9e9e", "fp8": "#3182bd", "fp4": "#b509ac"}
    for i, (name, bf, f8, f4, hbm, net) in enumerate(chips):
        gx = x0 + i * gw + 10
        bw = (gw - 20) / 3
        for j, (lab, c) in enumerate((("bf16", bf), ("fp8", f8), ("fp4", f4))):
            if c is None:
                continue
            v = min(c / hbm, ymax)
            hh = v / ymax * (y1 - y0)
            p.append(f"<rect x='{gx + j * bw:.1f}' y='{y1 - hh:.1f}' width='{bw - 2:.1f}' height='{hh:.1f}' fill='{colors[lab]}'/>")
            p.append(f"<text x='{gx + j * bw + bw / 2 - 1:.1f}' y='{y1 - hh - 4:.1f}' text-anchor='middle' font-size='9.5' fill='#333' {FONT}>{c / hbm:.0f}</text>")
        p.append(f"<text x='{gx + (gw - 20) / 2:.1f}' y='{y1 + 18}' text-anchor='middle' font-size='12' fill='#222' {FONT}>{name}</text>")
    p.append(f"<line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#888'/>")
    lx = x0 + 10
    for lab, col in colors.items():
        p.append(f"<rect x='{lx}' y='{y0 + 6}' width='12' height='12' fill='{col}'/>")
        p.append(f"<text x='{lx + 17}' y='{y0 + 16}' font-size='12' fill='#222' {FONT}>{lab}</text>")
        lx += 70
    p.append(f"<text x='390' y='{y1 + 48}' text-anchor='middle' font-size='11.5' fill='#444' {FONT}>The bf16 bars barely move across four hardware generations; the fp4 bars are where the new compute went. TPUs publish no fp4 rate.</text>")
    p.append(f"<text x='390' y='{y1 + 66}' text-anchor='middle' font-size='11.5' fill='#444' {FONT}>Every decode batch-size threshold in Section 7 scales with the height of the bar for the precision you compute in.</text>")
    p.append(legend_line)
    p.append("</svg>")
    with open(os.path.join(OUT, "chip-intensity.svg"), "w") as f:
        f.write("".join(p))
    print("wrote chip-intensity.svg")



def fig_hero():
    """Landing-page banner: the three machines the 2026 open models actually run
    on, drawn to a common scale of HBM bytes per fast domain."""
    W, H = 900, 300
    P = []
    P.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='#fbfbfd'/>")
    def tile(x, y, w, h, fill, stroke="#666"):
        P.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='2' fill='{fill}' stroke='{stroke}' stroke-width='0.8'/>")
    def label(x, y, t, size=13, bold=False, color="#222", anchor="middle"):
        fw = " font-weight='bold'" if bold else ""
        P.append(f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='{anchor}' font-size='{size}'{fw} fill='{color}' {FONT}>{esc(t)}</text>")
    # ---- panel 1: an 8-GPU H800 node (2024 serving: DeepSeek-V3 on 18 of these) ----
    x0, y0 = 40, 60
    label(x0 + 110, 40, "8 x H800 node", 14, True)
    label(x0 + 110, 56, "80GB each, NVLink 200GB/s each way (160 measured), one 400Gb/s NIC per GPU", 10, color="#666")
    for i in range(8):
        tile(x0 + i * 27, y0 + 30, 22, 60, "#dbe9f6")
    tile(x0 - 4, y0 + 100, 8 * 27 + 4, 10, "#9ecae1")
    label(x0 + 110, y0 + 128, "NVSwitch", 10, color="#666")
    for i in range(8):
        P.append(f"<line x1='{x0 + i * 27 + 11}' y1='{y0 + 110}' x2='{x0 + i * 27 + 11}' y2='{y0 + 150}' stroke='#bbb' stroke-width='1' stroke-dasharray='2,2'/>")
    label(x0 + 110, y0 + 168, "InfiniBand, 50GB/s per GPU", 10, color="#666")
    label(x0 + 110, y0 + 200, "640GB of HBM per fast domain", 12)
    label(x0 + 110, y0 + 218, "DeepSeek-V3 decode: 18 nodes, 144-way EP", 11, color="#b509ac")
    # ---- panel 2: GB200 NVL72 rack ----
    x1 = 340
    label(x1 + 110, 40, "GB200 NVL72 rack", 14, True)
    label(x1 + 110, 56, "72 GPUs, 186GB each, NVLink 900GB/s to every other GPU", 10, color="#666")
    for r in range(18):
        for c in range(4):
            tile(x1 + 25 + c * 44, y0 + 24 + r * 8.6, 38, 6.6, "#d2e7fd" if r not in (8, 9) else "#fbfbfd", "#7fa8d4" if r not in (8, 9) else "#fbfbfd")
    tile(x1 + 25 + 4 * 44 - 176, y0 + 24 + 8 * 8.6, 176, 2 * 8.6 - 2, "#3182bd", "#3182bd")
    label(x1 + 110, y0 + 24 + 9 * 8.6 + 3, "9 NVSwitch trays", 8, color="#fff")
    label(x1 + 110, y0 + 200, "13.4TB of HBM per fast domain", 12)
    label(x1 + 110, y0 + 218, "Nemotron 3 training, SGLang and Dynamo serving", 11, color="#b509ac")
    # ---- panel 3: TPU7x 4x4x4 cube (one of 144 in a pod) ----
    x2, y2 = 660, y0 + 30
    label(x2 + 100, 40, "TPU7x 4x4x4 cube", 14, True)
    label(x2 + 100, 56, "64 chips, 192GB each, 3D torus ICI, 90GB/s one way per link, six links", 10, color="#666")
    s_ = 30
    for k in range(4):           # depth layers drawn back to front
        off = (3 - k) * 9
        for i in range(4):
            for j in range(4):
                tile(x2 + 20 + i * s_ + off, y2 + 20 + j * s_ * 0.75 - off * 0.6 + 30, s_ - 4, s_ * 0.75 - 4, ("#efe3f2", "#e6cdeb", "#dcb6e3", "#d29fdb")[k], "#b98cc4")
    label(x2 + 100, y0 + 200, "12.3TB of HBM per cube, 1.8PB per 9,216-chip pod", 12)
    label(x2 + 100, y0 + 218, "Gemma 4 training (v6e); the TPU plan in Section 16", 11, color="#b509ac")
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>" + "".join(P) + "</svg>"
    open(os.path.join(OUT, "hero-hardware.svg"), "w").write(svg)
    print("wrote hero-hardware.svg")


if __name__ == "__main__":
    fig_decode_breakdown()
    fig_rl_loop()
    fig_chip_intensity()
    fig_hero()
