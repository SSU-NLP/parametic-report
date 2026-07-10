#!/usr/bin/env python3
"""Parametic Studio architecture figure (EMNLP demo) — clean SVG, computed coords."""

W, H = 1240, 690
SANS = "Helvetica, Arial, sans-serif"
MONO = "Menlo, monospace"
INK, MUT, LINE = "#1a1a1a", "#556070", "#3a3a3a"
PANEL, NEUT = "#f7f7f5", "#ececea"
AMB_S, AMB_F = "#b45309", "#fdf1e2"
BLU_S, BLU_F = "#1d4ed8", "#e8eefb"
GRN_S, GRN_F = "#15803d", "#e7f6ec"

s = []
def raw(x): s.append(x)
def rect(x, y, w, h, fill, stroke=LINE, sw=1.5, rx=7, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def txt(x, y, t, size=17, fill=INK, weight="normal", anchor="start", italic=False, mono=False):
    st = ' font-style="italic"' if italic else ""
    raw(f'<text x="{x}" y="{y}" font-family="{MONO if mono else SANS}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{st}>{t}</text>')
def badge(cx_, cy_, n, color):  # numbered step badge (renders everywhere, unlike U+2460)
    raw(f'<circle cx="{cx_}" cy="{cy_}" r="11" fill="{color}"/>')
    raw(f'<text x="{cx_}" y="{cy_+5}" font-family="{SANS}" font-size="15" font-weight="bold" fill="#ffffff" text-anchor="middle">{n}</text>')

raw(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">')
rect(0, 0, W, H, "#ffffff", stroke="none", sw=0, rx=0)
raw(f'''<defs>
  <marker id="aN" markerWidth="12" markerHeight="10" refX="8.5" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4 L0,8 Z" fill="{LINE}"/></marker>
  <marker id="aNs" markerWidth="12" markerHeight="10" refX="0.5" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M9,0 L0,4 L9,8 Z" fill="{LINE}"/></marker>
  <marker id="aB" markerWidth="12" markerHeight="10" refX="8.5" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4 L0,8 Z" fill="{BLU_S}"/></marker>
  <marker id="aA" markerWidth="12" markerHeight="10" refX="8.5" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4 L0,8 Z" fill="{AMB_S}"/></marker>
</defs>''')

txt(24, 40, "Parametic Studio &#8212; interactive discovery &amp; causal validation of the Coding Spot", 25, INK, weight="bold")
TOP, BOT = 62, H-18

# bands (explicit, non-overlapping)
A  = (24, 340)          # client   24..364
Bx = (376, 104)         # ws       376..480
Cx = (492, 472)         # kernel   492..964
Dx = (976, 250)         # storage  976..1226

# ---------- BAND A : CLIENT ----------
ax, aw = A
rect(ax, TOP, aw, BOT-TOP, "none", stroke=LINE, dash="6 4")
txt(ax+14, TOP+30, "Desktop app &#183; Tauri (Rust shell)", 20, INK, weight="bold")
txt(ax+14, TOP+52, "spawns / kills the kernel process", 15, MUT, italic=True)
txt(ax+14, TOP+71, "SSH tunnel &#8594; remote GPU box", 15, MUT, italic=True)
rpx, rpy, rpw = ax+12, TOP+84, aw-24
rph = BOT-10 - rpy
rect(rpx, rpy, rpw, rph, PANEL)
txt(rpx+14, rpy+30, "Interactive workspace (React)", 19, INK, weight="bold")
ix, iw = rpx+14, rpw-28
ey = rpy+44
rect(ix, ey, iw, 60, NEUT); txt(ix+12, ey+25, "Explorer", 17, INK, weight="bold")
txt(ix+12, ey+47, "models &#183; datasets &#183; regions", 15, INK)
vy = ey+72
rect(ix, vy, iw, 132, NEUT); txt(ix+12, vy+25, "Canvas &#8212; views", 17, INK, weight="bold")
txt(ix+12, vy+49, "output &#183; attention &#183; logit lens", 15, INK, mono=True)
raw(f'<text x="{ix+12}" y="{vy+71}" font-family="{MONO}" font-size="15" fill="{INK}"><tspan fill="{AMB_S}">importance</tspan> &#183; activations &#183; usage</text>')
raw(f'<text x="{ix+12}" y="{vy+93}" font-family="{MONO}" font-size="15" fill="{INK}"><tspan fill="{AMB_S}">spot</tspan> &#183; <tspan fill="{AMB_S}">tensors</tspan> &#183; <tspan fill="{AMB_S}">region control</tspan></text>')
txt(ix+12, vy+115, "train &#183; eval &#183; log", 15, INK, mono=True)
dy = vy+144
dh = rpy+rph-12 - dy
rect(ix, dy, iw, dh, AMB_F, stroke=AMB_S)
txt(ix+12, dy+25, "Prompt + knob dock", 17, INK, weight="bold")
txt(ix+12, dy+47, "&#945;-knobs: zero / scale / mean / rand", 15, INK)
txt(ix+12, dy+69, "top-k% slider &#183; A/B suspend&#8596;resume", 15, INK)
txt(ix+12, dy+91, "clean vs deleted compare", 15, INK)

# ---------- BAND B : WEBSOCKET ----------
bx, bw = Bx; cxB = bx+bw/2
rect(bx, TOP, bw, BOT-TOP, PANEL)
txt(cxB, TOP+30, "WebSocket", 19, INK, weight="bold", anchor="middle")
txt(cxB, TOP+51, ":8000/ws", 16, INK, anchor="middle", mono=True)
txt(cxB, TOP+69, "(or over SSH)", 14, MUT, anchor="middle", italic=True)
ry1 = TOP+96
raw(f'<line x1="{bx+10}" y1="{ry1}" x2="{bx+bw-10}" y2="{ry1}" stroke="{LINE}" stroke-width="2.5" marker-end="url(#aN)"/>')
txt(cxB, ry1-9, "requests", 14, MUT, anchor="middle", italic=True)
for i, t in enumerate(["generate &#183; spot", "intervene &#183; ppl", "eval &#183; sweep", "tensor_values"]):
    txt(cxB, ry1+21+i*19, t, 13, INK, anchor="middle", mono=True)
ry2 = ry1+140
raw(f'<line x1="{bx+bw-10}" y1="{ry2}" x2="{bx+10}" y2="{ry2}" stroke="{BLU_S}" stroke-width="2.5" marker-end="url(#aB)"/>')
txt(cxB, ry2-9, "streamed", 14, BLU_S, anchor="middle", italic=True)
for i, t in enumerate(["token &#183; attn", "spot_progress", "eval_progress", "contrast/sweep"]):
    txt(cxB, ry2+21+i*19, t, 13, BLU_S, anchor="middle", mono=True)
txt(cxB, BOT-28, "one WS / model", 13, MUT, anchor="middle", italic=True)
txt(cxB, BOT-11, "non-blocking", 13, MUT, anchor="middle", italic=True)

# ---------- BAND C : KERNEL ----------
cx, cw = Cx
rect(cx, TOP, cw, BOT-TOP, "none", stroke=LINE, dash="6 4")
txt(cx+14, TOP+30, "Analysis kernel &#183; Python (FastAPI / asyncio)", 19, INK, weight="bold")
txt(cx+14, TOP+51, "ModelSession &#8212; resident LLM (Hugging Face Transformers)", 15, MUT, italic=True)
kx, kw = cx+12, cw-24
gap = 12
colw = (kw-gap)/2
c1, c2 = kx, kx+colw+gap
r1y, rh = TOP+64, 116
# Row1: Locate | Observe
rect(c1, r1y, colw, rh, AMB_F, stroke=AMB_S)
badge(c1+24, r1y+18, "1", AMB_S); txt(c1+42, r1y+24, "Locate", 18, INK, weight="bold")
txt(c1+12, r1y+50, "&#931; | grad &#215; weight |", 16, INK)
txt(c1+12, r1y+73, "per-param top-k% &#8594; mask", 15, INK)
txt(c1+12, r1y+95, "= the Coding Spot", 15, AMB_S, italic=True)
rect(c2, r1y, colw, rh, BLU_F, stroke=BLU_S)
txt(c2+12, r1y+26, "Observe", 18, INK, weight="bold")
txt(c2+12, r1y+50, "token-by-token generate", 15, INK)
txt(c2+12, r1y+73, "attention / activation /", 15, INK)
txt(c2+12, r1y+95, "logit-lens / spot / tensors", 15, INK)
# Row2: Edit | Evaluate
r2y = r1y+rh+24
rect(c1, r2y, colw, rh, AMB_F, stroke=AMB_S)
badge(c1+24, r2y+18, "2", AMB_S); txt(c1+42, r2y+24, "Edit (reversible)", 18, INK, weight="bold")
txt(c1+12, r2y+50, "knobs: zero / scale /", 15, INK)
txt(c1+12, r2y+72, "mean / random", 15, INK)
raw(f'<text x="{c1+12}" y="{r2y+96}" font-family="{SANS}" font-size="15" fill="{INK}">controls: <tspan fill="{AMB_S}" font-weight="bold">random &#183; bottom</tspan></text>')
rect(c2, r2y, colw, rh, GRN_F, stroke=GRN_S)
badge(c2+24, r2y+18, "3", GRN_S); txt(c2+42, r2y+24, "Evaluate", 18, INK, weight="bold")
txt(c2+12, r2y+50, "PPL: code vs general", 15, INK)
txt(c2+12, r2y+72, "HumanEval pass@1", 15, INK)
txt(c2+12, r2y+96, "sandboxed unit-test exec", 15, GRN_S, italic=True)
# ①→② down arrow (left col), ②→③ right arrow (row2)
raw(f'<line x1="{c1+colw/2}" y1="{r1y+rh+2}" x2="{c1+colw/2}" y2="{r2y-2}" stroke="{AMB_S}" stroke-width="2" marker-end="url(#aA)"/>')
raw(f'<line x1="{c1+colw+2}" y1="{r2y+rh/2}" x2="{c2-2}" y2="{r2y+rh/2}" stroke="{AMB_S}" stroke-width="2" marker-end="url(#aA)"/>')
# Row3: causal experiment (full width)
r3y, r3h = r2y+rh+14, 84
rect(kx, r3y, kw, r3h, "#fbfbf9", stroke=LINE)
txt(kx+12, r3y+26, "Causal experiment (paper Table 1)", 18, INK, weight="bold")
txt(kx+12, r3y+50, "damage spot vs matched random / bottom across a top-k% sweep", 15, INK)
txt(kx+12, r3y+72, "&#8594; &#916;PPL (code / general) and &#916;pass@1", 15, INK)
# Row4: adapt
r4y, r4h = r3y+r3h+12, 60
rect(kx, r4y, kw, r4h, NEUT)
txt(kx+12, r4y+25, "Adapt (optional)", 18, INK, weight="bold")
txt(kx+12, r4y+47, "train: full &#183; spot-freeze &#183; spot-only &#183; LoRA &#8212; reversible", 15, INK)
# model strip
msy = r4y+r4h+12
msh = BOT-10 - msy
rect(kx, msy, kw, msh, PANEL)
mcy = msy+msh/2
txt(kx+kw/2, mcy-4, "resident model &#183; bf16 &#183; CUDA / MPS", 15, INK, anchor="middle")
txt(kx+kw/2, mcy+18, "multi-GPU parallel &#8212; spot &amp; eval split across cards", 15, INK, anchor="middle")

# ---------- BAND D : COMPUTE & STORAGE ----------
dx, dw = Dx; cxD = dx+dw/2
gy, gh = TOP, 68
rect(dx, gy, dw, gh, GRN_F, stroke=GRN_S)
txt(cxD, gy+30, "GPU &#215; N", 20, INK, weight="bold", anchor="middle")
txt(cxD, gy+53, "CUDA &#183; multi-card", 15, INK, anchor="middle")
hy, hh = gy+gh+22, 60
rect(dx, hy, dw, hh, PANEL)
txt(cxD, hy+27, "Hugging Face Hub", 17, INK, weight="bold", anchor="middle")
txt(cxD, hy+48, "models &#183; datasets", 15, INK, anchor="middle")
txt(cxD, hy+hh+30, "~/.parametic_studio/", 15, INK, anchor="middle", mono=True)
def cyl(y, h, label, sub=None):
    rxc = dw/2
    raw(f'<path d="M{dx},{y} v{h} a{rxc},11 0 0 0 {dw},0 v{-h}" fill="{NEUT}" stroke="{LINE}" stroke-width="1.5"/>')
    raw(f'<ellipse cx="{cxD}" cy="{y}" rx="{rxc}" ry="11" fill="{PANEL}" stroke="{LINE}" stroke-width="1.5"/>')
    txt(cxD, y+h/2+(2 if not sub else -4), label, 15, INK, anchor="middle", mono=True)
    if sub: txt(cxD, y+h/2+18, sub, 13, MUT, anchor="middle", italic=True)
sy = hy+hh+44
cyl(sy+11, 62, "regions/*.pt", "masks + importance")
cyl(sy+11+96, 46, "datasets/*.jsonl")
cyl(sy+11+96+82, 46, "runs/*.json.gz")

# ---------- inter-band data arrows ----------
midY = (TOP+BOT)/2
gmx = (cx+cw+dx)/2
raw(f'<line x1="{cx+cw}" y1="{midY}" x2="{dx}" y2="{midY}" stroke="{LINE}" stroke-width="2.5" marker-start="url(#aNs)" marker-end="url(#aN)"/>')
raw(f'<text transform="rotate(-90 {gmx} {midY-22})" x="{gmx}" y="{midY-22}" font-family="{SANS}" font-size="13" fill="{MUT}" text-anchor="middle" font-style="italic">load / save</text>')

raw('</svg>')
open("/root/doyun/demo/parametic-report/docs/figures/fig_studio_architecture.svg", "w").write("\n".join(s))
print("wrote", len("\n".join(s)), "bytes")
