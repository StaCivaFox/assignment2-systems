"""Build the Section 8 review PDF and a matching, standalone LaTeX source.

Dependencies: matplotlib, reportlab, pymupdf.
Run with any Python environment containing those packages. Equations are written
in LaTeX syntax, rendered with Matplotlib, and embedded as vector PDF objects.
The .tex file can also be compiled independently with pdflatex.
"""

from __future__ import annotations

import html
import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cs336-section8-matplotlib")

import matplotlib
import pymupdf
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent
STEM = "section8_parallelism_derivations"
W, H = A4
MARGIN = 48
WIDTH = W - 2 * MARGIN
INK = colors.HexColor("#193044")
ACCENT = colors.HexColor("#087F8C")
MUTED = colors.HexColor("#536576")
PALE = colors.HexColor("#EDF6F6")

matplotlib.rcParams.update({
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "savefig.transparent": True,
})
font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, filename in [
    ("Body", "DejaVuSans.ttf"),
    ("BodyBold", "DejaVuSans-Bold.ttf"),
    ("BodyItalic", "DejaVuSans-Oblique.ttf"),
    ("BodyBoldItalic", "DejaVuSans-BoldOblique.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
pdfmetrics.registerFontFamily(
    "Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBoldItalic"
)

styles = {
    "title": ParagraphStyle("title", fontName="BodyBold", fontSize=21, leading=26,
                            textColor=INK, spaceAfter=11),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=10, leading=15,
                               textColor=MUTED, spaceAfter=12),
    "h": ParagraphStyle("h", fontName="BodyBold", fontSize=11, leading=15,
                        textColor=ACCENT, spaceBefore=10, spaceAfter=5, keepWithNext=True),
    "p": ParagraphStyle("p", fontName="Body", fontSize=10, leading=15,
                        textColor=INK, spaceAfter=7),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8.6, leading=12.5,
                            textColor=MUTED, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=9, leading=13,
                           textColor=INK, alignment=TA_LEFT),
}

PAGES: list[dict] = []


def page(title, subtitle):
    PAGES.append({"title": title, "subtitle": subtitle, "items": []})


def p(text, small=False):
    PAGES[-1]["items"].append(("small" if small else "p", text))


def h(text):
    PAGES[-1]["items"].append(("h", text))


def eq(latex, result=False, size=13):
    PAGES[-1]["items"].append(("eq", latex, result, size))


def table(rows, widths):
    PAGES[-1]["items"].append(("table", rows, widths))


page("Section 8: parallelism derivations", "CS336 Assignment 2 · Review notes · Sections 8.1–8.5")
p("These notes collect our derivations for one gated feed-forward layer. They retain the exact ring factors, distinguish full tensors from shards, and separate exact bounds from large-degree approximations.")
table([
    ["Symbol", "Meaning"],
    ["B", "Global number of input rows (the handout’s batch size)"],
    ["D; D_FF", "Model dimension; feed-forward hidden dimension"],
    ["N", "Number of GPUs in the collective or parallelism group"],
    ["C", "Compute throughput per GPU, in FLOP/s"],
    ["W", "Egress bandwidth per GPU, in bytes/s"],
    ["S; s", "Full collective payload in bytes; bytes per element"],
], [0.18, 0.82])
h("Assumptions and the bottleneck criterion")
p("Use FP16 weights, activations, and communicated gradients: s = 2 bytes per element. Count only matrix multiplications. Ignore communication latency, reduction arithmetic, optimizer work, and memory-bandwidth limits. GPUs communicate concurrently.")
eq(r"t_{\mathrm{comp}}=\frac{\mathrm{FLOPs\ per\ GPU}}{C},\qquad t_{\mathrm{comm}}\leq t_{\mathrm{comp}}", True)
p("The inequality is the handout’s ideal compute-bound criterion. We compare forward and backward separately where requested. In Section 8.5, the problem explicitly specifies whether communication along the two axes overlaps.")
h("Why matrix multiplication costs approximately 2mnk FLOPs")
p("A product with shapes (m, k) and (k, n) has mn output elements. Each is a dot product requiring k multiplications and k − 1 additions.")
eq(r"\mathrm{FLOPs}=mn\,[k+(k-1)]=mn(2k-1)\simeq 2mnk")
p("A fused multiply-add counts as two floating-point operations. FLOPs is an operation count; FLOP/s is a rate. Using m, n, k here avoids confusing a matrix dimension with the hardware throughput C.")
p("Source: cs336_assignment2_systems.pdf, Section 8, pp. 40–46. Notation follows the handout: W₁ and W₂ are the two input projections; W₃ is the output projection. Prepared from our derivation discussion.", small=True)

page("8.1 · Ring communication", "Count rounds × bytes per round ÷ per-GPU bandwidth")
h("All-gather and reduce-scatter")
p("For all-gather, each GPU starts with one shard of size S/N and ends with the full tensor of size S. Each of N − 1 rounds sends one shard. Reduce-scatter has the same communication volume, with partial sums accumulated as shards circulate.")
eq(r"t_{\mathrm{AG}}=t_{\mathrm{RS}}=(N-1)\frac{S/N}{W}=\frac{N-1}{N}\frac{S}{W}", True)
p("For reduce-scatter, S is the full input tensor size on each GPU; the output shard has size S/N. Sending and receiving proceed concurrently in this model, so do not add another factor of two for reception.")
h("Standard ring all-reduce")
p("A reduce-scatter computes each GPU’s shard of the global sum. An all-gather distributes those reduced shards to every GPU.")
eq(r"t_{\mathrm{AR}}=t_{\mathrm{RS}}+t_{\mathrm{AG}}=2\frac{N-1}{N}\frac{S}{W}", True)
h("Problem: alternate ring all-reduce")
p("The alternate algorithm circulates whole original tensors. There are N − 1 rounds and each GPU sends S bytes per round.")
eq(r"t_{\mathrm{alt}}=(N-1)\frac{S}{W},\qquad \frac{t_{\mathrm{alt}}}{t_{\mathrm{AR}}}=\frac{N}{2}\quad(N>1)", True)
p("The algorithms tie at N = 2. Standard ring all-reduce is strictly faster for N > 2. At N = 1, both communication times are zero and the ratio is undefined.")
h("Full size versus shard size")
eq(r"S_{\mathrm{shard}}=\frac{S_{\mathrm{full}}}{N},\qquad t_{\mathrm{AG}}=\frac{N-1}{N}\frac{S_{\mathrm{full}}}{W}=(N-1)\frac{S_{\mathrm{shard}}}{W}")
p("Both forms are correct. Using a shard-sized S inside the first form divides by N twice. For an 8 GB full tensor on four GPUs, each GPU starts with 2 GB and receives three other shards: 6 GB in total.")
p("These are per-GPU elapsed-time models. Do not multiply by N to account for all devices: their transfers happen concurrently.")

page("8.2 · Data parallelism", "Shard the batch; replicate the weights; synchronize weight gradients")
h("The gated FFN and its matrix products")
eq(r"x_1=xW_1,\quad x_2=xW_2,\quad z=f(x_1)\odot x_2,\quad y=zW_3")
eq(r"W_1,W_2\in\mathbb{R}^{D\times D_{\mathrm{FF}}},\quad W_3\in\mathbb{R}^{D_{\mathrm{FF}}\times D},\quad P=3DD_{\mathrm{FF}}")
p("Each GPU processes B/N_DP rows. Backward contains six matmuls: one for dz, two contributing to dx, and three weight-gradient products. The elementwise derivatives are excluded from the FLOP count.")
eq(r"dz=dyW_3^\top,\qquad dx=dx_1W_1^\top+dx_2W_2^\top")
eq(r"dW_1=x^\top dx_1,\quad dW_2=x^\top dx_2,\quad dW_3=z^\top dy")
h("(a) Backward computation")
eq(r"\mathrm{FLOPs}_{\mathrm{bwd}}=6\left(2\frac{B}{N_{\mathrm{DP}}}DD_{\mathrm{FF}}\right)=\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{DP}}}", True)
h("(b) Backward communication")
p("All-reduce the three full weight gradients. Their combined full payload is independent of the batch size. Summing their costs is valid because this model ignores collective launch latency.")
eq(r"S_{\mathrm{grad}}=3DD_{\mathrm{FF}}\cdot 2=6DD_{\mathrm{FF}}\ \mathrm{bytes}")
eq(r"t_{\mathrm{comm,bwd}}=2\frac{N_{\mathrm{DP}}-1}{N_{\mathrm{DP}}}\frac{S_{\mathrm{grad}}}{W}=\frac{12(N_{\mathrm{DP}}-1)DD_{\mathrm{FF}}}{N_{\mathrm{DP}}W}", True)
h("(c) Compute-bound scaling limit")
eq(r"\frac{12(N_{\mathrm{DP}}-1)DD_{\mathrm{FF}}}{N_{\mathrm{DP}}W}\leq\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{DP}}C}\quad\Longrightarrow\quad N_{\mathrm{DP}}\leq 1+\frac{BW}{C}", True, 12)
p("D and D_FF cancel: widening the layer adds computation and gradient bytes in the same proportion. Increasing B adds computation while each parameter still contributes only one locally accumulated gradient to all-reduce.")

page("8.3 · Fully sharded data parallelism", "Shard persistent model state; reconstruct weights for local-batch computation")
h("(a) Forward and backward computation")
p("Each GPU still processes B/N_FSDP input rows. All-gather restores the complete weight matrices before computation, so the matmul counts match DP: three forward and six backward.")
eq(r"\mathrm{FLOPs}_{\mathrm{fwd}}=\frac{6BDD_{\mathrm{FF}}}{N_{\mathrm{FSDP}}},\qquad \mathrm{FLOPs}_{\mathrm{bwd}}=\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}", True)
h("(b) Communication")
p("The combined full size of the three weights is S = 6DD_FF bytes. Each GPU initially stores S/N_FSDP bytes. Use the full S in the ring formula; its division by group size is already included.")
p("Forward all-gathers the three weights. After forward, the full weights are discarded. Backward all-gathers them again, then reduce-scatters the three full gradient tensors into reduced local shards.")
eq(r"t_{\mathrm{comm,fwd}}=\frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}}\frac{6DD_{\mathrm{FF}}}{W}", True)
eq(r"t_{\mathrm{comm,bwd}}=2\frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}}\frac{6DD_{\mathrm{FF}}}{W}", True)
h("(c) Scaling limits")
eq(r"\frac{6(N_{\mathrm{FSDP}}-1)DD_{\mathrm{FF}}}{N_{\mathrm{FSDP}}W}\leq\frac{6BDD_{\mathrm{FF}}}{N_{\mathrm{FSDP}}C}\quad(\mathrm{forward})")
eq(r"\frac{12(N_{\mathrm{FSDP}}-1)DD_{\mathrm{FF}}}{N_{\mathrm{FSDP}}W}\leq\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{FSDP}}C}\quad(\mathrm{backward})")
eq(r"N_{\mathrm{FSDP}}\leq 1+\frac{BW}{C}\qquad\mathrm{in\ both\ phases}", True)
h("Why the two bounds agree")
p("Backward has twice the computation and twice the communication of forward. Their communication/computation ratios therefore match. The backward communication cost also matches DP’s all-reduce cost: one all-gather plus one reduce-scatter has the same ring cost as one all-reduce.")
p("Equal phase-wise scaling bounds do not imply equal step times. FSDP adds forward communication and reduces persistent memory per GPU. These formulas assume the weights are gathered again for backward.")

page("8.4 · Tensor parallel mechanics", "Column-shard the two input projections; row-shard the output projection")
p("Each GPU processes all B rows, but only a slice of the hidden dimension. The input and final output are replicated within the TP group.")
eq(r"W_1^{(i)},W_2^{(i)}\in\mathbb{R}^{D\times(D_{\mathrm{FF}}/N_{\mathrm{TP}})},\qquad W_3^{(i)}\in\mathbb{R}^{(D_{\mathrm{FF}}/N_{\mathrm{TP}})\times D}")
h("Forward")
eq(r"x_1^{(i)}=xW_1^{(i)},\qquad x_2^{(i)}=xW_2^{(i)}")
eq(r"z^{(i)}=f(x_1^{(i)})\odot x_2^{(i)},\qquad y^{(i)}=z^{(i)}W_3^{(i)}")
eq(r"y=\operatorname{AllReduce}_{\mathrm{sum}}\!\left(y^{(i)}\right)=\sum_{i=0}^{N_{\mathrm{TP}}-1}y^{(i)}")
p("The output projection consumes the local hidden slice directly, so no intermediate all-gather is needed. Only the partial outputs must be summed.")
h("(a) Backward, given the same dy on every TP rank")
eq(r"dz^{(i)}=dy\left(W_3^{(i)}\right)^\top")
eq(r"dx_1^{(i)}=dz^{(i)}\odot f'(x_1^{(i)})\odot x_2^{(i)},\qquad dx_2^{(i)}=dz^{(i)}\odot f(x_1^{(i)})", size=12)
eq(r"dW_1^{(i)}=x^\top dx_1^{(i)},\quad dW_2^{(i)}=x^\top dx_2^{(i)},\quad dW_3^{(i)}=\left(z^{(i)}\right)^\top dy", size=12)
eq(r"dx_{\mathrm{partial}}^{(i)}=dx_1^{(i)}\left(W_1^{(i)}\right)^\top+dx_2^{(i)}\left(W_2^{(i)}\right)^\top")
eq(r"dx=\operatorname{AllReduce}_{\mathrm{sum}}\!\left(dx_{\mathrm{partial}}^{(i)}\right)", True)
p("Each local weight gradient is already complete for that weight shard, so it needs no TP reduction. Each partial input gradient contains only one hidden slice’s contribution; summing across ranks reconstructs dx. This reduction is a sum, not an average.")
p("dy is the gradient of one logical replicated output. Under the handout’s backward convention, it is supplied to each rank without an additional TP all-reduce.", small=True)

page("8.4 · Tensor parallel costs", "Activation communication changes which dimension controls the scaling limit")
h("(b) Computation")
p("There are three forward and six backward matmuls. Each costs 2BDD_FF/N_TP FLOPs because the hidden dimension is sharded.")
eq(r"\mathrm{FLOPs}_{\mathrm{fwd}}=\frac{6BDD_{\mathrm{FF}}}{N_{\mathrm{TP}}},\qquad \mathrm{FLOPs}_{\mathrm{bwd}}=\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{TP}}}", True)
h("(c) Communication")
p("Forward all-reduces the partial output; backward all-reduces the partial input gradient. Both tensors have shape (B, D), hence a full payload of 2BD bytes.")
eq(r"t_{\mathrm{comm,fwd}}=t_{\mathrm{comm,bwd}}=2\frac{N_{\mathrm{TP}}-1}{N_{\mathrm{TP}}}\frac{2BD}{W}=\frac{4(N_{\mathrm{TP}}-1)BD}{N_{\mathrm{TP}}W}", True)
h("(d) Forward scaling limit")
eq(r"\frac{4(N_{\mathrm{TP}}-1)BD}{N_{\mathrm{TP}}W}\leq\frac{6BDD_{\mathrm{FF}}}{N_{\mathrm{TP}}C}")
eq(r"N_{\mathrm{TP}}\leq 1+\frac{3D_{\mathrm{FF}}W}{2C}\qquad(\mathrm{forward})", True)
h("Backward scaling limit")
eq(r"\frac{4(N_{\mathrm{TP}}-1)BD}{N_{\mathrm{TP}}W}\leq\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{TP}}C}")
eq(r"N_{\mathrm{TP}}\leq 1+\frac{3D_{\mathrm{FF}}W}{C}\qquad(\mathrm{backward})", True)
p("Forward is more restrictive: it has half the computation but the same communication. To keep both phases compute-bound, use the forward bound.")
h("Why the batch size cancels")
p("Increasing B increases both computation and activation payload proportionally. Increasing D_FF adds computation without enlarging the communicated tensors of shape (B, D). Thus TP benefits from a wide hidden dimension, while DP/FSDP benefit from a large batch.")

page("8.5 · Combining FSDP and TP", "Forward costs and the optimum when the two communication axes overlap")
eq(r"F=N_{\mathrm{FSDP}},\qquad T=N_{\mathrm{TP}},\qquad N=FT")
p("A rank (i, j) stores the following shards. FSDP all-gather reconstructs a TP weight slice, not the entire original weight matrix.")
eq(r"W_{1,2}^{(i,j)}:\left(\frac{D}{F},\frac{D_{\mathrm{FF}}}{T}\right),\quad W_3^{(i,j)}:\left(\frac{D_{\mathrm{FF}}}{T},\frac{D}{F}\right),\quad x^{(j)}:\left(\frac{B}{F},D\right)", size=12)
h("(a) Forward computation")
eq(r"\mathrm{FLOPs}_{\mathrm{fwd}}=\frac{6BDD_{\mathrm{FF}}}{FT},\qquad t_{\mathrm{comp}}=\frac{6BDD_{\mathrm{FF}}}{FTC}", True)
h("(b) Forward communication")
p("The full payload reconstructed within one FSDP group is 6DD_FF/T bytes. The TP all-reduce covers the local batch, so its full payload is 2BD/F bytes.")
eq(r"t_F=\frac{F-1}{F}\frac{6DD_{\mathrm{FF}}/T}{W}=\frac{6(F-1)DD_{\mathrm{FF}}}{FTW}")
eq(r"t_T=2\frac{T-1}{T}\frac{2BD/F}{W}=\frac{4(T-1)BD}{FTW}")
eq(r"t_{\mathrm{comm}}=\max(t_F,t_T)", True)
h("(c) Optimal degrees and maximum total GPU count")
eq(r"a=\frac{BW}{C},\qquad b=\frac{3D_{\mathrm{FF}}W}{2C},\qquad \frac{t_F}{t_{\mathrm{comp}}}=\frac{F-1}{a},\quad\frac{t_T}{t_{\mathrm{comp}}}=\frac{T-1}{b}", size=12)
p("Both communication times must be at most the compute time. The constraints are independent, so maximize both degrees.")
eq(r"F\leq 1+a,\quad T\leq 1+b\quad\Longrightarrow\quad F_*=1+a,\quad T_*=1+b")
eq(r"N\leq N_{\max}^{\mathrm{overlap}}=(1+a)(1+b)=\left(1+\frac{BW}{C}\right)\left(1+\frac{3D_{\mathrm{FF}}W}{2C}\right)", True, 12)
p("These are continuous optima. Actual degrees must be integers and support the required batch/weight partitioning. The max assumes the independent, overlapping communication resources specified in the problem.", small=True)

page("8.5 · When communication adds", "Part (d): exact continuous optimization, including single-axis boundary cases")
p("Keep a = BW/C and b = 3D_FF W/(2C). If the two collective families share network resources, their times add. The compute-bound constraint becomes:")
eq(r"t_F+t_T\leq t_{\mathrm{comp}}\quad\Longleftrightarrow\quad\frac{F-1}{a}+\frac{T-1}{b}\leq 1,\qquad F,T\geq 1", True, 12)
h("Interior optimum")
p("At the maximum the constraint is tight. Substitute T into N = FT; the result is a concave quadratic in F.")
eq(r"T=1+b-\frac{b}{a}(F-1),\qquad N(F)=F\left[1+b+\frac{b}{a}-\frac{b}{a}F\right]", size=12)
eq(r"\frac{dN}{dF}=1+b+\frac{b}{a}-\frac{2b}{a}F=0,\qquad\frac{d^2N}{dF^2}=-\frac{2b}{a}<0")
eq(r"F_* =\frac{a+1+a/b}{2},\qquad T_* =\frac{b+1+b/a}{2}")
eq(r"N_{\max}^{\mathrm{serial}}=\frac{(ab+a+b)^2}{4ab}\qquad(F_*\geq 1,\ T_*\geq 1)", True)
h("Boundary cases: a parallelism degree cannot be below 1")
eq(r"a(b+1)<b:\qquad F_*=1,\quad T_*=1+b,\quad N_{\max}=1+b", size=12)
eq(r"b(a+1)<a:\qquad T_*=1,\quad F_*=1+a,\quad N_{\max}=1+a", size=12)
p("Otherwise the interior formula applies. Equality at a boundary gives the same answer in either formula. An equivalent universal construction is:", small=True)
eq(r"F_*=\min\!\left(1+a,\max\!\left(1,\frac{a+1+a/b}{2}\right)\right),\quad T_*=1+b-\frac{b}{a}(F_*-1)", size=11.5)
h("Large-degree approximation: a and b both much greater than 1")
eq(r"F_*\simeq\frac{a}{2},\quad T_*\simeq\frac{b}{2},\quad N_{\max}^{\mathrm{serial}}\simeq\frac{ab}{4}=\frac{3BD_{\mathrm{FF}}W^2}{8C^2}", True, 12)
p("With overlap, the maximum is approximately ab. Thus overlap supports about four times as many GPUs in this limit. The factor of four is asymptotic, not an exact statement for every a and b. Equal communication-time allocation is also an approximation in this limit.", small=True)

page("Review sheet · What the bounds say", "A compact reference for checking future derivations")
h("Computation and communication at a glance")
p("For each one-dimensional strategy, write n for its group size. All three have forward FLOPs 6BDD_FF/n and backward FLOPs 12BDD_FF/n. With s = 2, their communication times are:")
eq(r"\mathrm{DP}:\qquad t_{\mathrm{fwd}}=0,\qquad t_{\mathrm{bwd}}=\frac{12(n-1)DD_{\mathrm{FF}}}{nW}")
eq(r"\mathrm{FSDP}:\qquad t_{\mathrm{fwd}}=\frac{6(n-1)DD_{\mathrm{FF}}}{nW},\qquad t_{\mathrm{bwd}}=\frac{12(n-1)DD_{\mathrm{FF}}}{nW}")
eq(r"\mathrm{TP}:\qquad t_{\mathrm{fwd}}=t_{\mathrm{bwd}}=\frac{4(n-1)BD}{nW}")
h("Generalizing to s bytes per communicated element")
p("Keeping the matmul throughput C fixed, replacing two-byte communication by s-byte communication gives these degree bounds:")
eq(r"N_{\mathrm{DP,bwd}},\ N_{\mathrm{FSDP,fwd}},\ N_{\mathrm{FSDP,bwd}}\ \leq\ 1+\frac{2BW}{sC}")
eq(r"N_{\mathrm{TP,fwd}}\leq 1+\frac{3D_{\mathrm{FF}}W}{sC},\qquad N_{\mathrm{TP,bwd}}\leq 1+\frac{6D_{\mathrm{FF}}W}{sC}")
p("For the 2D formulas, use a = 2BW/(sC) and b = 3D_FF W/(sC). Hardware throughput itself can change with precision; the formulas isolate the communication element-size effect.", small=True)
h("Why fixed-batch data parallelism eventually stops scaling")
eq(r"\rho_{\mathrm{DP}}\equiv\frac{t_{\mathrm{comm}}}{t_{\mathrm{comp}}}=\frac{s(n-1)C}{2BW}")
p("At fixed global B, adding GPUs shrinks the local computation while each rank still synchronizes the full gradient. The ratio grows with n. At fixed local batch q = B/n instead:")
eq(r"\rho_{\mathrm{DP}}=\frac{sC}{2qW}\left(1-\frac{1}{n}\right)\ \longrightarrow\ \frac{sC}{2qW}")
p("The ratio approaches a constant. Growing the global batch with the GPU count can preserve systems efficiency, but changes the optimization setup. Wider layers alone do not improve the ideal DP/FSDP ratio; a wider hidden dimension does improve the TP ratio.")
h("A reusable derivation method")
p("1. Write the local tensor shapes.  2. Count matmul FLOPs per rank.  3. Identify each collective’s <b>full</b> payload and group size.  4. Convert FLOPs and bytes into times.  5. Compare communication with computation, then optimize the degrees if there is more than one axis.")
p("Real performance also depends on utilization, latency, topology, memory capacity, and achievable overlap. The bounds above describe the handout’s simplified bandwidth model.", small=True)


PLACEMENTS = []
MATH_CACHE = {}


class MathBlock(Flowable):
    def __init__(self, latex, result=False, size=13):
        super().__init__()
        self.latex = latex
        self.result = result
        key = (latex, size)
        if key not in MATH_CACHE:
            buf = io.BytesIO()
            mathtext.math_to_image(
                "$" + latex.replace(r"\frac", r"\dfrac") + "$",
                buf, prop=FontProperties(size=size),
                format="pdf", color="#193044",
            )
            data = buf.getvalue()
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                rect = doc[0].rect
                MATH_CACHE[key] = (data, rect.width, rect.height)
        self.data, self.mw, self.mh = MATH_CACHE[key]
        self.spaceAfter = 4
        self.spaceBefore = 2

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        scale = min(1, (availWidth - 24) / self.mw)
        self.dw, self.dh = self.mw * scale, self.mh * scale
        self.pad = 8 if self.result else 3
        self.height = self.dh + 2 * self.pad
        return self.width, self.height

    def draw(self):
        if self.result:
            self.canv.setFillColor(PALE)
            self.canv.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
            self.canv.setStrokeColor(ACCENT)
            self.canv.setLineWidth(2)
            self.canv.line(0, 3, 0, self.height - 3)
        x, y = self.canv.absolutePosition((self.width - self.dw) / 2, self.pad)
        PLACEMENTS.append((self.canv.getPageNumber() - 1, x, y, self.dw, self.dh, self.data))


def furniture(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E3E8"))
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, H - 32, W - MARGIN, H - 32)
    canvas.setFillColor(MUTED)
    canvas.setFont("Body", 8)
    canvas.drawString(MARGIN, H - 24, "CS336 / SYSTEMS")
    canvas.drawRightString(W - MARGIN, H - 24, "SECTION 8 · DERIVATION NOTES")
    canvas.line(MARGIN, 35, W - MARGIN, 35)
    canvas.drawString(MARGIN, 22, "Ring collectives · DP · FSDP · TP · 2D parallelism")
    canvas.drawRightString(W - MARGIN, 22, str(doc.page))
    canvas.restoreState()


def tex_text(text):
    # Convert the limited paragraph markup used above into valid LaTeX.
    text = html.unescape(text)
    text = text.replace("<b>", "@@B@@").replace("</b>", "@@E@@")
    text = text.replace("<i>", "@@I@@").replace("</i>", "@@E@@")
    subs = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "−": "-", "×": r"$\times$",
            "÷": r"$\div$", "₁": r"$_1$", "₂": r"$_2$", "₃": r"$_3$",
            "’": "'", "–": "--", "—": "---", "·": r"\textperiodcentered{}"}
    text = "".join(subs.get(c, c) for c in text)
    return text.replace("@@B@@", r"\textbf{").replace("@@I@@", r"\emph{").replace("@@E@@", "}")


def write_tex():
    chunks = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,array,xcolor}
\usepackage[margin=19mm,headheight=14pt]{geometry}
\usepackage{fancyhdr,hyperref}
\definecolor{ink}{HTML}{193044}
\definecolor{accent}{HTML}{087F8C}
\hypersetup{colorlinks=true,linkcolor=accent,pdftitle={Section 8: Parallelism Derivations}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small CS336 / SYSTEMS}
\fancyhead[R]{\small SECTION 8: DERIVATION NOTES}
\fancyfoot[C]{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\abovedisplayskip}{7pt}\setlength{\belowdisplayskip}{7pt}
\color{ink}
\begin{document}
"""]
    for idx, pg in enumerate(PAGES):
        if idx:
            chunks.append(r"\clearpage")
        chunks.append(r"\section*{" + tex_text(pg["title"]) + "}")
        chunks.append(r"\textit{" + tex_text(pg["subtitle"]) + r"}\par")
        for item in pg["items"]:
            kind = item[0]
            if kind in ("p", "small"):
                chunks.append((r"{\small " if kind == "small" else "") + tex_text(item[1]) + ("}" if kind == "small" else "") + r"\par")
            elif kind == "h":
                chunks.append(r"\subsection*{" + tex_text(item[1]) + "}")
            elif kind == "eq":
                chunks.append(r"\[" + (r"\boxed{" if item[2] else "") + item[1] + ("}" if item[2] else "") + r"\]")
            elif kind == "table":
                rows = item[1]
                chunks.append(r"\begin{center}\begin{tabular}{p{0.14\linewidth}p{0.76\linewidth}}\toprule")
                for j, row in enumerate(rows):
                    chunks.append(" & ".join(tex_text(c) for c in row) + r"\\" + (r"\midrule" if j == 0 else ""))
                chunks.append(r"\bottomrule\end{tabular}\end{center}")
    chunks.append(r"\end{document}")
    (OUT / f"{STEM}.tex").write_text("\n\n".join(chunks) + "\n")


def build_pdf():
    PLACEMENTS.clear()
    story = []
    for idx, pg in enumerate(PAGES):
        if idx:
            story.append(PageBreak())
        story.extend([Paragraph(pg["title"], styles["title"]), Paragraph(pg["subtitle"], styles["subtitle"])])
        for item in pg["items"]:
            kind = item[0]
            if kind in styles:
                story.append(Paragraph(item[1], styles[kind]))
            elif kind == "eq":
                story.append(MathBlock(*item[1:]))
            elif kind == "table":
                rows = [[Paragraph(html.escape(c), styles["cell"]) for c in row] for row in item[1]]
                t = Table(rows, colWidths=[WIDTH * v for v in item[2]], hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), PALE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE6EB")),
                ]))
                story.extend([t, Spacer(1, 6)])

    base = io.BytesIO()
    doc = SimpleDocTemplate(base, pagesize=A4, rightMargin=MARGIN, leftMargin=MARGIN,
                            topMargin=49, bottomMargin=48,
                            title="Section 8: Parallelism Derivations", author="CS336 Review Notes")
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)
    with pymupdf.open(stream=base.getvalue(), filetype="pdf") as pdf:
        if len(pdf) != len(PAGES):
            starts = [p.get_text().splitlines()[3:5] for p in pdf]
            raise RuntimeError(f"Layout overflow: expected {len(PAGES)} pages, got {len(pdf)}; {starts}")
        for idx, x, y, w, h, data in PLACEMENTS:
            with pymupdf.open(stream=data, filetype="pdf") as formula:
                pdf[idx].show_pdf_page(pymupdf.Rect(x, H - y - h, x + w, H - y), formula, 0)
        pdf.set_toc([[1, pg["title"], i + 1] for i, pg in enumerate(PAGES)])
        pdf.set_metadata({"title": "Section 8: Parallelism Derivations",
                          "author": "CS336 Review Notes",
                          "subject": "Ring collectives, DP, FSDP, TP, and optimal 2D parallelism",
                          "keywords": "CS336, Section 8, FSDP, TP, data parallelism, derivations"})
        pdf.save(OUT / f"{STEM}.pdf", garbage=4, deflate=True)
        print(f"Saved {len(pdf)} pages, {len(PLACEMENTS)} vector equations")


if __name__ == "__main__":
    write_tex()
    build_pdf()
