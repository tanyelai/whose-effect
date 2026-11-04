"""Shared drawing kit for the paper's figures: hand-built SVG at print size, rendered to PDF by headless Chrome.

Colour roles (the same in every figure; tones after Paul Tol's palettes for scientific figures, colour-blind safe):
  net  sapphire   the network itself: its own interchange, the reference
  das  orange     what a trained objective reports beyond it (DAS and other learned directions)
  off  vermilion  off-route: an effect where the network's own interchange carries almost nothing
  nat  teal       natural directions read from the representation (mean difference, probe)
  unt  violet     untrained directions built from natural ones (the bisector, the readout gradient)
  grays           T, the input's whole effect (dark), axes and absent cells (light)
Type: Avenir Next for labels, Iowan Old Style italic for readings. Units are points at final print size (the paper's text
width is 397 pt), so a 7 pt label prints at 7 pt.
"""
import html
import math
import os
import shutil
import subprocess
from pathlib import Path

C = dict(ink="#1E1E1E", net="#2E5EAA", netl="#D7E2F3", das="#E07A1F", dasl="#FBE4CF", off="#C8372D", offl="#F6D8D5",
         nat="#2A9D8F", unt="#7B5EA7", tgray="#4D4D4D", gray="#9C9C9C", muted="#666666",
         line="#D9D9D9", soft="#ECECEC", paper="#FFFFFF", faint="#F6F6F6")
C.update(grayl="#C4C4C4")   # marks outside a claim's condition
SANS = "Avenir Next, Avenir, Helvetica Neue, Apple Symbols, sans-serif"   # Apple Symbols: the arrows Avenir lacks, as TrueType
SERIF = "Iowan Old Style, Charter, Georgia, serif"
CHROME = os.environ.get("CHROME") or next((c for c in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                                                     shutil.which("google-chrome"), shutil.which("chromium"),
                                                     shutil.which("chromium-browser")) if c and os.path.exists(c)), None)
NIPSW = 397.0                       # the paper's text width: one column, 5.5 in (397.48 pt)
HALFW, SIXW = 190.8, 238.5          # the appendix's 0.48 and 0.6 \textwidth floats, so those figures print at design size too
# One mark size for the scatters, at print size (in a 74 pt wide panel, discs of 2.6 pt would merge into a solid band):
# a disc of radius r and opacity op; the ring around a disc; the stroke of squared-error DAS's open square.
MK = dict(r=0.85, op=0.45, ring=1.5, ring_w=0.55, sq_w=0.4)
# One type scale for every figure, in points at print size (figures are designed at the width they print at, so no scaling):
# panel letter, panel title, subtitle, axis label, tick label, annotation: four sizes in all, the same in every figure.
# render() refuses any other size, and any text below MIN_FONT.
FS = dict(letter=9.0, title=8.0, sub=6.4, axis=6.4, tick=6.0, note=6.0)
SIZES = sorted(set(FS.values()))      # the only sizes a figure prints: 6, 6.4, 8 and 9 pt
MIN_FONT = 6.0
# small counts are spelled out in labels, as in the text
WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
        12: "twelve", 13: "thirteen", 14: "fourteen"}


def esc(s):
    return html.escape(s, quote=False)


_FACES = {}


def tw(s, size=7, weight=400):
    """The printed width of s in points, measured in the figures' font (Avenir Next; faces as Chrome maps the weights), so
    that text placed beside text does not collide. Needs PIL; the figures that place text by measure use it."""
    from PIL import ImageFont
    idx = {400: 7, 500: 5, 600: 2, 700: 0}[weight]
    if idx not in _FACES:
        _FACES[idx] = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 100, index=idx)
    return _FACES[idx].getlength(s) * size / 100.0


def T(x, y, s, size=7, fill=None, anchor="start", weight=400, serif=False, italic=False, ls=None, raw=False, op=None,
      rotate=None):
    a = f' font-family="{SERIF}"' if serif else ""
    a += ' font-style="italic"' if italic else ""
    a += f' letter-spacing="{ls}"' if ls else ""
    a += f' opacity="{op}"' if op is not None else ""
    a += f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill or C["ink"]}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{a}>{s if raw else esc(s)}</text>')


def L(x1, y1, x2, y2, stroke=None, w=0.8, dash=None, cap="round", op=None, marker=None, marker_start=None):
    a = f' stroke-dasharray="{dash}"' if dash else ""
    a += f' opacity="{op}"' if op is not None else ""
    a += f' marker-end="url(#{marker})"' if marker else ""
    a += f' marker-start="url(#{marker_start})"' if marker_start else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke or C["ink"]}" '
            f'stroke-width="{w}" stroke-linecap="{cap}"{a}/>')


def P(d, stroke=None, w=0.8, fill="none", dash=None, marker=None, op=None):
    a = f' stroke-dasharray="{dash}"' if dash else ""
    a += f' marker-end="url(#{marker})"' if marker else ""
    a += f' opacity="{op}"' if op is not None else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke or C["ink"]}" stroke-width="{w}" stroke-linecap="round" '
            f'stroke-linejoin="round"{a}/>')


def R(x, y, w, h, fill, r=1.5, stroke=None, sw=0.8, dash=None, op=None):
    a = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    a += f' stroke-dasharray="{dash}"' if dash else ""
    a += f' opacity="{op}"' if op is not None else ""
    if h < 0:
        y, h = y + h, -h
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{max(h, 0):.2f}" rx="{r}" fill="{fill}"{a}/>'


def circ(x, y, r, fill="none", stroke=None, sw=0.8):
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}"{s}/>'


def arrows(*named, size=5):
    out = "<defs>"
    for name, col in named:
        out += (f'<marker id="{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="{size}" markerHeight="{size}" '
                f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
    return out + "</defs>"


def head(x, letter, title, subs=(), y=9.0, sub_col=None):
    """A panel head: the letter, a title stating the finding, and at most two muted subtitle lines carrying the counts, at
    y + 8.6 and y + 16.0."""
    assert len(subs) <= 2, "at most two subtitle lines; the rest goes to the caption"
    off = 9.5 if letter else 0
    s = (T(x, y, letter, FS["letter"], weight=700) if letter else "") + T(x + off, y, title, FS["title"], weight=600)
    return s + "".join(T(x + off, y + 8.6 + 7.4 * k, t, FS["sub"], sub_col or C["muted"]) for k, t in enumerate(subs))


def key(rows, xs, y, pitch=7.6):
    """A key next to the marks it explains: rows of (mark, width, text) items, item c of every row starting at xs[c] so the
    rows align. mark(x, cy) draws the mark as in the plot, from x to x + width, centred on cy (2 pt above the baseline); the
    text follows, 6 pt muted regular."""
    out = ""
    for r, row in enumerate(rows):
        yy = y + r * pitch
        for kx, (mark, w, txt) in zip(xs, row):
            out += mark(kx, yy - 2.0) + T(kx + w + 2.5, yy, txt, FS["note"], C["muted"])
    return out


def kdisc(col, r=1.5, op=0.8):
    """Key mark: a disc (a scatter's point)."""
    return lambda x, cy: f'<circle cx="{x + r:.2f}" cy="{cy:.2f}" r="{r}" fill="{col}" fill-opacity="{op}"/>'


def kring(col, r=1.6, sw=0.6, fill="none"):
    """Key mark: an open circle or a ring."""
    return lambda x, cy: f'<circle cx="{x + r:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" stroke="{col}" stroke-width="{sw}"/>'


def kswatch(col, w=8.0, h=4.0, stroke=None):
    """Key mark: a filled swatch (a bar, a band, a cell)."""
    return lambda x, cy: R(x, cy - h / 2, w, h, col, r=0.4, stroke=stroke, sw=0.6)


def kline(col, w=8.0, sw=0.8, dash=None):
    """Key mark: a line (a top: dark grey dashed; a reference: solid)."""
    return lambda x, cy: L(x, cy, x + w, cy, col, sw, dash=dash)


def logscale(lo, hi, a0, a1):
    """Map [lo, hi] on a log scale to [a0, a1]; values outside are clamped to the edge."""
    l0, l1 = math.log10(lo), math.log10(hi)
    return lambda v: a0 + (math.log10(min(max(v, lo), hi)) - l0) / (l1 - l0) * (a1 - a0)


def linscale(lo, hi, a0, a1):
    return lambda v: a0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (a1 - a0)


def xaxis(x0, x1, y, X, ticks, title=None, sub=None, color=None):
    """A light baseline from x0 to x1 at y, tick labels (value, text) below it, then an axis title and an optional second line."""
    s = L(x0, y, x1, y, C["line"], 0.6)
    for v, t in ticks:
        s += T(X(v), y + 7.6, t, FS["tick"], C["gray"], anchor="middle")
    if title:
        s += T((x0 + x1) / 2, y + 16.2, title, FS["axis"], color or C["muted"], anchor="middle", weight=600 if color else 400)
    if sub:
        s += T((x0 + x1) / 2, y + 23.4, sub, FS["tick"], C["gray"], anchor="middle")
    return s


PREVIEW = Path(os.environ.get("FIG_PREVIEW") or Path(__file__).resolve().parents[2] / "build/figpreview")   # outside paper/


def render(name, w, h, body, outdir, min_font=None):
    """Print the figure to <outdir>/<name>.pdf at exactly w x h pt; the SVG source and a 300-dpi PNG preview go to PREVIEW
    (build/figpreview by default, which git ignores), so that paper/figures holds only the PDFs the paper includes. Refuses any
    text below min_font (default MIN_FONT) points: the figure is included at the width it is designed at, so this is its
    printed size."""
    import re
    small = sorted({float(m) for m in re.findall(r'font-size="([0-9.]+)"', body) if float(m) < (min_font or MIN_FONT)})
    if small:
        raise ValueError(f"{name}: text below {min_font or MIN_FONT} pt at print size: {small}")
    off = sorted({float(m) for m in re.findall(r'font-size="([0-9.]+)"', body)} - set(SIZES))
    if off:
        raise ValueError(f"{name}: text sizes off the figures' scale {SIZES}: {off}")
    outdir = Path(outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True); PREVIEW.mkdir(parents=True, exist_ok=True)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}pt" height="{h}pt" viewBox="0 0 {w} {h}" '
           f'font-family="{SANS}">{body}</svg>')
    (PREVIEW / f"{name}.svg").write_text(svg)
    page = (f'<!doctype html><html><head><meta charset="utf-8"><style>@page{{size:{w}pt {h}pt;margin:0}}'
            f'html,body{{margin:0;padding:0;background:#fff}}svg{{display:block}}</style></head><body>{svg}</body></html>')
    hp = PREVIEW / f"{name}.html"; hp.write_text(page)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--hide-scrollbars",
                    f"--print-to-pdf={outdir / (name + '.pdf')}", f"file://{hp}"], check=True, capture_output=True)
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(outdir / f"{name}.pdf"), str(PREVIEW / name)],
                   check=True)
    # the paper's PDF may hold only Type 1 or embedded TrueType fonts: a glyph that falls back to a Type 3 font stops here
    fonts = subprocess.run(["pdffonts", str(outdir / f"{name}.pdf")], check=True, capture_output=True, text=True).stdout
    if "Type 3" in fonts:
        raise ValueError(f"{name}: a Type 3 font in the PDF:\n{fonts}")
    return outdir / f"{name}.pdf"
