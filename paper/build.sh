#!/usr/bin/env bash
# The paper from the scorers' outputs (bash experiments/score/score_all.sh first): every number in the text as LaTeX macros, the tables,
# the figures, then paper/main.pdf. Each step's log goes to build/logs/paper_<step>.log; the first failure stops the build.
# Needs Python with numpy, scipy, Pillow, huggingface_hub and transformers; the pinned CausalGym snapshot and Pythia's
# tokenizer in the Hugging Face cache (python -I experiments/run/download.py); experiments/data/ and experiments/logs/ besides the scorers' outputs;
# and latexmk with pdfLaTeX, as the journal compiles the paper. The figures need Google Chrome or Chromium (paper/scripts/svgfig.py),
# pdftoppm (poppler) and the macOS fonts Avenir Next and Iowan Old Style, in which they measure and set their text.
# Usage: bash paper/build.sh            (PY: the Python to use, default python3; LATEXMK: the latexmk binary, default latexmk;
#        FIGS=0 keeps the figures as they are)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"; PY="${PY:-python3}"; LATEXMK="${LATEXMK:-latexmk}"; LOG="$REPO/build/logs"
mkdir -p "$LOG"
step() { local name=$1; shift; echo "== $name"
  "$@" > "$LOG/paper_$name.log" 2>&1 || { echo "STOPPED at $name"; tail -20 "$LOG/paper_$name.log"; exit 1; }; }

step numbers_main     "$PY" -I paper/scripts/numbers_main.py
step numbers_cgmap    "$PY" -I paper/scripts/numbers_cgmap.py
step table_causalgym  "$PY" -I paper/scripts/table_causalgym.py "$REPO"
for t in table_models table_readout table_scale table_sentiment table_tasks; do step "$t" "$PY" -I "paper/scripts/$t.py"; done
[ "${FIGS:-1}" = 0 ] || for f in fig_law fig_readout fig_trace fig_npi_decomp fig_cgmap fig_benchmarks fig_indexswap \
         fig_guide_a fig_guide_b fig_guide_c fig_guide_d fig_guide_e fig_guide_f; do
  step "$f" "$PY" -I "paper/scripts/$f.py"
done
# Figure 1 as the README's picture (GitHub does not show a PDF inline)
[ "${FIGS:-1}" = 0 ] || pdftoppm -r 450 -png -singlefile paper/figures/fig_law.pdf paper/figures/fig_law
[ -f paper/main.tex ] || { echo "numbers, tables and figures written; the paper's LaTeX source is not in this tree"; exit 0; }
cd paper
"$LATEXMK" -pdf -interaction=nonstopmode -halt-on-error main.tex > "$LOG/paper_main.log" 2>&1 ||
  { echo "STOPPED at main.pdf"; tail -20 "$LOG/paper_main.log"; exit 1; }
# main.log is the last pass's log: what is left there is left in the PDF
if grep -E "^! |undefined" main.log; then echo "PROBLEMS in main.pdf's log: paper/main.log"; exit 1; fi
echo "paper/main.pdf written, $(pdfinfo main.pdf | awk '/^Pages/ {print $2}') pages"
pdffonts main.pdf | grep -q "Type 3" && echo "PROBLEM: a Type 3 font in main.pdf (pdffonts paper/main.pdf)"
grep -q '^\\def\\deicGrant{}' main.tex && echo "NOTE: the DeiC allocation number (\\deicGrant in main.tex) is not set"
grep -q '^\\def\\competing{}' main.tex && echo "NOTE: the competing-interests statement (\\competing in main.tex) is not set"
exit 0
