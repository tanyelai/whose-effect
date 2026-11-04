"""One named set of inside maps (experiments/matrix/inside_sets.json: its models and tasks), scored by the inside-map scorer
(experiments/score/score_inside.py) and kept apart from every other set. The scorer always writes experiments/results/score_inside.json; here
it runs on the set's tasks and models, its output is moved to experiments/results/inside_sets/<set>.json beside a record of the set
(tasks, models, the tasks scored, the sha256 of the output, of the scorer and of the logit scales its instruments read, the
time), and whatever experiments/results/score_inside.json held before is put back byte for byte. A set already scored is kept unless
--again=1.
Usage: python -I experiments/score/score_inside_set.py --set=<name> [--tasks=<t1,t2,...>] [--models=<m1,m2,...>] [--again=1]
       (--tasks and --models replace the set's own lists)
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
NAME = A["set"]
SET = json.load(open(ROOT / "experiments/matrix/inside_sets.json")).get(NAME, {})
TASKS = A["tasks"].split(",") if A.get("tasks") else SET["tasks"]
MODELS = A.get("models") or ",".join(SET["models"])
SCORER, OUT = ROOT / "experiments/score/score_inside.py", ROOT / "experiments/results/score_inside.json"
SCALES = ROOT / "experiments/results/logit_scale.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()


dest = ROOT / "experiments/results/inside_sets" / f"{NAME}.json"
if dest.exists() and A.get("again") != "1":
    sys.exit(f"{dest.relative_to(ROOT)} exists: kept (--again=1 scores the set again)")
before = OUT.read_bytes() if OUT.is_file() else None
try:
    r = subprocess.run([sys.executable, "-I", str(SCORER), f"--tasks={','.join(TASKS)}", f"--models={MODELS}"],
                       capture_output=True, text=True)
    print(r.stdout[-4000:]); print(r.stderr[-2000:], file=sys.stderr)
    if r.returncode != 0:
        sys.exit(f"the scorer stopped (exit {r.returncode}); nothing kept")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(OUT), dest)
finally:
    if before is not None:
        OUT.write_bytes(before)
    elif OUT.exists() and not dest.exists():
        OUT.unlink()
got = json.load(open(dest)); scored = sorted({c["task"] for m in got["cells"] for c in got["cells"][m]})
meta = dict(set_name=NAME, tasks=TASKS, models=MODELS.split(","), scored_tasks=scored, output_sha256=sha(dest.read_bytes()),
            scorer_sha256=sha(SCORER.read_bytes()), logit_scale_sha256=sha(SCALES.read_bytes()) if SCALES.is_file() else None,
            finished=time.strftime("%Y-%m-%d %H:%M:%S"))
json.dump(meta, open(dest.with_suffix(".meta.json"), "w"), indent=1)
missing = sorted(set(TASKS) - set(scored))
print(f"kept {dest.relative_to(ROOT)}: {len(scored)} of {len(TASKS)} tasks scored" + (f"; not scored: {missing}" if missing else ""))
