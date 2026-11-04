"""Shared pieces of the experiment scripts (experiments/scripts/): arguments, device and precision, model loading, data sources, and
the run record.

Every script that uses this module calls start(name) first and finish(...) last. start() fixes the precision (float32, TF32
off; on a GPU, deterministic kernels where they exist), loads the model with transformers (float32, eager attention), records
its family (owneffect.families), and opens a run directory experiments/logs/<run id>/ with the command, the environment (Python,
torch, transformers, CUDA, GPU), the model's resolved revision and the git state. finish() writes the outputs' paths and
sizes and the sha256 of the code the run executed, and appends one row to experiments/RUNS.csv.
"""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from owneffect import families  # noqa: E402
import pins  # noqa: E402

FG_REV = pins.KUMON[1]                    # Kumon & Yanaka's constructions at their pinned snapshot
CG_REV = pins.CAUSALGYM[1]                # CausalGym's tasks at their pinned snapshot
RUN = {}


def parse(argv):
    pos = [a for a in argv if not a.startswith("--")]
    kw = dict(a.lstrip("-").split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    return pos, kw


def start(name, argv):
    """Parse the command line, fix the precision, load the model and open the run record. Returns (pos, args, model, tok)."""
    pos, args = parse(argv)
    dev = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if dev.startswith("cuda"):                         # on the CPU the default kernels are deterministic
        torch.use_deterministic_algorithms(True, warn_only=True)
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    model_name = args.get("model", "gpt2")
    tok = AutoTokenizer.from_pretrained(model_name, revision=args.get("revision"))
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=args.get("revision"), dtype=torch.float32,
                                                 attn_implementation="eager").eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    tag = model_name.rstrip("/").split("/")[-1]
    run_id = f"{name}_{tag}_{'_'.join(pos)}_{args.get('tag', '')}_{time.strftime('%Y%m%d-%H%M%S')}".replace("__", "_").strip("_")
    out = REPO / "experiments/logs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    RUN.update(id=run_id, dir=out, t0=time.time(), name=name, model=model_name, tag=tag, dev=dev, argv=list(argv))
    env = dict(command=[sys.executable] + sys.argv, python=platform.python_version(), torch=torch.__version__,
               transformers=__import__("transformers").__version__, device=dev, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0) if dev.startswith("cuda") else None, host=platform.node(),
               model=model_name, model_revision=_revision(model), family=families.name(model),
               git=_git(), started=time.strftime("%Y-%m-%d %H:%M:%S"), tf32=False, dtype="float32")
    json.dump(env, open(out / "env.json", "w"), indent=1)
    print(f"run {run_id} on {dev}; model {model_name} ({env['family']}, revision {env['model_revision']})", flush=True)
    return pos, args, model, tok


def _revision(model):
    p = getattr(model.config, "_name_or_path", "")
    snap = Path(p) if Path(p).is_dir() else None
    if snap is None:
        try:
            snap = Path(hf_hub_download(p, "config.json")).parent
        except Exception:
            return None
    return snap.name if snap.parent.name == "snapshots" else str(snap)


def _git():
    def run(*c):
        try:
            return subprocess.run(["git", "-C", str(REPO), *c], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    diff = run("diff", "HEAD") or ""
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(diff), diff_sha1=hashlib.sha1(diff.encode()).hexdigest() if diff else None)


def _code():
    """sha256 of every file of this tree the run executed: the script and each local module it imported (provenance.py)."""
    files = {Path(sys.argv[0]).resolve()} | {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
    return {f.relative_to(REPO).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)
            if f.is_file() and REPO in f.parents and not {".venv", "site-packages"} & set(f.parts)}


def rows(source, task):
    """Train and test rows in CausalGym's format. source: causalgym | fg | <a local directory with <task>/{train,test}.json>.
    Returns dict(train=[...], test=[...])."""
    if source == "causalgym":
        return {s: [r for r in json.load(open(hf_hub_download("aryaman/causalgym", f"{s}.json", repo_type="dataset", revision=CG_REV)))
                    if r["task"] == task] for s in ("train", "test")}
    if source == "fg":
        return {s: json.load(open(hf_hub_download("kumoryo9/shared-mech", f"{task}/{f}.json", repo_type="dataset",
                                                  revision=FG_REV)))
                for s, f in (("train", "train"), ("test", "test_id"))}
    base = Path(source) if Path(source).is_absolute() else REPO / source
    return {s: json.load(open(base / task / f"{s}.json")) for s in ("train", "test")}


def causalgym_tasks():
    return sorted({r["task"] for r in json.load(open(hf_hub_download("aryaman/causalgym", "test.json", repo_type="dataset", revision=CG_REV)))})


def finish(outputs, summary=None):
    """Record the outputs and append the run to experiments/RUNS.csv."""
    files = [dict(path=str(Path(o).resolve().relative_to(REPO)), bytes=Path(o).stat().st_size) for o in outputs if Path(o).is_file()]
    secs = time.time() - RUN["t0"]
    json.dump(dict(outputs=files, seconds=round(secs), summary=summary, finished=time.strftime("%Y-%m-%d %H:%M:%S"), code_sha256=_code()),
              open(RUN["dir"] / "done.json", "w"), indent=1)
    runs_csv = REPO / "experiments/RUNS.csv"; first_row = not runs_csv.is_file()
    with open(runs_csv, "a", newline="") as f:
        w = csv.writer(f)
        if first_row:
            w.writerow(["run", "script", "model", "device", "seconds", "outputs", "args"])
        w.writerow([RUN["id"], RUN["name"], RUN["model"], RUN["dev"], round(secs), ";".join(x["path"] for x in files),
                    " ".join(RUN["argv"])])
    print(f"done {secs:.0f}s; record {RUN['dir'].relative_to(REPO)}", flush=True)
