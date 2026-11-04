"""Run the readout units (experiments/matrix/registry.py): every unit by default, or a subset named by filters.

Each unit is one script run, one model process. A unit whose output counts (registry.unverified) is skipped, so a stopped
run resumes. A done record (experiments/matrix/done/<id>.json) holds the unit, the device, the time taken, the output's sha256, the
run record's id and the code fingerprint (registry.code_sha, read before and after the run: a change in between fails the
unit); a failed unit leaves experiments/matrix/failed/<id>.json with the end of its log, and the run goes on. Nothing runs while a
unit's data is not built or a model's pinned weights are not in the Hugging Face cache (experiments/run/download.py fetches them);
software other than the paper's (registry.STACK) is reported, not refused. Every call that gets past these checks writes a
manifest (experiments/matrix/runs/<time>.json) with its filters, its units and the ids it is to run.
On a GPU several small-model processes can share a device (--workers=N).
Usage: python -I experiments/matrix/run.py [--device=cuda] [--package=<name>] [--workers=1] [--models=a,b] [--kinds=a,b] [--groups=a,b]
       [--dry=1] [--status=1]
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry  # noqa: E402

REPO = registry.REPO
a = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
DEV, WORKERS = a.get("device", "cuda"), int(a.get("workers", "1"))
FILT = {k: a[k].split(",") for k in ("models", "kinds", "groups") if a.get(k)}
PKG = a.get("package")
if a.get("status") == "1":
    every = registry.units(device=DEV)
    bad = registry.unverified(None)
    for name in registry.PACKAGES:
        mine = [u for u in every if registry.package_of(u) == name]
        done = sum(u["output"] not in bad for u in mine)
        failed = sum((REPO / "experiments/matrix/failed" / f"{u['id']}.json").is_file() for u in mine)
        print(f"{name:17s} {len(mine):4d} units  {done:4d} done  {failed:3d} failed  {len(mine) - done:4d} to run")
    sys.exit(0)
if PKG:
    ms, ks = registry.PACKAGES[PKG]
    U = [u for u in registry.units(FILT.get("models") or ms, FILT.get("kinds") or ks, FILT.get("groups"), DEV)
         if u["model"] in ms and u["kind"] in ks]
else:
    U = registry.units(FILT.get("models"), FILT.get("kinds"), FILT.get("groups"), DEV)
DONE, FAILED, RUNS, LOGS = (REPO / "experiments/matrix" / d for d in ("done", "failed", "runs", "logs"))
for d in (DONE, FAILED, RUNS, LOGS):
    d.mkdir(parents=True, exist_ok=True)
BAD = registry.unverified(None, sorted({u["model"] for u in U}))
todo = [u for u in U if u["output"] in BAD]


def hub():
    try:
        from huggingface_hub.constants import HF_HUB_CACHE as h
    except ImportError:
        h = os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub"
    return Path(h).expanduser()


def cached(kind, name, rev):
    """True when the Hugging Face cache holds this repository's snapshot at this revision."""
    return (hub() / f"{kind}s--{name.replace('/', '--')}" / "snapshots" / rev).is_dir()


stop = [f"{registry.tag(m)}: {len(registry.missing_data(m))} tasks without data (bash experiments/run/data.sh {m})"
        for m in sorted({u["model"] for u in todo}) if registry.missing_data(m)]
stop += [f"{m}: revision {registry.REVISION[m]} not in the cache (python -I experiments/run/download.py {m})"
         for m in sorted({u["model"] for u in todo}) if not cached("model", m, registry.REVISION[m])]
if any(registry.reads_causalgym(u) for u in todo) and not cached("dataset", registry.CAUSALGYM_HF, registry.CAUSALGYM_REV):
    stop.append(f"{registry.CAUSALGYM_HF}: revision {registry.CAUSALGYM_REV} not in the cache (python -I experiments/run/download.py)")
if todo:
    r = subprocess.run([sys.executable, "-I", "-c", "import json, torch, transformers; "
                        "print(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__)))"],
                       capture_output=True, text=True)
    try:
        have = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
    except (ValueError, IndexError):
        have = {}
    for k, v in registry.STACK.items():
        if have.get(k) != v:
            print(f"NOTE: {k} {have.get(k)} installed; the paper's runs used {v}", flush=True)
if stop:
    print(f"{'NOTE' if a.get('dry') == '1' else 'STOP'}:\n  " + "\n  ".join(stop), flush=True)
    if a.get("dry") != "1":
        sys.exit(2)
stamp = time.strftime("%Y%m%d-%H%M%S")
json.dump(dict(started=stamp, device=DEV, workers=WORKERS, package=PKG, filters=FILT, units=[u["id"] for u in U],
               to_run=[u["id"] for u in todo]), open(RUNS / f"{stamp}.json", "w"), indent=1)
scope = (f"package {PKG}" if PKG else "every package") + ("" if not FILT else ", subset: " + json.dumps(FILT))
print(f"{len(U)} units in {scope}, {len(todo)} to run, device {DEV}", flush=True)
if a.get("dry") == "1":
    for u in todo:
        print(u["id"], u["kind"], registry.tag(u["model"]), u["group"])
    sys.exit(0)


def run(u):
    t0 = time.time(); log = LOGS / f"{u['id']}_{u['kind']}_{registry.tag(u['model'])}_{u['group']}.log"
    code = registry.code_sha(u["kind"])
    env = dict(os.environ, HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, "-I", str(REPO / u["script"]), *u["args"], f"--device={DEV}"], cwd=REPO, env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
    out = REPO / u["output"]
    rec = dict(log=str(log.relative_to(REPO)))
    rid = registry.run_id(rec)
    if p.returncode != 0 or not out.is_file() or registry.code_sha(u["kind"]) != code or not rid:
        tail = open(log).read().splitlines()[-15:]
        why = None if p.returncode != 0 or not out.is_file() else (
            "the code changed during the run" if registry.code_sha(u["kind"]) != code else "no run line in the log")
        json.dump(dict(unit=u, device=DEV, returncode=p.returncode, why=why, log=rec["log"], tail=tail),
                  open(FAILED / f"{u['id']}.json", "w"), indent=1)
        return u, False, time.time() - t0
    json.dump(dict(unit=u, device=DEV, seconds=round(time.time() - t0), log=rec["log"], run=rid, code=code,
                   output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), finished=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(DONE / f"{u['id']}.json", "w"), indent=1)
    (FAILED / f"{u['id']}.json").unlink(missing_ok=True)
    return u, True, time.time() - t0


with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for u, ok, secs in ex.map(run, todo):
        print(f"{'done' if ok else 'FAILED'} {u['id']} {u['kind']} {registry.tag(u['model'])} {u['group']} {secs:.0f}s", flush=True)
print("finished", time.strftime("%Y-%m-%d %H:%M:%S"))
