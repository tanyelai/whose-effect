"""The readout experiments as units: every kind x model x task group, each a single script run with a stable id.

A unit is one script run: a kind (the script and its fixed options), a model, and a task group (a data folder and an ordered
list of tasks, experiments/matrix/tasks.json). readout.py and readout_rank.py seed DAS start s (s = 0, 1, 2) with 2000 + 1000 s
plus a task's index in its group's list (with --seedfrom=causalgym, in CausalGym's sorted task list), so each list's order is
part of the unit. A unit's id is a hash of the script's bytes, its arguments (the model and its pinned revision among them)
and the bytes of every local data file it reads; a change to any of them gives the unit another id.
An output counts when its unit's done record (written by experiments/matrix/run.py) holds the output's sha256 and this tree's code
fingerprint (CODE plus the kind's CODE_EXTRA), and its run record in experiments/logs/<run>/ names the pinned model revision.
unverified() lists the outputs that do not count. The device and the software versions are recorded and reported, not
checked. With ANY_CODE=1 (experiments/lib/provenance.py) every output that exists counts, whatever code made it.
Usage: python -I experiments/matrix/registry.py [--models=a,b] [--kinds=a,b] [--groups=a,b] [--device=cuda|cpu] [--list]
   (with --device=cpu, GPT-2's units read out through a copy of the unembedding, --clone_unembed=1, and have other ids)
"""
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/lib"))
import pins  # noqa: E402
import provenance  # noqa: E402

SMALL = ["gpt2", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B",
         "EleutherAI/pythia-1b"]
LARGE = ["meta-llama/Llama-3.1-8B", "google/gemma-2-9b", "Qwen/Qwen2.5-7B", "EleutherAI/pythia-6.9b"]
SCALE = ["Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]   # sizes within the families above
QWEN32 = ["Qwen/Qwen2.5-32B"]                                                          # the Qwen family's largest size
ALL = SMALL + LARGE + SCALE + QWEN32
REVISION, STACK = pins.MODELS, pins.STACK
CAUSALGYM_HF, CAUSALGYM_REV = pins.CAUSALGYM
CODE = ("experiments/lib/common.py", "experiments/lib/pins.py", "owneffect/__init__.py", "owneffect/core.py", "owneffect/families.py")
TASKS = json.load(open(REPO / "experiments/matrix/tasks.json"))      # model tag -> group -> {data, tasks (None: all of CausalGym), options}

# kind -> (script, fixed options, the groups it runs on, suffix of the output's name)
KINDS = {"readout": ("experiments/scripts/readout.py", [], "all", ""),
         "readout_uncapped": ("experiments/scripts/readout.py", ["--uncapped=1"], "all", "__uncapped"),   # Gemma-2, its soft cap removed
         "readout_ce_law": ("experiments/scripts/readout_ce_law.py", [], "all", ""),                      # reads the readout unit's output
         "readout_rank": ("experiments/scripts/readout_rank.py", [], "main", "")}
CODE_EXTRA = {"readout": ("experiments/scripts/readout.py", "experiments/lib/measure.py"),
              "readout_uncapped": ("experiments/scripts/readout.py", "experiments/lib/measure.py"),
              "readout_ce_law": ("experiments/scripts/readout_ce_law.py", "experiments/lib/measure.py"),
              "readout_rank": ("experiments/scripts/readout_rank.py", "experiments/lib/rankmath.py", "experiments/lib/measure.py")}
OUTDIR = {"readout": "readout", "readout_uncapped": "readout", "readout_ce_law": "readout_ce_law", "readout_rank": "readout_rank"}
# the groups of the readout's main population (experiments/score/score_readout.py, MAIN): the rank sweep runs on these
MAIN = ("causalgym", "number_only", "kumon", "sst", "ioi_wang", "ioi_wang_label_signed", "ravel_attributes", "sentiment_templates",
        "ioi_ours_label_signed", "ioi_ours", "ravel_attributes_model_specific", "sentiment_templates_model_specific")
# package -> (models, kinds): the parts of the matrix, in the order they run (the CE law reads the readout's outputs)
PACKAGES = {"readout": (ALL, ["readout"]),
            "readout_uncapped": (["google/gemma-2-9b"], ["readout_uncapped"]),
            "readout_ce_law": (ALL, ["readout_ce_law"]),
            "readout_rank": (ALL, ["readout_rank"])}


def tag(model):
    return model.rstrip("/").split("/")[-1]


def groups(model):
    """Every task group of one model: name -> dict(data=..., tasks=[...] or None, options=[...])."""
    return TASKS[tag(model)]


def missing_data(model):
    """The (group, task) pairs whose train.json or test.json is not built yet (experiments/run/data.sh builds them)."""
    return [(g, t) for g, v in groups(model).items() if v["data"] not in ("causalgym", "fg")
            for t in v["tasks"] for s in ("train", "test") if not (REPO / v["data"] / t / f"{s}.json").is_file()]


def code_sha(kind=None):
    """The code fingerprint of a kind: sha256 over CODE and the kind's CODE_EXTRA, as they are in this tree."""
    h = hashlib.sha256()
    for p in CODE + CODE_EXTRA.get(kind, ()):
        h.update(p.encode() + b"\0" + hashlib.sha256((REPO / p).read_bytes()).digest())
    return h.hexdigest()


def reads_causalgym(u):
    """True when a unit reads CausalGym: its rows (--data=causalgym), or its task list for the DAS seed (--seedfrom=causalgym)."""
    return "--data=causalgym" in u["args"] or "--seedfrom=causalgym" in u["args"]


def run_id(done):
    """The run record (experiments/logs/<id>/) behind a done record: stored by run.py, or the script's "run <id> on" line in its log."""
    if done.get("run"):
        return done["run"]
    log = REPO / done.get("log", "")
    m = re.search(r"^run (\S+) on ", log.read_text(errors="replace"), re.M) if log.is_file() else None
    return m.group(1) if m else None


def why_not(u, done):
    """Why an output that its unit's done record names does not count, or None when it does."""
    if done.get("code") != code_sha(u["kind"]):
        return "made with other code than this tree's"
    rid = run_id(done)
    env = REPO / "experiments/logs" / rid / "env.json" if rid else None
    if not env or not env.is_file():
        return "no run record"
    if json.load(open(env)).get("model_revision") != REVISION[u["model"]]:
        return f"model revision {json.load(open(env)).get('model_revision')}, pinned {REVISION[u['model']]}"
    return None


def unverified(kinds, models=None, only_groups=None):
    """{output path: why it does not count} over the units of these kinds and models. With ANY_CODE=1 only a missing output
    is listed."""
    bad = {}
    for u in units(models, kinds, only_groups):
        out = REPO / u["output"]
        if provenance.ANY_CODE:
            if not out.is_file():
                bad[u["output"]] = "missing"
            continue
        done = REPO / "experiments/matrix/done" / f"{u['id']}.json"
        if not done.is_file():
            failed = (REPO / "experiments/matrix/failed" / f"{u['id']}.json").is_file()
            bad[u["output"]] = f"unit {u['id']} {'failed' if failed else 'has not run'}"
            continue
        d = json.load(open(done))
        if not out.is_file() or d["output_sha256"] != hashlib.sha256(out.read_bytes()).hexdigest():
            bad[u["output"]] = f"not the output unit {u['id']}'s done record holds"
        elif why := why_not(u, d):
            bad[u["output"]] = f"unit {u['id']}: {why}"
    return bad


def _files(data, tasks):
    if data in ("causalgym", "fg"):
        return []                                                      # read from the pinned Hub snapshots (experiments/lib/pins.py)
    return [REPO / data / t / f"{s}.json" for t in tasks or [] for s in ("train", "test") if (REPO / data / t / f"{s}.json").is_file()]


def units(models=None, kinds=None, only_groups=None, device="cuda"):
    out = []
    for model in models or ALL:
        for gname, g in groups(model).items():
            if only_groups and gname not in only_groups:
                continue
            for kind, (script, fixed, applies, suffix) in KINDS.items():
                if kinds and kind not in kinds:
                    continue
                if not any(model in ms and kind in ks for ms, ks in PACKAGES.values()):
                    continue                                           # a model-kind pair no package runs is not a unit
                if applies == "main" and gname not in MAIN:
                    continue
                opts = list(g["options"]) + (["--clone_unembed=1"] if tag(model) == "gpt2" and device == "cpu" else [])
                args = [f"--model={model}", f"--revision={REVISION[model]}", f"--data={g['data']}"] + \
                    ([f"--tasks={','.join(g['tasks'])}"] if g["tasks"] else []) + fixed + opts + [f"--tag=m_{gname}{suffix}"]
                h = hashlib.sha1((REPO / script).read_bytes())
                h.update(json.dumps(args).encode())
                for f in _files(g["data"], g["tasks"]):
                    h.update(f.read_bytes())
                out.append(dict(id=h.hexdigest()[:16], kind=kind, model=model, group=gname, script=script, args=args,
                                output=f"experiments/results/{OUTDIR[kind]}/{Path(script).stem}_{tag(model)}_m_{gname}{suffix}.json"))
    return out


def package_of(unit):
    names = [k for k, (ms, ks) in PACKAGES.items() if unit["model"] in ms and unit["kind"] in ks]
    assert len(names) == 1, (unit["id"], names)
    return names[0]


if __name__ == "__main__":
    a = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
    U = units(a["models"].split(",") if a.get("models") else None, a["kinds"].split(",") if a.get("kinds") else None,
              a["groups"].split(",") if a.get("groups") else None, a.get("device", "cuda"))
    if "--list" in sys.argv:
        for u in U:
            print(u["id"], u["kind"], tag(u["model"]), u["group"], " ".join(u["args"][3:])[:150])
    from collections import Counter
    print(len(U), "units;", dict(Counter(u["kind"] for u in U)), "| groups:", len({u["group"] for u in U}))
