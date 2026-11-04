"""Download what the GPU runs read into the Hugging Face cache, at the pinned revisions (experiments/lib/pins.py): CausalGym's tasks,
Kumon & Yanaka's constructions and the models' weights. Llama and Gemma are gated: HF_TOKEN must name an account with access.
Usage: python -I experiments/run/download.py [<model> ...]      (default: every model of the paper; the two datasets always)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import pins  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

for name, rev in (pins.CAUSALGYM, pins.KUMON):
    print(name, rev, snapshot_download(name, repo_type="dataset", revision=rev), flush=True)
for m in sys.argv[1:] or list(pins.MODELS):
    print(m, pins.MODELS[m], snapshot_download(m, revision=pins.MODELS[m],
                                               allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"]), flush=True)
