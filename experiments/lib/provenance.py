"""Which code made an output. Every run records the sha256 of each file it executed; an output counts when each of those
files is byte-identical in this tree, so that results made by another copy of the code are never mixed with this one's.

ANY_CODE=1 in the environment counts outputs whatever code made them (their records still say which). It scores outputs made
by another copy of this code whose computations are the same, for example to check that this tree reproduces the paper's
numbers from the authors' own runs. Imports nothing heavy."""
import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANY_CODE = os.environ.get("ANY_CODE") == "1"


def sha(path):
    """sha256 of a file of this tree (a path relative to the repository)."""
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()


def not_this_code(recorded):
    """None when every file a run recorded ({path: sha256}) is byte-identical in this tree, or when ANY_CODE is set; else
    the reason the output does not count."""
    if ANY_CODE:
        return None
    if not recorded:
        return "its run recorded no code"
    for f, h in sorted(recorded.items()):
        if not (REPO / f).is_file() or sha(f) != h:
            return f"{f} is not the file its run executed"
    return None
