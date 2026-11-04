"""OwnEffect: how much of a learned intervention's effect is the network's own?

The class OwnEffect (owneffect/core.py, whose docstring is the guide; owneffect/README.md shows its use) runs the paper's check
for a direction or subspace from any method; owneffect/families.py holds the supported model families and owneffect/stats.py
the intervals and spectra. OwnEffect is imported on first use, so owneffect.stats, which needs only numpy, loads without torch.
"""


def __getattr__(name):
    if name == "OwnEffect":
        from .core import OwnEffect
        return OwnEffect
    raise AttributeError(f"module 'owneffect' has no attribute {name!r}")
