"""The readout scorer's grouping rule, kept apart from experiments/score/score_readout.py so that experiments/tests/readout_groups_test.py
can test it: one name per construction.

construction(group, task): RAVEL's plain and JSON prompts of an attribute and an attribute's value form (gender, gender_value)
are one construction; SST's two framings of the same sentences are one; every other task is its own.
"""
import re


def construction(group, task):
    if group in ("ravel_attributes", "ravel_attributes_model_specific"):
        return "ravel:" + re.sub(r"_value$", "", task.replace("_json", ""))
    if group == "sst":
        return "sst"
    return f"{group}:{task}"
