"""The `multi` question type: every option that applies, from zero to many, in one forward pass.

A `multi` question is sent to the model as an ordinary `choice` question with one extra option, the
cutoff (`REF`). The model scores the cutoff in the same softmax as the labels, so the cutoff is a
threshold the model sets for each input. An option is selected when it outscores the cutoff.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

REF = "REF"
REF_TEXT = "Cutoff: no remaining option applies"
MAX_OPTIONS = 254
MULTI_RULE = (
    "Every option that applies is a correct answer. There can be several, or none. "
    "Option REF is the cutoff. Score every option that applies above REF and every other option below it."
)
DEFAULT_INSTRUCTIONS = "Which options apply?"


def instructions(question: Mapping[str, Any]) -> str:
    base = question.get("instructions") or DEFAULT_INSTRUCTIONS
    return f"{base}\n{MULTI_RULE}"


def validate(question: Mapping[str, Any]) -> None:
    criteria = question.get("criteria")
    if not isinstance(criteria, Mapping) or not criteria:
        raise ValueError("a multi question needs a non-empty criteria object")
    if len(criteria) > MAX_OPTIONS:
        raise ValueError(f"a multi question takes at most {MAX_OPTIONS} options, got {len(criteria)}")
    if REF in criteria:
        raise ValueError(f"{REF!r} is reserved for the cutoff option")
    offset = question.get("offset", 0)
    if not isinstance(offset, (int, float)) or not math.isfinite(offset):
        raise ValueError("offset must be a finite number")


def to_choice(question: Mapping[str, Any]) -> dict[str, Any]:
    validate(question)
    return {
        "type": "choice",
        "instructions": instructions(question),
        "criteria": {**question["criteria"], REF: REF_TEXT},
    }


def margins(probabilities: Mapping[str, float]) -> dict[str, float]:
    ref = probabilities[REF]
    floor = 1e-30
    return {k: math.log(max(p, floor) / max(ref, floor)) for k, p in probabilities.items() if k != REF}


def decode(probabilities: Mapping[str, float], offset: float = 0.0) -> dict[str, Any]:
    m = margins(probabilities)
    ranked = sorted(m, key=m.get, reverse=True)
    return {
        "type": "multi",
        "selected": [k for k in ranked if m[k] > offset],
        "scores": {k: 1.0 / (1.0 + math.exp(-m[k])) for k in ranked},
        "cutoff": probabilities[REF],
    }


def _self_check() -> None:
    q = {"type": "multi", "criteria": {"a": "x", "b": None, "c": "z"}}
    c = to_choice(q)
    assert list(c["criteria"]) == ["a", "b", "c", REF] and MULTI_RULE in c["instructions"]
    out = decode({"a": 0.5, "b": 0.1, "c": 0.15, REF: 0.25})
    assert out["selected"] == ["a"], out
    assert abs(out["scores"]["a"] - 2 / 3) < 1e-9
    assert decode({"a": 0.5, "b": 0.1, "c": 0.15, REF: 0.25}, offset=-1.0)["selected"] == ["a", "c", "b"]
    assert decode({"a": 0.1, REF: 0.9})["selected"] == []
    for bad in ({"criteria": {}}, {"criteria": {REF: "x"}}, {"criteria": {"a": 1}, "offset": float("nan")}):
        try:
            validate(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


if __name__ == "__main__":
    _self_check()
    print("ok")
