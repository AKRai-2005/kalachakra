"""The console makes claims on screen. This checks the data behind them.

`frontend/index.html` prints "Completeness holds, so the ranking is the model's
own arithmetic" under every attribution chart. That sentence is only true if
the shipped numbers actually satisfy the axiom, and the console decides which
sentence to print by recomputing the residual in the browser - so a payload
that quietly drifted would flip forty cards to "indicative only" in front of a
judge, with no warning anywhere upstream.

These tests re-run the browser's own arithmetic in Python. They skip when the
payload has not been generated, because `exp13` takes minutes and CI should not
be forced to train a model to run a unit test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalachakra.state.graph import GLOBAL_FEATURES, NODE_FEATURES  # noqa: E402

PAYLOAD = ROOT / "results" / "demo_trajectories.json"
DATA_JS = ROOT.parent / "frontend" / "data.js"

# The console's rule, character for character:
#   resid < Math.max(1e-3, 0.02 * Math.abs(gap))
ABS_FLOOR = 1e-3
REL_TOL = 0.02


def _tolerance(gap: float) -> float:
    return max(ABS_FLOOR, REL_TOL * abs(gap))


@pytest.fixture(scope="module")
def payload():
    if not PAYLOAD.exists():
        pytest.skip("run experiments/exp13_demo_trajectories.py to generate it")
    return json.loads(PAYLOAD.read_text(encoding="utf-8"))


def test_every_case_carries_an_attribution(payload):
    for c in payload["cases"]:
        assert "attribution" in c, f"{c['id']} has no attribution"


def test_attributions_name_every_feature(payload):
    """Truncating the ranking would break completeness silently - the sum would
    no longer reach the gap, and the console would start calling every case
    loose. Hosts are the one list that *is* truncated, because they are a
    re-aggregation of the same node attributions rather than a separate term."""
    for c in payload["cases"]:
        a = c["attribution"]
        assert [k for k, _ in a["node"]] and \
            set(k for k, _ in a["node"]) == set(NODE_FEATURES), c["id"]
        assert set(k for k, _ in a["global"]) == set(GLOBAL_FEATURES), c["id"]


def test_completeness_holds_for_every_shipped_case(payload):
    """The sentence the console prints, checked against the numbers it prints it
    from."""
    loose = []
    for c in payload["cases"]:
        a = c["attribution"]
        total = sum(v for _, v in a["node"]) + sum(v for _, v in a["global"])
        gap = a["risk"] - a["baseline_risk"]
        if abs(total - gap) > _tolerance(gap):
            loose.append(f"{c['id']}: resid {abs(total - gap):.2e} on gap {gap:+.4f}")
    assert not loose, "completeness violated on:\n  " + "\n  ".join(loose)


def test_the_converged_flag_does_not_overclaim(payload):
    """A case may legitimately fail to converge - the ceiling is finite. What it
    may not do is say it converged when it did not, because that is the flag
    deciding whether a judge sees a caveat."""
    for c in payload["cases"]:
        a = c["attribution"]
        total = sum(v for _, v in a["node"]) + sum(v for _, v in a["global"])
        gap = a["risk"] - a["baseline_risk"]
        if a["converged"]:
            assert abs(total - gap) <= _tolerance(gap), \
                f"{c['id']} claims convergence it does not have"


def test_the_baseline_is_recorded_and_is_not_zeros(payload):
    """An attribution whose reference point is hidden is not an explanation, and
    zeros is the wrong reference for this model - an empty graph scores about
    0.77 compromise risk, so zeros would explain only the sliver above it."""
    for c in payload["cases"]:
        a = c["attribution"]
        assert a["baseline"] == "quiet-reference", c["id"]
        assert a["baseline_risk"] < 0.2, \
            f"{c['id']}: quiet reference scores {a['baseline_risk']:.3f}, which is " \
            "not quiet - the benign mean has drifted and attributions now read " \
            "against an alarming state"


def test_the_attribution_caveat_ships_with_the_data(payload):
    """The console renders this text; if it goes missing the panel renders a
    ranked feature list with nothing saying it explains the model rather than
    the network."""
    note = payload["limits"].get("attribution", "")
    assert "spurious" in note and "SHAP" in note


def test_data_js_matches_the_json(payload):
    """The console loads `data.js`, not the JSON, because `fetch()` of a sibling
    file is blocked under `file://`. Two copies means they can disagree."""
    if not DATA_JS.exists():
        pytest.skip("frontend/data.js not generated")
    text = DATA_JS.read_text(encoding="utf-8")
    body = text[text.index("=") + 1:].rstrip().rstrip(";")
    assert json.loads(body) == payload
