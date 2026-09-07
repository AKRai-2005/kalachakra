"""Attribution must be verifiable, not decorative.

SIH26153 asks for "SHAP values or attention mechanisms identifying which
traffic attributes drive predictions". We have neither, so we use integrated
gradients - and the reason that is a defensible substitute rather than a
hand-wave is **completeness**: the attributions provably sum to
`f(input) - f(baseline)`, the same axiom SHAP's efficiency property gives.

Unlike SHAP we can check it on every call, so these tests do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalachakra.data.episodes import EpisodeConfig, EpisodeGenerator  # noqa: E402
from kalachakra.dynamics.world_model import WorldModel  # noqa: E402
from kalachakra.explain import attribute, explain_counterfactual  # noqa: E402
from kalachakra.state.graph import (  # noqa: E402
    GLOBAL_FEATURES, GraphBuilder, N_GLOBAL_F, N_NODE_F, NODE_FEATURES,
    hosts_from_config, stack_episode,
)

HORIZON = 6


@pytest.fixture(scope="module")
def fixture():
    """An untrained model is enough: completeness is a property of the method,
    not of whether the model is any good."""
    torch.manual_seed(0)
    cfg = EpisodeConfig(n_windows=14, n_internal=8)
    gen = EpisodeGenerator(cfg, seed=7)
    hosts = hosts_from_config(cfg.internal_prefix, cfg.n_internal)
    builder = GraphBuilder(hosts)
    ep = gen.generate(n_attack=2, n_benign=1, n_contained=0)[0]
    x, adj, novel, g, stage = stack_episode(builder.build_episode(ep))
    wm = WorldModel(N_NODE_F, N_GLOBAL_F)
    wm.eval()
    return (wm, torch.tensor(x)[None], torch.tensor(adj)[None],
            torch.tensor(g)[None], hosts)


def test_completeness_holds(fixture):
    """The axiom that makes this a substitute for SHAP rather than a guess."""
    wm, xs, adjs, gs, hosts = fixture
    a = attribute(wm, xs, adjs, gs, t=6, horizon=HORIZON, hosts=hosts, steps=48)
    gap = abs(a.risk - a.baseline_risk)
    assert a.completeness_error < max(1e-3, 0.02 * gap), (
        f"completeness violated: err {a.completeness_error:.2e} on a gap of {gap:.4f}")


def test_attribution_names_every_real_feature(fixture):
    wm, xs, adjs, gs, hosts = fixture
    a = attribute(wm, xs, adjs, gs, t=6, horizon=HORIZON, hosts=hosts, steps=8)
    assert set(a.node_features) == set(NODE_FEATURES)
    assert set(a.global_features) == set(GLOBAL_FEATURES)
    assert set(a.hosts) == set(hosts)


def test_windows_after_the_decision_contribute_nothing(fixture):
    """Only windows up to t are encoded. If a future window ever influenced an
    attribution, the rollout would be leaking data it has not seen."""
    wm, xs, adjs, gs, hosts = fixture
    early = attribute(wm, xs, adjs, gs, t=3, horizon=HORIZON, hosts=hosts, steps=8)
    xs2 = xs.clone()
    xs2[:, 4:] = xs2[:, 4:] * 7.0 + 3.0          # scramble everything after t=3
    late = attribute(wm, xs2, adjs, gs, t=3, horizon=HORIZON, hosts=hosts, steps=8)
    for k in NODE_FEATURES:
        assert early.node_features[k] == pytest.approx(late.node_features[k], abs=1e-5), \
            f"{k} changed when only post-decision windows were altered"


def test_batch_of_more_than_one_is_refused(fixture):
    wm, xs, adjs, gs, hosts = fixture
    with pytest.raises(ValueError, match="one episode"):
        attribute(wm, torch.cat([xs, xs]), torch.cat([adjs, adjs]),
                  torch.cat([gs, gs]), t=6, horizon=HORIZON, hosts=hosts)


def test_gradient_x_input_runs_but_carries_no_guarantee(fixture):
    """Kept as the fast path. It has no completeness property and the result
    records which method produced it, so the two can never be confused."""
    wm, xs, adjs, gs, hosts = fixture
    a = attribute(wm, xs, adjs, gs, t=6, horizon=HORIZON, hosts=hosts,
                  method="gradient_x_input")
    assert a.method == "gradient_x_input"
    assert set(a.node_features) == set(NODE_FEATURES)


def test_film_starts_as_the_identity(fixture):
    """FiLM is zero-initialised, so at init the action provably does nothing.

    This is deliberate - the modulation starts as an identity and learns to
    deviate (exp10 measured gamma drifting 0.23 and beta 0.15 over twenty
    epochs). Pinning it means a future change to the initialisation announces
    itself, instead of silently making every untrained branch comparison look
    like a working intervention.
    """
    wm, xs, adjs, gs, hosts = fixture
    out = explain_counterfactual(wm, xs, adjs, gs, t=6, horizon=HORIZON,
                                 hosts=hosts, action=1, steps=8)
    assert set(out) == {"none", "action"}
    assert out["none"].risk == pytest.approx(out["action"].risk, abs=1e-9), \
        "FiLM is no longer the identity at initialisation"


def test_the_action_reaches_the_rollout_once_film_is_not_identity(fixture):
    """The counterfactual is the whole product. If the action ever stopped
    reaching the transition, every branch comparison would silently return the
    same number - which is precisely what an untrained model does."""
    wm, xs, adjs, gs, hosts = fixture
    with torch.no_grad():                    # nudge FiLM off the identity
        for p_ in list(wm.film.to_gamma.parameters()) + \
                  list(wm.film.to_beta.parameters()):
            p_.add_(torch.randn_like(p_) * 0.1)
    out = explain_counterfactual(wm, xs, adjs, gs, t=6, horizon=HORIZON,
                                 hosts=hosts, action=1, steps=8)
    assert out["none"].risk != out["action"].risk, \
        "the action is not reaching the rollout"


def test_a_custom_baseline_changes_the_reference(fixture):
    wm, xs, adjs, gs, hosts = fixture
    zero = attribute(wm, xs, adjs, gs, t=6, horizon=HORIZON, hosts=hosts, steps=8)
    quiet = attribute(wm, xs, adjs, gs, t=6, horizon=HORIZON, hosts=hosts, steps=8,
                      baseline=(xs.mean(dim=(1, 2), keepdim=True), gs.mean(dim=1, keepdim=True)))
    assert zero.baseline == "zeros" and quiet.baseline == "quiet-reference"
    assert zero.baseline_risk != quiet.baseline_risk
