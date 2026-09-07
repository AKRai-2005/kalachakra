"""The horizon is now an argument. These pin what that must not break.

Experiment 16 sweeps the rollout horizon, which used to be two module constants
that had to agree. Turning them into arguments creates exactly one new way to be
wrong: training at one depth and scoring at another, which would not raise - it
would return a plausible number. The first two tests pin the defaults so no
previously reported result moves, and the rest pin the horizon's actual effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp01_leadtime import (  # noqa: E402
    HORIZON, ROLLOUT_DEPTH, build_dataset, train_world_model,
)
from exp14_ctu13_leadtime import wm_scores  # noqa: E402

from kalachakra.data.episodes import EpisodeConfig, EpisodeGenerator  # noqa: E402
from kalachakra.dynamics.world_model import TrainConfig  # noqa: E402
from kalachakra.state.graph import GraphBuilder, hosts_from_config  # noqa: E402


@pytest.fixture(scope="module")
def episodes():
    cfg = EpisodeConfig(n_windows=20, n_internal=6)
    gen = EpisodeGenerator(cfg, seed=0)
    return gen.generate(n_attack=4, n_benign=4), cfg


@pytest.fixture(scope="module")
def builder(episodes):
    cfg = episodes[1]
    return GraphBuilder(hosts_from_config(cfg.internal_prefix, cfg.n_internal))


def test_the_defaults_are_still_the_published_constants():
    """Every number in RESULTS.md predating experiment 16 was measured at these.
    If a default drifts, those results silently stop being reproducible."""
    assert HORIZON == 6
    assert ROLLOUT_DEPTH == 6


def test_omitting_the_horizon_gives_the_same_labels_as_before(episodes, builder):
    eps, _ = episodes
    _, _, _, _, y_default, _ = build_dataset(eps, builder)
    _, _, _, _, y_explicit, _ = build_dataset(eps, builder, horizon=HORIZON)
    assert np.array_equal(y_default, y_explicit)


def test_a_longer_horizon_labels_strictly_more_windows(episodes, builder):
    """The label is 'will compromise happen within H windows', so widening H can
    only add positives. This is also why AUC is not comparable across horizons -
    the task itself changes - and experiment 16 reports none."""
    eps, _ = episodes
    _, _, _, _, y2, meta = build_dataset(eps, builder, horizon=2)
    _, _, _, _, y8, _ = build_dataset(eps, builder, horizon=8)
    assert y8.sum() > y2.sum()
    # every positive at the short horizon is still positive at the long one
    assert np.all(y8[y2 == 1] == 1)


def test_horizon_cannot_label_past_the_compromise_window(episodes, builder):
    """`max(0, c - H)` clamps at the episode start; a horizon longer than the
    lead-up must not wrap round and label post-compromise windows."""
    eps, _ = episodes
    _, _, _, _, y, meta = build_dataset(eps, builder, horizon=999)
    for i, m in enumerate(meta):
        c = m["compromise"]
        if m["is_attack"] and c is not None:
            assert y[i][:c].all()
            assert not y[i][c:].any()
        else:
            assert not y[i].any()


def test_a_deeper_rollout_leaves_fewer_start_positions(episodes, builder):
    """Training at depth R uses T-R start positions, so R must stay under T. A
    silent clamp here would train a shallower model than the caller asked for
    and the sweep would compare two identical arms."""
    eps, cfg = episodes
    X, A, G, S, _, _ = build_dataset(eps, builder)
    wm = train_world_model(X, A, G, S, TrainConfig(epochs=1), verbose=False,
                           rollout_depth=4)
    import torch
    out = wm(torch.tensor(X), torch.tensor(A), torch.tensor(G),
             torch.zeros(X.shape[:2], dtype=torch.long), rollout_depth=4)
    assert out["rollout_depth"] == 4
    assert out["n_starts"] == cfg.n_windows - 4


def test_scoring_horizon_changes_the_scores(episodes, builder):
    """Risk accumulates as 1 - prod(1 - p_k) over the rollout, so a longer
    horizon must not return the identical series - if it did, `wm_scores` would
    be ignoring its argument and the whole sweep would be a no-op."""
    eps, _ = episodes
    X, A, G, S, _, _ = build_dataset(eps, builder)
    wm = train_world_model(X, A, G, S, TrainConfig(epochs=1), verbose=False)
    short = wm_scores(wm, X, A, G, horizon=2)
    long_ = wm_scores(wm, X, A, G, horizon=10)
    assert not np.allclose(short, long_)
    # accumulating over more steps cannot lower the risk at a given window
    assert (long_[:, :-1] >= short[:, :-1] - 1e-6).all()


def test_wm_scores_defaults_to_the_published_horizon(episodes, builder):
    eps, _ = episodes
    X, A, G, S, _, _ = build_dataset(eps, builder)
    wm = train_world_model(X, A, G, S, TrainConfig(epochs=1), verbose=False)
    assert np.allclose(wm_scores(wm, X, A, G),
                       wm_scores(wm, X, A, G, horizon=HORIZON))
