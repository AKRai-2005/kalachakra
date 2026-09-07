"""World-model invariants.

The most important test here is causality. A forecaster that can see the future
is trivially good at forecasting, and the failure is silent - the number just
comes out high. `test_rollout_cannot_see_the_future` mutates every window after
the rollout origin and asserts the prediction does not move. It is the
KALACHAKRA equivalent of EKAGRA's socket test: a structural claim made
falsifiable.

The rest pin the things two rounds of debugging already got wrong once: that
rollout feeds its own predictions back rather than observations, that the
ATT&CK head is reachable from predicted latents, and that the anti-collapse
term actually responds to collapse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalachakra.data.episodes import ACTIONS, STAGES
from kalachakra.dynamics.world_model import (  # noqa: E402
    GraphEncoder, WorldModel, collapse_diagnostic, vicreg,
)
from kalachakra.state.graph import N_GLOBAL_F, N_NODE_F  # noqa: E402

B, T, N = 4, 12, 6
R = 3


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    np.random.seed(0)


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand(B, T, N, N_NODE_F, generator=g)
    adjs = torch.rand(B, T, N, N, generator=g)
    gs = torch.rand(B, T, N_GLOBAL_F, generator=g)
    acts = torch.zeros(B, T, dtype=torch.long)
    return xs, adjs, gs, acts


def model(conditioning="film"):
    return WorldModel(N_NODE_F, N_GLOBAL_F,
                      action_conditioning=conditioning).eval()


def activate_conditioning(m):
    """Give the action pathway something to say.

    FiLM is identity-initialised on purpose, so an *untrained* FiLM model is
    genuinely action-blind - gamma = 1, beta = 0, every action identical. That
    is correct behaviour, not a plumbing fault, so testing that the action
    reaches the rollout means perturbing the projections first. A concat model
    needs no help: its random embedding perturbs the output for free, which is
    a difference between the two worth remembering when reading early-epoch
    behaviour.
    """
    if m.action_conditioning == "film":
        torch.nn.init.normal_(m.film.to_gamma.weight, std=0.2)
        torch.nn.init.normal_(m.film.to_beta.weight, std=0.2)
    return m


# ------------------------------------------------------------------ encoder

def test_encoder_output_shape_and_finiteness():
    enc = GraphEncoder(N_NODE_F, N_GLOBAL_F, hidden=16, latent=8)
    xs, adjs, gs, _ = batch()
    z = enc(xs[:, 0], adjs[:, 0], gs[:, 0])
    assert z.shape == (B, 8)
    assert torch.isfinite(z).all()


def test_encoder_handles_an_all_zero_adjacency():
    """A window with no internal traffic is normal, and row-normalising a
    zero-sum adjacency is a division by zero waiting to happen."""
    enc = GraphEncoder(N_NODE_F, N_GLOBAL_F, hidden=16, latent=8)
    xs, _, gs, _ = batch()
    z = enc(xs[:, 0], torch.zeros(B, N, N), gs[:, 0])
    assert torch.isfinite(z).all()


def test_encoder_is_permutation_sensitive_but_not_order_dependent_on_batch():
    enc = GraphEncoder(N_NODE_F, N_GLOBAL_F, hidden=16, latent=8).eval()
    xs, adjs, gs, _ = batch()
    with torch.no_grad():
        z = enc(xs[:, 0], adjs[:, 0], gs[:, 0])
        z_rev = enc(xs[:, 0].flip(0), adjs[:, 0].flip(0), gs[:, 0].flip(0))
    assert torch.allclose(z, z_rev.flip(0), atol=1e-5), \
        "batch order changed per-sample encodings"


# ------------------------------------------------------------------ forward

def test_forward_shapes():
    m = model()
    xs, adjs, gs, acts = batch()
    out = m(xs, adjs, gs, acts, rollout_depth=R)
    S = T - R
    assert out["z"].shape == (B, T, m.latent)
    assert out["pred_z"].shape == (B, S, R, m.latent)
    assert out["pred_stage"].shape == (B, S, R, len(STAGES))
    assert out["rollout_depth"] == R and out["n_starts"] == S


def test_predicted_latents_reach_the_attck_head():
    """The bug that cost 0.18 AUC: the head was supervised only on encoded
    latents, then applied at inference to predicted ones - a distribution it
    had never seen. `pred_stage` existing at all is what keeps that fixed."""
    m = model()
    xs, adjs, gs, acts = batch()
    out = m(xs, adjs, gs, acts, rollout_depth=R)
    assert torch.isfinite(out["pred_stage"]).all()
    probs = torch.softmax(out["pred_stage"], dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5)


def test_rollout_depth_is_clamped_to_available_horizon():
    m = model()
    xs, adjs, gs, acts = batch()
    out = m(xs, adjs, gs, acts, rollout_depth=T + 50)
    assert out["rollout_depth"] <= T - 1
    assert out["n_starts"] >= 1


# ---------------------------------------------------------------- causality

def test_rollout_cannot_see_the_future():
    """THE test. A forecaster that reads future windows is trivially accurate
    and fails silently - the number just comes out high.

    Everything after the rollout origin is replaced with garbage; the
    prediction must be bit-identical.
    """
    m = model()
    xs, adjs, gs, _ = batch()
    t = 5
    with torch.no_grad():
        before = m.compromise_risk(xs, adjs, gs, t=t, horizon=4)

    xs2, adjs2, gs2 = xs.clone(), adjs.clone(), gs.clone()
    xs2[:, t + 1:] = 999.0
    adjs2[:, t + 1:] = 999.0
    gs2[:, t + 1:] = -42.0
    with torch.no_grad():
        after = m.compromise_risk(xs2, adjs2, gs2, t=t, horizon=4)

    assert torch.allclose(before, after, atol=0), \
        "rollout output changed when future windows changed - it is reading ahead"


def test_changing_the_past_does_change_the_rollout():
    """The complement: if nothing moves the prediction, the causality test
    above would pass for a model that ignores its input entirely."""
    m = model()
    xs, adjs, gs, _ = batch()
    t = 5
    with torch.no_grad():
        before = m.compromise_risk(xs, adjs, gs, t=t, horizon=4)
    xs2 = xs.clone()
    xs2[:, : t + 1] += 5.0
    with torch.no_grad():
        after = m.compromise_risk(xs2, adjs, gs, t=t, horizon=4)
    assert not torch.allclose(before, after, atol=1e-6), \
        "changing observed history had no effect - the model is ignoring input"


# ------------------------------------------------------------------ rollout

def test_rollout_shape_and_probability_range():
    m = model()
    xs, adjs, gs, _ = batch()
    with torch.no_grad():
        probs = m.rollout(xs, adjs, gs, t=4, horizon=5)
    assert probs.shape == (B, 5, len(STAGES))
    assert (probs >= 0).all() and (probs <= 1).all()
    assert torch.allclose(probs.sum(-1), torch.ones(B, 5), atol=1e-5)


def test_compromise_risk_is_a_probability():
    m = model()
    xs, adjs, gs, _ = batch()
    with torch.no_grad():
        r = m.compromise_risk(xs, adjs, gs, t=3, horizon=6)
    assert r.shape == (B,)
    assert (r >= 0).all() and (r <= 1).all()


def test_longer_horizons_do_not_reduce_risk():
    """Risk is 1 - prod(1 - p_k): the attacker only has to get there once, so
    extending the horizon cannot make compromise less likely."""
    m = model()
    xs, adjs, gs, _ = batch()
    with torch.no_grad():
        short = m.compromise_risk(xs, adjs, gs, t=2, horizon=2)
        long = m.compromise_risk(xs, adjs, gs, t=2, horizon=6)
    assert (long >= short - 1e-6).all()


def test_rollout_is_deterministic_in_eval_mode():
    m = model()
    xs, adjs, gs, _ = batch()
    with torch.no_grad():
        a = m.compromise_risk(xs, adjs, gs, t=4, horizon=4)
        b = m.compromise_risk(xs, adjs, gs, t=4, horizon=4)
    assert torch.equal(a, b)


# ----------------------------------------------------------------- actions

def test_action_is_plumbed_through_the_rollout():
    """A rollout that ignores its action argument cannot answer a
    counterfactual, and the failure is invisible - it just returns the same
    number twice. This was broken before the action-conditioned training."""
    m = activate_conditioning(model())
    xs, adjs, gs, _ = batch()
    with torch.no_grad():
        none = m.compromise_risk(xs, adjs, gs, t=4, horizon=5, action=0)
        iso = m.compromise_risk(xs, adjs, gs, t=4, horizon=5,
                                action=ACTIONS.index("isolate"), action_at=0)
    assert not torch.allclose(none, iso, atol=1e-7), \
        "action had no effect on the rollout"


def test_action_at_controls_when_the_action_applies():
    m = activate_conditioning(model())
    xs, adjs, gs, _ = batch()
    iso = ACTIONS.index("isolate")
    with torch.no_grad():
        immediate = m.compromise_risk(xs, adjs, gs, t=4, horizon=6,
                                      action=iso, action_at=0)
        later = m.compromise_risk(xs, adjs, gs, t=4, horizon=6,
                                  action=iso, action_at=4)
        never = m.compromise_risk(xs, adjs, gs, t=4, horizon=6, action=iso)
    assert not torch.allclose(immediate, later, atol=1e-7)
    assert torch.allclose(never, m.compromise_risk(xs, adjs, gs, t=4, horizon=6,
                                                   action=0), atol=0), \
        "action_at=None should mean the action is never applied"


def test_forward_uses_the_action_at_each_rollout_depth():
    """Using one action for a whole rollout makes an intervention that lands
    part-way through invisible - which is exactly the case the counterfactual
    asks about."""
    m = activate_conditioning(model())
    xs, adjs, gs, acts = batch()
    late = acts.clone()
    late[:, T // 2:] = ACTIONS.index("isolate")
    with torch.no_grad():
        a = m(xs, adjs, gs, acts, rollout_depth=R)["pred_z"]
        b = m(xs, adjs, gs, late, rollout_depth=R)["pred_z"]
    assert not torch.allclose(a, b, atol=1e-7)


def test_film_is_identity_at_initialisation():
    """Dropping FiLM into an existing model must change nothing until it has
    learned something. The alternative is a randomly-rescaled hidden state at
    step zero and an unstable first few epochs."""
    m = model("film")
    h = torch.randn(5, m.hidden_size)
    for a in range(len(ACTIONS)):
        act = torch.full((5,), a, dtype=torch.long)
        assert torch.allclose(m.film(h, act), h, atol=1e-6),             f"FiLM is not the identity at init for action {a}"


def test_both_conditioning_modes_produce_valid_rollouts():
    xs, adjs, gs, _ = batch()
    for mode in ("film", "concat"):
        m = model(mode)
        with torch.no_grad():
            r = m.compromise_risk(xs, adjs, gs, t=3, horizon=4)
        assert r.shape == (B,) and (r >= 0).all() and (r <= 1).all()


def test_conditioning_mode_is_validated():
    with pytest.raises(ValueError):
        WorldModel(N_NODE_F, N_GLOBAL_F, action_conditioning="attention")


def test_film_multiplies_rather_than_only_shifting():
    """The point of FiLM over concat: the action rescales the state, so the
    same action necessarily has a different effect on a different state. A
    purely additive pathway would move both states by the same vector."""
    m = activate_conditioning(model("film"))
    a = torch.zeros(2, dtype=torch.long)
    b = torch.ones(2, dtype=torch.long)
    h1 = torch.randn(2, m.hidden_size)
    h2 = h1 * 4.0
    d1 = (m.film(h1, b) - m.film(h1, a)).abs().mean()
    d2 = (m.film(h2, b) - m.film(h2, a)).abs().mean()
    assert d2 > d1 * 1.5,         "action effect did not scale with the state - modulation is purely additive"


# ------------------------------------------------------------- regularisers

def test_vicreg_penalises_a_collapsed_latent():
    """Latent collapse is the known failure mode of a predictive objective: the
    encoder can satisfy the loss by mapping everything to a constant."""
    collapsed = torch.ones(64, 16) * 0.5
    spread = torch.randn(64, 16)
    assert vicreg(collapsed) > vicreg(spread)


def test_vicreg_penalises_correlated_dimensions():
    d = torch.randn(128, 1).repeat(1, 8)          # every dim identical
    indep = torch.randn(128, 8)
    assert vicreg(d) > vicreg(indep)


def test_collapse_diagnostic_detects_collapse():
    """Reported alongside every result because a collapsed latent produces a
    low predictive loss and a model that has learned nothing."""
    collapsed = torch.ones(50, 32) * 0.3
    healthy = torch.randn(50, 32)
    c = collapse_diagnostic(collapsed)
    h = collapse_diagnostic(healthy)
    assert c["effective_rank"] < 2.0
    assert h["effective_rank"] > 8.0
    assert c["mean_std"] < h["mean_std"]
    assert c["dim"] == h["dim"] == 32


def test_collapse_diagnostic_accepts_a_sequence_shaped_latent():
    z = torch.randn(B, T, 24)
    d = collapse_diagnostic(z)
    assert d["dim"] == 24 and np.isfinite(d["effective_rank"])


# -------------------------------------------------------------- integration

def test_the_model_can_learn_a_signal_that_is_actually_there():
    """A smoke test that the whole graph optimises, not a quality claim.

    Two ways to get this wrong, both of which I did first:

      - regressing only the predictive term means chasing `sg(z_{t+1})`, a
        target the encoder is simultaneously moving. It is not guaranteed to
        fall and in this fixture it rises. That instability is precisely why
        the real objective anchors the latent with a stage cross-entropy and a
        VICReg term.
      - supervising the stage head on *random* labels floors the loss at
        ln(7) = 1.946, so nothing can decrease and the test measures nothing.

    So the stage labels here are a deterministic function of the input. There
    is a signal to find, and a connected model must find some of it.
    """
    torch.manual_seed(0)
    xs, adjs, gs, acts = batch()
    # learnable labels: bucket a global feature into stage indices
    stages = (gs[:, :, 0] * len(STAGES)).long().clamp(0, len(STAGES) - 1)

    m = WorldModel(N_NODE_F, N_GLOBAL_F)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        out = m(xs, adjs, gs, acts, rollout_depth=2)
        z, pz = out["z"], out["pred_z"]
        S, Rd = out["n_starts"], out["rollout_depth"]
        tgt = torch.stack([z[:, d + 1:d + 1 + S] for d in range(Rd)], 2)
        loss = (torch.nn.functional.mse_loss(pz, tgt.detach())
                + torch.nn.functional.cross_entropy(
                    m.stage_logits(z).reshape(-1, len(STAGES)), stages.reshape(-1))
                + 0.5 * vicreg(z))
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())

    assert all(np.isfinite(x) for x in losses), "training produced non-finite loss"
    assert min(losses[-5:]) < losses[0] - 0.05, (
        f"objective did not fall on a learnable signal: "
        f"{losses[0]:.4f} -> {min(losses[-5:]):.4f}")


def test_encoder_receives_gradient():
    m = WorldModel(N_NODE_F, N_GLOBAL_F)
    xs, adjs, gs, acts = batch()
    out = m(xs, adjs, gs, acts, rollout_depth=2)
    loss = out["pred_z"].pow(2).mean() + out["z"].pow(2).mean()
    loss.backward()
    grads = [p.grad for p in m.encoder.parameters() if p.grad is not None]
    assert grads, "no gradient reached the graph encoder"
    assert any(g.abs().sum() > 0 for g in grads)
