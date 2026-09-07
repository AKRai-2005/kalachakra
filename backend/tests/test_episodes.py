"""Episode-generator invariants.

Experiment 02's entire counterfactual claim rests on one property of this file:
that a matched pair shares a bit-identical prefix and differs only in the
intervention. If that is not true, "the model predicted the effect of the
action" means nothing, because something else also differed.

These tests also pin the attacker-skill latent, which exists because an earlier
version made attacks fail by a coin flip with no observable correlate - making
the forecasting task not merely hard but impossible in principle.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalachakra.data.episodes import (  # noqa: E402
    ACTIONS, COMPROMISE_STAGE, STAGE_IDX, STAGES, EpisodeConfig,
    EpisodeGenerator, stage_sequence,
)

CFG = EpisodeConfig(n_windows=40, n_internal=24)


def flows_signature(ep, upto):
    """A comparable rendering of every flow in the first `upto` windows."""
    out = []
    for w in ep.windows[:upto]:
        out.append(tuple(sorted(
            (f.src, f.dst, f.dport, f.packets, f.bytes, f.syn, f.rst, f.novel)
            for f in w.flows)))
    return tuple(out)


# ------------------------------------------------------------------ structure

def test_stage_order_is_a_kill_chain():
    assert STAGES[0] == "BENIGN"
    assert STAGE_IDX["RECON"] < STAGE_IDX["LATERAL_MOVEMENT"]
    assert STAGE_IDX["LATERAL_MOVEMENT"] == COMPROMISE_STAGE
    assert STAGE_IDX["EXFILTRATION"] == len(STAGES) - 1


def test_generation_is_deterministic():
    a = EpisodeGenerator(CFG, seed=7).generate(n_attack=6, n_benign=6)
    b = EpisodeGenerator(CFG, seed=7).generate(n_attack=6, n_benign=6)
    assert [stage_sequence(x) for x in a] == [stage_sequence(x) for x in b]
    assert [flows_signature(x, CFG.n_windows) for x in a] == \
           [flows_signature(x, CFG.n_windows) for x in b]


def test_benign_episodes_never_leave_the_benign_stage():
    eps = EpisodeGenerator(CFG, seed=3).generate(n_attack=0, n_benign=12)
    for e in eps:
        assert not e.is_attack
        assert set(stage_sequence(e)) == {0}
        assert e.compromise_window is None
        assert all(w.flows for w in e.windows), "benign episodes still carry traffic"


def test_compromise_window_is_the_first_crossing():
    eps = EpisodeGenerator(CFG, seed=11).generate(n_attack=30, n_benign=0)
    for e in eps:
        seq = stage_sequence(e)
        crossings = [i for i, s in enumerate(seq) if s >= COMPROMISE_STAGE]
        if crossings:
            assert e.compromise_window == crossings[0]
            assert e.compromised
        else:
            assert e.compromise_window is None and not e.compromised


def test_attack_stages_do_not_run_backwards_while_active():
    """The attacker advances or stalls; it does not regress. Containment drops
    the episode back to BENIGN, which is a different thing and is allowed."""
    eps = EpisodeGenerator(CFG, seed=5).generate(n_attack=25, n_benign=0)
    for e in eps:
        active = [s for s in stage_sequence(e) if s > 0]
        assert active == sorted(active), f"{e.id} regressed: {active}"


# --------------------------------------------------------------- skill latent

def test_skill_is_recorded_and_bounded():
    eps = EpisodeGenerator(CFG, seed=9).generate(n_attack=20, n_benign=5)
    for e in eps:
        assert e.skill is not None
        assert 0.0 <= e.skill <= 1.0


def test_skill_drives_both_observable_behaviour_and_outcome():
    """The whole point of the latent: failure must have an observable cause.

    Low-skill attackers scan more broadly (visible in the traffic) and reach
    compromise less often (the thing to forecast). If these were independent,
    progression would be unpredictable in principle and the experiment would
    measure noise - which is exactly what an earlier version did.
    """
    gen = EpisodeGenerator(CFG, seed=13)
    lo = [gen._run(__import__("random").Random(1000 + i), f"lo{i}", True, skill=0.05)
          for i in range(25)]
    hi = [gen._run(__import__("random").Random(1000 + i), f"hi{i}", True, skill=0.95)
          for i in range(25)]

    def recon_breadth(eps):
        widths = []
        for e in eps:
            for w in e.windows:
                if w.stage == "RECON":
                    widths.append(len({f.dst for f in w.flows}))
        return sum(widths) / max(1, len(widths))

    assert recon_breadth(lo) > recon_breadth(hi), \
        "low-skill attackers should scan more broadly"
    lo_rate = sum(e.compromised for e in lo) / len(lo)
    hi_rate = sum(e.compromised for e in hi) / len(hi)
    assert hi_rate > lo_rate, \
        f"high skill should compromise more often ({hi_rate:.2f} vs {lo_rate:.2f})"


def test_low_skill_population_fails_more_often_than_the_general_one():
    """`generate_contained` draws from the low-skill end; it does not guarantee
    containment, and the name oversells it.

    Measured: 37.5% of the low-skill population never reaches compromise
    against 20.0% of the general population (n=120 each). Both numbers matter.
    The gap is the design working - failure has an observable cause. That it is
    a gap and not a guarantee is why these episodes are mixed into the label
    set rather than assumed to be negatives.
    """
    gen = EpisodeGenerator(CFG, seed=17)
    low = gen.generate_contained(120)
    general = gen.generate(n_attack=120, n_benign=0)

    def fail_rate(eps):
        return sum(1 for e in eps if e.compromise_window is None) / len(eps)

    assert fail_rate(low) > fail_rate(general) + 0.05, (
        f"low-skill draws should fail more often "
        f"({fail_rate(low):.2f} vs {fail_rate(general):.2f})")
    assert (sum(e.skill for e in low) / len(low)
            < sum(e.skill for e in general) / len(general))


def test_contained_negatives_still_carry_attack_activity():
    """Negatives must not be quiet episodes.

    If every episode containing attack traffic also ends in compromise, a model
    that fires on the first port scan scores a perfect false-alarm rate while
    having forecast nothing. These are what make "will this progress" a
    different question from "is something happening".
    """
    con = EpisodeGenerator(CFG, seed=17).generate_contained(40)
    negatives = [e for e in con if e.compromise_window is None]
    assert negatives, "no contained negatives generated"
    for e in negatives:
        assert e.is_attack
    with_activity = [e for e in negatives
                     if any(w.stage != "BENIGN" for w in e.windows)]
    assert len(with_activity) >= len(negatives) // 2, (
        "most contained negatives should show attack stages - otherwise they "
        "are indistinguishable from benign traffic and prove nothing")


# ------------------------------------------------------------- counterfactual

def test_counterfactual_pairs_share_a_bitwise_identical_prefix():
    """THE property experiment 02 depends on.

    Any divergence before the fork means a measured effect could be caused by
    something other than the intervention.
    """
    pairs = EpisodeGenerator(CFG, seed=4242).generate_counterfactual_pairs(25)
    assert pairs
    for factual, counter in pairs:
        f = factual.fork_window
        assert f is not None and f >= 1
        assert flows_signature(factual, f) == flows_signature(counter, f), \
            f"{factual.id}: prefix differs before the fork"
        assert stage_sequence(factual)[:f] == stage_sequence(counter)[:f]


def test_the_two_branches_differ_only_in_the_action():
    pairs = EpisodeGenerator(CFG, seed=4242).generate_counterfactual_pairs(15)
    for factual, counter in pairs:
        assert factual.action == "none"
        assert counter.action == "isolate"
        assert counter.parent_id == factual.id
        assert factual.fork_window == counter.fork_window


def test_intervention_lands_before_compromise():
    """An intervention after the attacker has already moved laterally asks a
    question with an obvious answer. An earlier version placed the fork at
    random and most interventions were too late to matter."""
    pairs = EpisodeGenerator(CFG, seed=4242).generate_counterfactual_pairs(40)
    for factual, _ in pairs:
        c = factual.compromise_window
        if c is not None:
            assert factual.fork_window < c, \
                f"{factual.id}: fork at {factual.fork_window} but compromise at {c}"


def test_intervention_changes_outcomes_without_always_working():
    """Containment is probabilistic and heterogeneous. If it always worked the
    counterfactual would be trivial; if it never did there would be no effect
    to predict."""
    pairs = EpisodeGenerator(CFG, seed=4242).generate_counterfactual_pairs(60)
    prevented = sum(1 for f, c in pairs if f.compromised and not c.compromised)
    failed = sum(1 for f, c in pairs if f.compromised and c.compromised)
    assert prevented > 0, "no intervention ever prevented compromise"
    assert failed > 0, "every intervention succeeded - no effect to predict"


def test_containment_probability_is_heterogeneous_and_recorded():
    pairs = EpisodeGenerator(CFG, seed=4242).generate_counterfactual_pairs(60)
    ps = [c.p_contain for _, c in pairs if c.p_contain is not None]
    assert len(ps) > 10
    assert min(ps) < max(ps) - 0.2, \
        "containment probability should vary; a constant effect is not worth predicting"
    assert all(0.0 <= p <= 1.0 for p in ps)


def test_action_sequence_is_zero_before_the_fork_and_set_after():
    pairs = EpisodeGenerator(CFG, seed=4242).generate_counterfactual_pairs(10)
    for factual, counter in pairs:
        assert set(factual.action_sequence()) == {0}, "no-action branch must be all zeros"
        seq = counter.action_sequence()
        f = counter.fork_window
        assert set(seq[:f]) == {0}
        assert set(seq[f:]) == {ACTIONS.index("isolate")}
        assert len(seq) == len(counter.windows)
