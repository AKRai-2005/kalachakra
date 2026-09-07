"""Lead-time metric invariants.

This metric is the project's headline comparison, so its definitions matter
more than its implementation. Two of them were wrong at some point and both
would have flattered us:

  - lead was not clamped, so an alarm raised *after* compromise could have
    subsidised the mean with a negative number
  - negatives were only quiet benign episodes, so a model that fired on the
    first port scan scored a perfect false-alarm rate while forecasting nothing

Both are pinned below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalachakra.metrics.leadtime import (  # noqa: E402
    EpisodeScores, curve, detection_rate, false_alarm_rate, lead_at_fpr,
    lead_time,
)


def ep(name, scores, compromise=None, is_attack=True):
    return EpisodeScores(episode_id=name, scores=np.asarray(scores, dtype=float),
                         is_attack=is_attack, compromise_window=compromise)


# -------------------------------------------------------------------- lead

def test_lead_is_the_gap_between_alarm_and_compromise():
    e = ep("a", [0.1, 0.2, 0.9, 0.9, 0.9], compromise=4)
    assert lead_time(e, 0.5) == 2.0          # fires at window 2, compromise at 4


def test_lead_is_zero_when_the_alarm_comes_too_late():
    """Firing after the attacker has already moved laterally is worth nothing.
    Letting this go negative would let late alarms subsidise the mean."""
    e = ep("a", [0.1, 0.1, 0.1, 0.1, 0.9], compromise=2)
    assert lead_time(e, 0.5) == 0.0


def test_lead_is_zero_when_no_alarm_fires():
    assert lead_time(ep("a", [0.1] * 5, compromise=3), 0.5) == 0.0


def test_non_compromised_episodes_have_no_lead():
    """Includes contained attacks, not just benign ones."""
    assert lead_time(ep("c", [0.9] * 5, compromise=None), 0.5) == 0.0
    assert lead_time(ep("b", [0.9] * 5, compromise=None, is_attack=False), 0.5) == 0.0


def test_earlier_alarms_give_more_lead():
    early = ep("e", [0.9, 0.9, 0.9, 0.1], compromise=3)
    late = ep("l", [0.1, 0.1, 0.9, 0.1], compromise=3)
    assert lead_time(early, 0.5) > lead_time(late, 0.5)


# ------------------------------------------------------------- false alarms

def test_negatives_include_contained_attacks_not_just_benign():
    """The definition that makes this forecasting rather than detection.

    A contained attack is a negative: it never reaches compromise. A model that
    fires on any attack activity must pay for it here.
    """
    contained = ep("con", [0.9, 0.9, 0.9], compromise=None, is_attack=True)
    quiet = ep("ben", [0.1, 0.1, 0.1], compromise=None, is_attack=False)
    assert false_alarm_rate([contained, quiet], 0.5) == 0.5


def test_false_alarms_are_counted_per_episode_not_per_window():
    """An analyst who gets one spurious page has been interrupted once."""
    noisy = ep("n", [0.9] * 100, compromise=None, is_attack=False)
    single = ep("s", [0.9] + [0.1] * 99, compromise=None, is_attack=False)
    assert false_alarm_rate([noisy], 0.5) == false_alarm_rate([single], 0.5) == 1.0


def test_false_alarm_rate_falls_as_the_threshold_rises():
    negs = [ep(f"n{i}", [0.1, 0.5, 0.8], compromise=None, is_attack=False)
            for i in range(4)]
    rates = [false_alarm_rate(negs, t) for t in (0.05, 0.6, 0.9, 0.99)]
    assert rates == sorted(rates, reverse=True)


def test_detection_rate_counts_only_pre_compromise_catches():
    caught = ep("c", [0.9, 0.1, 0.1], compromise=2)
    missed = ep("m", [0.1, 0.1, 0.9], compromise=2)
    assert detection_rate([caught, missed], 0.5) == 0.5


# ------------------------------------------------------------------- curve

def test_curve_is_monotone_in_the_threshold():
    eps = [ep(f"a{i}", np.linspace(0.1, 0.95, 10), compromise=8) for i in range(5)]
    eps += [ep(f"n{i}", np.linspace(0.1, 0.7, 10), compromise=None,
               is_attack=False) for i in range(5)]
    c = curve(eps, n_thresholds=25)
    assert list(c["fpr"]) == sorted(c["fpr"], reverse=True)
    assert list(c["detection_rate"]) == sorted(c["detection_rate"], reverse=True)
    assert len(c["tau"]) == 25


def test_curve_handles_a_population_with_no_negatives():
    c = curve([ep("a", [0.1, 0.9], compromise=1)], n_thresholds=5)
    assert (c["fpr"] == 0).all()


# --------------------------------------------------------- operating point

def test_lead_at_fpr_respects_the_budget():
    attacks = [ep(f"a{i}", [0.2, 0.6, 0.9, 0.9], compromise=3) for i in range(10)]
    negs = [ep(f"n{i}", [0.1, 0.55, 0.55, 0.1], compromise=None, is_attack=False)
            for i in range(10)]
    op = lead_at_fpr(attacks + negs, 0.05)
    assert op["achieved_fpr"] <= 0.05 + 1e-9
    assert op["mean_lead"] >= 0.0


def test_a_detector_that_always_fires_gets_no_credit():
    """Maximal lead at 100% false alarms is not a detector. If the budget
    cannot be met the operating point must return nothing usable."""
    attacks = [ep(f"a{i}", [1.0] * 5, compromise=4) for i in range(5)]
    negs = [ep(f"n{i}", [1.0] * 5, compromise=None, is_attack=False)
            for i in range(5)]
    op = lead_at_fpr(attacks + negs, 0.0)
    assert op["mean_lead"] == 0.0 or op["achieved_fpr"] <= 1e-9


def test_a_perfect_detector_beats_a_blind_one_at_matched_fpr():
    attacks = [ep(f"a{i}", [0.9, 0.9, 0.9, 0.9], compromise=3) for i in range(8)]
    negs = [ep(f"n{i}", [0.0] * 4, compromise=None, is_attack=False)
            for i in range(8)]
    good = lead_at_fpr(attacks + negs, 0.05)

    blind_a = [ep(f"a{i}", [0.0] * 4, compromise=3) for i in range(8)]
    blind = lead_at_fpr(blind_a + negs, 0.05)
    assert good["mean_lead"] > blind["mean_lead"]
    assert good["detection_rate"] > blind["detection_rate"]


def test_operating_point_reports_lead_conditioned_on_detection():
    """Mean lead over all attacks and mean lead over *detected* attacks are
    different numbers; reporting only the first hides a detector that catches
    few episodes very early."""
    attacks = [ep("hit", [0.9, 0.9, 0.9, 0.9], compromise=3),
               ep("miss", [0.0] * 4, compromise=3)]
    negs = [ep("n", [0.0] * 4, compromise=None, is_attack=False)]
    op = lead_at_fpr(attacks + negs, 0.5)
    assert op["mean_lead_when_detected"] >= op["mean_lead"]
