"""Lead time — the metric that makes this project different.

Every competing team will report accuracy or F1 on a per-window attack label.
That number answers "did you notice?", which a defender already knows by the
time it matters. The question a defender actually buys an answer to is:

    "How much warning did I get, and how many false alarms did it cost me?"

Definitions used here, stated precisely because the number is meaningless
otherwise
-----------------------------------------------------------------------------
For an attack episode with compromise window `c` (the first window at or beyond
LATERAL_MOVEMENT) and a per-window risk score `s_t`:

    alarm(tau)  = min{ t : s_t >= tau }          (or None)
    lead(tau)   = c - alarm(tau)   if alarm exists and alarm < c
                = 0                otherwise

Lead is clamped at zero: firing *after* the attacker has already moved
laterally is worth nothing, and letting it go negative would let late alarms
subsidise the mean.

A **negative episode is one that never reaches compromise** - which includes
both quiet benign episodes and episodes with genuine attack activity that was
contained or stalled out. This definition is the whole point. If negatives were
only attack-free episodes, a model that fires on the first port scan would post
a perfect false-alarm rate and huge lead time while having forecast nothing;
the task would collapse into detection. Counting contained attacks as negatives
is what forces a method to predict *progression*.

A false alarm is *any* window crossing tau, counted per-episode rather than
per-window: an analyst who gets one spurious page has been interrupted once.

    FPR(tau) = |{non-compromised episodes with any s_t >= tau}|
             / |non-compromised episodes|

Sweeping tau gives a lead-time-versus-false-alarm curve. Comparing methods at
*matched* FPR is the only fair comparison; comparing raw lead times at
different alarm rates is how you make a bad detector look good.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class EpisodeScores:
    """Per-window risk scores for one episode, plus its ground truth."""

    episode_id: str
    scores: np.ndarray          # (T,)
    is_attack: bool
    compromise_window: Optional[int]


def _alarm_window(scores: np.ndarray, tau: float) -> Optional[int]:
    idx = np.flatnonzero(scores >= tau)
    return int(idx[0]) if idx.size else None


def lead_time(ep: EpisodeScores, tau: float) -> float:
    if ep.compromise_window is None:
        return 0.0
    a = _alarm_window(ep.scores, tau)
    if a is None or a >= ep.compromise_window:
        return 0.0
    return float(ep.compromise_window - a)


def false_alarm_rate(negatives: Sequence[EpisodeScores], tau: float) -> float:
    """`negatives` must be episodes that never reach compromise - including
    contained attacks. See the module docstring."""
    if not negatives:
        return 0.0
    fired = sum(1 for e in negatives if _alarm_window(e.scores, tau) is not None)
    return fired / len(negatives)


def detection_rate(attacks: Sequence[EpisodeScores], tau: float) -> float:
    """Fraction of attack episodes caught *before* compromise."""
    if not attacks:
        return 0.0
    caught = sum(1 for e in attacks if lead_time(e, tau) > 0)
    return caught / len(attacks)


def curve(episodes: Sequence[EpisodeScores],
          n_thresholds: int = 60) -> Dict[str, np.ndarray]:
    """Lead-time / detection / false-alarm curve over a threshold sweep."""
    attacks = [e for e in episodes if e.compromise_window is not None]
    benign = [e for e in episodes if e.compromise_window is None]

    all_scores = np.concatenate([e.scores for e in episodes]) if episodes else np.zeros(1)
    lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
    taus = np.linspace(lo, hi, n_thresholds)

    leads, fprs, dets, leads_detected = [], [], [], []
    for tau in taus:
        ls = [lead_time(e, tau) for e in attacks]
        leads.append(float(np.mean(ls)) if ls else 0.0)
        pos = [x for x in ls if x > 0]
        leads_detected.append(float(np.mean(pos)) if pos else 0.0)
        fprs.append(false_alarm_rate(benign, tau))
        dets.append(detection_rate(attacks, tau))

    return {"tau": taus,
            "mean_lead": np.array(leads),
            "mean_lead_when_detected": np.array(leads_detected),
            "fpr": np.array(fprs),
            "detection_rate": np.array(dets)}


def lead_at_fpr(episodes: Sequence[EpisodeScores], target_fpr: float) -> Dict[str, float]:
    """Operating point: the highest-lead threshold whose FPR is within budget.

    This is how the comparison is reported. Picking the best lead time without
    an FPR constraint is meaningless - a detector that always fires at window 0
    has maximal lead and is useless.
    """
    c = curve(episodes)
    ok = c["fpr"] <= target_fpr + 1e-12
    if not ok.any():
        return {"fpr_budget": target_fpr, "achieved_fpr": float("nan"),
                "mean_lead": 0.0, "detection_rate": 0.0, "tau": float("nan")}
    idx_pool = np.flatnonzero(ok)
    best = idx_pool[int(np.argmax(c["mean_lead"][ok]))]
    return {"fpr_budget": target_fpr,
            "achieved_fpr": float(c["fpr"][best]),
            "mean_lead": float(c["mean_lead"][best]),
            "mean_lead_when_detected": float(c["mean_lead_when_detected"][best]),
            "detection_rate": float(c["detection_rate"][best]),
            "tau": float(c["tau"][best])}
