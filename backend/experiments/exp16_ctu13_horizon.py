"""Experiment 16 — sweeping the one knob experiment 15 refused to touch.

Why this experiment exists
--------------------------
Experiment 14 ran the world model on real botnet captures: **1 of 12** infections
caught, against 7 for gradient boosting. Experiment 15 tuned epochs, batch size
and loss weights under nested selection and left it at 1 of 12, so the loss is
not a tuning artefact.

But experiment 15 explicitly did not tune the **rollout horizon**, and said so:
`HORIZON` and `ROLLOUT_DEPTH` were module constants that had to match, so
changing them inside a selection loop would have compared models trained on
different objectives. That is the last untested explanation for the result, and
it is the most plausible one - the horizon is the parameter that decides how far
ahead the model is asked to see, and lead time is exactly what we are measuring.

So it is swept here, properly. Both constants are now arguments
(`build_dataset(..., horizon=)`, `train_world_model(..., rollout_depth=)`,
`wm_scores(..., horizon=)`), passed together, so training and inference can
never silently disagree.

The measurement problem this experiment had to solve first
-----------------------------------------------------------
**Changing the horizon changes the question, not just the model.** The label is

    y[max(0, c - H):c] = 1        "will compromise happen within H windows?"

so at H=2 the model is asked a rare, hard question and at H=18 a common, easy
one. **AUC is therefore not comparable across horizons** - it is measured
against a different target in every arm - and neither is F1. Reporting an AUC
curve over H would be meaningless, and a rising one would look like progress.

Lead time is different. `lead_at_fpr` reads only the compromise window and the
score series; it never sees the label. **Detection count, mean lead and
false-alarm rate are directly comparable across horizons**, and they are the
numbers this project reports anyway. So this experiment reports those and
deliberately publishes no AUC.

Two answers, one cheap and unfair, one honest
----------------------------------------------
**A. The oracle sweep.** Every horizon is run through the full
leave-one-scenario-out protocol, and *all six results are printed*. Taking the
best of six is selection on the test set and would be indefensible as a shipped
number - but as an **upper bound** it is exactly what we want: if the world
model loses even when allowed to pick its horizon knowing the answers, then no
honest procedure could have rescued it, and the question is closed.

**B. The nested selection.** The horizon is chosen on inner validation captures
the test fold is not part of, then applied. This is the number that would be
defensible to ship. It cannot beat the oracle; the gap between them is the cost
of not knowing the future.

B reuses A's trained model for its chosen (fold, horizon) rather than retraining
an identical one. That is not leakage: A's model for fold `s` is trained on the
twelve scenarios that are not `s`, which is precisely what B's outer step calls
for. Only the *selection* differs, and it uses inner captures only.

The inner validation set
------------------------
Three scenarios, rotated per fold, extended until the set holds **at least ten
benign episodes** and one attack with room to earn lead. Ten is not arbitrary:
the operating point is "mean lead at FPR <= 10%", and over fewer than ten
negatives that budget cannot resolve a single false alarm. Three CTU-13
scenarios contain no benign episodes at all, so a fixed single-scenario
validation split - what experiment 15 used, where the criterion was AUC and this
did not bite - would sometimes have had no false-alarm rate to measure.

A control that is not a prediction
----------------------------------
Gradient boosting is deterministic on this data, so at H=6 it must reproduce
experiment 14's row exactly (7 of 12 caught, mean lead 8.08). That check is
asserted and reported separately from the predictions below, because it is what
distinguishes "the world model behaved differently" from "the pipeline is not
the one experiment 14 ran". The world model cannot do that job: its training is
stochastic and unseeded per fold, so it can move by a whole detection on noise
alone - which is worth remembering when reading H1.

PRE-REGISTERED PREDICTIONS  (written before the first run, unedited since)

  H1  CONTROL. At H=6 this sweep reproduces experiment 14's world-model
      operating point: 1 of 12 caught. If it does not, the two experiments do
      not share a pipeline and nothing else here is comparable to anything
      previously reported.

  H2  Mean lead time increases with the horizon. Compromise risk accumulates
      over the rollout (1 - prod(1 - p_k)), so a longer rollout should fire
      earlier. This is the mechanism the sweep exists to test.

  H3  THE QUESTION. Even at its best horizon - chosen with full knowledge of
      the test results, which nothing deployable could do - the world model
      catches fewer infections than gradient boosting at *its* best horizon.
      If this fails, experiments 14 and 15 understated the model and the
      horizon was the missing knob.

  H4  The honest nested selection does no better than the untuned H=6 result
      of 1 of 12. Twelve noisy folds cannot identify the oracle's horizon.

  H5  The best horizon differs across folds: no single horizon is best on a
      majority of them. Same instability as experiment 15's configurations.

Run:  python experiments/exp16_ctu13_horizon.py        (~45 min)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp01_leadtime import (  # noqa: E402
    build_dataset, flat_features, to_scores, train_world_model,
)
from exp14_ctu13_leadtime import MAX_FPR, N_HOSTS, wm_scores  # noqa: E402

from kalachakra.dynamics.world_model import TrainConfig  # noqa: E402
from kalachakra.metrics.leadtime import lead_at_fpr  # noqa: E402
from kalachakra.state.graph import GraphBuilder  # noqa: E402
from kalachakra.provenance import environment, stamped  # noqa: E402

RESULTS = ROOT / "results"
CACHE = RESULTS / "ctu13_episodes_full.pkl"

HORIZONS = (2, 4, 6, 9, 12, 18)

# Held fixed at experiment 14's configuration on purpose. Experiment 15 already
# swept these; sweeping them again alongside the horizon would confound the two
# and make H1's control check meaningless.
TRAIN_CFG = TrainConfig(epochs=12, lr=2e-3)

MIN_VAL_BENIGN = 10

# Experiment 14's gradient-boosting row, used as a deterministic control.
GBDT_H6_CONTROL = {"caught": 7, "mean_lead": 8.083333333333334}


def operating_point(scores) -> dict:
    """Detection count, lead and FPR at the reported operating point.

    Deliberately returns no AUC: the label depends on the horizon, so an AUC
    from one arm cannot be compared with an AUC from another.
    """
    op = lead_at_fpr(scores, MAX_FPR)
    n = sum(1 for e in scores if e.compromise_window is not None)
    return {"caught": int(round(op["detection_rate"] * n)), "n_attack": n,
            "mean_lead": float(op["mean_lead"]),
            "fpr": float(op["achieved_fpr"])}


def inner_split(groups, scenarios, held, oi):
    """Validation scenarios for one outer fold, and the inner training set.

    Rotated by fold index so the same captures are not always used to choose,
    and widened until the set can actually measure a 10% false-alarm rate.
    """
    pool = [s for s in scenarios if s != held]
    n = len(pool)
    for shift in range(n):
        start = (oi + shift) % n
        for size in (3, 4, 5, 6):
            val = [pool[(start + k) % n] for k in range(size)]
            eps = [e for s in val for e in groups[s]]
            benign = sum(1 for e in eps if not e.is_attack)
            usable = [e for e in eps if e.is_attack and e.compromise_window]
            if benign >= MIN_VAL_BENIGN and usable:
                return val, [s for s in pool if s not in val]
    return None, None


def fit_wm(eps_tr, eps_te, builder, h):
    """Train at rollout depth h and score at horizon h - always together."""
    Xtr, Atr, Gtr, Str, _, _ = build_dataset(eps_tr, builder, horizon=h)
    Xte, Ate, Gte, _, _, mte = build_dataset(eps_te, builder, horizon=h)
    wm = train_world_model(Xtr, Atr, Gtr, Str, TRAIN_CFG,
                           verbose=False, rollout_depth=h)
    return to_scores(mte, wm_scores(wm, Xte, Ate, Gte, horizon=h))


def fit_gbdt(eps_tr, eps_te, builder, h):
    Ftr, Fte = flat_features(eps_tr, builder), flat_features(eps_te, builder)
    _, _, _, _, Ytr, _ = build_dataset(eps_tr, builder, horizon=h)
    _, _, _, _, Yte, mte = build_dataset(eps_te, builder, horizon=h)
    ytr = Ytr.reshape(-1)
    if ytr.sum() == 0:
        return None
    gb = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.1,
                       subsample=0.9, eval_metric="logloss",
                       verbosity=0, n_jobs=4)
    gb.fit(Ftr.reshape(-1, Ftr.shape[-1]), ytr)
    p = gb.predict_proba(Fte.reshape(-1, Fte.shape[-1]))[:, 1].reshape(Yte.shape)
    return to_scores(mte, p)


def main() -> int:
    global HORIZONS, TRAIN_CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="two horizons, one epoch - wiring check only, the "
                         "numbers it prints mean nothing")
    args = ap.parse_args()
    if args.smoke:
        HORIZONS = (2, 6)
        TRAIN_CFG = TrainConfig(epochs=1, lr=2e-3)

    t0 = time.time()
    print("=" * 78)
    print("KALACHAKRA  ·  Experiment 16  ·  Rollout-horizon sweep on CTU-13")
    print("=" * 78)
    if args.smoke:
        print("\n  *** SMOKE MODE - results are meaningless ***")

    if 6 not in HORIZONS:
        print("\nH=6 must be in the grid - it is the control against exp 14.")
        return 2
    if not CACHE.exists():
        print(f"\n{CACHE} not found. Run experiment 14 first.")
        return 2
    with CACHE.open("rb") as fh:
        groups = pickle.load(fh)
    scenarios = sorted(groups, key=lambda s: int(s.split("-")[0]))
    builder = GraphBuilder([f"n{i}" for i in range(N_HOSTS)])

    print(f"\n      {len(scenarios)} scenarios, "
          f"{sum(len(v) for v in groups.values())} episodes")
    print(f"      horizons: {', '.join(str(h) for h in HORIZONS)}")
    print("      no AUC is reported - the label changes with the horizon, so")
    print("      AUCs from different arms are not measuring the same task")

    # ---------------------------------------------------------- A. oracle sweep
    print("\n[1/3] Full leave-one-scenario-out at every horizon ...")
    print(f"\n      {'H':>3s}  {'fold':34s} {'WM lead':>8s} {'GBDT lead':>10s}")

    wm_by_h: dict = {h: {} for h in HORIZONS}   # h -> fold -> EpisodeScores list
    gb_by_h: dict = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        for held in scenarios:
            te_eps = groups[held]
            if not any(e.is_attack and e.compromise_window for e in te_eps):
                continue
            tr_eps = [e for s in scenarios if s != held for e in groups[s]]
            ws = fit_wm(tr_eps, te_eps, builder, h)
            gs = fit_gbdt(tr_eps, te_eps, builder, h)
            if gs is None:
                continue
            wm_by_h[h][held] = ws
            gb_by_h[h][held] = gs
            wl = operating_point(ws)["mean_lead"]
            gl = operating_point(gs)["mean_lead"]
            print(f"      {h:3d}  {held:34s} {wl:8.2f} {gl:10.2f}", flush=True)

    # Every horizon must have scored the same folds, or the columns below are
    # averages over different test sets and the comparison is not a comparison.
    fold_sets = {h: frozenset(wm_by_h[h]) for h in HORIZONS}
    if len(set(fold_sets.values())) != 1:
        print("\n  ABORT: horizons scored different folds:")
        for h in HORIZONS:
            print(f"    H={h}: {len(fold_sets[h])} folds")
        return 3

    print("\n[2/3] Pooled result at each horizon")
    print(f"\n      {'H':>3s} | {'WM caught':>10s} {'lead':>7s} {'FPR':>6s} "
          f"| {'GBDT caught':>12s} {'lead':>7s} {'FPR':>6s}")
    sweep = {}
    for h in HORIZONS:
        w = operating_point([e for f in wm_by_h[h].values() for e in f])
        g = operating_point([e for f in gb_by_h[h].values() for e in f])
        sweep[h] = {"wm": w, "gbdt": g}
        print(f"      {h:3d} | {w['caught']:6d}/{w['n_attack']:<3d} "
              f"{w['mean_lead']:7.2f} {w['fpr']:6.3f} "
              f"| {g['caught']:8d}/{g['n_attack']:<3d} "
              f"{g['mean_lead']:7.2f} {g['fpr']:6.3f}")

    # Deterministic pipeline control. Gradient boosting has no training noise
    # here, so at H=6 it must reproduce experiment 14's row exactly. It is not
    # a prediction - it is the check that tells us whether a *world model*
    # difference from experiment 14 is a real difference or a broken pipeline.
    # (The world model cannot serve this purpose: its training is stochastic
    # and unseeded per fold, so it can move by a whole detection on noise.)
    g6 = sweep[6]["gbdt"]
    ctrl = (g6["caught"] == GBDT_H6_CONTROL["caught"]
            and abs(g6["mean_lead"] - GBDT_H6_CONTROL["mean_lead"]) < 0.01)
    print(f"\n      control: GBDT at H=6 reproduces experiment 14 "
          f"({g6['caught']}/{g6['n_attack']} caught, lead {g6['mean_lead']:.2f}) "
          f"-> {'PASS' if ctrl else 'FAIL'}")
    if not ctrl:
        print("      the label, feature and scoring path is NOT shared with")
        print("      experiment 14 - nothing below is comparable to it")

    best_wm_h = max(HORIZONS, key=lambda h: (sweep[h]["wm"]["caught"],
                                             sweep[h]["wm"]["mean_lead"]))
    best_gb_h = max(HORIZONS, key=lambda h: (sweep[h]["gbdt"]["caught"],
                                             sweep[h]["gbdt"]["mean_lead"]))
    print(f"\n      oracle horizon  WM {best_wm_h:2d} -> "
          f"{sweep[best_wm_h]['wm']['caught']}/{sweep[best_wm_h]['wm']['n_attack']} caught, "
          f"lead {sweep[best_wm_h]['wm']['mean_lead']:.2f}")
    print(f"      oracle horizon  GBDT {best_gb_h:2d} -> "
          f"{sweep[best_gb_h]['gbdt']['caught']}/{sweep[best_gb_h]['gbdt']['n_attack']} caught, "
          f"lead {sweep[best_gb_h]['gbdt']['mean_lead']:.2f}")
    print("      (both are upper bounds - the horizon was picked knowing the")
    print("       test results, which nothing deployable can do)")

    # Per-fold oracle, for the stability question. Restricted to folds that
    # have benign episodes: with no negatives the FPR budget is vacuous, every
    # horizon is free to fire at window 0, and the "best" horizon would be an
    # artefact of the fold rather than a fact about the horizon.
    fold_best: Counter = Counter()
    for held in scenarios:
        if held not in wm_by_h[HORIZONS[0]]:
            continue
        if not any(not e.is_attack for e in groups[held]):
            continue
        cand = [(operating_point(wm_by_h[h][held])["mean_lead"], -h, h)
                for h in HORIZONS]
        fold_best[max(cand)[2]] += 1

    # ------------------------------------------------------ B. nested selection
    print("\n[3/3] Nested selection - horizon chosen on inner captures only")
    print(f"\n      {'held-out (test)':34s} {'val caps':>8s} {'chosen H':>9s} "
          f"{'val lead':>9s}")
    chosen: Counter = Counter()
    nested_scores, rows = [], []
    for oi, held in enumerate(scenarios):
        if held not in wm_by_h[HORIZONS[0]]:
            continue
        val, inner = inner_split(groups, scenarios, held, oi)
        if val is None:
            print(f"      {held:34s} skipped - no usable validation split")
            continue
        in_eps = [e for s in inner for e in groups[s]]
        va_eps = [e for s in val for e in groups[s]]

        best_h, best_lead = None, -1.0
        for h in HORIZONS:
            lead = operating_point(fit_wm(in_eps, va_eps, builder, h))["mean_lead"]
            # Strict >: on a tie the shorter horizon wins. A tie means the
            # evidence does not distinguish them, and the shallower rollout is
            # the weaker claim and the cheaper one to serve.
            if lead > best_lead:
                best_h, best_lead = h, lead

        chosen[best_h] += 1
        nested_scores.extend(wm_by_h[best_h][held])
        rows.append({"test": held, "val": val, "chosen_h": best_h,
                     "val_lead": round(best_lead, 3)})
        print(f"      {held:34s} {len(val):8d} {best_h:9d} {best_lead:9.2f}",
              flush=True)

    nested = operating_point(nested_scores)
    print("\n      nested (honest):    "
          f"caught {nested['caught']}/{nested['n_attack']}  "
          f"mean lead {nested['mean_lead']:.2f}  FPR {nested['fpr']:.3f}")
    print(f"      oracle WM (upper):  caught {sweep[best_wm_h]['wm']['caught']}"
          f"/{sweep[best_wm_h]['wm']['n_attack']}  "
          f"mean lead {sweep[best_wm_h]['wm']['mean_lead']:.2f}  H={best_wm_h}")
    print("      untuned (exp 14):   caught 1/12  mean lead 2.50  H=6")
    print("      exp 15 tuning:      caught 1/12  mean lead 2.25  H=6")
    print("      GBDT   (exp 14):    caught 7/12  mean lead 8.08  H=6")

    print("\n      horizon chosen per fold:")
    for h, n in chosen.most_common():
        print(f"        H={h:<3d} {n:2d} / {len(rows)} folds")

    # ------------------------------------------------------------- predictions
    h6 = sweep[6]["wm"]
    leads = [sweep[h]["wm"]["mean_lead"] for h in HORIZONS]
    # A flat lead curve has no correlation to compute, and numpy would return
    # nan with a divide warning. Zero variation is a real answer to H2 - the
    # horizon did nothing - so it is reported as that rather than as a number.
    rising = (float(np.corrcoef(np.array(HORIZONS, dtype=float), leads)[0, 1])
              if float(np.std(leads)) > 0 else float("nan"))
    top_fold = fold_best.most_common(1)[0][1] if fold_best else 0
    n_fold_best = sum(fold_best.values())

    verdicts = {
        "H1": h6["caught"] == 1,
        "H2": bool(rising > 0),
        "H3": sweep[best_wm_h]["wm"]["caught"] < sweep[best_gb_h]["gbdt"]["caught"],
        "H4": nested["caught"] <= 1,
        "H5": bool(n_fold_best) and top_fold <= n_fold_best / 2,
    }
    detail = {
        "H1": f"H=6 reproduces {h6['caught']}/{h6['n_attack']} (exp 14: 1/12)",
        "H2": (f"lead-vs-horizon correlation r = {rising:+.3f}"
               if rising == rising else
               f"mean lead is {leads[0]:.2f} at every horizon - the rollout "
               f"depth changed nothing to correlate"),
        "H3": f"oracle WM {sweep[best_wm_h]['wm']['caught']} (H={best_wm_h}) vs "
              f"oracle GBDT {sweep[best_gb_h]['gbdt']['caught']} (H={best_gb_h})",
        "H4": f"nested caught {nested['caught']}/{nested['n_attack']} vs untuned 1/12",
        "H5": f"most-common per-fold best horizon won {top_fold} of "
              f"{n_fold_best} folds",
    }
    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 78)
    for k in ("H1", "H2", "H3", "H4", "H5"):
        print(f"  {k}  {'HOLDS ' if verdicts[k] else 'FAILS '}  {detail[k]}")

    payload = {
        "question": "does the rollout horizon - the one knob experiment 15 left "
                    "untouched - explain the CTU-13 result?",
        "horizons": list(HORIZONS),
        "train_config": vars(TRAIN_CFG),
        "no_auc_because": "the label y[c-H:c] depends on H, so AUC measures a "
                          "different task in every arm; lead time and detection "
                          "count read only the compromise window and are "
                          "comparable across horizons",
        "sweep": {str(h): sweep[h] for h in HORIZONS},
        "pipeline_control": {"gbdt_h6_matches_exp14": bool(ctrl),
                             "measured": g6, "expected": GBDT_H6_CONTROL,
                             "why": "gradient boosting is deterministic here, so "
                                    "this separates a real world-model difference "
                                    "from a broken pipeline"},
        "oracle": {"wm_horizon": best_wm_h, "gbdt_horizon": best_gb_h,
                   "note": "selected on the test folds - an upper bound, not a "
                           "shippable result"},
        "per_fold_best_horizon": dict(fold_best),
        "nested": {**nested, "folds": rows, "chosen_counts": dict(chosen),
                   "protocol": "horizon chosen by mean lead at FPR<=10% on "
                               "rotated inner validation captures; the scored "
                               "model is trained on the 12 non-test scenarios"},
        "reference": {"exp14_untuned": {"caught": 1, "mean_lead": 2.50},
                      "exp15_tuned": {"caught": 1, "mean_lead": 2.25},
                      "exp14_gbdt": {"caught": 7, "mean_lead": 8.08}},
        "verdicts": verdicts,
        "detail": detail,
    }
    payload = stamped(payload)
    (RESULTS / "exp16_verdicts.json").write_text(json.dumps(payload, indent=1))
    print("\n  wrote results/exp16_verdicts.json")
    print(f"  total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
