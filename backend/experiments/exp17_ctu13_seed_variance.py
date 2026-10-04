"""Experiment 17 — how much of "1 of 12" was the random seed?

Why this experiment exists
--------------------------
Experiment 16 swept the rollout horizon and, at H=6, ran exactly experiment 14's
configuration on exactly experiment 14's folds. It should have reproduced
experiment 14's world-model row. It did not:

    experiment 14   world model, H=6, epochs=12   caught 1 of 12
    experiment 16   world model, H=6, epochs=12   caught 4 of 12

Experiment 16 carried a deterministic control for exactly this moment. Gradient
boosting has no training randomness here, and at H=6 it reproduced experiment
14's row to the digit - 7 of 12 caught, mean lead 8.08. So the episode
construction, the labels, the features, the fold set, the pooling and the
operating-point rule are all provably the same. The only thing left that can
differ is the world model's own initialisation and batch shuffling, which are
drawn from a global RNG whose state depends on how much training happened
earlier in the process.

That is a hypothesis, and this project does not publish hypotheses as findings.
So it is measured: the same configuration, the same folds, six different seeds.

Why this matters more than the horizon did
-------------------------------------------
"The world model catches 1 of 12 real infections" is currently on the pitch deck,
in RESULTS.md, and in the judge Q&A. If that number moves by three detections
when nothing changes but a seed, then it is not a measurement of the model - it
is one draw from a distribution we never characterised, and we have been
reporting it as though it were a constant.

The direction of the error does not rescue it either. A number that flatters us
would be a problem; this one is *harsher* than the truth, which is a problem of
the same kind. Publishing "1 of 12" when the honest answer is a spread around 4
is inaccurate in a way that happens to look humble, and that is still inaccurate.

PRE-REGISTERED PREDICTIONS  (written before the first run, unedited since)

  S1  The world model's detection count varies by at least 2 detections across
      seeds. If it does not, the seed cannot explain a gap of 3 and something
      else differs between experiments 14 and 16 that we have not found.

  S2  Experiment 14's published 1 of 12 falls inside the observed range. If the
      spread is real but never reaches 1, the discrepancy is still unexplained
      and both numbers are suspect.

  S3  CONTROL. Gradient boosting catches the identical number of infections on
      every seed. This is what shows the instability belongs to the world model
      rather than to the harness around it.

  S4  The world model's mean detection count across seeds remains below
      gradient boosting's 7 of 12. We expect the instability to change the
      number we should publish, not the ranking it supports.

Run:  python experiments/exp17_ctu13_seed_variance.py      (~25 min)
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp01_leadtime import (  # noqa: E402
    HORIZON, build_dataset, flat_features, to_scores, train_world_model,
)
from exp14_ctu13_leadtime import MAX_FPR, N_HOSTS, wm_scores  # noqa: E402

from kalachakra.dynamics.world_model import TrainConfig  # noqa: E402
from kalachakra.metrics.leadtime import lead_at_fpr  # noqa: E402
from kalachakra.state.graph import GraphBuilder  # noqa: E402
from kalachakra.provenance import environment, stamped  # noqa: E402

RESULTS = ROOT / "results"
CACHE = RESULTS / "ctu13_episodes_full.pkl"

SEEDS = (0, 1, 2, 3, 4, 5)
TRAIN_CFG = TrainConfig(epochs=12, lr=2e-3)   # experiment 14's, unchanged

EXP14_WM_CAUGHT = 1
EXP14_GBDT_CAUGHT = 7


def op(scores) -> dict:
    o = lead_at_fpr(scores, MAX_FPR)
    n = sum(1 for e in scores if e.compromise_window is not None)
    return {"caught": int(round(o["detection_rate"] * n)), "n_attack": n,
            "mean_lead": float(o["mean_lead"]), "fpr": float(o["achieved_fpr"])}


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("KALACHAKRA  ·  Experiment 17  ·  Seed variance of the CTU-13 result")
    print("=" * 78)

    if not CACHE.exists():
        print(f"\n{CACHE} not found. Run experiment 14 first.")
        return 2
    with CACHE.open("rb") as fh:
        groups = pickle.load(fh)
    scenarios = sorted(groups, key=lambda s: int(s.split("-")[0]))
    builder = GraphBuilder([f"n{i}" for i in range(N_HOSTS)])

    folds = [s for s in scenarios
             if any(e.is_attack and e.compromise_window for e in groups[s])]
    print(f"\n      {len(folds)} scored folds, horizon {HORIZON}, "
          f"config {vars(TRAIN_CFG)}")
    print(f"      seeds: {', '.join(str(s) for s in SEEDS)}")
    print("      nothing varies between rows below except torch's RNG seed")

    # Features and labels do not depend on the seed, so build them once. This
    # also removes them as a possible source of the variation being measured.
    cache = {}
    for held in folds:
        tr_eps = [e for s in scenarios if s != held for e in groups[s]]
        te_eps = groups[held]
        cache[held] = (build_dataset(tr_eps, builder), build_dataset(te_eps, builder),
                       flat_features(tr_eps, builder), flat_features(te_eps, builder))

    print(f"\n      {'seed':>5s} | {'WM caught':>10s} {'lead':>7s} {'FPR':>6s} "
          f"{'AUC':>6s} | {'GBDT caught':>12s} {'lead':>7s} {'AUC':>6s}")
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        wm_scored, gb_scored, wm_auc, gb_auc = [], [], [], []
        for held in folds:
            (Xtr, Atr, Gtr, Str, Ytr, _), (Xte, Ate, Gte, _, Yte, mte), Ftr, Fte = \
                cache[held]
            yte = Yte.reshape(-1)

            wm = train_world_model(Xtr, Atr, Gtr, Str, TRAIN_CFG, verbose=False)
            pw = wm_scores(wm, Xte, Ate, Gte)
            wm_scored.extend(to_scores(mte, pw))
            if len(np.unique(yte)) > 1:
                wm_auc.append(roc_auc_score(yte, pw.reshape(-1)))

            gb = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.1,
                               subsample=0.9, eval_metric="logloss",
                               verbosity=0, n_jobs=4)
            gb.fit(Ftr.reshape(-1, Ftr.shape[-1]), Ytr.reshape(-1))
            pg = gb.predict_proba(
                Fte.reshape(-1, Fte.shape[-1]))[:, 1].reshape(Yte.shape)
            gb_scored.extend(to_scores(mte, pg))
            if len(np.unique(yte)) > 1:
                gb_auc.append(roc_auc_score(yte, pg.reshape(-1)))

        w, g = op(wm_scored), op(gb_scored)
        w["auc"], g["auc"] = float(np.mean(wm_auc)), float(np.mean(gb_auc))
        rows.append({"seed": seed, "wm": w, "gbdt": g})
        print(f"      {seed:5d} | {w['caught']:6d}/{w['n_attack']:<3d} "
              f"{w['mean_lead']:7.2f} {w['fpr']:6.3f} {w['auc']:6.3f} "
              f"| {g['caught']:8d}/{g['n_attack']:<3d} {g['mean_lead']:7.2f} "
              f"{g['auc']:6.3f}", flush=True)

    wm_caught = [r["wm"]["caught"] for r in rows]
    wm_lead = [r["wm"]["mean_lead"] for r in rows]
    wm_auc_all = [r["wm"]["auc"] for r in rows]
    gb_caught = [r["gbdt"]["caught"] for r in rows]
    n_atk = rows[0]["wm"]["n_attack"]

    print(f"\n      world model  caught {min(wm_caught)}-{max(wm_caught)} of "
          f"{n_atk}  (mean {np.mean(wm_caught):.1f}, sd {np.std(wm_caught):.2f})")
    print(f"      world model  mean lead {min(wm_lead):.2f}-{max(wm_lead):.2f}"
          f"   AUC {min(wm_auc_all):.3f}-{max(wm_auc_all):.3f}")
    print(f"      gradient boosting caught {min(gb_caught)}-{max(gb_caught)} "
          f"of {n_atk}  (deterministic control)")
    print(f"\n      experiment 14 published: world model {EXP14_WM_CAUGHT}/12, "
          f"gradient boosting {EXP14_GBDT_CAUGHT}/12")

    verdicts = {
        "S1": max(wm_caught) - min(wm_caught) >= 2,
        "S2": min(wm_caught) <= EXP14_WM_CAUGHT <= max(wm_caught),
        "S3": len(set(gb_caught)) == 1,
        "S4": float(np.mean(wm_caught)) < EXP14_GBDT_CAUGHT,
    }
    detail = {
        "S1": f"world model spans {min(wm_caught)}-{max(wm_caught)} detections "
              f"across {len(SEEDS)} seeds",
        "S2": f"experiment 14's {EXP14_WM_CAUGHT}/12 "
              f"{'is inside' if verdicts['S2'] else 'is OUTSIDE'} the observed "
              f"range {min(wm_caught)}-{max(wm_caught)}",
        "S3": f"gradient boosting caught {sorted(set(gb_caught))} across seeds",
        "S4": f"world model mean {np.mean(wm_caught):.1f} vs gradient "
              f"boosting {EXP14_GBDT_CAUGHT}",
    }
    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 78)
    for k in ("S1", "S2", "S3", "S4"):
        print(f"  {k}  {'HOLDS ' if verdicts[k] else 'FAILS '}  {detail[k]}")

    payload = {
        "question": "is experiment 14's '1 of 12' a property of the model or of "
                    "the random seed?",
        "protocol": "experiment 14's config and folds, re-run under six seeds; "
                    "episodes, labels and features built once and shared, so "
                    "only torch's RNG differs between rows",
        "horizon": HORIZON,
        "train_config": vars(TRAIN_CFG),
        "seeds": list(SEEDS),
        "rows": rows,
        "world_model": {"caught_min": min(wm_caught), "caught_max": max(wm_caught),
                        "caught_mean": float(np.mean(wm_caught)),
                        "caught_sd": float(np.std(wm_caught)),
                        "lead_min": min(wm_lead), "lead_max": max(wm_lead),
                        "auc_min": min(wm_auc_all), "auc_max": max(wm_auc_all)},
        "gbdt_control": {"caught": sorted(set(gb_caught)),
                         "deterministic": verdicts["S3"]},
        "published": {"exp14_wm_caught": EXP14_WM_CAUGHT,
                      "exp14_gbdt_caught": EXP14_GBDT_CAUGHT},
        "verdicts": verdicts,
        "detail": detail,
    }
    payload = stamped(payload)
    (RESULTS / "exp17_verdicts.json").write_text(json.dumps(payload, indent=1))
    print("\n  wrote results/exp17_verdicts.json")
    print(f"  total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
