"""Experiment 15 — a fair-effort tuning pass for the world model on CTU-13.

Why this experiment exists
--------------------------
Experiment 14 ran the world model on real botnet captures and it caught **1 of
12** infections, against 7 for gradient boosting. But it ran on experiment 01's
hyperparameters, which were chosen on *generated* data. Reporting a loss from an
untuned model would be as unfair as reporting a win from an overtuned one, so
this gives the model a genuine attempt.

The trap this experiment is built to avoid
-------------------------------------------
The obvious way to tune is to try configurations and keep whichever scores best
under leave-one-scenario-out. **That is tuning on the test set**, and it would
manufacture an improvement out of nothing but search. With 12 attack episodes,
trying six configurations and keeping the luckiest would very likely "improve"
1/12 to 3/12 with no real change in capability.

So selection is **nested**. For each outer fold:

    outer test   = scenario s                  (never seen during selection)
    inner val    = one of the other 12         (rotated, so it is not always
                                                the same capture)
    inner train  = the remaining 11

The configuration is chosen by AUC on the inner validation scenario, then the
model is retrained on all 12 non-test scenarios with that configuration and
scored once on s. **No number reported here has seen its own test fold.**

The cost of doing it honestly is that the winning configuration differs between
folds, so there is no single "tuned model" to ship - which is itself worth
knowing, and is reported.

What is and is not tuned
------------------------
Tuned: epochs, batch size, and the loss weights `w_stage` and `w_vic`.

**Not tuned: the rollout horizon.** `HORIZON` and `ROLLOUT_DEPTH` are module
constants that must match — training and inference disagree otherwise — so
changing them is a code change rather than a config, and doing it inside a
selection loop would silently compare models trained on different objectives.
Flagged as unexplored rather than quietly skipped.

PRE-REGISTERED PREDICTIONS  (written before the first run, unedited since)

  T1  Honest tuning improves the world model: it catches more than 1 of 12.
      If it does not, experiment 14's result was not a tuning artefact.

  T2  Even tuned, the world model does not reach gradient boosting's 7 of 12.
      We expect a real but modest improvement, not a reversal.

  T3  The selected configuration differs from experiment 01's defaults on a
      majority of folds - i.e. hyperparameters chosen on generated data were
      not right for real data.

  T4  The selection is unstable: no single configuration wins a majority of
      folds. Twelve attack episodes across thirteen very different captures
      should not support a confident choice.

Run:  python experiments/exp15_ctu13_tuning.py      (~35 min)
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp01_leadtime import build_dataset, to_scores, train_world_model  # noqa: E402
from exp14_ctu13_leadtime import MAX_FPR, N_HOSTS, wm_scores  # noqa: E402

from kalachakra.dynamics.world_model import TrainConfig  # noqa: E402
from kalachakra.metrics.leadtime import lead_at_fpr  # noqa: E402
from kalachakra.state.graph import GraphBuilder  # noqa: E402

RESULTS = ROOT / "results"
CACHE = RESULTS / "ctu13_episodes_full.pkl"

# Small and defensible. Each config targets a specific hypothesis about why the
# untuned model caught almost nothing, rather than sweeping blindly.
GRID = {
    # experiment 01's settings, carried over from generated data
    "baseline":    TrainConfig(epochs=25, lr=2e-3, w_stage=1.0, w_vic=0.5, batch=16),
    # under-trained? 105 episodes at batch 16 is very few gradient steps
    "longer":      TrainConfig(epochs=40, lr=2e-3, w_stage=1.0, w_vic=0.5, batch=16),
    # more steps per epoch rather than more epochs
    "small_batch": TrainConfig(epochs=40, lr=2e-3, w_stage=1.0, w_vic=0.5, batch=8),
    # the risk score comes from the stage head - weight what we actually read
    "stage_heavy": TrainConfig(epochs=40, lr=2e-3, w_stage=3.0, w_vic=0.5, batch=16),
}


def fit_and_score(Xtr, Atr, Gtr, Str, Xte, Ate, Gte, cfg):
    wm = train_world_model(Xtr, Atr, Gtr, Str, cfg, verbose=False)
    return wm_scores(wm, Xte, Ate, Gte)


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("KALACHAKRA  ·  Experiment 15  ·  Fair-effort tuning on CTU-13")
    print("=" * 78)

    if not CACHE.exists():
        print(f"\n{CACHE} not found. Run experiment 14 first.")
        return 2
    with CACHE.open("rb") as fh:
        groups = pickle.load(fh)
    scenarios = sorted(groups, key=lambda s: int(s.split("-")[0]))
    builder = GraphBuilder([f"n{i}" for i in range(N_HOSTS)])

    print(f"\n      {len(scenarios)} scenarios, "
          f"{sum(len(v) for v in groups.values())} episodes")
    print(f"      grid: {', '.join(GRID)}")
    print("      selection is nested - no config is chosen using its own test fold")

    print("\n[1/2] Nested leave-one-scenario-out ...")
    print(f"\n      {'held-out (test)':34s} {'inner val':22s} {'chosen':12s} {'AUC':>6s}")
    chosen_counts: Counter = Counter()
    scores, aucs, rows = [], [], []

    for oi, held in enumerate(scenarios):
        pool = [s for s in scenarios if s != held]
        val = pool[oi % len(pool)]            # rotate, never always the same
        inner = [s for s in pool if s != val]

        te_eps = groups[held]
        if not any(e.is_attack for e in te_eps):
            print(f"      {held:34s} skipped - no attack episode")
            continue

        in_eps = [e for s in inner for e in groups[s]]
        va_eps = groups[val]
        Xi, Ai, Gi, Si, Yi, _ = build_dataset(in_eps, builder)
        Xv, Av, Gv, Sv, Yv, _ = build_dataset(va_eps, builder)
        yv = Yv.reshape(-1)

        best_name, best_auc = None, -1.0
        if len(np.unique(yv)) < 2:
            # This validation capture has no usable labels, so it cannot choose.
            # Falling back to the baseline is the honest move: inventing a
            # selection from a degenerate fold would be worse than not selecting.
            best_name, best_auc = "baseline", float("nan")
        else:
            for name, cfg in GRID.items():
                pv = fit_and_score(Xi, Ai, Gi, Si, Xv, Av, Gv, cfg).reshape(-1)
                a = roc_auc_score(yv, pv)
                if a > best_auc:
                    best_name, best_auc = name, a

        # Retrain on ALL non-test scenarios with the chosen config, then test.
        tr_eps = [e for s in pool for e in groups[s]]
        Xt, At, Gt, St, Yt, _ = build_dataset(tr_eps, builder)
        Xe, Ae, Ge, Se, Ye, me = build_dataset(te_eps, builder)
        pe = fit_and_score(Xt, At, Gt, St, Xe, Ae, Ge, GRID[best_name])
        ye = Ye.reshape(-1)
        if len(np.unique(ye)) < 2:
            print(f"      {held:34s} skipped - degenerate test labels")
            continue

        a_out = roc_auc_score(ye, pe.reshape(-1))
        aucs.append(a_out)
        scores.extend(to_scores(me, pe))
        chosen_counts[best_name] += 1
        rows.append({"test": held, "val": val, "chosen": best_name,
                     "val_auc": None if best_auc != best_auc else round(best_auc, 4),
                     "test_auc": round(a_out, 4)})
        print(f"      {held:34s} {val[:20]:22s} {best_name:12s} {a_out:6.3f}",
              flush=True)

    print("\n[2/2] Result")
    op = lead_at_fpr(scores, MAX_FPR)
    n_atk = sum(1 for e in scores if e.compromise_window is not None)
    caught = int(round(op["detection_rate"] * n_atk))
    mean_auc = float(np.mean(aucs))
    print(f"\n      tuned world model:  AUC {mean_auc:.3f} (sd {np.std(aucs):.3f})  "
          f"caught {caught}/{n_atk}  mean lead {op['mean_lead']:.2f}  "
          f"FPR {op['achieved_fpr']:.3f}")
    print("      untuned (exp 14):   AUC 0.663 (sd 0.330)  caught 1/12  "
          "mean lead 2.50  FPR 0.020")
    print("      GBDT   (exp 14):    AUC 0.624 (sd 0.322)  caught 7/12  "
          "mean lead 8.08  FPR 0.088")

    print("\n      configuration chosen per fold:")
    for name, n in chosen_counts.most_common():
        print(f"        {name:12s} {n:2d} / {len(rows)} folds")

    non_baseline = sum(n for k, n in chosen_counts.items() if k != "baseline")
    top = chosen_counts.most_common(1)[0][1] if chosen_counts else 0
    verdicts = {
        "T1": caught > 1,
        "T2": caught < 7,
        "T3": non_baseline > len(rows) / 2,
        "T4": top <= len(rows) / 2,
    }
    detail = {
        "T1": f"tuned caught {caught}/{n_atk} vs untuned 1/12",
        "T2": f"tuned caught {caught}/{n_atk} vs GBDT 7/12",
        "T3": f"{non_baseline} of {len(rows)} folds chose a non-default config",
        "T4": f"most-chosen config won {top} of {len(rows)} folds",
    }
    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 78)
    for k in ("T1", "T2", "T3", "T4"):
        print(f"  {k}  {'HOLDS ' if verdicts[k] else 'FAILS '}  {detail[k]}")

    payload = {
        "protocol": "nested leave-one-scenario-out; config chosen on an inner "
                    "validation scenario, never on the test fold",
        "grid": {k: vars(v) for k, v in GRID.items()},
        "not_tuned": ["rollout horizon (HORIZON/ROLLOUT_DEPTH are coupled module "
                      "constants; changing them inside a selection loop would "
                      "compare models trained on different objectives)"],
        "folds": rows,
        "chosen_counts": dict(chosen_counts),
        "tuned": {"auc": round(mean_auc, 4), "auc_std": round(float(np.std(aucs)), 4),
                  "caught": caught, "n_attack": n_atk,
                  "mean_lead": round(op["mean_lead"], 3),
                  "fpr": round(op["achieved_fpr"], 4)},
        "untuned_exp14": {"auc": 0.663, "caught": 1, "mean_lead": 2.50},
        "gbdt_exp14": {"auc": 0.624, "caught": 7, "mean_lead": 8.08},
        "verdicts": verdicts,
    }
    (RESULTS / "exp15_verdicts.json").write_text(json.dumps(payload, indent=1))
    print("\n  wrote results/exp15_verdicts.json")
    print(f"  total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
