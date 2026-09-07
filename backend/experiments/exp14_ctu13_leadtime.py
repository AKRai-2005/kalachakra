"""Experiment 14 — lead time on REAL botnet captures (CTU-13).

The gap this closes
-------------------
Every KALACHAKRA number until now was measured on `episodes.py`, our own
generator. Experiment 01 refuted the lead-time claim there — a tuned LSTM won
3.35 windows of warning to the world model's 0.26. That refutation has never
been checked against an intrusion nobody wrote for us, so it could in principle
have been an artefact of our own dynamics.

CTU-13 is thirteen captures of real botnet traffic (CTU Prague, 2011) with real
background and normal traffic mixed in. `data/ctu13.py` turns each scenario into
the same `Episode` objects the generator produces, so `GraphBuilder` and every
model here run **unchanged** — if the adapter built its own features, a
synthetic-vs-real difference could be the adapter rather than the data.

Protocol: leave-one-scenario-out
---------------------------------
There are **thirteen independent infection events**, one per scenario, because a
scenario's infected hosts all wake within minutes of each other. A random
episode split would put episodes from the same capture on both sides and measure
memorisation. So the model is trained on twelve scenarios and tested on the
thirteenth, rotated — every test episode comes from a network and a malware
family the model has never seen.

That is the strongest protocol the data supports and also the hardest. It is
chosen deliberately: cross-network generalisation is what a deployed sensor
actually faces.

What this cannot test
---------------------
**The counterfactual.** No real capture contains the branch that did not happen,
so the intervention claim stays validated on constructed pairs. **Intermediate
ATT&CK stages**, too: CTU-13 labels flows botnet / normal / background, with no
kill-chain ladder, so the stage head degenerates to compromised-or-not here.

PRE-REGISTERED PREDICTIONS  (written before the first run, unedited since)

  E1  All four methods beat chance on real data: window-level AUC > 0.60 for
      every method. If they do not, the adapter or the labels are wrong and no
      comparison below means anything.

  E2  The world model does **not** beat the LSTM on mean lead time. We expect
      experiment 01's refutation to replicate on real data rather than reverse.

  E3  Real lead times are **shorter** than synthetic ones: the best real mean
      lead time is below exp01's 3.35 windows. Real botnet onsets are abrupt,
      where our generator ramps.

  E4  At least one method achieves mean lead time > 0 at a false-alarm rate
      <= 10%. If none does, forecasting on this corpus is not possible at all
      at a usable operating point, whoever is doing it.

  E5  Cross-network transfer is poor: no method exceeds 0.80 AUC under
      leave-one-scenario-out. Held-out means a different network and a
      different malware family.

  E6  Logistic regression is not last. On synthetic data the world model
      already lost window-F1 to it (0.412 vs 0.606); we expect the simplest
      baseline to remain competitive on real data too.

Run:  python experiments/exp14_ctu13_leadtime.py        (~15 min)
      python experiments/exp14_ctu13_leadtime.py --quick
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp01_leadtime import (  # noqa: E402
    HORIZON, LSTMBaseline, TrainConfig, build_dataset, flat_features,
    to_scores, train_world_model, window_f1,
)

from kalachakra.data.ctu13 import ScenarioConfig, load_scenario  # noqa: E402
from kalachakra.metrics.leadtime import lead_at_fpr  # noqa: E402
from kalachakra.state.graph import GraphBuilder  # noqa: E402

RESULTS = ROOT / "results"
DATA = ROOT / "data" / "ctu13" / "CSV-originals"
# Cache key includes the mode: a --quick cache holds capped episodes, and
# silently reusing it for a full run would report a real-looking result
# from 400k rows per scenario instead of all 19.9M.
def cache_path(quick: bool) -> Path:
    return RESULTS / ("ctu13_episodes_quick.pkl" if quick
                      else "ctu13_episodes_full.pkl")
N_HOSTS = 24
MAX_FPR = 0.10
T_EPISODE = 40
PRE_WINDOWS = 30


def scenario_key(ep) -> str:
    """Which capture an episode came from - the leave-one-out unit."""
    return ep.id.split(".binetflow")[0]


def load_all(quick: bool) -> dict:
    """Episodes grouped by scenario, cached because parsing 19.9M rows is slow."""
    cache = cache_path(quick)
    if cache.exists():
        print(f"      reusing {cache.name}")
        with cache.open("rb") as fh:
            return pickle.load(fh)

    cfg = ScenarioConfig(n_hosts=N_HOSTS,
                         max_rows=400_000 if quick else None)
    groups: dict = {}
    paths = sorted(DATA.glob("*.csv"), key=lambda p: int(p.name.split("-")[0]))
    for p in paths:
        eps, _, meta = load_scenario(p, cfg)
        if not eps:
            continue
        groups[p.stem] = eps
        print(f"      {p.name:38s} {len(eps):3d} eps "
              f"({meta['n_attack']} attack)", flush=True)
    RESULTS.mkdir(exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump(groups, fh)
    return groups


def wm_scores(wm, X, A, G, horizon: int = HORIZON) -> np.ndarray:
    """Per-window compromise risk over the rollout horizon.

    `horizon` must match the depth the model was trained at; experiment 16
    passes both together, and nothing else passes either.
    """
    xs, adjs, gs = torch.tensor(X), torch.tensor(A), torch.tensor(G)
    T = xs.shape[1]
    out = np.zeros((xs.shape[0], T), dtype=np.float32)
    with torch.no_grad():
        for t in range(T - 1):
            r = wm.compromise_risk(xs, adjs, gs, t, horizon, 0)
            out[:, t] = r.numpy().reshape(-1)
    out[:, -1] = out[:, -2]
    return out


def train_lstm(F, Y, epochs: int) -> LSTMBaseline:
    m = LSTMBaseline(F.shape[-1])
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    x, y = torch.tensor(F, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    pos = float(y.mean().clamp(1e-4, 1 - 1e-4))
    w = torch.tensor((1 - pos) / pos)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=w)
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 8):
            idx = perm[i:i + 8]
            opt.zero_grad()
            loss = lossf(m(x[idx]), y[idx])
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pass
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="cap rows per scenario; for wiring checks only")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("KALACHAKRA  ·  Experiment 14  ·  Lead time on REAL captures (CTU-13)")
    print("=" * 78)

    if not DATA.exists():
        print(f"\n{DATA} not found. Run:  python fetch_ctu13.py --version 1")
        return 2

    print("\n[1/4] Loading scenarios ...")
    groups = load_all(args.quick)
    scenarios = sorted(groups, key=lambda s: int(s.split("-")[0]))
    total = sum(len(v) for v in groups.values())
    n_atk = sum(1 for v in groups.values() for e in v if e.is_attack)
    print(f"      {len(scenarios)} scenarios, {total} episodes, "
          f"{n_atk} attack / {total - n_atk} benign")

    hosts = [f"n{i}" for i in range(N_HOSTS)]
    builder = GraphBuilder(hosts)

    print("\n[2/4] Leave-one-scenario-out ...")
    print("      every test episode is a network and a malware family the")
    print("      model has never seen")
    methods = ("WorldModel", "LSTM", "GBDT", "LogReg")
    fold_scores: dict = {m: [] for m in methods}
    auc_rows: dict = {m: [] for m in methods}
    f1_rows: dict = {m: [] for m in methods}

    for held in scenarios:
        tr_eps = [e for s in scenarios if s != held for e in groups[s]]
        te_eps = groups[held]
        if not any(e.is_attack for e in te_eps):
            continue

        Xtr, Atr, Gtr, Str, Ytr, mtr = build_dataset(tr_eps, builder)
        Xte, Ate, Gte, Ste, Yte, mte = build_dataset(te_eps, builder)
        Ftr, Fte = flat_features(tr_eps, builder), flat_features(te_eps, builder)
        ytr, yte = Ytr.reshape(-1), Yte.reshape(-1)
        if ytr.sum() == 0 or len(np.unique(yte)) < 2:
            print(f"      {held:34s} skipped - no usable labels in this fold")
            continue

        preds = {}

        wm = train_world_model(Xtr, Atr, Gtr, Str,
                               TrainConfig(epochs=args.epochs, lr=2e-3),
                               verbose=False)
        preds["WorldModel"] = wm_scores(wm, Xte, Ate, Gte)

        lstm = train_lstm(Ftr, Ytr, args.epochs)
        with torch.no_grad():
            preds["LSTM"] = torch.sigmoid(
                lstm(torch.tensor(Fte, dtype=torch.float32))).numpy()

        flat_tr = Ftr.reshape(-1, Ftr.shape[-1])
        flat_te = Fte.reshape(-1, Fte.shape[-1])
        gb = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.1,
                           subsample=0.9, eval_metric="logloss",
                           verbosity=0, n_jobs=4)
        gb.fit(flat_tr, ytr)
        preds["GBDT"] = gb.predict_proba(flat_te)[:, 1].reshape(Yte.shape)

        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        mu, sd = flat_tr.mean(0), flat_tr.std(0) + 1e-8
        lr.fit((flat_tr - mu) / sd, ytr)
        preds["LogReg"] = lr.predict_proba((flat_te - mu) / sd)[:, 1].reshape(Yte.shape)

        line = f"      {held:34s}"
        for m in methods:
            p = preds[m].reshape(-1)
            auc = roc_auc_score(yte, p)
            auc_rows[m].append(auc)
            # train-side predictions for the F1 threshold sweep
            if m == "WorldModel":
                ptr = wm_scores(wm, Xtr, Atr, Gtr).reshape(-1)
            elif m == "LSTM":
                with torch.no_grad():
                    ptr = torch.sigmoid(lstm(torch.tensor(
                        Ftr, dtype=torch.float32))).numpy().reshape(-1)
            elif m == "GBDT":
                ptr = gb.predict_proba(flat_tr)[:, 1]
            else:
                ptr = lr.predict_proba((flat_tr - mu) / sd)[:, 1]
            f1_rows[m].append(window_f1(ytr, ptr, yte, p)["f1_test"])
            fold_scores[m].extend(to_scores(mte, preds[m]))
            line += f"  {m[:4]} {auc:.3f}"
        print(line, flush=True)

    print("\n[3/4] Lead time at matched false-alarm rate")
    n_atk_scored = sum(1 for e in fold_scores[methods[0]]
                       if e.compromise_window is not None)
    print("")
    print(f"      {'method':12s} {'AUC':>7s} {'sd':>6s} {'F1':>7s} "
          f"{'caught':>8s} {'lead|det':>9s} {'mean lead':>10s} {'FPR':>6s}")
    summary = {}
    for m in methods:
        scores = fold_scores[m]
        if not scores:
            continue
        op = lead_at_fpr(scores, MAX_FPR)
        best = {"lead": op["mean_lead"],
                "lead_when_detected": op.get("mean_lead_when_detected", 0.0),
                "detect": op["detection_rate"],
                "fpr": op["achieved_fpr"],
                "tau": op["tau"]}
        summary[m] = {"auc": float(np.mean(auc_rows[m])),
                      "auc_std": float(np.std(auc_rows[m])),
                      "f1": float(np.mean(f1_rows[m])), **best}
        caught = int(round(best["detect"] * n_atk_scored))
        print(f"      {m:12s} {summary[m]['auc']:7.3f} {summary[m]['auc_std']:6.3f} "
              f"{summary[m]['f1']:7.3f} {caught:4d}/{n_atk_scored:<3d} "
              f"{best['lead_when_detected']:9.2f} {best['lead']:10.2f} "
              f"{best['fpr']:6.3f}")

    # Both lead columns are printed because neither alone is honest. "mean
    # lead" averages over ALL attack episodes, so catching one at maximum
    # lead scores the same as catching half at moderate lead. "lead|det" is
    # the mean over only those caught, which flatters a method that catches
    # almost nothing.
    print(f"\n      Episodes are {T_EPISODE} windows with compromise at window "
          f"{PRE_WINDOWS}, so {PRE_WINDOWS} is the maximum achievable lead - a")
    print("      method showing it alarmed on the very first window.")
    print(f"      Only {n_atk_scored} attack episodes exist. Treat the counts, "
          "not the rates, as the sample.")

    print("\n[4/4] Verdicts")
    best_lead = max((v["lead"], k) for k, v in summary.items())
    wm_lead = summary.get("WorldModel", {}).get("lead", 0.0)
    lstm_lead = summary.get("LSTM", {}).get("lead", 0.0)
    lr_rank = sorted(summary, key=lambda k: -summary[k]["auc"]).index("LogReg")
    verdicts = {
        "E1": all(v["auc"] > 0.60 for v in summary.values()),
        "E2": not (wm_lead > lstm_lead),
        "E3": best_lead[0] < 3.35,
        "E4": best_lead[0] > 0.0,
        "E5": all(v["auc"] <= 0.80 for v in summary.values()),
        "E6": lr_rank < len(summary) - 1,
    }
    detail = {
        "E1": f"min AUC {min(v['auc'] for v in summary.values()):.3f} > 0.60",
        "E2": f"WM lead {wm_lead:.2f} does not beat LSTM {lstm_lead:.2f}",
        "E3": f"best real lead {best_lead[0]:.2f} < synthetic 3.35",
        "E4": f"best lead {best_lead[0]:.2f} > 0 at FPR <= {MAX_FPR:.0%}",
        "E5": f"max AUC {max(v['auc'] for v in summary.values()):.3f} <= 0.80",
        "E6": f"LogReg ranks {lr_rank + 1} of {len(summary)} by AUC",
    }
    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 78)
    for k in ("E1", "E2", "E3", "E4", "E5", "E6"):
        print(f"  {k}  {'HOLDS ' if verdicts[k] else 'FAILS '}  {detail[k]}")

    payload = {
        "corpus": "CTU-13 (Stratosphere Lab, CTU Prague), Kaggle mirror v1",
        "protocol": "leave-one-scenario-out",
        "n_scenarios": len(scenarios),
        "n_episodes": total,
        "n_attack_episodes": n_atk,
        "horizon": HORIZON,
        # Recorded because omitting it made this file's own result
        # irreproducible. The published row was measured at an --epochs value
        # this script no longer defaults to, and nothing here said which one;
        # experiments 16 to 18 exist because of that omission. Every knob that
        # can change a number now goes in the payload.
        "epochs": args.epochs,
        "quick": bool(args.quick),
        "max_fpr": MAX_FPR,
        "per_method": summary,
        "per_fold_auc": {m: [round(a, 4) for a in auc_rows[m]] for m in methods},
        "verdicts": verdicts,
        "cannot_test": ["counterfactual intervention (no real capture has the "
                        "branch that did not happen)",
                        "intermediate ATT&CK stages (CTU-13 labels are "
                        "botnet / normal / background only)"],
    }
    (RESULTS / "exp14_verdicts.json").write_text(json.dumps(payload, indent=1))
    print("\n  wrote results/exp14_verdicts.json")
    print(f"  total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
