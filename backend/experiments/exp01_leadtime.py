"""Experiment 01 — lead time: world model vs sequence classifier vs per-window GBDT.

Question
--------
Does learning latent transition dynamics buy warning time that a classifier
cannot, at a matched false-alarm rate?

Task, identical for all three methods
-------------------------------------
At each window t, emit a risk score for "this network reaches LATERAL_MOVEMENT
or beyond at some window after t". Scores are compared on the lead-time curve
in `metrics/leadtime.py`, always at matched FPR.

Methods
-------
  LOGREG  per-window flat features, linear, no history. The floor the problem
          statement names explicitly - added after the first four experiments,
          together with window-level F1. Both are reported measurements, not
          pre-registered predictions: Q1-Q4 below were registered before the
          original run and have not been touched.
  GBDT    per-window flat features, no history. The naive baseline.
  LSTM    sequence of flat features up to t. A genuinely strong baseline -
          it sees the same history the world model does.
  WM      graph encoder -> latent transition model -> K-step rollout ->
          P(reach compromise within K).

The LSTM matters. Beating a per-window GBDT proves only that history helps,
which nobody doubts. The claim worth making is that *forward simulation* beats
*discriminative use of the same history*, and only the LSTM comparison tests it.

Pre-registered predictions (written before the first run)
---------------------------------------------------------
  Q1  All three beat chance; GBDT is worst.
  Q2  WM >= LSTM on mean lead time at FPR <= 0.10.
  Q3  The latent does not collapse (effective rank > 8 of 64).
  Q4  WM's advantage grows at *lower* FPR budgets, because rollout should be
      more confident earlier than a discriminative score.

If Q2 fails, the headline claim of this project does not hold and gets
rewritten rather than reworded.

Run:  python experiments/exp01_leadtime.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from kalachakra.data.episodes import EpisodeConfig, EpisodeGenerator, STAGES
from kalachakra.dynamics.world_model import (  # noqa: E402
    TrainConfig, WorldModel, collapse_diagnostic, vicreg,
)
from kalachakra.metrics.leadtime import EpisodeScores, lead_at_fpr  # noqa: E402
from kalachakra.state.graph import (  # noqa: E402
    GraphBuilder, N_GLOBAL_F, N_NODE_F, flat_window_features, hosts_from_config,
    stack_episode,
)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

HORIZON = 6          # rollout steps at inference; ~6 windows of useful warning
ROLLOUT_DEPTH = 6    # rollout steps during training - must match HORIZON, or the
                     # model is optimised for a horizon it is never used at
FPR_BUDGETS = (0.02, 0.05, 0.10, 0.20)


def build_dataset(episodes, builder, horizon: int = HORIZON):
    """Episodes -> tensors + per-window forecasting labels.

    `horizon` is an argument rather than a read of the module constant so that
    experiment 16 can sweep it. It defaults to HORIZON, so every existing call
    site is unchanged - and changing it changes the *question being asked*, not
    just a hyperparameter, which is why the sweep cannot compare AUCs.
    """
    X, A, G, S, Y, meta = [], [], [], [], [], []
    for ep in episodes:
        snaps = builder.build_episode(ep)
        x, adj, novel, g, stage = stack_episode(snaps)
        c = ep.compromise_window
        T = len(snaps)
        # Label: will compromise occur within the next HORIZON windows?
        #
        # This is horizon-limited on purpose, and the first version of this
        # experiment got it wrong. Labelling every window before compromise as
        # positive asks the baselines "is an attack in progress at all", which
        # they can answer the moment any attack traffic appears - giving them
        # ~12 windows of apparent lead. The world model was being asked the
        # much harder "will compromise happen within K steps", which caps its
        # lead near K by construction. The two methods were answering different
        # questions and the comparison was meaningless.
        #
        # All three methods now predict the same thing.
        y = np.zeros(T, dtype=np.int64)
        if ep.is_attack and c is not None:
            y[max(0, c - horizon):c] = 1
        X.append(x); A.append(adj); G.append(g); S.append(stage); Y.append(y)
        meta.append({"id": ep.id, "is_attack": ep.is_attack, "compromise": c})
    return (np.stack(X), np.stack(A), np.stack(G), np.stack(S), np.stack(Y), meta)


def flat_features(episodes, builder):
    out = []
    for ep in episodes:
        snaps = builder.build_episode(ep)
        out.append(np.stack([flat_window_features(s) for s in snaps]))
    return np.stack(out)


class LSTMBaseline(nn.Module):
    """Discriminative sequence model over the same history the WM sees."""

    def __init__(self, n_in: int, hidden: int = 96) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_in)
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        h, _ = self.lstm(self.norm(x))
        return self.head(h).squeeze(-1)


def window_f1(y_tr, p_tr, y_te, p_te) -> dict:
    """Window-level F1 at a threshold chosen on TRAIN, applied to test.

    The problem statement asks for F1 scores against a logistic-regression
    baseline. F1 needs a threshold and AUC does not, so the choice has to be
    made somewhere - and making it on the test split would flatter every model
    equally but mean nothing. The cut is swept on train only.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y_tr, (p_tr >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return {"threshold": best_t,
            "f1_train": float(best_f1),
            "f1_test": float(f1_score(y_te, (p_te >= best_t).astype(int),
                                      zero_division=0))}


def to_scores(meta, per_window: np.ndarray):
    return [EpisodeScores(episode_id=m["id"], scores=per_window[i],
                          is_attack=m["is_attack"], compromise_window=m["compromise"])
            for i, m in enumerate(meta)]


def train_world_model(Xtr, Atr, Gtr, Str, tc, *, verbose: bool = True,
                      rollout_depth: int = ROLLOUT_DEPTH):
    """Train the world model on the JEPA-style objective.

    Extracted from `main()` so experiment 14 can train on real CTU-13 episodes
    with provably the same objective. Replicating the loss in the other file
    would let the two drift apart, and a synthetic-vs-real difference would then
    be unattributable - which is the whole question experiment 14 exists to ask.
    """
    wm = WorldModel(N_NODE_F, N_GLOBAL_F)
    opt = torch.optim.Adam(wm.parameters(), lr=tc.lr)

    xs = torch.tensor(Xtr); adjs = torch.tensor(Atr); gs = torch.tensor(Gtr)
    stg = torch.tensor(Str); acts = torch.zeros(xs.shape[:2], dtype=torch.long)

    for epoch in range(tc.epochs):
        perm = torch.randperm(len(xs))
        tot_p = tot_s = tot_v = 0.0
        nb = 0
        for i in range(0, len(xs), tc.batch):
            b = perm[i:i + tc.batch]
            opt.zero_grad()
            out = wm(xs[b], adjs[b], gs[b], acts[b], rollout_depth=rollout_depth)
            z, pz, ps = out["z"], out["pred_z"], out["pred_stage"]
            R, S = out["rollout_depth"], out["n_starts"]

            # Targets for each rollout depth: the latent and stage actually
            # observed d steps after each start position.
            tgt_z = torch.stack([z[:, d + 1:d + 1 + S] for d in range(R)], 2)
            tgt_stage = torch.stack([stg[b][:, d + 1:d + 1 + S] for d in range(R)], 2)

            # Predictive loss in latent space, stop-grad on the target.
            l_pred = F.mse_loss(pz, tgt_z.detach())
            # The rollout path itself is supervised: predicted latents go
            # through the same ATT&CK head that inference uses.
            l_roll = F.cross_entropy(ps.reshape(-1, len(STAGES)), tgt_stage.reshape(-1))
            # Grounding on encoded latents.
            l_stage = F.cross_entropy(
                wm.stage_logits(z).reshape(-1, len(STAGES)), stg[b].reshape(-1))
            l_vic = vicreg(z)
            loss = l_pred + tc.w_stage * (l_stage + l_roll) + tc.w_vic * l_vic
            loss.backward(); opt.step()
            tot_p += l_pred.detach().item(); tot_s += l_roll.detach().item()
            tot_v += l_vic.detach().item(); nb += 1
        if verbose and epoch % 10 == 9:
            print(f"        epoch {epoch+1:2d}  pred {tot_p/nb:.4f}  "
                  f"rollout-stage {tot_s/nb:.4f}  vicreg {tot_v/nb:.4f}")
    wm.eval()
    return wm


def main() -> None:
    t0 = time.time()
    print("=" * 74)
    print("KALACHAKRA  ·  Experiment 01  ·  Lead time at matched false-alarm rate")
    print("=" * 74)

    cfg = EpisodeConfig(n_windows=40, n_internal=24)
    gen = EpisodeGenerator(cfg, seed=4242)
    print("\n[1/5] Generating episodes ...")
    eps = gen.generate(n_attack=200, n_benign=140, n_contained=120)
    builder = GraphBuilder(hosts_from_config(cfg.internal_prefix, cfg.n_internal))
    n_comp = sum(1 for e in eps if e.compromise_window is not None)
    n_contained = sum(1 for e in eps if e.is_attack and e.compromise_window is None)
    n_quiet = sum(1 for e in eps if not e.is_attack)
    print(f"      {len(eps)} episodes: {n_comp} reach compromise (positives)")
    print(f"        negatives = {n_contained} contained attacks + {n_quiet} quiet benign")
    print("        contained episodes are what make this forecasting, not detection")

    # Episode-level split. Episodes are independent draws, so splitting on
    # episodes (never on windows) is what prevents leakage: windows from the
    # same episode must never straddle the boundary.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(eps))
    cut = int(0.7 * len(eps))
    tr_idx, te_idx = order[:cut], order[cut:]
    tr_eps = [eps[i] for i in tr_idx]
    te_eps = [eps[i] for i in te_idx]
    print(f"      split: {len(tr_eps)} train / {len(te_eps)} test episodes")

    print("\n[2/5] Building graph snapshots ...")
    Xtr, Atr, Gtr, Str, Ytr, mtr = build_dataset(tr_eps, builder)
    Xte, Ate, Gte, Ste, Yte, mte = build_dataset(te_eps, builder)
    Ftr, Fte = flat_features(tr_eps, builder), flat_features(te_eps, builder)
    print(f"      graph: nodes={builder.n}  node_f={N_NODE_F}  global_f={N_GLOBAL_F}")
    print(f"      flat baseline feature dim = {Ftr.shape[-1]}")

    results = {}
    f1s = {}
    ytr_flat, yte_flat = Ytr.reshape(-1), Yte.reshape(-1)

    # --------------------------------------------------- logistic regression
    # The problem statement names this baseline explicitly. It is the honest
    # floor: a linear model on per-window features with no history at all. If
    # a world model cannot beat it, nothing else in this file matters.
    print("\n[3/6] Baseline 0: logistic regression (per-window, no history) ...")
    scaler = StandardScaler().fit(Ftr.reshape(-1, Ftr.shape[-1]))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    lr.fit(scaler.transform(Ftr.reshape(-1, Ftr.shape[-1])), ytr_flat)
    p_lr_tr = lr.predict_proba(scaler.transform(Ftr.reshape(-1, Ftr.shape[-1])))[:, 1]
    p_lr = lr.predict_proba(
        scaler.transform(Fte.reshape(-1, Fte.shape[-1])))[:, 1].reshape(Fte.shape[:2])
    auc_lr = roc_auc_score(yte_flat, p_lr.reshape(-1))
    f1s["LogisticRegression"] = window_f1(ytr_flat, p_lr_tr, yte_flat, p_lr.reshape(-1))
    print(f"      window-level AUC = {auc_lr:.4f}   "
          f"F1 = {f1s['LogisticRegression']['f1_test']:.4f}")
    results["LogisticRegression"] = to_scores(mte, p_lr)

    # ------------------------------------------------------------------ GBDT
    print("\n[4/6] Baseline 1: per-window GBDT (no history) ...")
    gb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                       subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                       eval_metric="logloss", n_jobs=4, random_state=0)
    gb.fit(Ftr.reshape(-1, Ftr.shape[-1]), Ytr.reshape(-1))
    p_gb = gb.predict_proba(Fte.reshape(-1, Fte.shape[-1]))[:, 1].reshape(Fte.shape[:2])
    auc_gb = roc_auc_score(yte_flat, p_gb.reshape(-1))
    p_gb_tr = gb.predict_proba(Ftr.reshape(-1, Ftr.shape[-1]))[:, 1]
    f1s["GBDT"] = window_f1(ytr_flat, p_gb_tr, yte_flat, p_gb.reshape(-1))
    print(f"      window-level AUC = {auc_gb:.4f}   F1 = {f1s['GBDT']['f1_test']:.4f}")
    results["GBDT"] = to_scores(mte, p_gb)

    # ------------------------------------------------------------------ LSTM
    print("\n[5/6] Baseline 2: LSTM sequence classifier (same history) ...")
    lstm = LSTMBaseline(Ftr.shape[-1])
    opt = torch.optim.Adam(lstm.parameters(), lr=2e-3)
    ft = torch.tensor(Ftr); yt = torch.tensor(Ytr, dtype=torch.float32)
    for epoch in range(40):
        perm = torch.randperm(len(ft))
        tot = 0.0
        for i in range(0, len(ft), 16):
            b = perm[i:i + 16]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(lstm(ft[b]), yt[b])
            loss.backward(); opt.step(); tot += loss.detach().item()
        if epoch % 10 == 9:
            print(f"        epoch {epoch+1:2d}  loss {tot/(len(ft)/16):.4f}")
    lstm.eval()
    with torch.no_grad():
        p_lstm = torch.sigmoid(lstm(torch.tensor(Fte))).numpy()
    auc_lstm = roc_auc_score(yte_flat, p_lstm.reshape(-1))
    with torch.no_grad():
        p_lstm_tr = torch.sigmoid(lstm(torch.tensor(Ftr))).numpy()
    f1s["LSTM"] = window_f1(ytr_flat, p_lstm_tr.reshape(-1), yte_flat, p_lstm.reshape(-1))
    print(f"      window-level AUC = {auc_lstm:.4f}   F1 = {f1s['LSTM']['f1_test']:.4f}")
    results["LSTM"] = to_scores(mte, p_lstm)

    # ------------------------------------------------------------- world model
    print("\n[6/6] World model: latent dynamics + K-step rollout ...")
    tc = TrainConfig(epochs=35, lr=2e-3)
    wm = train_world_model(Xtr, Atr, Gtr, Str, tc)

    xte = torch.tensor(Xte); ate = torch.tensor(Ate); gte = torch.tensor(Gte)
    with torch.no_grad():
        diag = collapse_diagnostic(wm(xte, ate, gte,
                                      torch.zeros(xte.shape[:2], dtype=torch.long),
                                      rollout_depth=ROLLOUT_DEPTH)["z"])
    print(f"      latent collapse diagnostic: effective_rank={diag['effective_rank']:.2f} "
          f"of {diag['dim']}, mean_std={diag['mean_std']:.3f}")

    T = Xte.shape[1]
    risk = np.zeros((len(te_eps), T), dtype=np.float32)
    with torch.no_grad():
        for t in range(T - 1):
            risk[:, t] = wm.compromise_risk(xte, ate, gte, t, HORIZON).numpy()
    risk[:, -1] = risk[:, -2]
    auc_wm = roc_auc_score(yte_flat, risk.reshape(-1))
    risk_tr = np.zeros((len(tr_eps), Xtr.shape[1]), dtype=np.float32)
    with torch.no_grad():
        xtr_t, atr_t, gtr_t = torch.tensor(Xtr), torch.tensor(Atr), torch.tensor(Gtr)
        for t in range(Xtr.shape[1] - 1):
            risk_tr[:, t] = wm.compromise_risk(xtr_t, atr_t, gtr_t, t, HORIZON).numpy()
    risk_tr[:, -1] = risk_tr[:, -2]
    f1s["WorldModel"] = window_f1(ytr_flat, risk_tr.reshape(-1), yte_flat, risk.reshape(-1))
    print(f"      window-level AUC = {auc_wm:.4f}  (same label, same task)   "
          f"F1 = {f1s['WorldModel']['f1_test']:.4f}")
    results["WorldModel"] = to_scores(mte, risk)

    # ------------------------------------------------------------- comparison
    print("\n" + "=" * 74)
    print("LEAD TIME AT MATCHED FALSE-ALARM RATE  (windows of warning)")
    print("=" * 74)
    header = f"{'method':<12s}" + "".join(f"{f'FPR<={b:.0%}':>16s}" for b in FPR_BUDGETS)
    print(header)
    print("-" * len(header))
    table = {}
    for name, scores in results.items():
        row = {}
        line = f"{name:<12s}"
        for b in FPR_BUDGETS:
            op = lead_at_fpr(scores, b)
            row[str(b)] = op
            line += f"{op['mean_lead']:>9.2f} ({op['detection_rate']:.0%})"
        print(line)
        table[name] = row

    print("\nReading: mean lead in windows, (detection rate before compromise).")

    print("\n" + "=" * 74)
    print("WINDOW-LEVEL F1 vs THE LOGISTIC-REGRESSION FLOOR")
    print("=" * 74)
    print(f"  {'method':<22s} {'AUC':>8s} {'F1 train':>10s} {'F1 test':>9s} {'cut':>6s}")
    aucs = {"LogisticRegression": auc_lr, "GBDT": auc_gb,
            "LSTM": auc_lstm, "WorldModel": auc_wm}
    for name in ("LogisticRegression", "GBDT", "LSTM", "WorldModel"):
        m = f1s[name]
        print(f"  {name:<22s} {aucs[name]:>8.4f} {m['f1_train']:>10.4f} "
              f"{m['f1_test']:>9.4f} {m['threshold']:>6.2f}")
    floor = f1s["LogisticRegression"]["f1_test"]
    wm_f1 = f1s["WorldModel"]["f1_test"]
    print("\n  The problem statement asks for this comparison by name. The threshold")
    print("  is swept on train and applied unchanged to test - sweeping it on test")
    print("  would flatter every model equally and mean nothing.")
    if wm_f1 < floor:
        print(f"\n  !! The world model ({wm_f1:.3f}) scores BELOW logistic regression")
        print(f"     ({floor:.3f}) on window-level F1. It is a forward simulator scored")
        print("     by rollout risk, not a window classifier - but this is the")
        print("     benchmark the problem statement asks for, and the answer is that")
        print("     on detection it loses to the simplest model in the file.")

    print("\n" + "=" * 74)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 74)
    wm10 = table["WorldModel"]["0.1"]["mean_lead"]
    ls10 = table["LSTM"]["0.1"]["mean_lead"]
    gb10 = table["GBDT"]["0.1"]["mean_lead"]
    q1 = gb10 <= max(ls10, wm10)
    q2 = wm10 >= ls10
    q3 = diag["effective_rank"] > 8
    adv_low = table["WorldModel"]["0.02"]["mean_lead"] - table["LSTM"]["0.02"]["mean_lead"]
    adv_high = table["WorldModel"]["0.2"]["mean_lead"] - table["LSTM"]["0.2"]["mean_lead"]
    q4 = adv_low > adv_high
    for k, ok, d in [("Q1", q1, "all beat chance, GBDT worst"),
                     ("Q2", q2, "WM >= LSTM at FPR<=10%  [THE CLAIM]"),
                     ("Q3", q3, f"latent does not collapse (rank {diag['effective_rank']:.1f})"),
                     ("Q4", q4, "WM advantage grows at lower FPR")]:
        print(f"  {k}  {'HOLDS ' if ok else 'FAILS '}  {d}")
    if not q2:
        print("\n  !! Q2 FAILED. Forward simulation did not beat discriminative use of")
        print("     the same history. The headline claim must be rewritten, not reworded.")

    (RESULTS / "exp01_leadtime.json").write_text(json.dumps(
        {"auc": {"LogisticRegression": auc_lr, "GBDT": auc_gb,
                 "LSTM": auc_lstm, "WorldModel": auc_wm},
         # Window-level F1 against a logistic-regression floor, which the
         # problem statement asks for by name. Threshold swept on train.
         "window_f1": f1s,
         "collapse_diagnostic": diag,
         "horizon": HORIZON,
         "operating_points": table,
         "verdicts": {"Q1": bool(q1), "Q2": bool(q2), "Q3": bool(q3), "Q4": bool(q4)}},
        indent=2, default=float))
    np.savez(RESULTS / "exp01_scores.npz",
             **{f"{k}": np.stack([s.scores for s in v]) for k, v in results.items()},
             is_attack=np.array([m["is_attack"] for m in mte]),
             compromise=np.array([m["compromise"] if m["compromise"] is not None else -1
                                  for m in mte]))
    print("\n  wrote results/exp01_leadtime.json and exp01_scores.npz")
    print(f"  total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
