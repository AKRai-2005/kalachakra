"""Experiment 02 — counterfactual intervention.

The question
------------
    "We see this. If we isolate the host now, what happens?"

This is the capability that survives from KALACHAKRA's original pitch after
experiment 01 refuted the lead-time claim. It is worth testing because a
world model can express it natively - `P(s_{t+1} | s_t, a)` - and because the
answer is *actionable* in a way a detection score is not.

The fair baseline
-----------------
It would be easy to compare against an action-blind classifier, note that its
counterfactual effect is identically zero, and declare victory. That is a
strawman. A competent team would simply feed the action in as a feature.

So the baseline here is an **action-aware LSTM**: the same sequence model from
experiment 01, with a per-window action channel appended to its inputs. It can
learn "isolate reduces compromise". The real question is whether the world
model does it *better* - specifically, whether it predicts **which**
interventions will work, not merely that interventions help on average.

Ground truth
------------
Only available because we generate the data. Each counterfactual pair shares a
bit-identical prefix and differs solely in the action at the fork. The true
containment probability at the moment of intervention is

    p_contain = clip(0.95 - 0.45*skill - 0.10*stage, 0.10, 0.98)

which is heterogeneous (0.30-0.83 in practice) and has observable correlates:
skill drives scan breadth and dwell time, stage drives the traffic pattern.
`p_contain` is never a model input - it is the causal parameter we score
against.

Evaluation protocol
-------------------
For each held-out pair, take the **factual** branch's windows up to and
including the fork (identical prefix; pre-intervention traffic), then ask each
model twice, varying only the action:

    risk_none = P(compromise within H | do(a = none))
    risk_iso  = P(compromise within H | do(a = isolate))
    effect    = risk_none - risk_iso

Pre-registered predictions (written before the first run)
---------------------------------------------------------
  R1  WM effect is positive on average (isolating helps).
  R2  WM says "isolating helps" on > 0.70 of pairs (effect > 0).
      NOTE: this is not accuracy - it never compares to an outcome, and a
      constant "always isolate" predictor would score 1.000. See RESULTS.md,
      "A metric correction".
  R3  corr(predicted effect, true p_contain) > 0.30 for the WM.
      **This is the claim.** It says the model predicts *which* interventions
      work, not just that intervention helps.
  R4  WM beats the action-aware LSTM on R3's correlation.

If R3 or R4 fails, the counterfactual claim does not hold either, and this
project has no differentiator left that we can evidence. That gets said
plainly rather than dressed up.

Run:  python experiments/exp02_counterfactual.py
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
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from kalachakra.data.episodes import (  # noqa: E402
    ACTIONS, EpisodeConfig, EpisodeGenerator, STAGES,
)
from kalachakra.dynamics.world_model import (  # noqa: E402
    TrainConfig, WorldModel, collapse_diagnostic, vicreg,
)
from kalachakra.state.graph import (  # noqa: E402
    GraphBuilder, N_GLOBAL_F, N_NODE_F, flat_window_features, hosts_from_config,
    stack_episode,
)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

HORIZON = 6
ROLLOUT_DEPTH = 6
ISOLATE = ACTIONS.index("isolate")


def pack(episodes, builder):
    """Episodes -> (X, A, G, S, ACT, Y, meta)."""
    X, Adj, G, St, Act, Y, meta = [], [], [], [], [], [], []
    for ep in episodes:
        snaps = builder.build_episode(ep)
        x, adj, novel, g, stage = stack_episode(snaps)
        c = ep.compromise_window
        T = len(snaps)
        y = np.zeros(T, dtype=np.int64)
        if c is not None:
            y[max(0, c - HORIZON):c] = 1
        X.append(x); Adj.append(adj); G.append(g); St.append(stage)
        Act.append(np.array(ep.action_sequence(), dtype=np.int64)); Y.append(y)
        meta.append({"id": ep.id, "compromise": c, "fork": ep.fork_window,
                     "action": ep.action, "skill": ep.skill,
                     "p_contain": ep.p_contain, "compromised": ep.compromised})
    return (np.stack(X), np.stack(Adj), np.stack(G), np.stack(St),
            np.stack(Act), np.stack(Y), meta)


def flat_feats(episodes, builder):
    return np.stack([np.stack([flat_window_features(s)
                               for s in builder.build_episode(ep)]) for ep in episodes])


class ActionAwareLSTM(nn.Module):
    """The honest baseline: a sequence classifier that can see the action."""

    def __init__(self, n_in: int, n_actions: int = len(ACTIONS), hidden: int = 96) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_in)
        self.act_emb = nn.Embedding(n_actions, 16)
        self.lstm = nn.LSTM(n_in + 16, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, act):
        z = torch.cat([self.norm(x), self.act_emb(act)], dim=-1)
        h, _ = self.lstm(z)
        return self.head(h).squeeze(-1)


def main() -> None:
    t0 = time.time()
    print("=" * 74)
    print("KALACHAKRA  ·  Experiment 02  ·  Counterfactual intervention")
    print("=" * 74)

    cfg = EpisodeConfig(n_windows=40, n_internal=24)
    gen = EpisodeGenerator(cfg, seed=4242)
    builder = GraphBuilder(hosts_from_config(cfg.internal_prefix, cfg.n_internal))

    print("\n[1/4] Generating episodes ...")
    base = gen.generate(n_attack=200, n_benign=140, n_contained=120)
    pairs = gen.generate_counterfactual_pairs(n_pairs=800, action="isolate")
    print(f"      {len(base)} observational episodes")
    print(f"      {len(pairs)} matched counterfactual pairs (factual / isolate)")

    pc = np.array([c.p_contain for _, c in pairs], dtype=np.float32)
    worked = np.array([1.0 if (f.compromised and not c.compromised) else 0.0
                       for f, c in pairs])
    print(f"      true p_contain: {pc.min():.2f}-{pc.max():.2f} (mean {pc.mean():.2f})")
    print(f"      interventions that prevented compromise: {100*worked.mean():.1f}%")

    # Split by PAIR, not by episode: both branches of a pair share a prefix, so
    # putting one in train and the other in test would leak that prefix wholesale.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    cut = int(0.7 * len(pairs))
    tr_pairs = [pairs[i] for i in order[:cut]]
    te_pairs = [pairs[i] for i in order[cut:]]
    print(f"      pair split: {len(tr_pairs)} train / {len(te_pairs)} test")

    train_eps = base + [e for pr in tr_pairs for e in pr]
    print(f"      training episodes: {len(train_eps)}")

    print("\n[2/4] Building tensors ...")
    Xtr, Atr, Gtr, Str, ACTtr, Ytr, _ = pack(train_eps, builder)
    Ftr = flat_feats(train_eps, builder)
    n_interv = int((ACTtr > 0).any(1).sum())
    print(f"      {n_interv} training episodes contain an intervention "
          f"({100*n_interv/len(train_eps):.0f}%)")

    # ------------------------------------------------------- action-aware LSTM
    print("\n[3/4] Baseline: action-aware LSTM ...")
    lstm = ActionAwareLSTM(Ftr.shape[-1])
    opt = torch.optim.Adam(lstm.parameters(), lr=2e-3)
    ft = torch.tensor(Ftr); at = torch.tensor(ACTtr); yt = torch.tensor(Ytr, dtype=torch.float32)
    for epoch in range(40):
        perm = torch.randperm(len(ft)); tot = 0.0
        for i in range(0, len(ft), 16):
            b = perm[i:i + 16]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(lstm(ft[b], at[b]), yt[b])
            loss.backward(); opt.step(); tot += loss.detach().item()
        if epoch % 20 == 19:
            print(f"        epoch {epoch+1:2d}  loss {tot/(len(ft)/16):.4f}")
    lstm.eval()

    # ------------------------------------------------------------- world model
    print("\n[4/4] World model with action-conditioned dynamics ...")
    tc = TrainConfig(epochs=35, lr=2e-3)
    wm = WorldModel(N_NODE_F, N_GLOBAL_F)
    opt = torch.optim.Adam(wm.parameters(), lr=tc.lr)
    xs = torch.tensor(Xtr); adjs = torch.tensor(Atr); gs = torch.tensor(Gtr)
    stg = torch.tensor(Str); acts = torch.tensor(ACTtr)

    for epoch in range(tc.epochs):
        perm = torch.randperm(len(xs)); tp = ts = tv = 0.0; nb = 0
        for i in range(0, len(xs), tc.batch):
            b = perm[i:i + tc.batch]
            opt.zero_grad()
            out = wm(xs[b], adjs[b], gs[b], acts[b], rollout_depth=ROLLOUT_DEPTH)
            z, pz, ps = out["z"], out["pred_z"], out["pred_stage"]
            R, S = out["rollout_depth"], out["n_starts"]
            tgt_z = torch.stack([z[:, d + 1:d + 1 + S] for d in range(R)], 2)
            tgt_st = torch.stack([stg[b][:, d + 1:d + 1 + S] for d in range(R)], 2)
            l_pred = F.mse_loss(pz, tgt_z.detach())
            l_roll = F.cross_entropy(ps.reshape(-1, len(STAGES)), tgt_st.reshape(-1))
            l_stage = F.cross_entropy(
                wm.stage_logits(z).reshape(-1, len(STAGES)), stg[b].reshape(-1))
            l_vic = vicreg(z)
            loss = l_pred + tc.w_stage * (l_stage + l_roll) + tc.w_vic * l_vic
            loss.backward(); opt.step()
            tp += l_pred.detach().item(); ts += l_roll.detach().item()
            tv += l_vic.detach().item(); nb += 1
        if epoch % 10 == 9:
            print(f"        epoch {epoch+1:2d}  pred {tp/nb:.4f}  "
                  f"rollout-stage {ts/nb:.4f}  vicreg {tv/nb:.4f}")
    wm.eval()

    # ------------------------------------------------------------- evaluation
    print("\n" + "=" * 74)
    print("COUNTERFACTUAL EVALUATION  (held-out pairs)")
    print("=" * 74)

    fact = [f for f, _ in te_pairs]
    Xe, Ae, Ge, Se, ACTe, Ye, me = pack(fact, builder)
    Fe = flat_feats(fact, builder)
    xe = torch.tensor(Xe); ae = torch.tensor(Ae); ge = torch.tensor(Ge)

    p_true = np.array([m["p_contain"] for m in me], dtype=np.float32)
    forks = [m["fork"] for m in me]
    actual = np.array([1.0 if (f.compromised and not c.compromised) else 0.0
                       for f, c in te_pairs])

    # Query both models at the fork window, on identical pre-intervention
    # context, varying only the action.
    wm_none, wm_iso, ls_none, ls_iso = [], [], [], []
    with torch.no_grad():
        for i, t in enumerate(forks):
            t = max(0, min(int(t), Xe.shape[1] - 2))
            xi, ai, gi = xe[i:i + 1], ae[i:i + 1], ge[i:i + 1]
            wm_none.append(float(wm.compromise_risk(xi, ai, gi, t, HORIZON, action=0)))
            wm_iso.append(float(wm.compromise_risk(xi, ai, gi, t, HORIZON,
                                                   action=ISOLATE, action_at=0)))
            # LSTM: same prefix, action channel set to the planned action at t.
            f_pre = torch.tensor(Fe[i:i + 1, :t + 1])
            a0 = torch.zeros(1, t + 1, dtype=torch.long)
            a1 = a0.clone(); a1[0, t] = ISOLATE
            ls_none.append(float(torch.sigmoid(lstm(f_pre, a0))[0, -1]))
            ls_iso.append(float(torch.sigmoid(lstm(f_pre, a1))[0, -1]))

    wm_eff = np.array(wm_none) - np.array(wm_iso)
    ls_eff = np.array(ls_none) - np.array(ls_iso)

    def report(name, eff):
        dir_acc = float((eff > 0).mean())
        r_p = pearsonr(eff, p_true)
        r_s = spearmanr(eff, p_true)
        r_a = pearsonr(eff, actual)
        print(f"\n  {name}")
        print(f"    mean predicted effect      {eff.mean():+.4f}   (sd {eff.std():.4f})")
        print(f"    says 'isolating helps'      {dir_acc:.3f}   (trivial always-isolate = 1.000)")
        print(f"    corr(effect, p_contain)    r={r_p[0]:+.3f}  p={r_p[1]:.2g}   "
              f"[spearman {r_s[0]:+.3f}]")
        print(f"    corr(effect, worked)       r={r_a[0]:+.3f}  p={r_a[1]:.2g}")
        return {"mean_effect": float(eff.mean()), "sd": float(eff.std()),
                "direction_accuracy": dir_acc,
                "pearson_p_contain": float(r_p[0]), "pearson_p_value": float(r_p[1]),
                "spearman_p_contain": float(r_s[0]),
                "pearson_worked": float(r_a[0])}

    print(f"\n  Ground truth on {len(te_pairs)} held-out pairs:")
    print(f"    true p_contain mean {p_true.mean():.3f}, "
          f"interventions that worked {100*actual.mean():.1f}%")

    res_wm = report("World model (action-conditioned rollout)", wm_eff)
    res_ls = report("Action-aware LSTM (baseline)", ls_eff)

    print("\n" + "=" * 74)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 74)
    r1 = res_wm["mean_effect"] > 0
    r2 = res_wm["direction_accuracy"] > 0.70
    r3 = res_wm["pearson_p_contain"] > 0.30
    r4 = res_wm["pearson_p_contain"] > res_ls["pearson_p_contain"]
    for k, ok, d in [("R1", r1, "WM effect positive on average"),
                     ("R2", r2, f"WM direction accuracy > 0.70 "
                                f"({res_wm['direction_accuracy']:.3f})"),
                     ("R3", r3, f"corr(effect, p_contain) > 0.30 "
                                f"({res_wm['pearson_p_contain']:+.3f})  [THE CLAIM]"),
                     ("R4", r4, "WM beats action-aware LSTM on that correlation")]:
        print(f"  {k}  {'HOLDS ' if ok else 'FAILS '}  {d}")

    if not (r3 and r4):
        print("\n  !! The counterfactual claim does not hold as stated.")
        print("     KALACHAKRA then has no evidenced differentiator, and that is")
        print("     what goes in the write-up.")

    with torch.no_grad():
        diag = collapse_diagnostic(wm(xe, ae, ge, torch.tensor(ACTe),
                                      rollout_depth=ROLLOUT_DEPTH)["z"])
    print(f"\n  latent effective rank {diag['effective_rank']:.1f} of {diag['dim']}")

    (RESULTS / "exp02_counterfactual.json").write_text(json.dumps(
        {"n_test_pairs": len(te_pairs), "horizon": HORIZON,
         "ground_truth": {"mean_p_contain": float(p_true.mean()),
                          "frac_worked": float(actual.mean())},
         "world_model": res_wm, "action_aware_lstm": res_ls,
         "collapse_diagnostic": diag,
         "verdicts": {"R1": bool(r1), "R2": bool(r2), "R3": bool(r3), "R4": bool(r4)}},
        indent=2, default=float))
    np.savez(RESULTS / "exp02_effects.npz", wm_effect=wm_eff, lstm_effect=ls_eff,
             p_contain=p_true, worked=actual)
    print("\n  wrote results/exp02_counterfactual.json")
    print(f"  total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
