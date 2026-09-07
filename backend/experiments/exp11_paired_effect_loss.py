"""Experiment 11 — supervise the treatment effect directly.

Where this comes from
---------------------
Experiment 02 found the counterfactual works directionally but predicts an
essentially constant effect. Experiment 10 tested the architectural
explanation - that the action entered the transition additively - by building
FiLM conditioning, and refuted it: FiLM trained (gamma drifts 0.23 from
identity) and made the correlation slightly *worse*.

That left one diagnosis. **The objective never touches the treatment effect.**
Each episode carries one action sequence, `none` or `isolate`, never both, so
the model learns marginal dynamics `P(s_{t+1} | s_t, a)` while the difference
between two rollouts - the thing we actually report - has no gradient path.
This experiment gives it one.

The loss
--------
For each matched pair, roll out from the window before the fork under both
actions on the *shared* prefix, and supervise the predicted delta:

    delta_hat = risk(a=none) - risk(a=isolate)
    L_effect  = MSE(delta_hat, delta_observed)

`delta_observed` is the **realised binary outcome difference**: 1 when the
factual branch reached compromise and the counterfactual one did not, else 0.

What the target is NOT
----------------------
It is not `p_contain`. That is the generator's internal causal parameter, it is
what the evaluation correlates against, and training on it would be leakage
dressed up as a result. The model sees one noisy Bernoulli draw per pair -
which is all a real programme of randomised interventions would ever give.

The honest limitation of this whole approach
---------------------------------------------
Paired counterfactual data does not exist in the field. You observe the branch
that happened, never the one that did not. So a loss on matched pairs is
trainable here only because we generate the data. The deployable analogue is a
randomised-intervention programme plus standard causal estimation from
*unpaired* outcomes (IPW, doubly-robust) - a weaker setting than this. That gap
is stated rather than papered over.

Pre-registered predictions (written before the first run)
---------------------------------------------------------
  AA1  The effect-supervised model reaches corr(predicted effect, true
       p_contain) > 0.30 - the target exp02 and exp10 both missed.
  AA2  It beats the unsupervised control on that correlation.
  AA3  Direction accuracy stays >= 0.75, i.e. heterogeneity is not bought by
       giving up the capability that already works.
  AA4  Predicted-effect spread increases against the control.

Run:  python experiments/exp11_paired_effect_loss.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from kalachakra.data.episodes import ACTIONS, EpisodeConfig, EpisodeGenerator, STAGES  # noqa: E402
from kalachakra.dynamics.world_model import TrainConfig, WorldModel, vicreg
from kalachakra.state.graph import (  # noqa: E402
    GraphBuilder, N_GLOBAL_F, N_NODE_F, hosts_from_config, stack_episode,
)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

HORIZON = 6
ROLLOUT_DEPTH = 6
N_PAIRS = 800
EPOCHS = 30
PAIRS_PER_EPOCH = 160          # subsample: each pair costs two full rollouts
W_EFFECT = 2.0
ISOLATE = ACTIONS.index("isolate")
SEED = 0


def pack(episodes, builder):
    X, A, G, S, ACT, meta = [], [], [], [], [], []
    for ep in episodes:
        x, adj, novel, g, stage = stack_episode(builder.build_episode(ep))
        X.append(x); A.append(adj); G.append(g); S.append(stage)
        ACT.append(np.array(ep.action_sequence(), dtype=np.int64))
        meta.append({"id": ep.id, "compromise": ep.compromise_window,
                     "fork": ep.fork_window, "p_contain": ep.p_contain,
                     "compromised": ep.compromised})
    return (np.stack(X), np.stack(A), np.stack(G), np.stack(S),
            np.stack(ACT), meta)


def effect_loss(wm, xf, af, gf, forks, delta_obs, idx, horizon=HORIZON):
    """MSE between the predicted counterfactual delta and the realised one.

    Pairs are grouped by fork window because a rollout starts at a per-sample
    index; grouping lets each distinct start be one batched rollout instead of
    one per pair.
    """
    by_t = defaultdict(list)
    for i in idx:
        t = max(0, min(int(forks[i]) - 1, xf.shape[1] - 2))
        by_t[t].append(i)

    total, n = xf.new_zeros(()), 0
    for t, group in by_t.items():
        sel = torch.tensor(group, dtype=torch.long)
        xs, adjs, gs = xf[sel], af[sel], gf[sel]
        risk_none = wm.compromise_risk_grad(xs, adjs, gs, t, horizon, action=0)
        risk_iso = wm.compromise_risk_grad(xs, adjs, gs, t, horizon,
                                           action=ISOLATE, action_at=0)
        delta_hat = risk_none - risk_iso
        total = total + F.mse_loss(delta_hat, delta_obs[sel], reduction="sum")
        n += len(group)
    return total / max(1, n)


def train(xs, adjs, gs, stg, acts, *, pair_data=None, epochs=EPOCHS, seed=SEED):
    """Train a world model; `pair_data` adds the paired-difference term."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tc = TrainConfig(epochs=epochs, lr=2e-3)
    wm = WorldModel(N_NODE_F, N_GLOBAL_F, action_conditioning="film")
    opt = torch.optim.Adam(wm.parameters(), lr=tc.lr)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        perm = torch.randperm(len(xs))
        tp = ts = te = 0.0
        nb = 0
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
            loss = l_pred + tc.w_stage * (l_stage + l_roll) + tc.w_vic * vicreg(z)
            loss.backward()
            opt.step()
            tp += l_pred.detach().item(); ts += l_roll.detach().item(); nb += 1

        if pair_data is not None:
            xf, af, gf, forks, delta_obs, n_pairs = pair_data
            sub = rng.choice(n_pairs, size=min(PAIRS_PER_EPOCH, n_pairs),
                             replace=False)
            opt.zero_grad()
            le = W_EFFECT * effect_loss(wm, xf, af, gf, forks, delta_obs, sub)
            le.backward()
            opt.step()
            te = le.detach().item()

        if epoch % 10 == 9:
            tag = "effect" if pair_data is not None else "  base"
            print(f"        [{tag}] epoch {epoch+1:2d}  pred {tp/nb:.4f}  "
                  f"rollout-stage {ts/nb:.4f}" +
                  (f"  effect {te:.4f}" if pair_data is not None else ""))
    wm.eval()
    return wm


def evaluate(wm, xe, ae, ge, forks, p_true, worked):
    none, iso = [], []
    with torch.no_grad():
        for i, t in enumerate(forks):
            t = max(0, min(int(t) - 1, xe.shape[1] - 2))
            xi, ai, gi = xe[i:i + 1], ae[i:i + 1], ge[i:i + 1]
            none.append(float(wm.compromise_risk(xi, ai, gi, t, HORIZON, action=0)))
            iso.append(float(wm.compromise_risk(xi, ai, gi, t, HORIZON,
                                                action=ISOLATE, action_at=0)))
    eff = np.array(none) - np.array(iso)
    r_p, r_s, r_w = pearsonr(eff, p_true), spearmanr(eff, p_true), pearsonr(eff, worked)
    return {
        "mean_effect": float(eff.mean()), "sd_effect": float(eff.std()),
        "direction_accuracy": float((eff > 0).mean()),
        "pearson_p_contain": float(r_p[0]), "p_value": float(r_p[1]),
        "spearman_p_contain": float(r_s[0]), "pearson_worked": float(r_w[0]),
    }, eff


def main() -> None:
    t_all = time.time()
    print("=" * 78)
    print("KALACHAKRA  ·  Experiment 11  ·  Supervising the counterfactual delta")
    print("=" * 78)

    cfg = EpisodeConfig(n_windows=40, n_internal=24)
    gen = EpisodeGenerator(cfg, seed=4242)
    builder = GraphBuilder(hosts_from_config(cfg.internal_prefix, cfg.n_internal))

    print("\n[1/4] Generating episodes ...")
    base = gen.generate(n_attack=200, n_benign=140, n_contained=120)
    pairs = gen.generate_counterfactual_pairs(N_PAIRS, action="isolate")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    cut = int(0.7 * len(pairs))
    tr_pairs = [pairs[i] for i in order[:cut]]
    te_pairs = [pairs[i] for i in order[cut:]]
    print(f"      {len(base)} observational + {len(pairs)} pairs "
          f"({len(tr_pairs)} train / {len(te_pairs)} test)")

    print("\n[2/4] Building tensors ...")
    train_eps = base + [e for pr in tr_pairs for e in pr]
    Xtr, Atr, Gtr, Str, ACTtr, _ = pack(train_eps, builder)
    xs, adjs, gs = torch.tensor(Xtr), torch.tensor(Atr), torch.tensor(Gtr)
    stg, acts = torch.tensor(Str), torch.tensor(ACTtr)

    # Training-pair tensors: the SHARED prefix, taken from the factual branch.
    tr_fact = [f for f, _ in tr_pairs]
    Xp, Ap, Gp, _, _, mp = pack(tr_fact, builder)
    xf, af, gf = torch.tensor(Xp), torch.tensor(Ap), torch.tensor(Gp)
    forks_tr = [m["fork"] for m in mp]
    # The supervision signal: one realised Bernoulli draw per pair. NOT p_contain.
    delta_obs = torch.tensor(
        [1.0 if (f.compromised and not c.compromised) else 0.0
         for f, c in tr_pairs], dtype=torch.float32)
    print(f"      effect target: realised binary difference, "
          f"{100*float(delta_obs.mean()):.1f}% positive  (p_contain never used)")

    te_fact = [f for f, _ in te_pairs]
    Xe, Ae, Ge, _, _, me = pack(te_fact, builder)
    xe, ae, ge = torch.tensor(Xe), torch.tensor(Ae), torch.tensor(Ge)
    forks_te = [m["fork"] for m in me]
    p_true = np.array([m["p_contain"] for m in me], dtype=np.float32)
    worked = np.array([1.0 if (f.compromised and not c.compromised) else 0.0
                       for f, c in te_pairs])

    pair_data = (xf, af, gf, forks_tr, delta_obs, len(tr_pairs))

    print("\n[3/4] Training control and effect-supervised models ...")
    results = {}
    t0 = time.time()
    ctrl = train(xs, adjs, gs, stg, acts, pair_data=None)
    results["control"], eff_c = evaluate(ctrl, xe, ae, ge, forks_te, p_true, worked)
    results["control"]["train_s"] = time.time() - t0
    print(f"      control done in {results['control']['train_s']:.0f}s")

    t0 = time.time()
    sup = train(xs, adjs, gs, stg, acts, pair_data=pair_data)
    results["effect_supervised"], eff_s = evaluate(
        sup, xe, ae, ge, forks_te, p_true, worked)
    results["effect_supervised"]["train_s"] = time.time() - t0
    print(f"      effect-supervised done in "
          f"{results['effect_supervised']['train_s']:.0f}s")

    print("\n[4/4] Held-out comparison")
    print(f"      {'':<18s} {'mean eff':>9s} {'sd eff':>8s} {'direction':>10s} "
          f"{'r(p_contain)':>13s} {'p':>8s}")
    for k in ("control", "effect_supervised"):
        r = results[k]
        print(f"      {k:<18s} {r['mean_effect']:>+9.4f} {r['sd_effect']:>8.4f} "
              f"{r['direction_accuracy']:>10.3f} {r['pearson_p_contain']:>+13.3f} "
              f"{r['p_value']:>8.2g}")

    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 78)
    sup_r, ctl_r = results["effect_supervised"], results["control"]
    aa1 = sup_r["pearson_p_contain"] > 0.30
    aa2 = sup_r["pearson_p_contain"] > ctl_r["pearson_p_contain"]
    aa3 = sup_r["direction_accuracy"] >= 0.75
    aa4 = sup_r["sd_effect"] > ctl_r["sd_effect"]
    for k, ok, d in [
        ("AA1", aa1, f"r(effect, p_contain) > 0.30 "
                     f"({sup_r['pearson_p_contain']:+.3f})  [THE CLAIM]"),
        ("AA2", aa2, f"beats the unsupervised control "
                     f"({sup_r['pearson_p_contain']:+.3f} vs "
                     f"{ctl_r['pearson_p_contain']:+.3f})"),
        ("AA3", aa3, f"direction accuracy held "
                     f"({sup_r['direction_accuracy']:.3f} vs "
                     f"{ctl_r['direction_accuracy']:.3f})"),
        ("AA4", aa4, f"effect spread increased "
                     f"({sup_r['sd_effect']:.4f} vs {ctl_r['sd_effect']:.4f})")]:
        print(f"  {k}  {'HOLDS ' if ok else 'FAILS '}  {d}")

    if not aa1:
        print("\n  !! AA1 FAILED. Supervising the delta did not produce a")
        print("     heterogeneous effect either. Report it and stop claiming the")
        print("     next mechanism will fix it.")

    (RESULTS / "exp11_verdicts.json").write_text(json.dumps({
        "n_pairs": N_PAIRS, "n_test_pairs": len(te_pairs),
        "epochs": EPOCHS, "pairs_per_epoch": PAIRS_PER_EPOCH,
        "w_effect": W_EFFECT, "horizon": HORIZON,
        "effect_target": "realised binary outcome difference (never p_contain)",
        "positive_rate_in_target": float(delta_obs.mean()),
        "results": results,
        "verdicts": {"AA1": bool(aa1), "AA2": bool(aa2),
                     "AA3": bool(aa3), "AA4": bool(aa4)},
    }, indent=2, default=float))
    np.savez(RESULTS / "exp11_effects.npz", control=eff_c,
             effect_supervised=eff_s, p_contain=p_true, worked=worked)
    print("\n  wrote results/exp11_verdicts.json")
    print(f"  total runtime {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
