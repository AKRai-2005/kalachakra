"""Experiment 10 — does FiLM conditioning buy a heterogeneous treatment effect?

The open problem from experiment 02
------------------------------------
The counterfactual worked directionally: asked "should we isolate this host
now?", the world model was right on 83% of held-out matched pairs while an
action-aware LSTM sat at chance. But the effect it predicted was essentially
**constant** - r = +0.007 against the true containment probability, and raising
the data from 260 to 800 pairs did not move it. It could say isolation helps;
it could not say which isolations would work, and the second is the more useful
product.

The hypothesis
--------------
The action entered the transition by concatenation: the predictor saw
`[h, e(a)]`. An MLP can learn an interaction from that, but nothing pushes it
to - the shortest path to lower loss is an additive shift, which *is* a constant
treatment effect. FiLM makes the interaction the primitive:

    h' = gamma(a) * h + beta(a)

so the action rescales the state instead of displacing it.

Both models are trained here on identical data, identical splits and an
identical budget. Only the conditioning differs. "We changed the architecture
and the number moved" is not a finding unless the old architecture ran on the
same data.

Pre-registered predictions (written before the first run)
---------------------------------------------------------
  Z1  FiLM reaches corr(predicted effect, true p_contain) > 0.30 - the target
      experiment 02 missed at +0.007.
  Z2  FiLM beats concat on that correlation.
  Z3  FiLM keeps direction accuracy >= 0.75, i.e. it does not buy heterogeneity
      by giving up the capability that already worked.
  Z4  FiLM's predicted effects are more spread out than concat's. If the
      variance does not move, the mechanism is not doing what it is supposed
      to and any correlation change is luck.

Run:  python experiments/exp10_film_counterfactual.py
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
import torch.nn.functional as F  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from kalachakra.data.episodes import ACTIONS, EpisodeConfig, EpisodeGenerator, STAGES  # noqa: E402
from kalachakra.dynamics.world_model import (  # noqa: E402
    TrainConfig, WorldModel, collapse_diagnostic, vicreg,
)
from kalachakra.state.graph import (  # noqa: E402
    GraphBuilder, N_GLOBAL_F, N_NODE_F, hosts_from_config, stack_episode,
)
from kalachakra.provenance import environment, stamped  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

HORIZON = 6
ROLLOUT_DEPTH = 6
ALPHA_SEED = 0
N_PAIRS = 800
ISOLATE = ACTIONS.index("isolate")


def pack(episodes, builder):
    X, A, G, S, ACT, meta = [], [], [], [], [], []
    for ep in episodes:
        snaps = builder.build_episode(ep)
        x, adj, novel, g, stage = stack_episode(snaps)
        X.append(x); A.append(adj); G.append(g); S.append(stage)
        ACT.append(np.array(ep.action_sequence(), dtype=np.int64))
        meta.append({"id": ep.id, "compromise": ep.compromise_window,
                     "fork": ep.fork_window, "skill": ep.skill,
                     "p_contain": ep.p_contain, "compromised": ep.compromised})
    return (np.stack(X), np.stack(A), np.stack(G), np.stack(S),
            np.stack(ACT), meta)


def train(mode, xs, adjs, gs, stg, acts, epochs=35, seed=0):
    """Train one world model. Identical everything except `mode`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tc = TrainConfig(epochs=epochs, lr=2e-3)
    wm = WorldModel(N_NODE_F, N_GLOBAL_F, action_conditioning=mode)
    opt = torch.optim.Adam(wm.parameters(), lr=tc.lr)

    for epoch in range(tc.epochs):
        perm = torch.randperm(len(xs))
        tp = ts = tv = 0.0
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
            loss.backward(); opt.step()
            tp += l_pred.detach().item(); ts += l_roll.detach().item()
            tv += vicreg(z).detach().item(); nb += 1
        if epoch % 10 == 9:
            print(f"        [{mode}] epoch {epoch+1:2d}  pred {tp/nb:.4f}  "
                  f"rollout-stage {ts/nb:.4f}  vicreg {tv/nb:.4f}")
    wm.eval()
    return wm


def counterfactual_effects(wm, xe, ae, ge, forks):
    """Ask each model twice on identical context, varying only the action."""
    none, iso = [], []
    with torch.no_grad():
        for i, t in enumerate(forks):
            t = max(0, min(int(t), xe.shape[1] - 2))
            xi, ai, gi = xe[i:i + 1], ae[i:i + 1], ge[i:i + 1]
            none.append(float(wm.compromise_risk(xi, ai, gi, t, HORIZON, action=0)))
            iso.append(float(wm.compromise_risk(xi, ai, gi, t, HORIZON,
                                                action=ISOLATE, action_at=0)))
    return np.array(none) - np.array(iso)


def main() -> None:
    t_all = time.time()
    print("=" * 78)
    print("KALACHAKRA  ·  Experiment 10  ·  FiLM vs concat action conditioning")
    print("=" * 78)

    cfg = EpisodeConfig(n_windows=40, n_internal=24)
    gen = EpisodeGenerator(cfg, seed=4242)
    builder = GraphBuilder(hosts_from_config(cfg.internal_prefix, cfg.n_internal))

    print("\n[1/4] Generating episodes ...")
    base = gen.generate(n_attack=200, n_benign=140, n_contained=120)
    pairs = gen.generate_counterfactual_pairs(N_PAIRS, action="isolate")
    print(f"      {len(base)} observational + {len(pairs)} matched pairs")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    cut = int(0.7 * len(pairs))
    tr_pairs = [pairs[i] for i in order[:cut]]
    te_pairs = [pairs[i] for i in order[cut:]]
    train_eps = base + [e for pr in tr_pairs for e in pr]
    print(f"      pair split: {len(tr_pairs)} train / {len(te_pairs)} test")

    print("\n[2/4] Building tensors ...")
    Xtr, Atr, Gtr, Str, ACTtr, _ = pack(train_eps, builder)
    xs, adjs, gs = torch.tensor(Xtr), torch.tensor(Atr), torch.tensor(Gtr)
    stg, acts = torch.tensor(Str), torch.tensor(ACTtr)

    fact = [f for f, _ in te_pairs]
    Xe, Ae, Ge, Se, ACTe, me = pack(fact, builder)
    xe, ae, ge = torch.tensor(Xe), torch.tensor(Ae), torch.tensor(Ge)
    p_true = np.array([m["p_contain"] for m in me], dtype=np.float32)
    forks = [m["fork"] for m in me]
    worked = np.array([1.0 if (f.compromised and not c.compromised) else 0.0
                       for f, c in te_pairs])
    print(f"      {len(te_pairs)} held-out pairs, true p_contain "
          f"{p_true.min():.2f}-{p_true.max():.2f}, {100*worked.mean():.1f}% worked")

    print("\n[3/4] Training both conditionings on identical data ...")
    results = {}
    for mode in ("concat", "film"):
        t0 = time.time()
        wm = train(mode, xs, adjs, gs, stg, acts, seed=ALPHA_SEED)
        eff = counterfactual_effects(wm, xe, ae, ge, forks)
        with torch.no_grad():
            diag = collapse_diagnostic(
                wm(xe, ae, ge, torch.tensor(ACTe),
                   rollout_depth=ROLLOUT_DEPTH)["z"])
        r_p = pearsonr(eff, p_true)
        r_s = spearmanr(eff, p_true)
        r_w = pearsonr(eff, worked)
        results[mode] = {
            "mean_effect": float(eff.mean()), "sd_effect": float(eff.std()),
            "direction_accuracy": float((eff > 0).mean()),
            "pearson_p_contain": float(r_p[0]), "p_value": float(r_p[1]),
            "spearman_p_contain": float(r_s[0]),
            "pearson_worked": float(r_w[0]),
            "effective_rank": diag["effective_rank"],
            "train_s": time.time() - t0,
        }
        print(f"      [{mode}] done in {results[mode]['train_s']:.0f}s")

    print("\n[4/4] Comparison on identical held-out pairs")
    print(f"      {'':<8s} {'mean eff':>9s} {'sd eff':>8s} {'direction':>10s} "
          f"{'r(p_contain)':>13s} {'p':>8s} {'rank':>6s}")
    for mode in ("concat", "film"):
        r = results[mode]
        print(f"      {mode:<8s} {r['mean_effect']:>+9.4f} {r['sd_effect']:>8.4f} "
              f"{r['direction_accuracy']:>10.3f} {r['pearson_p_contain']:>+13.3f} "
              f"{r['p_value']:>8.2g} {r['effective_rank']:>6.1f}")

    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 78)
    f, c = results["film"], results["concat"]
    z1 = f["pearson_p_contain"] > 0.30
    z2 = f["pearson_p_contain"] > c["pearson_p_contain"]
    z3 = f["direction_accuracy"] >= 0.75
    z4 = f["sd_effect"] > c["sd_effect"]
    for k, ok, d in [
        ("Z1", z1, f"FiLM r(effect, p_contain) > 0.30 "
                   f"({f['pearson_p_contain']:+.3f})  [THE CLAIM]"),
        ("Z2", z2, f"FiLM beats concat on that correlation "
                   f"({f['pearson_p_contain']:+.3f} vs {c['pearson_p_contain']:+.3f})"),
        ("Z3", z3, f"direction accuracy held at >= 0.75 "
                   f"({f['direction_accuracy']:.3f} vs concat {c['direction_accuracy']:.3f})"),
        ("Z4", z4, f"FiLM effects are more spread out "
                   f"(sd {f['sd_effect']:.4f} vs {c['sd_effect']:.4f})")]:
        print(f"  {k}  {'HOLDS ' if ok else 'FAILS '}  {d}")

    if not z1:
        print("\n  !! Z1 FAILED. FiLM did not deliver a heterogeneous effect either.")
        print("     The constant-effect limit is not about how the action enters")
        print("     the transition, and the write-up must say so.")

    (RESULTS / "exp10_verdicts.json").write_text(json.dumps({
        "environment": environment(),
        "n_test_pairs": len(te_pairs), "n_pairs": N_PAIRS, "horizon": HORIZON,
        "ground_truth": {"mean_p_contain": float(p_true.mean()),
                         "frac_worked": float(worked.mean())},
        "results": results,
        "verdicts": {"Z1": bool(z1), "Z2": bool(z2), "Z3": bool(z3), "Z4": bool(z4)},
    }, indent=2, default=float))
    print("\n  wrote results/exp10_verdicts.json")
    print(f"  total runtime {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
