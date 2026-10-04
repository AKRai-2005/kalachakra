"""Precompute the trajectories the console replays.

Why precompute
--------------
Training the world model takes about four minutes. That is fine once and
unacceptable in front of a judge, so this runs the model exactly as experiment
11 does, then writes every number the console needs to a JSON file. The console
is a *replay* of a real run, not a simulation of one - the same choice EKAGRA's
console makes with its evidence log, and for the same reason: a demo that
recomputes under time pressure is a demo that fails under time pressure.

What it writes
--------------
For each held-out counterfactual case:

  - the observed ATT&CK stage sequence up to the fork (what the sensor saw)
  - the model's predicted compromise risk at each step of the horizon, twice:
    once rolling forward under "do nothing", once under "isolate"
  - the predicted effect, which is the difference between those two risks
  - the realised outcome of both branches, which is ground truth we only have
    because we generated the data
  - an integrated-gradients attribution: which traffic attributes, which
    window-level features and which hosts drove the "do nothing" forecast,
    together with the completeness residual that says whether to trust the
    ranking at all

The two risk curves diverging from a shared prefix are the whole point. A
detector scores the trajectory it observes; only a model with an action input
can be asked what the *other* branch would have looked like.

What the console must not imply
-------------------------------
Experiments 10 to 12 established that this model predicts an **average**
treatment effect, not a per-host one: it says isolating helps in 94% of cases,
with essentially zero correlation with which specific interventions work. That
94% is *not* accuracy - it never compares against an outcome, and a constant
"always isolate" predictor scores 100% on the same measure. So this file writes
all three numbers, and the console shows all three. A demo that ranked hosts by
predicted effect would be claiming something four experiments say we cannot do.

The attribution panel is the one ranking that is allowed, because it ranks
*features within one case* rather than cases against each other - and it ships
with the residual that proves the ranking is the model's own arithmetic.

Run:  python experiments/exp13_demo_trajectories.py     (~9 min cold)
      the trained model is cached in results/; reruns take about a minute.
      Pass --retrain to rebuild it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp11_paired_effect_loss import (  # noqa: E402
    HORIZON, ISOLATE, N_PAIRS, pack, train,
)

from kalachakra.data.episodes import (  # noqa: E402
    COMPROMISE_STAGE, EpisodeConfig, EpisodeGenerator, STAGES,
)
from kalachakra.dynamics.world_model import WorldModel  # noqa: E402
from kalachakra.explain import attribute  # noqa: E402
from kalachakra.state.graph import (  # noqa: E402
    GraphBuilder, N_GLOBAL_F, N_NODE_F, hosts_from_config,
)
from kalachakra.provenance import environment, stamped  # noqa: E402

RESULTS = ROOT / "results"
N_CASES = 40          # how many cases the console offers; the rest score the stats
IG_STEPS = 32         # integrated-gradients path resolution, before refinement
IG_MAX = 2048         # refinement ceiling; past this the case is marked loose
IG_TOL = 0.02         # residual allowed, as a fraction of the gap explained
N_HOSTS_SHOWN = 6     # per-host contributions the console lists


def cumulative_risk(probs: torch.Tensor) -> list:
    """P(reached compromise by step j), for j = 1..horizon.

    The attacker only has to get there once, so risk accumulates as
    1 - prod(1 - p_k) - the same combination `compromise_risk` uses, exposed
    per step so the console can draw it as a curve rather than a single number.
    """
    p_bad = probs[..., COMPROMISE_STAGE:].sum(-1).clamp(0, 1)
    keep = torch.cumprod(1.0 - p_bad, dim=1)
    return (1.0 - keep).squeeze(0).tolist()


def main() -> None:
    t0 = time.time()
    print("=" * 74)
    print("KALACHAKRA  ·  Experiment 13  ·  precompute console trajectories")
    print("=" * 74)

    cfg = EpisodeConfig(n_windows=40, n_internal=24)
    gen = EpisodeGenerator(cfg, seed=4242)
    hosts = hosts_from_config(cfg.internal_prefix, cfg.n_internal)
    builder = GraphBuilder(hosts)

    print("\n[1/5] Generating episodes ...")
    base = gen.generate(n_attack=200, n_benign=140, n_contained=120)
    pairs = gen.generate_counterfactual_pairs(N_PAIRS, action="isolate")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    cut = int(0.7 * len(pairs))
    tr_pairs = [pairs[i] for i in order[:cut]]
    te_pairs = [pairs[i] for i in order[cut:]]
    print(f"      {len(base)} base episodes, {len(tr_pairs)} train / "
          f"{len(te_pairs)} held-out pairs")

    print("\n[2/5] Training (the paired-difference model from exp11) ...")
    train_eps = base + [e for pr in tr_pairs for e in pr]
    Xtr, Atr, Gtr, Str, ACTtr, _ = pack(train_eps, builder)
    xs, adjs, gs = torch.tensor(Xtr), torch.tensor(Atr), torch.tensor(Gtr)
    stg, acts = torch.tensor(Str), torch.tensor(ACTtr)

    tr_fact = [f for f, _ in tr_pairs]
    Xp, Ap, Gp, _, _, mp = pack(tr_fact, builder)
    delta = torch.tensor(
        [1.0 if (f.compromised and not c.compromised) else 0.0
         for f, c in tr_pairs], dtype=torch.float32)

    # Cached so the attribution pass can be re-tuned without paying six minutes
    # of training each time. Everything upstream of this is seeded, so the cache
    # is reproducible - `--retrain` throws it away.
    ckpt = RESULTS / "demo_model.pt"
    if ckpt.exists() and "--retrain" not in sys.argv:
        wm = WorldModel(N_NODE_F, N_GLOBAL_F)
        wm.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"      reusing {ckpt.name} (pass --retrain to rebuild it)")
    else:
        wm = train(xs, adjs, gs, stg, acts,
                   pair_data=(torch.tensor(Xp), torch.tensor(Ap), torch.tensor(Gp),
                              [m["fork"] for m in mp], delta, len(tr_pairs)))
        torch.save(wm.state_dict(), ckpt)
        print(f"      trained in {time.time()-t0:.0f}s")
    wm.eval()

    print("\n[3/5] Rolling out both branches for every held-out case ...")
    te_fact = [f for f, _ in te_pairs]
    Xe, Ae, Ge, _, _, me = pack(te_fact, builder)
    xe, ae, ge = torch.tensor(Xe), torch.tensor(Ae), torch.tensor(Ge)

    cases, effects, p_true, worked = [], [], [], []
    forks = []            # (row in the held-out tensor, decision index) per case
    with torch.no_grad():
        for i, (fact, cf) in enumerate(te_pairs):
            fork = int(me[i]["fork"])
            t = max(0, min(fork - 1, xe.shape[1] - 2))
            one = slice(i, i + 1)

            r_none = cumulative_risk(
                wm.rollout(xe[one], ae[one], ge[one], t, HORIZON, 0, None))
            r_iso = cumulative_risk(
                wm.rollout(xe[one], ae[one], ge[one], t, HORIZON, ISOLATE, 0))
            effect = r_none[-1] - r_iso[-1]

            did_work = bool(fact.compromised and not cf.compromised)
            effects.append(effect)
            p_true.append(float(me[i]["p_contain"]))
            worked.append(1.0 if did_work else 0.0)

            if len(cases) < N_CASES:
                forks.append((i, t))
                # The stages the sensor actually observed before the decision.
                # WindowSnapshot.stage is already the stage name.
                observed = [w.stage for w in fact.windows[:fork]]
                cases.append({
                    "id": fact.id,
                    "fork": fork,
                    "observed_stages": observed[-12:],
                    "risk_none": [round(v, 4) for v in r_none],
                    "risk_isolate": [round(v, 4) for v in r_iso],
                    "effect": round(effect, 4),
                    "recommend": "isolate" if effect > 0 else "hold",
                    "truth": {
                        "factual_compromised": bool(fact.compromised),
                        "counterfactual_compromised": bool(cf.compromised),
                        "isolation_worked": did_work,
                        "p_contain": round(float(me[i]["p_contain"]), 4),
                    },
                })

    effects = np.array(effects)
    worked = np.array(worked)
    p_true = np.array(p_true)

    # Three numbers, because the one this project has been quoting is not what
    # its name suggests.
    #
    # `says_helps` is exp11's "direction_accuracy": the fraction of cases where
    # the model predicts a positive effect. It never looks at an outcome. This
    # generator makes isolation genuinely helpful in every pair (p_contain runs
    # 0.31 to 0.75, never negative), so the true sign is always positive and a
    # constant "always isolate" predictor scores 1.000 on it. Quoting 0.94 as
    # "accuracy" invites the reading "right about which cases", which is false.
    says_helps = float((effects > 0).mean())
    trivial = 1.0                      # what "always isolate" would score
    vs_outcome = float(((effects > 0) == (worked > 0)).mean())
    r_pc = float(pearsonr(effects, p_true)[0])

    print(f"      {len(cases)} cases written, {len(effects)} scored")
    print(f"      predicts 'isolating helps' in {says_helps:.3f} of cases")
    print(f"      trivial always-isolate baseline scores {trivial:.3f}")
    print(f"      agreement with realised outcome {vs_outcome:.3f} (chance)")
    print(f"      r(effect, p_contain) {r_pc:+.4f}")

    # ---------------------------------------------------------------- step 4
    # SIH26153 asks the prediction engine to report "contributing features".
    # Integrated gradients against a quiet-network reference, so each number
    # reads as "what makes this window different from a normal day".
    #
    # The reference is the mean of *benign* windows, not zeros. An empty graph
    # scores about 0.76 risk on this model - the model finds "nothing is
    # happening" alarming - so attributing against zeros would explain only the
    # sliver above 0.76 and reference a state the network is never in.
    print("\n[4/5] Attributing each case to its contributing features ...")
    benign = [e for e in base if not e.compromised]
    Xb, _, Gb, _, _, _ = pack(benign, builder)
    quiet_x = torch.tensor(Xb).mean(dim=(0, 1), keepdim=True)   # (1,1,N,F)
    quiet_g = torch.tensor(Gb).mean(dim=(0, 1), keepdim=True)   # (1,1,G)
    print(f"      quiet reference from {len(benign)} benign episodes")

    worst_rel, n_loose, total_steps = 0.0, 0, 0
    for case, (i, t) in zip(cases, forks):
        one = slice(i, i + 1)
        # Refine until the residual is small *relative to the gap being
        # explained*. A flat 32 steps left a residual near 1e-2 on every case,
        # which is negligible against a gap of 0.97 and larger than the whole
        # gap on a quiet one - the risk head clamps and saturates, so the path
        # integrand has kinks that coarse quadrature walks straight past.
        steps = IG_STEPS
        while True:
            a = attribute(wm, xe[one], ae[one], ge[one], t, HORIZON, hosts=hosts,
                          action=0, action_at=None, steps=steps,
                          baseline=(quiet_x, quiet_g))
            gap = abs(a.risk - a.baseline_risk)
            if a.completeness_error <= max(1e-3, IG_TOL * gap) or steps >= IG_MAX:
                break
            steps *= 2
        total_steps += steps
        rel = a.completeness_error / max(gap, 1e-9)
        worst_rel = max(worst_rel, rel)
        loose = a.completeness_error > max(1e-3, IG_TOL * gap)
        n_loose += loose
        rank = lambda d, n=None: [                     # noqa: E731
            [k, round(v, 5)] for k, v in
            sorted(d.items(), key=lambda kv: -abs(kv[1]))[:n]]
        case["attribution"] = {
            "risk": round(a.risk, 4),
            "baseline_risk": round(a.baseline_risk, 4),
            "node": rank(a.node_features),
            "global": rank(a.global_features),
            "hosts": rank(a.hosts, N_HOSTS_SHOWN),
            "completeness_error": round(a.completeness_error, 6),
            "steps": steps,
            "converged": not bool(loose),
            "method": a.method,
            "baseline": a.baseline,
        }

    # Completeness is the property that makes IG a defensible stand-in for the
    # SHAP the problem statement names. If it ever drifts, the ranking is not
    # trustworthy and the console must not present it as one - so it is printed
    # here and shown on every card rather than hidden.
    print(f"      {len(cases)} attributions, {total_steps // len(cases)} steps "
          f"on average (cap {IG_MAX})")
    # Two numbers because they can disagree: the tolerance has a 1e-3 absolute
    # floor, so a case with a tiny gap can sit above IG_TOL in relative terms
    # and still be well inside it in absolute ones. That is intended - chasing
    # 2% of a 0.008 gap is chasing quadrature noise, not signal.
    print(f"      worst residual {100 * worst_rel:.2f}% of its gap "
          f"(absolute tolerance floor {1e-3:g})")
    print(f"      {n_loose} case(s) failed the check and are flagged on screen")

    print("\n[5/5] Writing results/demo_trajectories.json ...")
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "horizon": HORIZON,
        "stages": list(STAGES),
        "compromise_stage": STAGES[COMPROMISE_STAGE],
        "headline": {
            "says_helps": round(says_helps, 4),
            "lstm_says_helps": 0.48,
            "trivial_baseline": trivial,
            "agreement_with_outcome": round(vs_outcome, 4),
            "r_effect_vs_p_contain": round(r_pc, 4),
            "n_scored": int(len(effects)),
        },
        # Carried so the console can state the limit next to the capability
        # rather than leaving a judge to discover it.
        "limits": {
            "predicts": "the average effect of isolating - whether it helps",
            "does_not_predict": "which host to isolate first",
            "evidence": "four experiments (exp02, 10, 11, 12) failed to produce a "
                        "per-host effect; an oracle trained on the target itself "
                        "still gave r = +0.011, so the limit is structural",
            "metric_caveat": "The headline is the fraction of cases where the model predicts isolating helps - not accuracy. Isolation genuinely helps in every pair this generator makes, so a constant always-isolate predictor scores 100%. The number is evidence that the action-aware LSTM failed to learn the effect at all, not that our model ranks cases.",
            "attribution": "Attributions explain this model's function, not the "
                           "network. If the model has learned a spurious "
                           "correlate, attribution reports the spurious "
                           "correlate faithfully - true of SHAP too. Read them "
                           "as what the model used, never as what caused the "
                           "attack.",
            "lead_time": "on lead time this model LOSES to a tuned LSTM: 0.26 vs "
                         "3.35 windows of warning. The interventional question is "
                         "the one it answers better, not detection.",
        },
        "cases": cases,
    }
    payload = stamped(payload)
    (RESULTS / "demo_trajectories.json").write_text(json.dumps(payload, indent=1))
    size = (RESULTS / "demo_trajectories.json").stat().st_size / 1024
    print(f"      {size:.0f} KB")

    # Also as a plain script assigning a global, so the console opens straight
    # from the filesystem. fetch() of a sibling JSON is blocked under file://,
    # and requiring a server for a replay would be a needless failure mode.
    console = ROOT.parent / "frontend"
    console.mkdir(exist_ok=True)
    (console / "data.js").write_text(
        "window.KALACHAKRA_DATA = " + json.dumps(payload) + ";\n", encoding="utf-8")
    print("      frontend/data.js written - open frontend/index.html directly")
    print(f"\n  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
