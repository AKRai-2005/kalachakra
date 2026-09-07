"""Experiment 12 — a deliberately-leaky diagnostic. NOT a reportable result.

Why this exists
---------------
Three attempts have now failed to produce a heterogeneous treatment effect:

  exp02  more data (260 -> 800 pairs)          r: +0.060 -> +0.007
  exp10  FiLM conditioning                     r: +0.032 -> -0.103
  exp11  supervising the paired delta          r: +0.009 -> +0.037

exp11 is the informative one: supervising the delta lifted *direction accuracy*
from 0.810 to 0.940 - a large gain on the average effect - while leaving the
correlation with the true containment probability at essentially zero. So the
model can learn that isolation helps, and learns it better when told, but not
which isolations help.

Two explanations remain and they call for opposite responses:

  A. **Not enough signal.** Each pair yields one Bernoulli draw of a probability
     in [0.30, 0.83]. Estimating a *conditional* effect from single draws is the
     classic CATE difficulty, and 560 training pairs may simply be too few - the
     MSE-optimal answer under that much noise is the mean. Response: more pairs,
     or a variance-reduction estimator.

  B. **Something structural.** The context the model conditions on may not carry
     the information, or the rollout may wash it out. Response: stop; the
     approach is wrong.

The test
--------
Train the same model with the same loss, but on a **noise-free target**: the
true `p_contain`. If the correlation jumps, the machinery works and (A) holds -
the limit is sample noise. If even an oracle target fails, (B) holds.

Why this is not a result
------------------------
`p_contain` is the generator's internal causal parameter, it is unobservable in
any real deployment, and it is exactly what the evaluation correlates against.
Training on it is leakage by construction. This run exists to tell us *which
way to spend the next effort*, and its number must never be quoted as
performance. The verdict below says so explicitly.

Run:  python experiments/exp12_oracle_diagnostic.py
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
from scipy.stats import pearsonr  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from exp11_paired_effect_loss import N_PAIRS, evaluate, pack, train

from kalachakra.data.episodes import EpisodeConfig, EpisodeGenerator  # noqa: E402
from kalachakra.state.graph import GraphBuilder, hosts_from_config  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    t_all = time.time()
    print("=" * 78)
    print("KALACHAKRA  ·  Experiment 12  ·  ORACLE DIAGNOSTIC (leaky by design)")
    print("=" * 78)
    print("  This trains on the evaluation target. Its number is NOT performance.")

    cfg = EpisodeConfig(n_windows=40, n_internal=24)
    gen = EpisodeGenerator(cfg, seed=4242)
    builder = GraphBuilder(hosts_from_config(cfg.internal_prefix, cfg.n_internal))

    print("\n[1/3] Rebuilding exp11's data ...")
    base = gen.generate(n_attack=200, n_benign=140, n_contained=120)
    pairs = gen.generate_counterfactual_pairs(N_PAIRS, action="isolate")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    cut = int(0.7 * len(pairs))
    tr_pairs = [pairs[i] for i in order[:cut]]
    te_pairs = [pairs[i] for i in order[cut:]]

    train_eps = base + [e for pr in tr_pairs for e in pr]
    Xtr, Atr, Gtr, Str, ACTtr, _ = pack(train_eps, builder)
    xs, adjs, gs = torch.tensor(Xtr), torch.tensor(Atr), torch.tensor(Gtr)
    stg, acts = torch.tensor(Str), torch.tensor(ACTtr)

    tr_fact = [f for f, _ in tr_pairs]
    Xp, Ap, Gp, _, _, mp = pack(tr_fact, builder)
    xf, af, gf = torch.tensor(Xp), torch.tensor(Ap), torch.tensor(Gp)
    forks_tr = [m["fork"] for m in mp]

    # THE LEAK, on purpose: the noise-free causal parameter as the target.
    oracle = torch.tensor([c.p_contain for _, c in tr_pairs], dtype=torch.float32)
    noisy = torch.tensor(
        [1.0 if (f.compromised and not c.compromised) else 0.0
         for f, c in tr_pairs], dtype=torch.float32)
    print(f"      oracle target: sd {float(oracle.std()):.4f} "
          f"(range {float(oracle.min()):.2f}-{float(oracle.max()):.2f})")
    print(f"      noisy  target: sd {float(noisy.std()):.4f} "
          f"(single Bernoulli draw per pair)")
    print(f"      correlation between them: "
          f"{pearsonr(oracle.numpy(), noisy.numpy())[0]:+.3f}")

    te_fact = [f for f, _ in te_pairs]
    Xe, Ae, Ge, _, _, me = pack(te_fact, builder)
    xe, ae, ge = torch.tensor(Xe), torch.tensor(Ae), torch.tensor(Ge)
    forks_te = [m["fork"] for m in me]
    p_true = np.array([m["p_contain"] for m in me], dtype=np.float32)
    worked = np.array([1.0 if (f.compromised and not c.compromised) else 0.0
                       for f, c in te_pairs])

    print("\n[2/3] Training on the ORACLE target ...")
    t0 = time.time()
    wm = train(xs, adjs, gs, stg, acts,
               pair_data=(xf, af, gf, forks_tr, oracle, len(tr_pairs)))
    res, eff = evaluate(wm, xe, ae, ge, forks_te, p_true, worked)
    res["train_s"] = time.time() - t0
    print(f"      done in {res['train_s']:.0f}s")

    print("\n[3/3] Diagnosis")
    prev = {}
    p11 = RESULTS / "exp11_verdicts.json"
    if p11.exists():
        prev = json.loads(p11.read_text())["results"]
    print(f"      {'':<20s} {'sd eff':>8s} {'direction':>10s} {'r(p_contain)':>13s}")
    for name, r in (("control (exp11)", prev.get("control")),
                    ("noisy target (exp11)", prev.get("effect_supervised")),
                    ("ORACLE target", res)):
        if r:
            print(f"      {name:<20s} {r['sd_effect']:>8.4f} "
                  f"{r['direction_accuracy']:>10.3f} "
                  f"{r['pearson_p_contain']:>+13.3f}")

    oracle_works = res["pearson_p_contain"] > 0.30
    print("\n" + "=" * 78)
    print("VERDICT  (diagnostic only - this number is not performance)")
    print("=" * 78)
    if oracle_works:
        print("  The machinery CAN represent a heterogeneous effect. Given a")
        print("  noise-free target it recovers one, so the limit in exp11 is")
        print("  sample noise: one Bernoulli draw per pair is too little signal")
        print("  at this scale.")
        print("  -> Next: far more pairs, or a variance-reduced CATE estimator.")
        print("     NOT another architecture change.")
    else:
        print("  Even a noise-free target does not produce a heterogeneous")
        print("  effect. The limit is structural, not statistical: the context")
        print("  the rollout conditions on does not carry the information, or")
        print("  the rollout washes it out.")
        print("  -> Next: stop pursuing heterogeneity through this rollout, and")
        print("     say plainly that the model predicts an average effect only.")

    (RESULTS / "exp12_verdicts.json").write_text(json.dumps({
        "IS_A_DIAGNOSTIC": True,
        "WARNING": "trained on the evaluation target; not a performance claim",
        "oracle_target_sd": float(oracle.std()),
        "noisy_target_sd": float(noisy.std()),
        "oracle_result": res,
        "oracle_recovers_heterogeneity": bool(oracle_works),
    }, indent=2, default=float))
    print("\n  wrote results/exp12_verdicts.json")
    print(f"  total runtime {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
