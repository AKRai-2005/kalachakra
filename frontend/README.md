# KALACHAKRA — frontend

The intervention console for SIH26153. One file, `index.html`, no build step,
no framework, no CDN.

## Run it

```bash
cd kalachakra/backend && python demo.py
```

**There is no server.** This console is a replay of a finished model run: the
trajectories are computed once and written to `data.js`, which `index.html`
loads with a plain script tag. Open the file directly and it works.

That is the honest shape of the thing — nothing here is live, and a server
would imply it was. If the first run has no `data.js`, `demo.py` computes it
(about seven minutes of training).

## The data contract

`data.js` assigns one global, `window.KALACHAKRA_DATA`:

```js
{
  generated: "2026-09-05 08:41:41",
  horizon: 6,
  stages: ["BENIGN","RECON","INITIAL_ACCESS","EXECUTION",
           "LATERAL_MOVEMENT","COLLECTION","EXFILTRATION"],
  compromise_stage: "LATERAL_MOVEMENT",
  headline: { says_helps, lstm_says_helps, trivial_baseline,
              agreement_with_outcome, r_effect_vs_p_contain, n_scored },
  limits:   { predicts, does_not_predict, evidence, metric_caveat,
              attribution, lead_time },
  cases: [{
    id, fork, observed_stages: [...],
    risk_none:    [6 floats],   // P(compromised by step j), do nothing
    risk_isolate: [6 floats],   // same, having isolated at the fork
    effect, recommend,
    truth: { factual_compromised, counterfactual_compromised,
             isolation_worked, p_contain },
    attribution: {
      risk, baseline_risk,        // what is being explained: risk − baseline
      node:   [[name, value]],    // all 15, ranked by |value|
      global: [[name, value]],    // all 8
      hosts:  [[addr, value]],    // top 6 only — see below
      completeness_error, steps, converged, method, baseline
    }
  }]
}
```

Regenerate with `python backend/experiments/exp13_demo_trajectories.py`.

### The attribution contract, and the one rule you must not break

`node` and `global` are shipped **complete**, never truncated, because the
console re-derives the completeness residual in the browser: it sums every
contribution and checks it against `risk − baseline_risk`. Ship a top-10 slice
and that sum stops reaching the gap, and all forty cards flip to "treat this
ranking as indicative only". `hosts` *is* truncated to 6 and that is safe —
host contributions are the same node attributions re-aggregated, not a third
additive term, so they are excluded from the check.

Why the residual is recomputed rather than read from `completeness_error`: the
number the generator printed is the generator marking its own homework. This is
the property that makes integrated gradients a defensible stand-in for the SHAP
the problem statement names, so the console checks it rather than trusting it.
`backend/tests/test_demo_payload.py` runs the same arithmetic in Python and
fails if a shipped payload violates it.

## The one thing this console exists to show

Two risk curves diverging from a **bit-identical prefix**. Both branches are the
same episode replayed, differing only in whether the host was isolated, so the
gap between the curves is caused by the intervention and nothing else. A
detector scores the trajectory it observed; only a model with an action input
can be asked what the other branch would have looked like.

If you redesign anything, keep that comparison as the visual centre.

## Do not build a ranked list

This is the constraint that matters most, and it is tempting to break.

The obvious "impressive" feature is a leaderboard of hosts to isolate. **The
model cannot support one.** Four experiments established that it predicts an
*average* treatment effect, not a per-host one — correlation with the true
containment probability is `+0.036`, and an oracle trained directly on that
target still managed only `+0.011`. Sorting 40 cases by predicted effect would
produce a confident-looking ranking that is close to noise.

Related: the headline `94%` is **not accuracy**. It is the share of cases where
the model says isolating helps, and it never compares against an outcome.
Isolation genuinely helps in every pair the generator makes, so a constant
"always isolate" predictor scores **100%**. That trivial baseline is displayed
next to our number on purpose. Do not remove it, and do not relabel `says
isolating helps` as accuracy.

The permanent "What this cannot do" panel carries these limits plus the fact
that the model **loses to an LSTM on lead time** (0.26 vs 3.35 windows). It
should stay on screen, not behind a disclosure toggle.

## Where the design has room

- **The case list is 40 flat rows.** Grouping by recommendation, or by whether
  the prediction matched the outcome, would help. 39 of 40 say "isolate", which
  is itself informative and currently easy to miss. Grouping by *top driver*
  would also be legitimate — that ranks features within a case, which is not the
  ranked list the section below forbids.
- **Each row shows only its single strongest driver.** That feature carries a
  median 30% of the case's attribution mass, never all of it, which is why the
  share is printed beside the name. Do not drop it: a bare feature name on a row
  reads as a cause, and 30% is not a cause.
- **The stage ladder is a row of chips.** A proper ATT&CK timeline would be
  better, and the PS explicitly speaks the kill-chain vocabulary.
- **No comparison view** — you cannot see two cases side by side.
- **The chart is hand-drawn SVG** at a fixed 560×230 viewBox. Deliberate: a
  charting library is 300KB for two polylines. If you replace it, vendor the
  library locally rather than fetching from a CDN.
- **The attribution panel is three stacked bar groups.** It works, but per-host
  and per-feature contributions are currently separate lists when they are two
  projections of one matrix — a small heatmap would say more in less space.
  Whatever replaces it must keep the shared scale: node, window and host bars
  are deliberately drawn against one maximum because integrated gradients puts
  them on one additive scale, and rescaling each group independently would make
  a negligible contribution look decisive.

## Accessibility, already done — please keep it

Case rows are real `<button>` elements, so keyboard and screen-reader support
comes for free. The chart carries an `aria-label` stating both endpoint values,
because a line drawing is invisible to a screen reader. Touch targets are 52px
under `pointer: coarse`. The curve-draw animation is disabled under
`prefers-reduced-motion`. There is a skip link.
