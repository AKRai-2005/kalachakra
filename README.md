# KALACHAKRA

**A model you can ask "what if we act?"**

Smart India Hackathon 2026 · Problem Statement **SIH26153** · National
Technical Research Organisation (NTRO) · Theme: Blockchain & Cybersecurity ·
Team **AlgoRhythms**

> *AI based Network Attack Forecasting from Network Traffic Data*

Companion project: **[EKAGRA](https://github.com/AKRai-2005/ekagra)**
(SIH26145) — *what is happening now, given I can only observe?* It supplies
roughly 60–70% of this project's data foundation.

---

## The idea

Rather than classifying traffic, KALACHAKRA learns the **transition dynamics**
of a host-interaction graph — `P(S_t+1 | S_t, action)` — and rolls them forward
K steps with no new observations. That makes possible a question a classifier
cannot even represent:

> **What happens if we isolate this host *now*?**

A discriminative model scores the trajectory it observed. It has nowhere to put
an action that was never taken, and there is no bolt-on that gives it one.

```
G_t -> GNN encoder -> (z_t, action) -> transition model -> K-step rollout
    -> MITRE ATT&CK stage scoring -> P(reach compromise)
```

## Read this before anything else

**Our headline forecasting claim is refuted — by our own experiments, twice.**

| Claim | Status |
|---|---|
| Lead time: a world model warns earlier than a classifier | **REFUTED** on generated data (a tuned LSTM won 3.35 windows to 0.26) and again on real botnet captures |
| Counterfactual: it can tell you whether isolating helps | **HOLDS, NARROWLY** — see the caveat below |
| It can tell you *which* interventions work | **REFUTED after four attempts** (r = −0.105, −0.103, +0.037, +0.011) |

On **CTU-13** — thirteen real botnet captures, 19.9 million flows,
leave-one-scenario-out — the world model catches **5 of 12** real infections at
≤10% false alarms. A plain gradient-boosted tree catches **7 to 9**.

These are on the slides too. They are the reason the surviving claims can be
trusted.

## The correction we made against our own interest

We published **1 of 12**. It was wrong. Here is how we found it, because the
method matters more than the number:

1. **Experiment 16** re-ran experiment 14's configuration on its own folds and
   got **4 of 12**, not 1.
2. **A deterministic control isolated the fault.** Gradient boosting has no
   training randomness here, and in the same run it reproduced experiment 14's
   row *to the digit* — 7 of 12, mean lead 8.08, AUC 0.624. So the data,
   labels, features, folds and scoring were provably identical.
3. **Experiment 17 ruled out noise.** Six seeds, 4 of 12 every time, zero
   variance.
4. **Experiment 18 found the cause: training length** — and the direction is
   the finding.

| Epochs | Caught (of 12) | Window-level AUC |
|---|---|---|
| 6 | 1.5 | 0.461 |
| **12** | **4.0** | 0.306 |
| 18 | 4.0 | 0.330 |
| 25 | 2.0 | 0.529 |
| 40 | 1.0 | **0.690** |

`epochs vs detections r = −0.451` · `epochs vs AUC r = +0.780`

**The two metrics move in opposite directions.** Our earlier tuning pass
selected on **AUC** — so it was steering away from the metric we actually
report, and its grid started at 25 epochs, putting the region that works (12–18)
outside the search entirely. That pass's *conclusion* was right; its *method*
was pointed backwards.

The honest figure is **5 of 12**. The error had made us look **worse** than we
were. We fixed it anyway.

## What survives, and is worth your attention

- **A capability class the competition structurally lacks.** Roll out under
  `action=none` and `action=isolate` from one shared prefix; the gap is the
  model's belief about the intervention. Validated on matched pairs whose
  prefixes are *bit-identical* before the fork, so any divergence is caused by
  the action and not by noise.
- **An evaluation instrument the field does not have.** Lead time at a *matched*
  false-alarm rate; leave-one-scenario-out across thirteen real captures;
  **contained attacks counted as negatives**, so a model that fires on the first
  port scan cannot post a perfect false-alarm rate while having forecast
  nothing.
- **A finding about the problem, not about us.** On CTU-13 all four methods span
  roughly **0.03 to 0.99 AUC** across captures. Cross-network transfer here is
  close to a coin flip per capture — which applies to every team on this corpus.
- **Attribution with a checkable axiom.** Not the SHAP or attention the PS names
  — the encoder is message passing, there is no attention to read off. We ship
  **integrated gradients** because its *completeness* property is verifiable on
  every call, and we verify it: in the library, in the shipped payload's tests,
  and again in the browser.

## Quickstart

```bash
cd backend
pip install -e .

python -m pytest tests/ -q                        # 107 tests
python fetch_ctu13.py --version 1                 # one-time corpus fetch
python experiments/exp14_ctu13_leadtime.py        # real-data evaluation
python experiments/exp16_ctu13_horizon.py         # horizon sweep (~24 min)
python experiments/exp18_ctu13_epochs.py          # the training-length finding
python demo.py                                    # builds the console payload
```

Then open `frontend/index.html` **directly** — no server, no CDN, no build
step. The console is a *replay* of a finished run, and we say so; running a
server would imply something that is not true.

### The corpus is not in this repository

**CTU-13** (Stratosphere Lab, CTU Prague, 2011) is third-party and ~2.6 GB.
`backend/fetch_ctu13.py` downloads it. **Use `--version 1`** — later versions of
the mirror drop the `StartTime`, `SrcAddr` and `DstAddr` columns, without which
the adapter cannot build episodes at all.

## How to read the evidence

Every experiment carries **pre-registered predictions** written into the file
before the first run. The run prints a verdict for each and persists it to
`backend/results/exp*_verdicts.json`. Predictions are never edited afterwards.

**40 predictions. 16 refuted — including the headline.**

| Document | What it is |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Design rationale (note: parts predate experiments 02–18 and are marked where stale) |
| [`RESULTS.md`](RESULTS.md) | Full experiment log, including every refutation and the correction above |
| [`submission/REPORT_KALACHAKRA_SIH26153.pdf`](submission/) | 42-page complete technical, functional and presentation reference |
| [`submission/JUDGE_QA.md`](submission/JUDGE_QA.md) | Anticipated questions with honest answers |

## Known limitations

- **Forecasting loses to a gradient-boosted tree on real data** (5 of 12 vs
  7–9). Even allowed to pick its rollout horizon *knowing the test results* —
  which nothing deployable can do — it reaches 5 against GBDT's 9. The upper
  bound loses, so no honest procedure could have rescued it.
- **It predicts an average intervention effect, not a per-host one.** Four
  attempts — more data, FiLM conditioning, a paired-difference loss, and an
  oracle trained on the target itself — all returned nulls. The limit is
  structural, not sample size.
- **The counterfactual headline is beaten by a trivial baseline.** Direction
  accuracy measures how often the model says isolating helps; a constant
  "always isolate" predictor scores 1.000 by construction. The figure is
  evidence about the LSTM's failure (48%, chance), not about ranking ability.
- **Widening the rollout horizon makes our lead time worse**, not better
  (r = −0.842), while gradient boosting's improves monotonically.
- **The counterfactual can never be validated on real captures** — no real
  capture contains the branch that did not happen.
- **No deployment path.** No inference service, no streaming graph builder, no
  authentication. What exists is a model, an evaluation, and a replay console.

## Licence and attribution

Built for Smart India Hackathon 2026 by Team AlgoRhythms.
CTU-13 is the work of the Stratosphere Laboratory, CTU Prague.
