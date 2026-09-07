# KALACHAKRA — Architecture

**SIH 2026 · PS SIH26153 · NTRO**
*AI based Network Attack Forecasting from Network Traffic Data*

---

## 1. What the PS actually asks for

NTRO's text is unusually specific, and it is a research brief:

> "Rather than classifying traffic, a world model learns the transition dynamics `P(S_t+1 | S_t)` … This enables forward simulation: roll out K steps ahead and identify whether the current trajectory converges to an infiltration state, **before the attacker completes the kill chain**."

It also asks for network state as **feature vectors or graphs**, sequence/GNN/latent state models, mapping to **MITRE ATT&CK** stages, and **explainability**.

Most teams will submit an LSTM that emits a benign/malicious label and call it a world model. The distinguishing question is one sentence:

> *"What does your model predict about **state**, as opposed to labels?"*

Everything below exists to answer that.

---

## 2. Classifier vs world model — the actual difference

| | Sequence classifier | World model |
|---|---|---|
| Learns | `P(y \| x_{1..t})` | `P(s_{t+1} \| s_t)` |
| Output | a label for now | a distribution over futures |
| Multi-step rollout | impossible | native |
| **Counterfactual** | impossible | native |
| Answers "how long do I have?" | no | yes |
| Answers "what if I isolate this host?" | **no** | **yes** |

The last two rows are the entire pitch. They are things a classifier is structurally incapable of, which makes them safe differentiators — a competing team cannot bolt them on late.

---

## 3. Two headline claims — and where they stand

> **Updated 30 Aug 2026 after experiment 01.** Claim 1 is **not supported by our own evidence**. See [`RESULTS.md`](RESULTS.md). It is left here because the reasoning is still right and the measurement is still the one to make — but the number currently goes against us.

### Claim 1 — Lead time · **REFUTED so far**
> *"We warn N windows before compromise at a fixed false-alarm rate."*

Every other team will report accuracy. Lead time is what a defender actually buys, and it produces a curve nobody else will have: **lead-time vs false-alarm-rate**, swept over the alerting threshold, against a sequence-classifier baseline on identical data and splits.

We built exactly that. The baseline wins: **LSTM 3.35 windows vs world model 0.26 at FPR ≤ 10%.** Two model bugs and two experiment-design errors were found and fixed on the way to that number; it is the corrected one.

### Claim 2 — Counterfactual intervention · **the surviving differentiator, untested**
Because the model learns dynamics, we can condition on an action:

```
P(s_{t+1} | s_t, a)      a ∈ {none, isolate, block_edge, rate_limit}
```

Roll out under `a = none` and under `a = isolate(h)`, and show the predicted compromise probability diverging.

**A sequence classifier structurally cannot do this.** There is no bolt-on that gives a discriminative model a counterfactual — which is what makes this a safe differentiator, and why it is now the project's centre of gravity rather than lead time.

Status: `rollout(action=..., action_at=...)` is implemented and matched-pair validation data exists (34/40 interventions prevent compromise; prefixes bit-identical before the fork). **The action embedding is untrained** — every training episode uses `action="none"` — so counterfactual rollout currently returns the same trajectory for every action. Training on the counterfactual pairs is the next step and the one that decides whether this project has a claim.

## 4. State representation

Network state at time `t` is a **host-interaction graph** `G_t`:

- **Nodes** = hosts (internal + external), with features: packet/byte rates in and out, distinct peers, distinct ports touched, flag distribution, protocol mix, novelty score.
- **Edges** = observed communication in window `t`, with features: volume, packet count, duration, flag bitmask summary, port class.
- **Global** = window-level aggregates: total flows, source-IP entropy, scan-ness, new-edge fraction.

Graphs, not flat vectors, because the phenomena we forecast are *relational*: lateral movement is an edge appearing between two internal hosts that never talked before. A flat vector destroys exactly that.

---

## 5. Model

```
G_t ──► GNN encoder ──► z_t  (latent state, d=64)
                          │
                          ├──► ATT&CK head:  P(tactic | z_t)
                          │
        z_t, a_t ──► Transition model ──► ẑ_{t+1}
                     (GRU / Transformer over latent)
                          │
                          └──► K-step rollout ──► P(reach compromise within K)
```

### Training objective — JEPA-style, deliberately

We predict in **latent space**, not raw traffic:

```
L = ‖ ẑ_{t+1} − sg(z_{t+1}) ‖²          (predictive, stop-grad target)
  + λ₁ · CE(ATT&CK tactic)               (grounds the latent in semantics)
  + λ₂ · VICReg(variance, covariance)    (prevents latent collapse)
```

**Why not reconstruct raw traffic?** Because packet-level detail is mostly noise for forecasting, and reconstruction spends all model capacity on it. Predicting a learned representation is both cheaper and the current direction in the literature. Latent collapse is the known failure mode, hence the variance/covariance term — and we report the collapse diagnostic, because a judge will ask.

---

## 6. ATT&CK grounding

Tactic states used (a deliberate subset, ordered as a kill chain):

`BENIGN → RECON → INITIAL_ACCESS → EXECUTION → LATERAL_MOVEMENT → COLLECTION → EXFILTRATION`

The ATT&CK head is what makes the latent interpretable — it is not decoration. Rollout probability of reaching `LATERAL_MOVEMENT` or beyond is the alerting signal.

---

## 6b. Explainability — what we ship, and what we are substituting for

The PS names two mechanisms: **SHAP values or attention**. We have neither, and pretending otherwise would not survive one question. The encoder is normalised-adjacency message passing — there is no attention to read off — and SHAP is a dependency plus thousands of forward passes per explanation.

`src/kalachakra/explain/attribution.py` ships **integrated gradients** instead. The argument for it is one property, not a preference: **completeness**, the guarantee that attributions sum to `f(input) − f(baseline)`. That is the same axiom SHAP's efficiency property provides, and unlike SHAP it is cheap enough to *verify on every call* — which we do, in the library, in the shipped payload's tests, and again in the browser before the console will call a ranking trustworthy.

Two design choices are load-bearing:

- **The reference is a quiet network, not zeros.** Measured: an empty graph scores **0.768** compromise risk on the trained model, and zeros with zero adjacency scores 0.980 — the model finds "nothing is happening" alarming. Against zeros we would be explaining only the sliver above 0.77, referenced to a state the network is never in. Against the benign mean (0.027) attributions read as *what makes this different from a normal day*. Every result records which reference produced it, because an attribution whose reference is hidden is not an explanation.
- **The step count adapts per case.** A flat 32 steps left residuals near `1e-2` — negligible against a gap of 0.97, larger than the whole gap on a quiet case, where the sum came out with the wrong sign. Refining until the residual fits the gap gives 40/40 converged at 52 steps on average. See `RESULTS.md`.

The limit is stated wherever the attributions appear: this explains **the model's function, not the network**. A learned spurious correlate is reported faithfully — true of SHAP too.

---

## 7. Evaluation protocol

Stated up front because this is where credibility is won or lost.

- **Temporal splits only.** No random shuffling of a time series. Train on earlier episodes, test on later.
- **Baselines**: (a) per-window GBDT classifier, (b) LSTM sequence classifier, (c) ours. Same features, same splits.
- **Metrics**:
  - Lead time at matched FPR *(headline)*
  - Rollout calibration over horizon K — Brier score per step
  - ATT&CK stage accuracy
  - **Cross-dataset transfer** — train on one corpus, test on another
  - Latent collapse diagnostic (rank / variance of `z`)
- **Counterfactual validation**: on synthetic episodes we control the ground truth, so we can actually branch an episode with and without an intervention and check whether predicted divergence matches observed divergence. This is the honest way to validate a counterfactual, and it is only possible because we generate the data.

---

## 8. Shared core with EKAGRA

Both problem statements consume the same network telemetry. Shared, by design:

- packet/flow record schema
- the streaming feature extractors
- dataset adapters (CIC-IDS2017/2018, UNSW-NB15)
- the traffic generator — extended here to emit **multi-stage attack episodes** with ATT&CK stage labels and branch points

EKAGRA answers *"what is happening now, given I can only observe."*
KALACHAKRA answers *"where is this going, and what if I act."*

Roughly 60–70% of KALACHAKRA's data foundation is EKAGRA's, which is why these two problem statements are the right pair to submit.

---

## 9. What we are deliberately not building

| Not building | Why |
|---|---|
| **An RL defender policy** | The obvious next step, and a research programme. Cannot be evaluated honestly before December. We provide the *simulator*; the policy is future work — and saying so is a strength. |
| Raw-traffic generative reconstruction | Wrong objective, enormous cost. |
| A full SOC / SIEM | Out of scope. |
| Real-time inference at line rate | KALACHAKRA operates on windowed graph snapshots (seconds), not packets. Different latency regime from EKAGRA, and we say so. |

---

## 10. Build order

1. Multi-stage episode generator with ATT&CK stage labels + branch points
2. Graph state builder
3. Baselines (GBDT, LSTM) — *before* the world model, so the comparison is honest
4. GNN encoder + latent transition model
5. `experiments/exp01_leadtime.py` — **the headline curve**
6. Rollout + `experiments/exp02_counterfactual.py` — **the demo moment**
7. Cross-dataset transfer
8. Trajectory console
