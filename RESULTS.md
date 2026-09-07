# KALACHAKRA — results log

## Experiments 16–18 — the horizon sweep that found a bug in our own headline

This started as the last loose end from experiment 15, which had swept epochs,
batch size and loss weights but deliberately left the **rollout horizon**
untouched, and said so. Sweeping it turned up something else: **our published
CTU-13 number was wrong, and the reason it was wrong is that we selected it with
a metric that points the wrong way.**

The corrected result is worse for the story and better for the science. It is
below in full, along with how we caught it.

### Why the horizon needed its own experiment

Changing the horizon changes the *question*, not just the model. The label is

    y[max(0, c − H):c] = 1        "will compromise happen within H windows?"

so at H=2 the model is asked a rare, hard question and at H=18 a common, easy
one. **AUC is therefore not comparable across horizons** — it is measured
against a different target in every arm, and a rising AUC curve would look like
progress while measuring nothing.

Lead time is different. `lead_at_fpr` reads only the compromise window and the
score series; it never sees the label. **Detection count, mean lead and
false-alarm rate are directly comparable across horizons.** Experiment 16
therefore reports those and publishes no AUC at all.

### The control that caught the bug

Experiment 16 carries a deterministic control. Gradient boosting has no training
randomness here, so at H=6 it must reproduce experiment 14's row exactly. It
did — 7 of 12 caught, mean lead 8.08, AUC 0.624, to the digit.

The world model, on the same folds, in the same run, with what the script says
is the same configuration, caught **4 of 12 — not the published 1 of 12**.

With the deterministic half of the pipeline matching perfectly, the episode
construction, labels, features, fold set, pooling and operating-point rule were
all provably identical. Something in the world-model path alone was different.

### Ruling out the obvious suspect (experiment 17)

The natural explanation is training noise. It is not:

| seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| world model caught | 4/12 | 4/12 | 4/12 | 4/12 | 4/12 | 4/12 |
| mean lead | 8.92 | 8.92 | 8.92 | 8.92 | 8.92 | 8.92 |
| GBDT caught | 7/12 | 7/12 | 7/12 | 7/12 | 7/12 | 7/12 |

Six seeds, zero variance in the detection count. **S1 and S2 both fail** — the
seed cannot explain a gap of three, and experiment 14's 1 of 12 lies outside the
observed range entirely. AUC did move (0.298–0.381), and every one of those is
far below experiment 14's published 0.663, which was the clue.

### The actual cause (experiment 18)

Experiment 14's script takes `--epochs`, defaults to 12, and **never recorded
which value produced the published row**. Sweeping it:

| epochs | caught | mean lead | AUC |
|---|---|---|---|
| 6 | 1.5 | 3.75 | 0.461 |
| **12** | **4.0** | **8.92** | 0.306 |
| 18 | 4.0 | 8.92 | 0.330 |
| 25 | 2.0 | 4.46 | 0.529 |
| 40 | 1.0 | 2.25 | **0.690** |

(means over two seeds; gradient boosting is a flat 7 of 12 in every row)

    epochs vs detections   r = −0.451
    epochs vs AUC          r = +0.780

**The two metrics move in opposite directions.** Training longer makes the AUC
steadily better and the actual detection count steadily worse. Experiment 14's
published row is reproduced at 40 epochs — AUC 0.663, 1 of 12 — so it was
measured at a long training length while the script's default says 12, and
nothing in the payload recorded the difference.

### Why this matters more than the horizon did

**Experiment 15 selected on AUC.** AUC rewards longer training. Longer training
destroys detection. So experiment 15's nested selection chose `longer` (40
epochs) on 6 of 12 folds and drove the detection count *down* — and then
correctly reported that tuning had not helped.

Experiment 15's conclusion was right. **Its method was pointed backwards**, and
it could not have discovered that, because its grid started at 25 epochs and
only went up. The region that actually works — 12 to 18 epochs — was outside the
search entirely. That is what P5 was pre-registered to test, and it holds.

Experiment 16's nested selection used **mean lead** instead, and chose horizons
on inner validation captures the test fold is not part of. It is the only
honestly-selected number we have:

| | caught | mean lead | FPR |
|---|---|---|---|
| world model, nested selection on lead | **5 / 12** | 9.00 | 0.069 |
| world model, oracle horizon (upper bound, H=12) | 5 / 12 | 7.17 | 0.088 |
| world model, as published in exp 14 | 1 / 12 | 2.50 | 0.020 |
| GBDT, H=6 | 7 / 12 | 8.08 | 0.088 |
| GBDT, oracle horizon (H=18) | **9 / 12** | 10.75 | 0.098 |

### The horizon itself did almost nothing — for us

| H | WM caught | WM lead | GBDT caught | GBDT lead |
|---|---|---|---|---|
| 2 | 4/12 | 8.92 | 5/12 | 5.58 |
| 4 | 4/12 | 8.92 | 7/12 | 7.25 |
| 6 | 4/12 | 8.92 | 7/12 | 8.08 |
| 9 | 3/12 | 6.42 | 9/12 | 10.58 |
| 12 | 5/12 | 7.17 | 9/12 | 10.67 |
| 18 | 3/12 | 6.42 | 9/12 | 10.75 |

**H2 fails**: the world model's lead does not rise with the horizon —
r = −0.842, it falls. **Gradient boosting's does**, monotonically, 5 of 12 to 9
of 12. Widening the forecasting window is a change to the task that helps
everyone, and it helps the world model least. That is the opposite of what a
model whose entire premise is multi-step rollout should show.

**H3 holds, and it is the one that closes the question.** Even allowed to pick
its horizon *knowing the test results* — which nothing deployable can do — the
world model reaches 5 of 12 against gradient boosting's 9. The upper bound
loses, so no honest procedure could have rescued it.

**H4 and H5 fail** for the same reason H1 did: they were written against the
1-of-12 baseline that turned out to be an artefact.

### What we are correcting, and what stands

**Corrected.** The world model catches **5 of 12** real infections under honest
selection, not 1 of 12. The old number was a low draw from an unstable procedure
steered by the wrong metric. It made us look worse than we are, which is not a
reason to leave it uncorrected — a number that flatters us would be a fault, and
so is this one.

**Strengthened.** *The world model loses to a gradient-boosted tree on real
botnet traffic* now holds across six horizons × five training lengths × up to
six seeds — every single configuration tried. Gradient boosting never once
dropped below 7 of 12; the world model never once reached it.

**Fixed.** `exp14_verdicts.json` now records `epochs` and `quick`. A published
row that its own script cannot regenerate is not reproducible whatever its
value, and that was a real fault in this repository.

**Still open.** Nothing here tests the counterfactual — no real capture contains
the branch that did not happen — so the intervention claim remains validated on
constructed pairs only.

## Experiment 15 — we tuned it properly, and it did not help

Experiment 14 ran the world model on real botnet captures with experiment 01's
hyperparameters, which were chosen on *generated* data. Reporting a loss from an
untuned model would be as unfair as reporting a win from an overtuned one, so
this gives the model a genuine attempt — and answers the obvious challenge,
*"but did you actually tune it?"*, with evidence rather than assertion.

### The trap, and the protocol built to avoid it

The obvious approach is to try configurations and keep whichever scores best
under leave-one-scenario-out. **That is tuning on the test set.** With twelve
attack episodes, trying four configurations and keeping the luckiest would very
likely "improve" 1/12 to 3/12 out of nothing but search.

Selection is therefore **nested**. For each outer fold the test scenario is held
out entirely; a configuration is chosen by AUC on a *different* inner validation
scenario (rotated, so it is never the same capture twice); the model is then
retrained on all twelve non-test scenarios with that choice and scored once.
**No number below has seen its own test fold.**

### Result

| | AUC | sd | caught | mean lead | FPR |
|---|---|---|---|---|---|
| world model, **tuned** | 0.677 | 0.347 | **1 / 12** | 2.25 | 0.088 |
| world model, untuned (exp 14) | 0.663 | 0.330 | **1 / 12** | 2.50 | 0.020 |
| GBDT, untuned (exp 14) | 0.624 | 0.322 | **7 / 12** | 8.08 | 0.088 |

**T1 fails: tuning changed nothing that matters.** AUC moved +0.014 against a
fold-to-fold standard deviation of 0.35 — noise. The detection count is
identical. The tuned model actually operates at a *higher* false-alarm rate
(0.088 against 0.020) for the same single catch, and its mean lead is slightly
lower.

**So experiment 14's result was not a tuning artefact.** The world model does
not work on real botnet traffic at a usable operating point, and it is not
because nobody tried.

### The two predictions that held are the interesting ones

**T3: eleven of twelve folds chose a non-default configuration.** The
hyperparameters selected on generated data were genuinely wrong for real data —
and correcting them still did not help. That is a sharper result than if the
defaults had simply been optimal: the model was mis-tuned *and* mis-tuning was
not the problem.

**T4: no configuration wins a majority.**

    longer        6 / 12 folds
    small_batch   3 / 12
    stage_heavy   2 / 12
    baseline      1 / 12

There is no single "tuned model" to ship. Twelve attack episodes across thirteen
very different captures do not support a confident choice, which is the same
message as the 0.03–0.99 per-fold AUC spread in experiment 14: **on this corpus
the variance between captures dwarfs the difference between configurations.**

### What was deliberately not tuned

The **rollout horizon**. `HORIZON` and `ROLLOUT_DEPTH` are coupled module
constants that must match, so varying them is a code change rather than a
config, and sweeping them inside a selection loop would silently compare models
trained on different objectives. It is flagged as unexplored rather than
quietly skipped, and it is the one remaining avenue that could plausibly change
this result.

    python experiments/exp15_ctu13_tuning.py

---

## Experiment 14 — lead time on REAL botnet captures (CTU-13)

**This is the first KALACHAKRA number that is not measured on our own
generator.** Everything before it ran on `episodes.py`: we invent the dynamics,
then show a model of the dynamics learns them. Experiment 01 refuted the
lead-time claim there — a tuned LSTM won 3.35 windows of warning to the world
model's 0.26 — but that refutation had never been checked against an intrusion
nobody wrote for us, so it could in principle have been an artefact.

CTU-13 is thirteen captures of real botnet traffic (Stratosphere Lab, CTU
Prague, 2011), 19.9M flows, with real background and normal traffic mixed in.
`data/ctu13.py` turns each scenario into the **same `Episode` objects** the
generator produces, so `GraphBuilder` and every model run unchanged — and
`train_world_model` was extracted from experiment 01 so both use provably the
same objective. If the adapter built its own features or its own loss, a
synthetic-versus-real difference would be unattributable, which is the entire
question.

**Protocol: leave-one-scenario-out.** Every test episode comes from a network
and a malware family the model has never seen.

> **Superseded in part — see experiments 16–18 above.** The world-model row
> below was measured at a training length this script no longer defaults to, and
> the payload did not record which. It is left here unaltered as the record of
> what that run produced; the honestly-selected number is **5 of 12**, not 1 of
> 12. Every other row, including gradient boosting's, reproduces exactly.

| method | AUC | sd | F1 | caught | lead when caught | mean lead | FPR |
|---|---|---|---|---|---|---|---|
| WorldModel | 0.663 | 0.330 | 0.101 | **1 / 12** | 30.00 | 2.50 | 0.020 |
| LSTM | 0.647 | 0.305 | 0.015 | 4 / 12 | 26.00 | **8.67** | 0.059 |
| **GBDT** | 0.624 | 0.322 | 0.132 | **7 / 12** | 13.86 | 8.08 | 0.088 |
| LogReg | **0.703** | 0.341 | **0.203** | 1 / 12 | 19.00 | 1.58 | 0.078 |

### The headline: the refutation replicates

**E2 holds.** The world model does not beat the LSTM on real data either —
2.50 against 8.67 windows. Experiment 01's result was not an artefact of our own
dynamics. **The world model catches 1 of 12 real infections; gradient boosting
catches 7; logistic regression still wins AUC and F1.**

### E3 failed, and the reasoning behind it was wrong

We predicted real lead times would be **shorter** than synthetic, on the
argument that real botnet onsets are abrupt where our generator ramps. They are
**longer**: 8.67 against 3.35. Real infected hosts are evidently doing
recognisable things well before the first flow CTU-13 labels as botnet, so there
is *more* warning available in real traffic than our generator offers, not less.
That is worth more than the prediction would have been if it had held — it says
the generator is pessimistic about the very quantity it was built to study.

### Two things the table would otherwise hide

**The lead ceiling is 30 by construction.** Episodes are 40 windows with
compromise at window 30, so the world model's `lead when caught = 30.00` means
it alarmed on the **very first window** of the single episode it caught. That is
not forecasting, and reporting mean lead alone would have disguised it. Both
columns are printed for this reason: mean lead averages over all attack
episodes, so catching one at maximum lead scores like catching half at moderate
lead; lead-when-caught flatters a method that catches almost nothing.

**Fold variance dominates.** Every method spans roughly 0.03 to 0.99 AUC across
the folds. With sd ≈ 0.33 on twelve folds the four mean AUCs are not
distinguishable from one another. **Cross-network transfer is close to a coin
flip per capture**, which is the honest summary of all four methods, ours
included.

### What CTU-13 cannot test, stated before the experiment rather than after

- **The counterfactual.** No real capture contains the branch that did not
  happen. The intervention claim stays validated on constructed pairs, and that
  is a property of reality rather than a gap in our evidence.
- **Intermediate ATT&CK stages.** CTU-13 labels flows botnet / normal /
  background, with no kill-chain ladder, so the stage head degenerates to
  compromised-or-not here.

### Three construction problems, and how each was resolved

**Pseudo-replication.** A scenario's infected hosts all wake within minutes —
in scenario 9, ten hosts across fourteen windows. One episode per infected host
would have given 130 attack episodes with almost entirely shared
pre-compromise context. **CTU-13 is thirteen independent infection events and
the experiment says thirteen.**

**Four benign episodes.** Infections start early (onsets at windows 0, 1, 8,
9 …), so pre-infection traffic barely exists. A false-alarm rate over four
episodes quantises to {0, .25, .5, .75, 1}, and lead time is measured *at a
matched false-alarm rate* — four is not a small sample, it is an unusable one.
Benign episodes are now drawn from hosts never infected in that scenario, giving
105.

**A 500× confound, the worst of the three.** The monitored set is ten quiet
infected hosts plus the fourteen busiest machines on the network. Matching the
clean set to the *infected* hosts' activity produced attack episodes of 167,000
flows against benign episodes of 300 — any model could have scored well by
learning "busy means attack". Nearest-activity matching per host failed too, and
instructively: the busiest hosts are *unique*, so once they are in the monitored
set there is nothing comparable left. Both sets are now **interleaved** down the
activity ranking, drawn from one distribution by construction: 273,763 against
260,037 flows, a ratio of 1.05.

### Reproduce

    python fetch_ctu13.py --version 1        # v1; later versions drop the IPs
    python experiments/exp14_ctu13_leadtime.py

Note on the mirror: `mcfp.felk.cvut.cz` is blocked at IP level by some ISPs (it
serves live malware samples). The Kaggle mirror's *latest* version is cleaned in
a way that **drops `StartTime`, `SrcAddr` and `DstAddr`** — precisely the columns
a windowed host graph is built from — so the tidiest version of the data is the
one version that cannot be used here. Licence: the mirror is CC BY-NC-SA 4.0
where original CTU-13 is CC-BY.

---

Every number here is reproducible with the scripts in `experiments/`.
Predictions were written into the experiment file **before** the first run and
have not been edited since.

---

## Attribution — the "contributing features" the PS asks for

SIH26153 asks for "SHAP values or attention mechanisms identifying which
traffic attributes drive predictions". We have **neither**. The encoder is
normalised-adjacency message passing, so there is no attention to read off, and
SHAP would be a dependency plus thousands of forward passes per explanation.

What ships instead is **integrated gradients** (Sundararajan et al., 2017),
and the reason that is a substitute rather than a hand-wave is
**completeness**: attributions provably sum to `f(input) − f(baseline)`, which
is the same axiom SHAP's efficiency property provides. Unlike SHAP, we can
*check* it — and do, on every call, in three places.

### The baseline is not zeros, and that is a finding

| reference state | compromise risk |
|---|---|
| zeros, zero adjacency | **0.980** |
| zeros, real adjacency (empty graph) | **0.768** |
| mean of 194 benign windows (what we use) | **0.027** |

**The model finds "nothing is happening" alarming.** That is a real calibration
weakness and it is worth stating on its own. It also settles the baseline
question: integrated gradients explains `risk − baseline`, so against zeros we
would be explaining only the sliver above 0.77, referenced to a state the
network is never in. Attributions are taken against a quiet-network reference
instead, so they read as *what makes this different from a normal day*. Every
result records which reference produced it.

### A flat step count was not good enough — and only completeness revealed it

The first run used 32 integration steps for every case. Residuals came out near
`1e-2` regardless of the case, which is negligible against a gap of 0.97 and
**larger than the entire gap** on a quiet one:

| case | gap explained | Σ contributions | residual |
|---|---|---|---|
| cf0422 | +0.9712 | +0.9655 | 5.7e-03 |
| cf0314 | **+0.0077** | **−0.1667** | **1.7e-01** |

Twelve of forty cases failed a 2%-of-gap tolerance, and on cf0314 the sum had
the wrong sign. The cause is the risk head: `1 − Π(1 − p)` over a six-step
rollout, with a `clamp(0, 1)`, has kinks and saturated regions that coarse
quadrature walks straight past. Refining the step count per case until the
residual fits the gap fixed it — **40/40 now converge, at 52 steps on average**
(cap 2048), worst residual 2.64% of its own gap.

The point worth keeping: a gradient×input attribution would have produced a
confident ranked list for cf0314 too, and nothing would have told us it was
noise. This is the whole argument for choosing a method with a checkable axiom.

### Where it is checked

- `tests/test_attribution.py` — completeness on the method itself
- `tests/test_demo_payload.py` — completeness on the forty shipped cases
- the console — recomputes the residual **in the browser** and prints
  "indicative only" instead of a ranking if it fails

### What this does not do

It explains **the model's function, not the network**. If the model has learned
a spurious correlate, attribution reports the spurious correlate faithfully —
true of SHAP too. The console says so on screen rather than letting a ranked
feature list imply causation.

---

## The logistic-regression benchmark the problem statement asks for

SIH26153 requires "F1 scores comparing against logistic regression baseline".
We had neither: no logistic-regression baseline, and no F1 anywhere - experiment
01 reported AUC and lead time only. Both are now in `exp01_leadtime.py`.

The threshold is swept on **train** and applied unchanged to test. F1 needs a
cut and AUC does not, so the choice has to be made somewhere; making it on test
would flatter every model equally and mean nothing.

| method | AUC | F1 train | F1 test |
|---|---|---|---|
| **LogisticRegression** *(the floor)* | 0.938 | 0.611 | **0.606** |
| GBDT | 0.921 | 0.971 | 0.516 |
| LSTM | **0.968** | 0.712 | **0.712** |
| **WorldModel** | 0.784 | 0.421 | **0.412** |

### The world model loses to logistic regression

On window-level detection it scores **0.412 F1 against the floor's 0.606**, and
0.784 AUC against 0.938. This is the second benchmark it loses - it already lost
lead time to a tuned LSTM (0.26 vs 3.35 windows of warning).

The defence, such as it is: the world model is a forward simulator scored by
rollout risk over a K-step horizon, not a window classifier, and it is being
asked a question it was not built for. That is a real distinction and it is also
not a good enough answer on its own, because **the problem statement asks for
exactly this benchmark**. The honest position is that on detection this
architecture is worse than a linear model, and the case for it rests entirely on
the interventional question in experiments 02 and 11-13 - which no baseline here
can answer at all.

Also worth naming: **GBDT overfits hard** - 0.971 train against 0.516 test. Its
AUC looks respectable and its F1 does not survive the split.

### Not a pre-registered prediction

Q1-Q4 were registered before the original run and have not been touched. This
benchmark was added afterwards to meet a stated requirement, so it is reported
as a measurement rather than dressed up as a prediction we made in advance.

## A metric correction — read before any number below

Every table in this file has a column labelled **"direction acc"**. That label is
wrong, and it was wrong from experiment 02 onwards. The quantity computed is:

```python
"direction_accuracy": float((eff > 0).mean())
```

which is **the fraction of cases where the model predicts a positive effect**.
It never compares against an outcome, so it is not accuracy.

Why that matters here specifically: the generator makes isolation genuinely
helpful in **every** pair — the true containment probability runs 0.306 to
0.749 and is never negative — so the true sign is always positive. Therefore:

| predictor | scores |
|---|---|
| a constant **"always isolate"** | **1.000** |
| our world model (exp11) | 0.940 |
| action-aware LSTM | 0.480 |
| against the **realised binary outcome** | 0.490 — chance, since the outcome is one noisy draw |

**So 0.940 is below the trivial baseline.** Read every "direction acc" column
below as *"how often the model says isolating helps"*, and read the comparison
as evidence about the **baseline's failure** — an action-aware LSTM given the
action as an input never learned that the intervention does anything — rather
than as a ranking ability we do not have.

The numbers themselves are unchanged and reproducible; only the label was
wrong.

**The pre-registered prediction text still says "direction accuracy"** in
experiments 02, 10 and 11 (R2, Z3, AA3). That is deliberate. Predictions are
written before a run and are not edited afterwards - rewriting them now to look
better informed would be precisely the dishonesty this discipline exists to
prevent. Read those predictions with this correction in hand. `experiments/exp13_demo_trajectories.py` computes and labels all three
quantities, and the KALACHAKRA console displays the trivial baseline next to
ours.

## Experiments 11 and 12 — the constant effect is structural. Read this first.

Experiment 10 refuted the architectural explanation for the constant treatment
effect. Two experiments closed the question.

### exp11 — supervise the paired difference

Each episode carries one action, so the counterfactual delta had no gradient
path. exp11 adds one: roll out from the shared prefix under both actions and
regress the predicted delta onto the **realised binary outcome difference**.

Not onto `p_contain` — that is the generator's internal parameter and the
evaluation target; training on it would be leakage. The model sees one noisy
Bernoulli draw per pair, which is all a real randomised-intervention programme
would ever give.

| | says helps¹ | r(effect, p_contain) | sd(effect) |
|---|---|---|---|
| control (no effect loss) | 0.810 | +0.009 | 0.286 |
| **paired-difference loss** | **0.940** | +0.037 | 0.302 |

| Prediction | Verdict |
|---|---|
| AA1 — r > 0.30 *(the claim)* | **FAILS** (+0.037) |
| AA2 — beats the control | HOLDS (+0.037 vs +0.009) |
| AA3 — direction accuracy held ≥ 0.75 | HOLDS (0.940) |
| AA4 — effect spread increased | HOLDS (0.302 vs 0.286) |

**Worth keeping**: the model went from saying *helps* on 0.810 of pairs to **0.940** — more consistent about the sign, though still under the 1.000 a constant predictor would score. A gain on
the capability that already worked. **Did not deliver**: heterogeneity.

### exp12 — the oracle diagnostic

Two explanations remained, calling for opposite responses: too little signal
(one Bernoulli draw per pair, 560 pairs) or something structural. So: train the
same model with the same loss on a **noise-free** target — `p_contain` itself.

This is leakage by construction. It is a diagnostic, its number is never
performance, and the JSON says so.

| | says helps¹ | r(effect, p_contain) |
|---|---|---|
| control | 0.810 | +0.009 |
| noisy target | 0.940 | +0.037 |
| **ORACLE target** | 0.880 | **+0.011** |

**Handed a perfect target, it still cannot represent a conditional effect.**

### The conclusion, after four attempts

| attempt | mechanism | r(effect, p_contain) |
|---|---|---|
| exp02 | 3× more data (260 → 800 pairs) | +0.060 → −0.105 |
| exp10 | FiLM conditioning | +0.032 → −0.103 |
| exp11 | supervising the paired delta | +0.009 → +0.037 |
| exp12 | oracle target *(diagnostic)* | +0.011 |

Every one of these is indistinguishable from zero, and the column wanders
across zero rather than converging on it — exp02's own value moved from +0.007
to −0.105 under an unrelated code change. Four mechanisms, four nulls.

The limit is **structural, not statistical**. Not sample noise, not the
conditioning mechanism, not the objective. A counterfactual computed as the
difference of two rollout risks — each squashed through a softmax and a product
over the horizon — collapses to an average effect regardless of what supervises
it.

> **This model predicts an average treatment effect, not a conditional one.** It
> can tell an operator *whether* isolating a host helps — it says so on 0.940
> of pairs, against an action-aware LSTM at chance. It cannot rank two hosts, and
> four experiments say that is not a tuning problem.

Recovering the conditional effect needs a different formulation of the
counterfactual — not another loss term. That is where the next effort should
go, and this repository does not claim to have it.

---

## Experiment 10 — FiLM conditioning does not fix the constant effect

Experiment 02 left one open problem: the counterfactual worked directionally
(82% vs an action-aware LSTM at chance) but predicted an essentially **constant**
effect — r indistinguishable from zero against the true containment probability,
unmoved by 3x more data. The suspect was how the action entered the transition. With `concat` the
predictor sees `[h, e(a)]`, and the shortest path to lower loss is an additive
shift, which *is* a constant effect. FiLM makes the interaction the primitive:
`h' = gamma(a) * h + beta(a)`.

Both trained on identical data, splits and budget. Only the conditioning differs.

| conditioning | says helps¹ | r(effect, p_contain) | sd(effect) | latent rank |
|---|---|---|---|---|
| concat | **0.905** | **+0.032** | **0.311** | 43.5 |
| film | 0.900 | −0.103 | 0.262 | 43.5 |

| Prediction | Verdict |
|---|---|
| Z1 — FiLM reaches r > 0.30 *(the claim)* | **FAILS** (−0.103) |
| Z2 — FiLM beats concat on that correlation | **FAILS** (−0.103 vs +0.032) |
| Z3 — direction accuracy held ≥ 0.75 | HOLDS (0.900) |
| Z4 — FiLM effects more spread out | **FAILS** (0.262 vs 0.311) |

### It is not that FiLM failed to train

The obvious explanation for "the new mechanism did nothing" is that it never
activated — FiLM is identity-initialised, so a model that barely trains it would
behave exactly like this. Measured over twenty epochs:

```
at init      |gamma-1| = 0.000    |beta| = 0.000
after 20 ep  |gamma-1| = 0.232    |beta| = 0.146
```

The modulation is learned and substantial. It simply does not encode *which*
interventions succeed.

### The diagnosis this leaves

**The training objective never supervises the treatment effect.** Every episode
carries one action sequence — `none` *or* `isolate`, never both. The model
learns marginal dynamics `P(s_{t+1} | s_t, a)`; the counterfactual difference
between two rollouts is a derived quantity that no gradient ever touches.
Changing how the action enters the transition cannot repair that, which is
exactly why both conditionings land in the same place.

So the constant-effect limit is **not an architecture problem**. The next thing
to try is a loss term on the paired difference — the matched pairs exist, the
true effect is known for them, and supervising the predicted delta directly is
what the causal-ML literature does (targeted regularisation, R-learner). Not
built; no claim made.

### What survives, unchanged

Direction accuracy held at 0.900 under FiLM against 0.905 under concat. The
capability experiment 02 established — telling an operator *whether* isolating
helps, where an action-aware LSTM is at chance — is intact. What is still
missing is *how much*, and it is now clear that the missing piece is in the
objective rather than the model.

---

## Experiment 01 — lead time at matched false-alarm rate

### Verdict: the headline claim does not currently hold

| Method | FPR ≤ 2% | FPR ≤ 5% | FPR ≤ 10% | FPR ≤ 20% | window AUC |
|---|---|---|---|---|---|
| GBDT (per-window, no history) | 0.74 (10%) | 0.74 (10%) | 3.48 (61%) | 6.17 (95%) | 0.921 |
| **LSTM (same history)** | **0.74 (22%)** | **0.74 (22%)** | **3.35 (76%)** | **9.70 (100%)** | **0.968** |
| World model (rollout) | 0.26 (12%) | 0.26 (12%) | 0.26 (12%) | 9.61 (100%) | 0.789 |

*Mean lead in windows; (detection rate before compromise).*

| Prediction | Verdict |
|---|---|
| Q1 — all beat chance, GBDT worst | **FAILS** — GBDT beats the world model at tight FPR |
| Q2 — WM ≥ LSTM at FPR ≤ 10% *(the claim)* | **FAILS** — 0.26 vs 3.35 |
| Q3 — latent does not collapse | HOLDS — effective rank 27.1 of 64 |
| Q4 — WM advantage grows at lower FPR | **FAILS** — the opposite happens |

**A well-tuned LSTM over the same features beats the world model on lead time.**
The world model reaches parity only at a 20% false-alarm budget (9.61 vs 9.70),
which is far too loose to be operationally useful.

### Two implementation bugs were found and fixed along the way

Both were real defects, not tuning. They are recorded because the corrected
result is the one that counts, and because the size of the correction shows how
easily this kind of model produces a number that means nothing.

1. **The ATT&CK head was never trained on predicted latents.** It was supervised
   only on encoded `z`, then applied at inference to predicted `ẑ` — a
   distribution it had never seen. Fixing this raised rollout AUC 0.61 → 0.79.
2. **Training was one-step, inference was six-step.** The model was optimised
   for a horizon it is never used at, so compounding error was a surprise rather
   than something it was trained against. Now trained at the inference horizon.

### Two experiment-design errors were also found and fixed

These mattered more than the model bugs, and both made earlier results
meaningless in ways that flattered us.

1. **The methods were answering different questions.** Baselines were labelled
   "is compromise anywhere in the future", the world model "within K steps".
   The baselines' apparent 12-window lead was an artefact. All three now
   predict the same horizon-limited target.
2. **The false-alarm denominator contained no attacks.** With only quiet benign
   episodes as negatives, a model that fires on the first port scan scores a
   perfect FPR and enormous lead while having forecast nothing — the task
   silently collapsed into detection. Negatives now include **attack episodes
   that never reach compromise**.
   The first attempt at those used a random coin flip to contain attacks, which
   made progression *unpredictable in principle*; every method scored noise.
   Failure now has an observable cause: an attacker-skill latent that drives
   scan breadth, dwell time and stall probability together.

---

## Experiment 02 — counterfactual intervention

### Verdict: the capability works directionally. The heterogeneous version does not.

800 matched pairs, split **by pair** (both branches of a pair stay together, or
the shared prefix leaks). 200 held-out pairs. Both models queried at the fork
window on identical pre-intervention context, varying only the action.

Numbers below are from the verification sweep of 5 Sep 2026, and supersede an
earlier run — see **The correlation is not stable across code versions** below,
which is a finding rather than a correction.

| | World model | Action-aware LSTM |
|---|---|---|
| Mean predicted effect | **+0.075** | +0.070 |
| **Says isolating helps¹** | **0.820** | 0.480 *(chance)* |
| corr(effect, true `p_contain`) | **−0.105** (p=0.14) | +0.001 (p=0.98) |
| corr(effect, intervention worked) | +0.076 | +0.014 |

| Prediction | Verdict |
|---|---|
| R1 — effect positive on average | **HOLDS** |
| R2 — direction accuracy > 0.70 | **HOLDS as written (0.820), but the prediction was badly worded** — see ¹ |
| R3 — corr(effect, p_contain) > 0.30 *(the claim)* | **FAILS** (−0.105) |
| R4 — WM beats action-aware LSTM on that correlation | **FAILS** — the earlier run had this holding at +0.007 vs +0.001; both were noise |

### The correlation is not stable across code versions — and that is the result

The 30 Aug run of this experiment recorded `corr(effect, p_contain) = +0.007`
and R4 holding. Re-running the identical, **seeded** script on 5 Sep 2026 gives
**−0.105** and R4 failing. Two consecutive re-runs are bit-identical to each
other, so the script is deterministic; what moved was the shared model code
underneath it between the two dates.

A quantity that swings from +0.007 to −0.105 under an unrelated code change,
with p = 0.14, is **not a small positive correlation. It is zero with noise
around it.** Reading either value as a finding would have been wrong in both
directions, and quoting the friendlier one would have been worse. R4 "holding"
in August was an artefact of which side of zero the noise happened to land on.

This strengthens rather than weakens the conclusion four experiments converge
on: the model predicts an average effect and has no conditional signal at all.

**What works — stated carefully, because the obvious phrasing is false.**
Asked "should we isolate this host now?", the world model answers *yes* on
**82% of held-out pairs**. That is **not** 82% accuracy: isolation genuinely
helps in every pair this generator makes, so the true sign is always positive
and a constant "always isolate" predictor scores **1.000**. Read as accuracy,
0.820 is *below* the trivial baseline.

What the number is evidence for is the comparison, and that survives. The
action-aware LSTM — a genuinely fair baseline that receives the action as an
input feature — sits at **48%, indistinguishable from chance**: given the same
information it did not learn that the action does anything at all. That gap is
real and hard for a competitor to reproduce, and it is a statement about the
LSTM's failure rather than our model's ranking ability. A discriminative model
conditioned on an action it has only ever seen correlated with outcomes does
not learn to *intervene*; a model of the dynamics does.

**What does not work.** The model has learned an approximately **constant**
treatment effect. It cannot tell which interventions will succeed
(r = −0.105 against the true containment probability, which varies 0.30–0.83,
and p = 0.14 — indistinguishable from zero). Raising the data from 260 to 800
pairs did not move it off zero either, so this is not a sample-size limit — it
is a modelling limit.

**Honest reading:** we can tell an operator *that* isolating helps here. We
cannot yet tell them *how much*, or which of two hosts to isolate first. The
second is the more valuable product and it is not evidenced.

---

## What this means for the problem statement

**Claim 1 (lead time) is not supported by our own evidence.** Presenting it
would be dishonest, and an NTRO evaluator running the repo would find out in
one command.

**Claim 2 (counterfactual intervention) is now tested, and partly holds.**
A sequence classifier is *structurally* poor at this even when handed the
action as a feature: the action-aware LSTM sits at chance (0.480) while the
world model says *helps* on 0.940 of pairs. The differentiator is that the LSTM
learned nothing about intervening¹.

What is **not** evidenced is heterogeneity — the model predicts a roughly
constant effect and cannot rank interventions by likely success.

### Honest position, if this had to be presented today

> "We built the forecasting model the PS asks for. On lead time it does **not**
> beat a well-tuned LSTM over the same features, and we can show you the curve.
> Where it does win is the counterfactual, and I want to be precise about what
> the number means. Asked whether isolating a host now changes the outcome, our
> model says yes on 83% of held-out matched pairs and an action-aware LSTM
> given the same information sits at chance. That is not 83% accuracy — a
> constant 'always isolate' answer would score 100% on this corpus. It is
> evidence that the LSTM never learned the action matters, which is the
> structural point. We validate on paired episodes with an identical prefix, so
> the ground truth is real. What we cannot yet do is tell you *which*
> interventions will work — the model has learned a constant effect, and more
> data did not change that."

That is a weaker pitch than the one this project set out to make, and a far
stronger one than a number we cannot defend.

---

## Reproducibility

- Episode generation, splits and torch seeds are fixed.
- Split is **episode-level**, never window-level: windows from one episode
  never straddle the train/test boundary.
- Runtime ~90s on CPU.

---

¹ See **A metric correction** at the top of this file: this column is the share of cases where the model predicts a positive effect, not accuracy against an outcome. A constant "always isolate" predictor scores 1.000 on it.
