# Design brief — KALACHAKRA deck (SIH 2026, PS ID SIH26153)

You are improving the visual design of a **Smart India Hackathon 2026 idea
submission**. The file is `SIH2026_SIH26153_KALACHAKRA_idea.pptx`. It is 6
slides, 16:9, 13.33 × 7.5 inches.

**Your job is visual only.** The content is finished, factually verified, and
every number is reproducible from a code repository. Treat the words and figures
as fixed copy. Your task is to make it *land in two minutes*.

---

## How this deck is actually scored — design for this, not for beauty

SIH evaluators spend **2–3 minutes per submission**. The official weighting:

| criterion | weight | what carries it on this deck |
|---|---|---|
| **Innovation & uniqueness** | **25%** | the comparison grid on slide 2 |
| Problem understanding & clarity | 20% | slide 2 headline + the branching-rollout idea |
| Technical feasibility | 20% | slide 3 architecture diagram |
| Impact & scalability | 20% | slide 5 numbers |
| Presentation quality | 15% | everything you are about to do |

Two consequences you must design around:

1. **The comparison grid on slide 2 is the single highest-value object in the
   deck.** It is the only thing answering "why can no existing tool do this".
2. **Diagrams beat text, always.** The official template's own instruction slide
   says: *"Try to avoid paragraphs and post your idea in points / diagrams /
   Infographics / pictures."*

---

## What this project is

KALACHAKRA forecasts network attacks with a **world model** rather than a
classifier. Instead of labelling traffic malicious or benign, it learns the
*transition dynamics* of network state and rolls them forward K steps to ask
whether a trajectory is heading toward compromise — and, crucially, **what would
happen if we intervened**. A classifier can score the trajectory it observed; it
cannot answer "what if we isolate this host now?"

The client is India's **National Technical Research Organisation**. Theme:
Blockchain & Cybersecurity. Category: Software. Team: **AlgoRhythms**.

**Read this before you touch anything.** This deck's defining quality is that it
reports its own failures. Headline claims were refuted by the team's own
experiments — including on **real botnet captures**, where their model catches
5 of 12 infections and a plain gradient-boosted tree catches 7 to 9. It also
reports a number the team **corrected against their own interest**: they had
published a *worse* result (1 of 12), found the error themselves, and say so on
the slide. Those admissions are the submission's credibility, not blemishes to
tidy away. A redesign that softens them destroys the thing the deck is for.

The refutations live on **slide 4**, where the official template asks for risks
and challenges. Do not move them forward and do not soften them.

**Palette (already applied — keep it):**

| role | hex |
|---|---|
| primary | `#2E4A7D` deep indigo |
| pale primary | `#EEF1F7` |
| warning / refutation | `#A3341F` rust |
| ink | `#141918` |
| muted | `#55605D` |

---

## Hard constraints — breaking any of these can disqualify the submission

1. **Exactly 6 slides.** The official template's instruction slide says
   *"maximum slides limit up to six (6), including the title slide"*.
2. **Do not change the slide titles.** They are the template's prescribed
   sections: title page, `IDEA TITLE`, `TECHNICAL APPROACH`,
   `FEASIBILITY AND VIABILITY`, `IMPACT AND BENEFITS`,
   `RESEARCH AND REFERENCES`.
3. **Do not alter, round, soften or delete any number, claim or caveat.**
   **Protected values — every one must survive verbatim:**

   | value | where | why it must not be "improved" |
   |---|---|---|
   | the four comparison rows (Splunk/Exabeam, LSTM/transformer, BloodHound, KALACHAKRA) | slide 2 | **the 25% criterion** — never compress to prose |
   | chart `48 / 94 / 100` | slide 2 | the **tallest bar is a trivial baseline that beats the team** |
   | "Red is the trivial baseline. It beats us, and we put it on the slide." | slide 2 | the caption that makes the chart honest |
   | "94% is not accuracy - see Feasibility" | slide 2 | the pointer that stops the chart being a boast |
   | `48% → 94%`, `19.9M flows`, `16 / 40`, `< 5e-3` | slide 2 stat cards | |
   | `5 of 12` real infections caught | slide 4 | the real-data loss |
   | `REFUTED — on generated data AND on real captures` | slide 4 | |
   | `REFUTED — after four attempts` | slide 4 | |
   | `HOLDS, NARROWLY` | slide 4 | the one surviving claim, already qualified |
   | "we had published 1 of 12" | slide 4 | a **self-correction**: they found their own published number was too harsh and fixed it. Do not trim it as redundant — it is the point |
   | `0.03 to 0.99 AUC` / "close to a coin flip" | slide 4 | applies to all methods, not just theirs |
   | `r = −0.105, −0.103, +0.037`, `+0.011` | slide 4 | four failed attempts |
   | `29.44 lakh` CERT-In incidents, `up 85%` | slide 5 | the national scale figure |
   | `60-70%` shared core | slide 5 | one core, two problem statements |

   If text must shrink to fit, cut connective words — never a figure, never a
   qualifier, never the word REFUTED.
4. **Keep the template's own furniture**: the SIH logo (top-right, slide 1), the
   grey footer bar, slide numbers, and the oval reading **AlgoRhythms** on
   slides 2–6.
5. **Nothing below y = 6.95 inches.** Minimum 0.5 inch margins elsewhere.
6. Body text never below **10 pt**. Preserve the subscripts and symbols in
   `P(Sₜ₊₁ | Sₜ, action)`, and `≤`, `→`, `ẑ`, `−` (true minus, not hyphen).
7. **Export a PDF alongside the .pptx.** The SIH portal accepts **PDF only**.

---

## Slide-by-slide direction

### Slide 1 — Title page
Leave the structure. Improve typographic hierarchy on the Problem Statement ID /
title / organisation / theme / category / team block. Restrained.

### Slide 2 — `KALACHAKRA — a model you can ask "what if we act?"`
**The slide that wins or loses the deck. Most of your effort goes here.**

Layout: one-line solution statement → comparison grid (left) + native bar chart
(right) → four stat cards → one-line uniqueness statement.

- **The comparison grid is the point of the slide.** Four rows, three columns
  (Approach / What it does / What it cannot do), with the KALACHAKRA row filled
  and bold. Its final cell — *"answers: what happens if we ISOLATE THIS HOST
  NOW?"* — is the entire differentiator. Make the grid scannable in ten seconds
  and let that last cell land. **Do not turn it into paragraphs.**
- **Keep the chart native.** Three bars doing something unusual on purpose:

  ```
  Action-aware LSTM        48%
  Our world model          94%
  Trivial always-isolate  100%   ← highlighted RUST, and it is the TALLEST bar
  ```

  **Do not reorder the bars to put the team's own result on top.** The whole
  point is that a trivial baseline beats them and they showed it unprompted. A
  judge who discovers this alone has caught the team out; a judge who is told is
  reading an honest submission. Restyle freely otherwise.
- Four stat cards: `48% → 94%`, `19.9M flows`, `16 / 40`, `< 5e-3`. These are
  short phrases, not bare digits — size them so the phrase still reads.

### Slide 3 — `TECHNICAL APPROACH`
A 5-stage pipeline: **Host graph Gₜ → GNN encoder → Transition model → K-step
rollout → ATT&CK scoring**, with the middle two emphasised.

**This slide carries the 20% technical-feasibility score. It is the biggest
visual opportunity in either deck.**

- Turn it into a proper process diagram with a flat line **icon** above each
  stage (node-graph, neural net, state-transition arrows, forward-branching
  timeline, kill-chain ladder).
- **The chevrons are the literal character `>`. Replace with real arrow shapes.**
- **The one thing you must draw:** the transition model takes an **action** as
  input, which is what makes the rollout answer a counterfactual. Show the
  rollout **branching into two futures** — one "do nothing", one "isolate" —
  with the two risk curves visibly diverging. That single idea is the project's
  entire differentiator and it is currently not drawn at all. If you add one
  original diagram to this deck, make it this one.
- The supporting text uses specific correct terminology — JEPA-style latent
  predictive objective, VICReg anti-collapse, GRU context encoder, Mondrian
  conformal calibration, MITRE ATT&CK tactic stages. These must stay. Hanging
  them off the diagram as small callouts is better than a text block.

### Slide 4 — `FEASIBILITY AND VIABILITY`
Three verdict rows:

| claim | status |
|---|---|
| Lead time | **REFUTED — on generated data AND on real captures** |
| Counterfactual | **HOLDS, NARROWLY** |
| Which interventions work | **REFUTED — after four attempts** |

- Rebuild as three cards or a clean table. **Colour the status labels
  semantically** — rust for REFUTED, indigo for HOLDS-NARROWLY — so the honesty
  is legible at a glance rather than buried in prose. Do not soften the wording.
- The Lead-time row contains the self-correction ("we had published 1 of 12 …
  we corrected it upward"). It is the most unusual sentence in either deck.
  Consider giving it a subtle distinct treatment so a skimming judge sees that
  the team audited itself — but do not make it look like an apology.
- The closing line carries a finding about the *problem*, not the team: on
  CTU-13 all four methods span 0.03 to 0.99 AUC across thirteen captures.
  Consider setting it apart — it is the most quotable sentence on the slide.

### Slide 5 — `IMPACT AND BENEFITS`
Three impact stat cards across the top (`29.44 lakh`, `60-70%`, `0`), then a
2 × 2 grid of benefit cards: *Decision support, not one more alert*, *Speaks the
SOC's existing language*, *Attribution with a checkable axiom*, *A benchmark the
field does not have*.

- Give each card an icon in a tinted circle.
- The fourth card reframes their evaluation harness as a deliverable in its own
  right — including that it caught an error in their own published result. Let
  it look like the others, not like a footnote.

### Slide 6 — `RESEARCH AND REFERENCES`
Academic citations, datasets, and scale sources. Keep it plain and readable.
Improve leading and hanging indents. Do not drop any citation.

---

## Style

- **Restrained and technical.** An instrument panel, not a startup pitch.
- Icons: flat, single-weight line icons in one palette colour. **No clip-art, no
  3-D, no stock photos, and absolutely no hooded-figure or
  padlock-on-green-binary cybersecurity clichés.**
- Subtle depth is welcome: soft shadows, light tints, generous whitespace.
- **Do not add** decorative colour bars, edge stripes, or accent lines under
  titles. Those read as AI-generated filler.
- Do not centre body text. Left-align paragraphs, lists and table cells; centre
  only titles and the stat cards.
- Keep the file **under 10 MB**.

---

## Before you return the file

- [ ] Still exactly 6 slides, section titles unchanged
- [ ] No text overflows its shape or the slide edge
- [ ] Nothing overlaps the footer bar (y > 6.95")
- [ ] The slide-2 comparison grid is still a **grid**, still four rows, with the
      KALACHAKRA row visually distinct
- [ ] The bar chart is still native, still 48 / 94 / 100, and the **trivial
      baseline is still the tallest bar and still rust**
- [ ] **Every value in the protected table above is present and identical**,
      including every `REFUTED` and the "we had published 1 of 12" correction
- [ ] Slide 3 has real arrows, not `>` characters, and **draws the two-future
      branch**
- [ ] Subscripts and symbols (`P(Sₜ₊₁ | Sₜ, action)`, `≤`, `→`, `ẑ`, `−`) render
      correctly
- [ ] Team oval still reads **AlgoRhythms** on slides 2–6
- [ ] Body text ≥ 10 pt everywhere
- [ ] **Exported a PDF** — the SIH portal accepts nothing else
