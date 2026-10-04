# KALACHAKRA — slide 6 references, with working links

Every URL below was fetched and returned HTTP 200 on 4 October 2026. Where a paper is
paywalled, an open-access route is given instead of the publisher's page.

Hand this file to whoever produced the beautified slide, or use it to attach the links
yourself (see **Putting these into the deck**, at the end).

---

## 01 · Prior art in ATT&CK forecasting

**Husák, Bartoš, Sokol & Gajdoš** — *Predictive Methods in Cyber Defense: Current
Experience and Research Challenges.* Future Generation Computer Systems **115** (2021)
517–530.
https://doi.org/10.1016/j.future.2020.10.006
Publisher page, Elsevier. **This is the one reference your teammate may hit a paywall
on** — no open-access copy exists. The record page at Masaryk University carries the
full citation if they only need to verify it: https://ics.muni.cz/en/research/publications/1682977

**Albasheer et al.** — *Cyber-Attack Prediction Based on Network Intrusion Detection
Systems for Alert Correlation Techniques: A Survey.* Sensors **22**(4):1494, 2022.
https://pmc.ncbi.nlm.nih.gov/articles/PMC8879519/
Open access, full text and PDF, no login.

> **Correct the author name on the slide.** The slide credits this to "Abdillahmed
> et al." (the deck source said "Abdlhamed"). PMC8879519 is Albasheer, Md Siraj,
> Mubarakali, Tayfour, Salih, Hamdan, Khan, Zainal & Kamarudeen. An evaluator who
> follows the PMC ID lands on a different author list than the slide claims.

**Srivastava, Thakkar, Valiveti, Shah & Raval** — *Anticipated Network Surveillance:
An extrapolated study to predict cyber-attacks using Machine Learning and Data
Analytics*, 2023.
https://arxiv.org/abs/2312.17270
Open access. PDF: https://arxiv.org/pdf/2312.17270

---

## 02 · Representation learning & ATT&CK

**Bardes, Ponce & LeCun** — *VICReg: Variance-Invariance-Covariance Regularization for
Self-Supervised Learning.* ICLR 2022.
https://arxiv.org/abs/2105.04906
Open access.

**LeCun** — *A Path Towards Autonomous Machine Intelligence*, 2022 (the JEPA
objective).
https://openreview.net/forum?id=BZ5a1r-kVsf
Open access; the PDF button is on that page. Direct PDF:
https://openreview.net/pdf?id=BZ5a1r-kVsf

**MITRE ATT&CK** — for tactic-stage labelling.
https://attack.mitre.org/
Open.

**Sundararajan, Taly & Yan** — *Axiomatic Attribution for Deep Networks.* ICML 2017
(integrated gradients).
https://arxiv.org/abs/1703.01365
Open access.

---

## 03 · Data, scale & our work

**CTU-13 botnet captures** — Stratosphere Laboratory, Czech Technical University in
Prague. The corpus every measured number in this submission comes from.
https://www.stratosphereips.org/datasets-ctu13
Open; the capture files download from that page.

**Garcia, Grill, Stiborek & Zunino** — *An empirical comparison of botnet detection
methods.* Computers & Security **45** (2014) 100–123. The paper that introduced
CTU-13; cite this when you cite the dataset.
https://doi.org/10.1016/j.cose.2014.05.011
Publisher page. Author copy and summary:
https://www.stratosphereips.org/publications/2014/5/11/an-empirical-comparison-of-botnet-detection-methods

**CERT-In / PIB** — 29.44 lakh incidents handled in 2025, up 85% on 2023.
https://static.pib.gov.in/WriteReadData/specificdocs/documents/2026/jan/doc2026123764501.pdf
The PIB release, 23 January 2026. Direct PDF, opens without a login.

**Our repository** — all experiments, pre-registered predictions and printed verdicts,
one-command reproduction.
https://github.com/AKRai-2005/kalachakra

> The beautified slide still shows the placeholder `[REPO URL]` in the "Our work" box.
> Replace it with the address above.

---

## 04 · Key technologies

Lower value as citations, but they are named on the slide, so:

| Named | Link |
|---|---|
| PyTorch | https://pytorch.org/ |
| XGBoost | https://xgboost.readthedocs.io/ |
| NumPy | https://numpy.org/ |
| pytest | https://docs.pytest.org/ |

---

## Putting these into the deck

**If the slide is a generated one** (`build_kalachakra_ppt.py`), it is already done —
slide 6 now carries all eleven links, each as a live hyperlink *and* as printed text.
Rebuild and export.

**If the slide is a beautified image**, PowerPoint cannot hyperlink text that is part of
a picture. Lay invisible click targets over it:

1. Insert → Shapes → Rectangle, drawn over the reference text you want clickable.
2. Shape Fill → **No Fill**, Shape Outline → **No Outline**.
3. Right-click → **Link** → paste the URL → OK.
4. Repeat for each reference. Group them so they move with the slide.

These survive **Save as PDF** and stay clickable for whoever opens the file.

**What does not survive:** any link, of either kind, once the deck is printed on paper
or screenshotted. That is why the generated slide prints the short address next to each
link as well — a reader with only a printout can still type it. If the beautified slide
has no room for printed addresses, send this file alongside the PDF.
