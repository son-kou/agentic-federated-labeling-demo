# 5-Minute Demo Script

**Agentic Federated Labeling Infrastructure for Trustworthy Prostate MRI AI**

---

## Before you start

- Run `python run_demo.py --generate-data --run-pipeline --run-fl`
- Open `streamlit run app.py` in a browser
- Navigate the sidebar as you speak

---

## Opening (30 s)

> "Prostate cancer is the most commonly diagnosed cancer in men in Europe. Clinically significant
> prostate cancer — the cases that actually need treatment — can be detected and risk-stratified on
> MRI. But to train or evaluate any AI model on prostate MRI, you need labelled data: someone has
> to draw a contour around the gland and mark suspicious lesions on every scan. That's expensive,
> slow, and requires an expert radiologist."

---

## The problem: private data scattered across sites (45 s)

> "The real challenge is that the best prostate MRI data sit in hospital archives at different
> institutions. Each site has its own scanner vendor, protocol, and patient population. No single
> site has enough cases to build a robust AI model on its own — but the data can't just be
> pooled centrally. Ethics approvals, data transfer agreements, and GDPR all create friction."

> "So the question is: can we build an AI-assisted labeling infrastructure where the images
> never leave the site, but the sites still benefit from each other's experience?"

---

## System idea (30 s)

Navigate to **📋 Project Overview**.

> "This is what we're proposing. Each participating site runs a local pipeline: preprocessing,
> auto-labeling, quality control, and human review. The only things that leave a site are
> model update summaries — scalar numbers — and aggregated quality statistics. Raw images,
> labels, and patient metadata stay on-site permanently. The central coordinator just
> aggregates these tiny summaries and sends back an improved model."

---

## Per-site local workflow (45 s)

Navigate to **📊 Site Dashboard**.

> "In this demo we simulate three sites. Site A is a clean reference scanner. Site B has
> higher noise — typical of an older MRI system. Site C has lower contrast and a bias field
> artifact. You can see these differences show up in the QC pass rates: site A passes
> most of its cases automatically, while site B and C have a higher review burden because
> our QC agent correctly detects the higher-risk scanner shift."

> "This is exactly what would happen in reality: a radiologist at a noisier site would need
> to spend more time correcting labels."

---

## Auto-labeling and QC (45 s)

Navigate to **🔬 Case Viewer**.

> "For each case, the auto-labeling agent generates a prostate gland mask and, where
> present, a lesion candidate mask. Each label comes with a provenance record: which model,
> which version, what timestamp, and a confidence score. The QC agent then evaluates the
> mask independently: Is the volume anatomically plausible? Is the boundary smooth? Is the
> lesion inside the gland? Are there disconnected fragments?"

> "The auto-label overlay tab shows the predicted masks in green and red. The ground
> truth tab shows the synthetic reference — in a real deployment that tab would not exist,
> because we wouldn't have ground truth until after human review."

---

## Human-in-the-loop review queue (30 s)

Navigate to **🏷 Label Quality**.

> "Not all cases need human review. The active learning agent ranks cases by priority —
> combining boundary uncertainty, confidence score, QC status, and site shift risk. A
> radiologist starting their review session would open this queue and work from the top.
> The infrastructure tells them *why* each case was flagged, so they can triage quickly."

---

## FL simulation (45 s)

Navigate to **🌐 Federated Rounds**.

> "After the local pipeline runs at each site, the federated coordinator simulates three
> rounds of federated learning. In each round, each site trains locally on its own data —
> which you can imagine as fine-tuning the auto-labeler on the corrected labels — and sends
> back only a gradient summary: a norm and a cosine similarity. No weights, no images, no
> labels. The coordinator aggregates these, updates the global model, and sends back new
> model parameters. You can see global confidence improving across rounds and review burden
> stabilising."

---

## Privacy boundary (30 s)

> "Let me point out the privacy boundary table here. Every row is a data type. The left
> column is what the coordinator sees. The right column is what stays on-site forever.
> This is the property we're designing around: the coordinator is deliberately kept ignorant
> of individual case content."

---

## What this demo proves (30 s)

Navigate to **📄 Audit Report**.

> "What this demo proves is that the *workflow* works end-to-end: data ingestion,
> auto-labeling with provenance, QC with risk flags, active-learning triage, federated
> rounds with audit logs, and a human-readable meeting report — all automated and
> reproducible. The audit report is generated automatically and documents what information
> left each site."

---

## What this demo does not prove

> "What it does *not* prove is clinical performance. The auto-labeler here uses simulated
> labels based on synthetic ground truth — it's a stand-in for MONAI Label or nnU-Net. The
> QC thresholds are heuristic. The FL simulation uses scalar summaries, not real gradient
> exchange. We haven't validated the uncertainty estimates against radiologist inter-reader
> variability. This is infrastructure, not a clinical model."

---

## Next steps with real data (30 s)

> "The next step is to connect this infrastructure to real local data. We're looking for
> collaborators willing to run the local pipeline — preprocessing, auto-labeling, QC — on
> their own DICOM archive, without sending any images outside. In return they get a
> prioritised review queue that saves radiologist time, a quality report for their own data,
> and eventually a model that improves with input from all participating sites. If that
> sounds interesting, the one-pager in the docs folder has the details."

---

*Total: ~5 minutes. Allow extra time for questions on the Case Viewer and FL pages.*
