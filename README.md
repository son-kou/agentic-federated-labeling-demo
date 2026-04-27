# Agentic Federated Labeling Infrastructure for Trustworthy Prostate MRI AI

A privacy-preserving, multi-site, AI-assisted labeling infrastructure demo for prostate MRI.
Uses **synthetic data only** — safe to run, share, and extend.

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data, run pipeline (autolabel + QC + active learning), simulate FL
python run_demo.py --generate-data --run-pipeline --run-fl

# Launch the Streamlit dashboard
streamlit run app.py
```

Run tests:

```bash
pytest tests/ -v
```

---

## What This Demo Shows

| Component | What it does |
|-----------|-------------|
| **Synthetic data** | 3 sites × 12 cases, 3D prostate MRI volumes with prostate and lesion masks, per-site scanner shift simulation |
| **AutoLabelingAgent** | Generates approximate prostate and lesion masks with full provenance (model name, version, confidence score, timestamp) |
| **QualityControlAgent** | Computes prostate volume, connected components, boundary uncertainty, lesion-inside-prostate check; assigns status and risk flags |
| **ActiveLearningAgent** | Ranks cases by review priority from uncertainty, QC flags, and site shift |
| **FederatedCoordinatorAgent** | Simulates 3 FL rounds — image data never leave the site; only gradient norms and aggregated QC stats are shared |
| **Streamlit dashboard** | 6-page interactive UI: overview, site stats, case viewer, label quality, FL rounds, audit report |
| **Audit report** | `outputs/meeting_report.md` + per-round JSON files documenting what information left each site |

---

## Scientific Framing

> **"Agentic Federated Labeling Infrastructure for Trustworthy Prostate MRI AI"**

This MVP demonstrates the infrastructure and workflow. Each component is designed to be replaced
by production-grade tools in a real hospital deployment:

| MVP component | Production replacement |
|---------------|----------------------|
| Synthetic NumPy volumes | Hospital DICOM/NIfTI data via MONAI or SimpleITK |
| Heuristic auto-labeler | MONAI Label, nnU-Net, MedSAM, or MedSAM2 |
| Simulated FL summaries | Flower (`flwr`) or NVFLARE |
| Heuristic QC thresholds | Radiologist correction + inter-reader variability studies |
| Demo-only privacy | Differential privacy (Gaussian mechanism), secure aggregation |
| JSON audit files | Immutable audit log (FHIR AuditEvent, distributed ledger) |

---

## Repository Layout

```
agentic-federated-labeling-demo/
├── README.md
├── requirements.txt
├── app.py                        ← Streamlit dashboard (6 pages)
├── run_demo.py                   ← CLI orchestrator
├── docs/
│   ├── demo_script.md            ← 5-minute meeting script
│   ├── collaborator_one_pager.md ← clinical/research collaborator brief
│   ├── real_deployment_plan.md   ← 7-phase upgrade path to real data
│   ├── model_replacement_plan.md ← how to swap in MONAI Label / nnU-Net / MedSAM
│   └── privacy_boundary.md      ← governance-level privacy definition
├── configs/
│   └── demo_config.yaml          ← sites, cases_per_site, volume_shape, fl_rounds
├── data/
│   └── sites/
│       ├── site_A/               ← clean scanner (reference)
│       ├── site_B/               ← noisy scanner
│       └── site_C/               ← low-contrast + bias field
├── outputs/
│   ├── labels/                   ← auto-label masks + provenance JSON
│   ├── qc_reports/               ← per-case QC JSON
│   ├── site_reports/             ← site summaries + active-learning queue
│   ├── fl_rounds/                ← round_001_summary.json … round_003_summary.json
│   └── meeting_report.md         ← human-readable audit report
├── src/
│   └── afl_labeling/
│       ├── schemas.py            ← Pydantic models (CaseMetadata, LabelRecord, QCRecord …)
│       ├── synthetic_data.py     ← 3D volume + mask generation with site shifts
│       ├── preprocessing.py      ← normalisation, site profiling, case listing
│       ├── visualization.py      ← axial slice helpers for Streamlit
│       ├── autolabel.py          ← AutoLabelingAgent
│       ├── qc.py                 ← QualityControlAgent
│       ├── active_learning.py    ← ActiveLearningAgent
│       ├── federated.py          ← FederatedCoordinatorAgent
│       ├── agents.py             ← per-site pipeline orchestration
│       ├── audit.py              ← audit trail builders
│       └── report.py             ← meeting_report.md generator
└── tests/
    └── test_demo_pipeline.py     ← pytest integration tests
```

---

## How to Present This Demo to Collaborators

### Recommended command sequence

```bash
# 1. Run once before the meeting
python run_demo.py --generate-data --run-pipeline --run-fl

# 2. Open the dashboard
streamlit run app.py
```

### Recommended dashboard page order

| Step | Page | What to show |
|------|------|-------------|
| 1 | 📋 Project Overview | Workflow diagram; explain the privacy invariant |
| 2 | 📊 Site Dashboard | Site heterogeneity; QC pass rates differ across scanners |
| 3 | 🔬 Case Viewer | Auto-label vs GT overlay; QC flags on a specific case |
| 4 | 🏷 Label Quality | Review priority queue; how radiologist time is focused |
| 5 | 🌐 Federated Rounds | Privacy boundary table; FL trend plots |
| 6 | 📄 Audit Report | Auto-generated report; JSON audit trail |

### 5 key talking points

1. **Images never leave the site.** The privacy boundary is enforced by design, not policy.
2. **Every automated label has a provenance record.** Model name, version, confidence, and timestamp.
3. **QC is automated but risk-stratified.** Clean-scanner cases pass automatically; noisy-scanner cases flag for review — matching real-world expectations.
4. **Human review is prioritised, not eliminated.** The active learning queue sends radiologists to the highest-uncertainty cases first.
5. **This is infrastructure, not a clinical model.** Auto-labels here are simulated. The workflow is identical with MONAI Label, nnU-Net, or MedSAM plugged in.

### Limitations (be explicit)

- Auto-labeling accuracy is not evaluated — labels are synthetic.
- Federated rounds are simulated; no real gradients are computed or exchanged.
- QC thresholds are heuristic; they have not been validated against radiologist agreement.
- The privacy model does not include differential privacy in this demo.
- Scanner shift is simulated by noise injection, not real multi-vendor data.

### Next-step collaboration checklist

- [ ] Ethics / IRB approval for local retrospective data use
- [ ] Identify 1–2 radiologists for review workflow testing (Phase 5)
- [ ] Confirm local compute: Linux server, ≥16 GB RAM (GPU optional for inference)
- [ ] Install MONAI Label server or nnU-Net environment locally
- [ ] Run Phase 1 ingestion on 5–10 de-identified test cases (see `docs/real_deployment_plan.md`)
- [ ] Sign a lightweight data processing agreement covering gradient summary sharing

---

## Privacy Invariant

```
✗  Raw MRI volumes           → never transmitted
✗  Per-voxel labels          → never transmitted
✗  Individual case metadata  → never transmitted
✗  Patient identifiers       → never transmitted

✓  Model weight delta norm   → scalar, sent to coordinator
✓  Aggregated QC statistics  → pass rate, mean Dice, flag counts
✓  Case count integers       → processed / needing review
```

---

## License

Demo only — not a clinical decision tool. Uses synthetic data throughout.
# agentic-federated-labeling-demo
