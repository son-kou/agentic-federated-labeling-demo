# Agentic Federated Labeling for Prostate MRI
## Collaborator Information Sheet

---

### Clinical and Research Motivation

Clinically significant prostate cancer (csPCa) can be detected and risk-stratified on multi-parametric MRI (T2W, DWI, ADC). Robust AI models for prostate gland segmentation, zonal anatomy, and lesion detection require large, well-labelled multi-site datasets. Manual labelling is expensive, slow, and subject to inter-reader variability.

A federated labeling infrastructure addresses this by:
- Reducing radiologist effort through AI-assisted pre-labeling
- Prioritising the cases most in need of human correction
- Enabling multi-site model improvement without sharing raw images
- Providing a documented audit trail for every label

---

### What the System Does

```
Your local server                          Central coordinator
────────────────                           ──────────────────
DICOM / NIfTI data                         (never sees images)
      │
      ▼
Preprocessing + QC profiling
      │
      ▼
AI auto-labeling (prostate gland + lesion candidates)
      │
      ▼
Quality control (volume, shape, uncertainty flags)
      │
      ▼
Review queue (ranked by urgency)     ───► model Δ norm + QC stats only ──►
      │                                                                   │
      ▼                                                                   ▼
Radiologist correction                                            FL aggregation
      │                                                                   │
      ▼                                                              improved model
Final label archive (stays local)    ◄───────────────────────────────────┘
```

---

### Why Local / Federated Labeling Matters

| Conventional central labeling | Federated labeling |
|-------------------------------|-------------------|
| Images leave your institution | Images stay on your server |
| Requires full data transfer agreement | Requires only model update protocol |
| Single-site bias in the resulting model | Multi-site diversity built in |
| No visibility into per-site QC | Per-site quality reports and audit logs |
| Radiologist reviews everything | Active learning filters to high-priority cases |

---

### What Collaborators Need to Provide

| Item | Notes |
|------|-------|
| Local compute server | Linux, ≥16 GB RAM, ≥4 CPU cores; GPU optional for inference |
| Prostate MRI data | T2W axial + DWI/ADC, DICOM or NIfTI format |
| Local IT approval | Standard research use; no cloud transfer needed |
| 1 radiologist for review | Part-time; active learning reduces review load |
| Ethics / IRB coverage | For local retrospective data use |

---

### What Stays Local — Always

- Raw MRI volumes (DICOM / NIfTI)
- Per-voxel label masks
- Individual case metadata (age, PSA, biopsy result)
- Patient identifiers
- Preprocessing artefacts

---

### What You Receive

- A prioritised review queue for your radiologist
- Per-case QC reports with risk flags
- A site-level data quality profile (de-identified)
- A locally-trained model fine-tuned on your data distribution
- A human-readable audit report of every automated decision
- Contribution to a shared federated model (improved over rounds)

---

### Future Integration

Once infrastructure is validated with synthetic data:

| Component | Tool |
|-----------|------|
| Prostate auto-labeler | MONAI Label, nnU-Net v2 |
| Lesion detection | MedSAM, MedSAM2, or site-fine-tuned nnU-Net |
| Federated framework | Flower (`flwr`) or NVIDIA FLARE |
| Privacy guarantee | Differential privacy (Gaussian mechanism) |
| Audit log | FHIR AuditEvent or institutional equivalent |

---

### Important: Current Demo Uses Synthetic Data Only

> **This system is currently a research infrastructure demo.**
>
> - All data shown in the dashboard are 3D synthetic volumes — no real patient images.
> - The auto-labeler simulates labels to demonstrate workflow behavior, not clinical performance.
> - No claims about segmentation accuracy, lesion detection sensitivity, or clinical utility are made.
> - The federated learning rounds are simulated; no real model weights are transmitted.
>
> Connecting to real hospital data is the subject of a future ethics-approved deployment phase.

---

*Contact: gs@cercare-medical.com*
