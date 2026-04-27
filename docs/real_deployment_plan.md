# Real Deployment Plan

**Agentic Federated Labeling Infrastructure — Upgrade Path from Demo to Production**

This document describes how to transition each demo component to real local hospital data,
phase by phase. Each phase can be validated independently before the next begins.

---

## Phase 1 — Local DICOM/NIfTI Ingestion and Visualization

**Goal:** Confirm the system can read real data and render it correctly, with no labeling yet.

| | Detail |
|-|--------|
| **Input** | Raw prostate MRI (DICOM series or NIfTI files); T2W axial required, DWI/ADC optional |
| **Output** | Loaded volumes in standardised orientation (RAS+); axial slice viewer in dashboard |
| **Collaborator involvement** | Provide a small test set (5–10 cases) from a de-identified export |
| **Technical risks** | DICOM tag inconsistency across scanners; slice thickness / spacing variation; oblique acquisitions |
| **Validation metrics** | Visual inspection; spacing consistency check; orientation sanity check (prostate visible in mid-axial slice) |
| **Key libraries** | `pydicom`, `nibabel`, `SimpleITK`, or MONAI `LoadImage` |

---

## Phase 2 — Prostate Gland Auto-Labeling

**Goal:** Generate a whole-gland segmentation mask automatically; no zonal detail yet.

| | Detail |
|-|--------|
| **Input** | T2W volume in standardised orientation, spacing normalised to 0.5×0.5×3 mm (or model's expected resolution) |
| **Output** | Binary prostate gland mask (.nii.gz or .npz); confidence score; provenance JSON |
| **Collaborator involvement** | Radiologist spot-checks on ~10 % of cases to estimate accuracy |
| **Technical risks** | Domain shift between training scanner and local scanner; large prostate edge cases; coil inhomogeneity |
| **Validation metrics** | Dice coefficient vs radiologist contour; Hausdorff distance; volume correlation (r²) |
| **Recommended models** | nnU-Net v2 trained on PI-CAI + Prostate158; MONAI Label TotalSegmentator prostate module |

---

## Phase 3 — Zonal Segmentation

**Goal:** Separate peripheral zone (PZ) and transition zone (TZ/CG); needed for PI-RADS grading context.

| | Detail |
|-|--------|
| **Input** | T2W volume + whole-gland mask from Phase 2 |
| **Output** | Per-zone binary masks; zonal volume estimates |
| **Collaborator involvement** | Radiologist review of zone boundary cases (~20 % expected review rate initially) |
| **Technical risks** | PZ/TZ boundary is poorly defined in enlarged glands (BPH); T2W signal overlap |
| **Validation metrics** | Per-zone Dice; PZ/TZ volume ratio plausibility; comparison with PI-CAI zone annotations |
| **Recommended models** | nnU-Net v2 with zone labels from PI-CAI; Prostate158 zone segmentation baseline |

---

## Phase 4 — Lesion Candidate Detection and Segmentation

**Goal:** Detect and delineate clinically significant lesion candidates (PI-RADS ≥ 3).

| | Detail |
|-|--------|
| **Input** | T2W + ADC volumes; prostate gland mask from Phase 2 |
| **Output** | Lesion candidate mask(s); per-lesion confidence; PI-RADS likelihood score (optional) |
| **Collaborator involvement** | Radiologist review of all lesion-positive predictions; site-specific false-positive rate audit |
| **Technical risks** | High false-positive rate on first deployment; lesion size varies widely (3–30 mm); ADC quality varies |
| **Validation metrics** | Lesion-level sensitivity / specificity at fixed false-positive rates; FROC curve; comparison with biopsy outcomes where available |
| **Recommended models** | MedSAM2 prompted with lesion bounding box; nnU-Net trained on PI-CAI csPCa labels |

---

## Phase 5 — Human Correction Workflow

**Goal:** Integrate a structured radiologist correction loop; capture inter-reader variability.

| | Detail |
|-|--------|
| **Input** | Auto-labels from Phases 2–4; active-learning review queue |
| **Output** | Corrected label archive; correction time log; inter-reader agreement metrics |
| **Collaborator involvement** | 1–2 radiologists; structured annotation tool (OHIF, 3D Slicer + MONAI Label plugin) |
| **Technical risks** | Reader fatigue; inconsistent correction granularity; missing correction metadata |
| **Validation metrics** | Correction rate (% of auto-labels modified); time-per-case; inter-reader Dice on double-annotated subset |
| **Key tool** | MONAI Label server on local site; annotation UI in OHIF or 3D Slicer |

---

## Phase 6 — Federated Training with Flower or NVFLARE

**Goal:** Replace simulated FL summaries with a real gradient exchange protocol.

| | Detail |
|-|--------|
| **Input** | Corrected label archives at participating sites; shared model initialisation |
| **Output** | Federated global model checkpoint; per-site performance metrics; training audit log |
| **Collaborator involvement** | Site IT team to open outbound port; data governance sign-off on gradient sharing; DPA with coordinator institution |
| **Technical risks** | Communication failures; model divergence under non-iid data; membership inference attacks on gradients |
| **Validation metrics** | Convergence curves; per-site holdout Dice before vs after FL; differential privacy budget tracking (ε, δ) |
| **Key libraries** | `flwr` (Flower) for simple setups; NVIDIA FLARE for hospital-grade deployment with audit and DP |

---

## Phase 7 — Downstream Label Utility Evaluation

**Goal:** Validate that labels produced by this infrastructure are useful for training downstream models.

| | Detail |
|-|--------|
| **Input** | Federated label archive from Phase 6; holdout test sets at each site |
| **Output** | Classification / detection performance report; comparison against manually-labelled baselines |
| **Collaborator involvement** | Radiologist review of test-set predictions; biopsy outcome linkage where available |
| **Technical risks** | Label noise amplification in downstream model; distribution mismatch between FL training and test population |
| **Validation metrics** | AUROC for csPCa detection; per-PI-RADS sensitivity at 10 % FPR; Dice on held-out manual labels; radiologist agreement on 50-case sample |
| **Publication target** | MICCAI, Medical Image Analysis, npj Digital Medicine |

---

## Summary Timeline (indicative)

| Phase | Prerequisite | Estimated effort |
|-------|-------------|-----------------|
| 1 — Ingestion | 1 site, ethics approval | 2–4 weeks |
| 2 — Gland labeling | Phase 1 complete | 4–6 weeks |
| 3 — Zonal segmentation | Phase 2 + zone GT | 4–6 weeks |
| 4 — Lesion detection | Phase 3 + csPCa GT | 6–10 weeks |
| 5 — Human correction | Phase 4 + radiologist time | Ongoing |
| 6 — Federated training | ≥2 sites at Phase 5 | 8–12 weeks |
| 7 — Utility evaluation | Phase 6 + test sets | 4–8 weeks |
