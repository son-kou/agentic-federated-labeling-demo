# Privacy Boundary Definition

**Agentic Federated Labeling Infrastructure — What leaves the site and what does not**

This document defines the privacy boundary enforced by the infrastructure. It is intended
for data governance review, ethics committee briefings, and collaborator onboarding.

---

## 1. Data That Never Leave the Site

These data items are processed entirely within the local site environment and are never
transmitted, copied, or referenced externally:

| Data item | Why it stays local |
|-----------|--------------------|
| Raw MRI volumes (DICOM / NIfTI) | Primary imaging data; protected under GDPR Art. 9 and institutional DPA |
| Per-voxel segmentation masks | Derived from images; can reconstruct anatomical information |
| Individual case metadata | May contain indirect identifiers (age, PSA level, scanner protocol) |
| Patient identifiers | Name, DOB, patient ID, accession number |
| Preprocessing artefacts | Normalised volumes, resampled grids |
| Human radiologist corrections | Correction strokes and timing logs |
| Local model fine-tune checkpoints | Full weight tensors; not suitable for transmission without DP |

---

## 2. Data That May Leave the Site

These items are designed to be shareable. They are aggregated at the site level before
transmission and contain no individual case content:

| Data item | Form | Notes |
|-----------|------|-------|
| Model weight delta norm | Scalar float, e.g. `∥Δw∥=0.0042` | No weight values, only the L2 norm |
| Cosine similarity to global model | Scalar float in [−1, 1] | Indicates update direction, not content |
| QC pass rate | Fraction, e.g. `0.73` | Aggregate over all cases processed this round |
| Review rate | Fraction | Proportion of cases flagged for human review |
| Mean confidence / mean Dice | Scalar | Average over all cases; no per-case breakdown |
| Cases processed count | Integer | Number of cases in this FL round |
| Cases needing review count | Integer | Aggregate count |

---

## 3. Data That Should Be Reviewed Before Sharing

These items are site-generated but require governance review before they can be shared
externally. They are stored locally in the audit log:

| Data item | Review consideration |
|-----------|---------------------|
| Site data quality profile | May reveal scanner protocol details; check with site radiologist |
| Risk flag type counts (e.g. `fragmented_mask × 3`) | Reveals systematic QC issues; acceptable in most cases |
| Site shift type annotation | May reveal scanner vendor; typically acceptable |
| Per-round FL summaries | Review for indirect identifiability before archiving externally |

---

## 4. How Audit Logs Are Generated

Every automated decision is recorded in a structured audit log:

```
outputs/
├── labels/<site>/<case_id>/prostate_provenance.json   ← per-label audit
├── qc_reports/<site>/<case_id>_qc.json               ← per-case QC audit
├── site_reports/<site>/site_summary.json              ← per-site summary
├── fl_rounds/round_001_summary.json                   ← per-round FL audit
└── meeting_report.md                                  ← human-readable summary
```

Each provenance JSON records:
- `model_name` and `model_version` — which model produced the label
- `generated_at` — UTC timestamp
- `input_case_id` — which case was processed
- `confidence_score` — model certainty estimate

Each QC JSON records:
- All computed features (volume, boundary uncertainty, connected components)
- The assigned status and risk flags
- The Dice score vs synthetic GT (demo mode only)

These logs are stored locally. In production, they should be stored in an immutable
append-only log (FHIR `AuditEvent` resources, or a site-controlled database).

---

## 5. How FL Summaries Differ from Image Sharing

| Property | Image sharing | FL summary sharing |
|----------|--------------|-------------------|
| Contains pixels | Yes | No |
| Contains patient metadata | Yes | No |
| Can reconstruct anatomy | Yes | No (scalar norm only) |
| Reversible to training data | Potentially | Not with current summaries |
| Requires DTA | Yes (full) | Lightweight data processing agreement |
| GDPR Art. 9 applies | Yes | Unlikely (no special category data) |
| Membership inference risk | High | Low (aggregate stats only) |

> **Note on gradient inversion attacks:** Full gradient tensors can be susceptible to
> reconstruction attacks (Zhu et al., 2019). This infrastructure intentionally transmits
> only the gradient *norm* and cosine similarity, not the full gradient tensor, removing
> this attack surface. In production, apply differential privacy noise (Gaussian mechanism)
> to gradient updates before computing the norm.

---

## 6. Why This Approach Is Suitable for Early Privacy-Preserving Collaboration

1. **No image data cross the network.** The privacy risk profile is fundamentally different
   from centralised data pooling or even pseudonymised transfer.

2. **Each site retains full control.** Sites can inspect and approve every item that leaves
   before transmission. The audit log makes this verifiable.

3. **The coordinator is deliberately ignorant.** The FL coordinator cannot reconstruct any
   individual case. Its only inputs are scalars and aggregate fractions.

4. **Incrementally deployable.** A site can participate in QC reporting and review-queue
   generation without ever participating in FL gradient sharing.

5. **Compatible with existing ethics frameworks.** Most ethics approvals for retrospective
   MRI data permit local algorithm development and quality assurance. Sharing scalar
   training summaries typically falls within the scope of multi-site research agreements.

6. **Auditable and reproducible.** Every automated step is logged. A data governance officer
   can review the full pipeline behavior from the JSON audit files without inspecting images.

---

*For questions about the privacy model or to review the audit logs, contact the system maintainer.*
