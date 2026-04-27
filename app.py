"""Streamlit dashboard — Agentic Federated Labeling Demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path constants ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "sites"
OUTPUTS = ROOT / "outputs"
QC_DIR = OUTPUTS / "qc_reports"
LABELS_DIR = OUTPUTS / "labels"
AL_DIR = OUTPUTS / "site_reports"
FL_DIR = OUTPUTS / "fl_rounds"
REPORT_PATH = OUTPUTS / "meeting_report.md"

SITES = ["site_A", "site_B", "site_C"]

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_all_qc() -> pd.DataFrame:
    rows = []
    for site in SITES:
        site_dir = QC_DIR / site
        if not site_dir.exists():
            continue
        for f in sorted(site_dir.glob("*_qc.json")):
            rows.append(json.loads(f.read_text()))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data
def load_all_labels() -> pd.DataFrame:
    rows = []
    for site in SITES:
        site_label_dir = LABELS_DIR / site
        if not site_label_dir.exists():
            continue
        for case_dir in sorted(site_label_dir.iterdir()):
            prov_path = case_dir / "prostate_provenance.json"
            if prov_path.exists():
                prov = json.loads(prov_path.read_text())
                rows.append(
                    {
                        "case_id": prov.get("input_case_id"),
                        "site_id": site,
                        "confidence_score": prov.get("confidence_score"),
                        "label_type": prov.get("label_type"),
                        "model_name": prov.get("model_name"),
                        "generated_at": prov.get("generated_at"),
                    }
                )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data
def load_al_queue() -> pd.DataFrame:
    path = AL_DIR / "active_learning_queue.json"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(path.read_text()))


@st.cache_data
def load_fl_rounds() -> List[Dict[str, Any]]:
    rounds = []
    for f in sorted(FL_DIR.glob("round_*_summary.json")):
        rounds.append(json.loads(f.read_text()))
    return rounds


def load_case_data(site_id: str, case_id: str):
    case_dir = DATA_DIR / site_id / case_id
    if not (case_dir / "volume.npz").exists():
        return None, None, None, None
    data = np.load(case_dir / "volume.npz")
    meta = json.loads((case_dir / "metadata.json").read_text())
    return data["volume"], data["gt_prostate"], data["gt_lesion"], meta


def load_pred_mask(site_id: str, case_id: str, label_type: str) -> Optional[np.ndarray]:
    path = LABELS_DIR / site_id / case_id / f"{label_type}_pred.npz"
    if not path.exists():
        return None
    return np.load(path)["mask"]


def _middle_slice_rgb(
    volume: np.ndarray,
    prostate: Optional[np.ndarray] = None,
    lesion: Optional[np.ndarray] = None,
) -> np.ndarray:
    from src.afl_labeling.visualization import slice_with_both_overlays, get_middle_axial_slice
    if prostate is None:
        sl = get_middle_axial_slice(volume)
        mn, mx = sl.min(), sl.max()
        norm = (sl - mn) / (mx - mn + 1e-8)
        return np.stack([norm, norm, norm], axis=-1).astype(np.float32)
    return slice_with_both_overlays(volume, prostate, lesion)


def _data_available() -> bool:
    return any((QC_DIR / s).exists() for s in SITES)


def _disclaimer() -> None:
    """Render a consistent synthetic-data disclaimer banner."""
    st.warning(
        "**Synthetic demo mode** — no real patient data are used. "
        "Labels are simulated to demonstrate infrastructure behaviour, "
        "**not clinical performance**. This is not a clinical decision tool."
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AFL Demo — Prostate MRI",
    page_icon="🏥",
    layout="wide",
)

page = st.sidebar.radio(
    "Navigation",
    [
        "📋 Project Overview",
        "📊 Site Dashboard",
        "🔬 Case Viewer",
        "🏷 Label Quality",
        "🌐 Federated Rounds",
        "📄 Audit Report",
    ],
)

if not _data_available():
    st.warning(
        "No pipeline outputs found yet. Run:\n\n"
        "```bash\npython run_demo.py --generate-data --run-pipeline --run-fl\n```"
    )

# ═══════════════════════════════════════════════════════════════════════════
# 1. PROJECT OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📋 Project Overview":
    st.title("Agentic Federated Labeling Infrastructure")
    st.subheader("Trustworthy Prostate MRI AI — MVP Demo")
    _disclaimer()

    st.info(
        "**Images stay local. Only model updates and quality summaries are shared.**\n\n"
        "This demo uses **synthetic data only** and is not a clinical tool."
    )

    st.markdown(
        """
## System Goal

Build a privacy-preserving, multi-site, AI-assisted labeling infrastructure for prostate MRI
that enables federated learning without sharing raw imaging data.

## End-to-End Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                        LOCAL SITE                            │
│                                                              │
│  Raw MRI  →  Preprocessing  →  AutoLabel  →  QC  →  Review  │
│                                                              │
│  ✗ Images never leave          ✓ Only model Δ + QC stats →  │
└──────────────────────────────────────────────────────────────┘
                                         │
                               ┌─────────▼──────────┐
                               │  FL Coordinator    │
                               │  (aggregates only) │
                               └────────────────────┘
```

## Components

| Component | MVP Implementation | Production Replacement |
|-----------|-------------------|----------------------|
| Data | Synthetic NumPy volumes | DICOM/NIfTI (MONAI/SimpleITK) |
| Auto-labeler | Heuristic degraded GT | MONAI Label, nnU-Net, MedSAM |
| QC | Heuristic thresholds | Radiologist QC + inter-reader |
| Active Learning | Priority scoring | Uncertainty + diversity sampling |
| FL Framework | Simulated summaries | Flower (flwr) or NVFLARE |
| Privacy | Demo only | Differential privacy, secure aggregation |

## Sites in This Demo

| Site | Scanner Shift | Description |
|------|--------------|-------------|
| site_A | Clean | Standard contrast, reference quality |
| site_B | Noisy | Higher noise level, different intensity scale |
| site_C | Low contrast + bias | Reduced contrast, mild bias field |
"""
    )

# ═══════════════════════════════════════════════════════════════════════════
# 2. SITE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Site Dashboard":
    st.title("Site Dashboard")
    _disclaimer()

    qc_df = load_all_qc()
    label_df = load_all_labels()

    if qc_df.empty:
        st.warning("No QC data found. Run the pipeline first.")
    else:
        # Build summary table
        rows = []
        for site in SITES:
            sdf = qc_df[qc_df["site_id"] == site] if "site_id" in qc_df.columns else pd.DataFrame()
            if sdf.empty:
                continue
            n = len(sdf)
            n_pass = (sdf["status"] == "auto_qc_pass").sum()
            n_review = (sdf["status"] == "review_required").sum()
            n_reject = (sdf["status"] == "reject_or_redraw").sum()
            mean_dice = sdf["dice_vs_gt"].mean() if "dice_vs_gt" in sdf.columns else None

            ldf = label_df[label_df["site_id"] == site] if not label_df.empty else pd.DataFrame()
            mean_conf = ldf["confidence_score"].mean() if not ldf.empty and "confidence_score" in ldf.columns else None

            rows.append(
                {
                    "Site": site,
                    "Cases": n,
                    "QC Pass": f"{n_pass} ({n_pass/n:.0%})",
                    "Review Required": f"{n_review} ({n_review/n:.0%})",
                    "Rejected": f"{n_reject} ({n_reject/n:.0%})",
                    "Mean Confidence": f"{mean_conf:.3f}" if mean_conf is not None else "—",
                    "Mean Dice": f"{mean_dice:.3f}" if mean_dice is not None else "—",
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Bar charts
        col1, col2 = st.columns(2)
        with col1:
            status_counts = qc_df.groupby(["site_id", "status"]).size().reset_index(name="count")
            fig = px.bar(
                status_counts,
                x="site_id",
                y="count",
                color="status",
                title="QC Status by Site",
                color_discrete_map={
                    "auto_qc_pass": "#2ca02c",
                    "review_required": "#ff7f0e",
                    "reject_or_redraw": "#d62728",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "dice_vs_gt" in qc_df.columns:
                fig2 = px.box(
                    qc_df,
                    x="site_id",
                    y="dice_vs_gt",
                    title="Dice Score Distribution by Site",
                    color="site_id",
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Risk flag heatmap
        st.subheader("Risk Flags Across Sites")
        flag_cols = ["high_boundary_uncertainty", "small_prostate_volume",
                     "fragmented_mask", "lesion_outside_prostate", "high_site_shift_risk"]
        if "risk_flags" in qc_df.columns:
            flag_data = {}
            for flag in flag_cols:
                flag_data[flag] = {
                    site: qc_df[qc_df["site_id"] == site]["risk_flags"]
                    .apply(lambda x: flag in x if isinstance(x, list) else False)
                    .sum()
                    for site in SITES
                }
            flag_df = pd.DataFrame(flag_data).T
            fig3 = px.imshow(
                flag_df,
                text_auto=True,
                title="Risk Flag Counts per Site",
                color_continuous_scale="Oranges",
            )
            st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. CASE VIEWER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Case Viewer":
    st.title("Case Viewer")
    _disclaimer()

    col_site, col_case = st.columns(2)
    with col_site:
        selected_site = st.selectbox("Site", SITES)
    site_dir = DATA_DIR / selected_site
    case_ids = sorted(
        d.name for d in site_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()
    ) if site_dir.exists() else []

    with col_case:
        selected_case = st.selectbox("Case", case_ids or ["(none)"])

    if case_ids and selected_case != "(none)":
        volume, gt_prostate, gt_lesion, meta = load_case_data(selected_site, selected_case)
        pred_prostate = load_pred_mask(selected_site, selected_case, "prostate")
        pred_lesion = load_pred_mask(selected_site, selected_case, "lesion")

        if volume is not None:
            col_img, col_info = st.columns([2, 1])

            with col_img:
                # Slice slider
                mid = volume.shape[0] // 2
                z = st.slider("Axial slice", 0, volume.shape[0] - 1, mid)

                def _rgb_at_z(
                    v: np.ndarray,
                    pm: Optional[np.ndarray],
                    lm: Optional[np.ndarray],
                    z: int,
                ) -> np.ndarray:
                    sl = v[z].astype(np.float32)
                    mn, mx = sl.min(), sl.max()
                    norm = (sl - mn) / (mx - mn + 1e-8)
                    rgb = np.stack([norm, norm, norm], axis=-1)
                    if pm is not None:
                        alpha_p = 0.25
                        green_mask = pm[z] > 0
                        rgb[:, :, 1] = np.where(green_mask, (1 - alpha_p) * rgb[:, :, 1] + alpha_p * 0.8, rgb[:, :, 1])
                        rgb[:, :, 0] = np.where(green_mask, rgb[:, :, 0] * (1 - alpha_p), rgb[:, :, 0])
                    if lm is not None:
                        alpha_l = 0.55
                        red_mask = lm[z] > 0
                        rgb[:, :, 0] = np.where(red_mask, (1 - alpha_l) * rgb[:, :, 0] + alpha_l, rgb[:, :, 0])
                        rgb[:, :, 1] = np.where(red_mask, rgb[:, :, 1] * (1 - alpha_l), rgb[:, :, 1])
                    return np.clip(rgb, 0, 1)

                rgb_pred = _rgb_at_z(volume, pred_prostate, pred_lesion, z)
                rgb_gt = _rgb_at_z(volume, gt_prostate, gt_lesion if gt_lesion.sum() > 0 else None, z)

                tab1, tab2 = st.tabs(["Auto-label overlay", "Ground truth overlay"])
                with tab1:
                    st.image(rgb_pred, caption="Green = prostate auto-label  |  Red = lesion auto-label", use_container_width=True)
                with tab2:
                    st.caption(
                        "⚠️ Synthetic ground truth — shown for demo validation only. "
                        "Not available in real deployment (ground truth requires radiologist annotation)."
                    )
                    st.image(rgb_gt, caption="Green = synthetic GT prostate  |  Red = synthetic GT lesion", use_container_width=True)

            with col_info:
                st.subheader("Metadata")
                st.json(meta)

                st.subheader("QC Status")
                qc_path = QC_DIR / selected_site / f"{selected_case}_qc.json"
                if qc_path.exists():
                    qc = json.loads(qc_path.read_text())
                    status = qc["status"]
                    color = {"auto_qc_pass": "green", "review_required": "orange", "reject_or_redraw": "red"}.get(status, "grey")
                    st.markdown(f"**Status:** :{color}[{status}]")
                    st.markdown(f"**Prostate volume:** {qc['prostate_volume_ml']} ml")
                    st.markdown(f"**Dice vs GT:** {qc.get('dice_vs_gt', '—')}")
                    st.markdown(f"**Boundary uncertainty:** {qc['boundary_uncertainty']:.3f}")
                    if qc["risk_flags"]:
                        st.warning("**Risk flags:** " + ", ".join(qc["risk_flags"]))
                    else:
                        st.success("No risk flags.")
                else:
                    st.info("QC not yet available — run the pipeline first.")

# ═══════════════════════════════════════════════════════════════════════════
# 4. LABEL QUALITY
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏷 Label Quality":
    st.title("Label Quality")
    _disclaimer()

    qc_df = load_all_qc()
    label_df = load_all_labels()
    al_df = load_al_queue()

    if label_df.empty and qc_df.empty:
        st.warning("No label data found. Run the pipeline first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if not label_df.empty and "confidence_score" in label_df.columns:
                fig = px.histogram(
                    label_df,
                    x="confidence_score",
                    color="site_id",
                    nbins=20,
                    title="Confidence Score Distribution",
                    barmode="overlay",
                    opacity=0.7,
                )
                fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Review threshold")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if not al_df.empty and "priority_score" in al_df.columns:
                fig2 = px.histogram(
                    al_df,
                    x="priority_score",
                    color="site_id",
                    nbins=20,
                    title="Review Priority Score Distribution",
                    barmode="overlay",
                    opacity=0.7,
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Top Cases for Human Review")
        if not al_df.empty:
            top = al_df.head(15)[["rank", "case_id", "site_id", "priority_score", "reasons"]]
            top = top.copy()
            top["reasons"] = top["reasons"].apply(
                lambda x: "; ".join(x[:2]) if isinstance(x, list) else str(x)
            )
            st.dataframe(top, use_container_width=True)

        if not qc_df.empty and "dice_vs_gt" in qc_df.columns:
            st.subheader("Dice Score vs Confidence")
            if not label_df.empty and "confidence_score" in label_df.columns:
                merged = qc_df.merge(label_df[["case_id", "confidence_score"]], on="case_id", how="left")
                if not merged.empty:
                    fig3 = px.scatter(
                        merged,
                        x="confidence_score",
                        y="dice_vs_gt",
                        color="site_id",
                        symbol="status",
                        title="Auto-label Confidence vs Dice (vs GT)",
                        hover_data=["case_id", "risk_flags"],
                    )
                    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. FEDERATED ROUNDS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌐 Federated Rounds":
    st.title("Federated Learning Rounds")
    _disclaimer()

    st.info(
        "**Privacy invariant:** Image data never leave any site. "
        "The coordinator receives only gradient summaries and aggregated QC stats."
    )

    fl_rounds = load_fl_rounds()

    if not fl_rounds:
        st.warning("No FL round data found. Run `python run_demo.py --run-fl`.")
    else:
        # Global trend
        trend_df = pd.DataFrame(
            [
                {
                    "Round": r["round_number"],
                    "Global Confidence": r["global_mean_confidence"],
                    "Review Rate": r["global_review_rate"],
                }
                for r in fl_rounds
            ]
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(trend_df, x="Round", y="Global Confidence", markers=True,
                          title="Global Label Confidence over FL Rounds")
            fig.update_traces(line_color="#2ca02c")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(trend_df, x="Round", y="Review Rate", markers=True,
                           title="Global Review Rate over FL Rounds")
            fig2.update_traces(line_color="#ff7f0e")
            st.plotly_chart(fig2, use_container_width=True)

        # Per-site per-round table
        st.subheader("Per-Site Contributions per Round")
        rows = []
        for rnd in fl_rounds:
            for contrib in rnd["site_contributions"]:
                rows.append(
                    {
                        "Round": rnd["round_number"],
                        "Site": contrib["site_id"],
                        "Cases Processed": contrib["cases_processed"],
                        "Needing Review": contrib["cases_needing_review"],
                        "Mean Confidence": contrib["mean_confidence"],
                        "Model Δ Summary": contrib["model_delta_summary"],
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Per-site review burden
        st.subheader("Review Burden Before / After FL")
        site_rows = []
        for site in SITES:
            before = next(
                (c for c in fl_rounds[0]["site_contributions"] if c["site_id"] == site),
                None,
            )
            after = next(
                (c for c in fl_rounds[-1]["site_contributions"] if c["site_id"] == site),
                None,
            )
            if before and after:
                site_rows.append(
                    {
                        "Site": site,
                        "Review cases (round 1)": before["cases_needing_review"],
                        "Review cases (final round)": after["cases_needing_review"],
                        "Δ": after["cases_needing_review"] - before["cases_needing_review"],
                    }
                )
        if site_rows:
            st.dataframe(pd.DataFrame(site_rows), use_container_width=True)

        st.subheader("Privacy Boundary — What Leaves Each Site?")
        privacy_rows = [
            {"Data item", "Leaves site?", "Rationale"},
        ]
        st.markdown(
            """
| Data item | Leaves site? | Rationale |
|-----------|:------------:|-----------|
| Raw MRI volume (DICOM / NIfTI) | **Never** | Primary imaging data; GDPR Art. 9 |
| Per-voxel label mask | **Never** | Derived from image; can reconstruct anatomy |
| Individual case metadata | **Never** | Indirect identifiers (age, scanner, PSA) |
| Patient identifiers | **Never** | Name, DOB, patient ID |
| Full model weight tensor | **Never** | Susceptible to gradient inversion attacks |
| Human correction details | **Never** | Correction strokes and timing logs |
| Model weight delta **norm** (scalar) | ✓ Shareable | No weight values; L2 norm only |
| Cosine similarity to global model | ✓ Shareable | Direction only, not gradient content |
| QC pass / review rate (aggregate) | ✓ Shareable | Fraction over all cases; no per-case info |
| Mean confidence / mean Dice | ✓ Shareable | Aggregate scalar |
| Cases processed / needing review | ✓ Shareable | Integer counts only |
| Site shift risk label | Review first | May reveal scanner vendor |
| Risk flag type counts | Review first | Reveals systematic QC issues |
"""
        )
        st.caption("See `docs/privacy_boundary.md` for the full governance reference.")

# ═══════════════════════════════════════════════════════════════════════════
# 6. AUDIT REPORT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📄 Audit Report":
    st.title("Audit Report")
    _disclaimer()

    if REPORT_PATH.exists():
        st.markdown(REPORT_PATH.read_text())
    else:
        st.warning("meeting_report.md not found. Run the full pipeline first.")

    st.divider()
    st.subheader("JSON Audit File Paths")
    audit_files = sorted(OUTPUTS.rglob("*audit*.json")) + sorted(FL_DIR.glob("round_*.json"))
    if audit_files:
        for p in audit_files:
            st.code(str(p.relative_to(ROOT)))
    else:
        st.info("No audit files found yet.")
