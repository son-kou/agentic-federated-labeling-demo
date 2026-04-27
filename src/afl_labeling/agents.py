"""Pipeline orchestration: run the full per-site pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .active_learning import ActiveLearningAgent
from .autolabel import AutoLabelingAgent
from .preprocessing import list_cases, load_case
from .qc import QualityControlAgent
from .schemas import ActiveLearningRecord, LabelRecord, QCRecord


def run_site_pipeline(
    site_dir: Path, outputs_root: Path
) -> Dict[str, Any]:
    """
    Execute autolabeling + QC + active-learning for all cases in *site_dir*.

    Returns a dict with per-case results and a site-level summary.
    """
    site_id = site_dir.name
    labels_dir = outputs_root / "labels"
    qc_dir = outputs_root / "qc_reports"
    al_dir = outputs_root / "site_reports"

    autolabeler = AutoLabelingAgent(labels_dir=labels_dir)
    qc_agent = QualityControlAgent(qc_dir=qc_dir)
    al_agent = ActiveLearningAgent(out_dir=al_dir)

    case_dirs = list_cases(site_dir)
    if not case_dirs:
        print(f"  [{site_id}] no cases found — skipping.")
        return {}

    label_records: List[LabelRecord] = []
    qc_records: List[QCRecord] = []
    confidence_map: Dict[str, float] = {}

    for case_dir in case_dirs:
        volume, gt_prostate, gt_lesion, meta = load_case(case_dir)
        case_id = meta["case_id"]

        # --- Auto-labeling ---
        p_rec, l_rec = autolabeler.label_case(
            case_id=case_id,
            site_id=site_id,
            volume=volume,
            gt_prostate=gt_prostate,
            gt_lesion=gt_lesion,
        )
        label_records.append(p_rec)
        confidence_map[case_id] = p_rec.provenance.confidence_score
        if l_rec is not None:
            label_records.append(l_rec)

        # Load the predictions back from disk for QC
        pred_prostate = _load_pred_mask(p_rec.label_path)
        pred_lesion: Optional[np.ndarray] = None
        if l_rec is not None:
            pred_lesion = _load_pred_mask(l_rec.label_path)

        # --- QC ---
        qc_rec = qc_agent.evaluate_case(
            case_id=case_id,
            site_id=site_id,
            gt_prostate=gt_prostate,
            gt_lesion=gt_lesion,
            pred_prostate=pred_prostate,
            pred_lesion=pred_lesion,
            metadata=meta,
        )
        qc_records.append(qc_rec)

    # --- Active learning ranking ---
    al_records = al_agent.rank_cases(qc_records, confidence_map)

    summary = _build_site_summary(site_id, qc_records, confidence_map, al_records)
    _save_site_report(al_dir / site_id, summary)
    print(
        f"  [{site_id}] {len(case_dirs)} cases — "
        f"pass={summary['qc_pass_rate']:.0%}  "
        f"review={summary['review_rate']:.0%}  "
        f"mean_conf={summary['mean_confidence']:.3f}"
    )
    return summary


def _load_pred_mask(path: str) -> np.ndarray:
    return np.load(path)["mask"].astype(np.uint8)


def _build_site_summary(
    site_id: str,
    qc_records: List[QCRecord],
    confidence_map: Dict[str, float],
    al_records: List[ActiveLearningRecord],
) -> Dict[str, Any]:
    n = len(qc_records)
    n_pass = sum(1 for r in qc_records if r.status == "auto_qc_pass")
    n_review = sum(1 for r in qc_records if r.status == "review_required")
    n_reject = sum(1 for r in qc_records if r.status == "reject_or_redraw")
    confs = list(confidence_map.values())
    dice_vals = [r.dice_vs_gt for r in qc_records if r.dice_vs_gt is not None]
    top_cases = [r.case_id for r in al_records[:5]]

    return {
        "site_id": site_id,
        "total_cases": n,
        "qc_pass": n_pass,
        "review_required": n_review,
        "rejected": n_reject,
        "qc_pass_rate": n_pass / n if n else 0.0,
        "review_rate": n_review / n if n else 0.0,
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "mean_dice": float(np.mean(dice_vals)) if dice_vals else None,
        "top_review_cases": top_cases,
    }


def _save_site_report(out_dir: Path, summary: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "site_summary.json").write_text(json.dumps(summary, indent=2))
