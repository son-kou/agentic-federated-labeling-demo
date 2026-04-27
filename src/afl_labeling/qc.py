"""QualityControlAgent — compute QC features and assign status/risk flags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.ndimage import binary_erosion, label as scipy_label

from .schemas import LabelRecord, QCRecord

# Plausibility thresholds (ml). Synthetic volumes are compact so min is low.
_PROSTATE_MIN_ML = 5.0
_PROSTATE_MAX_ML = 150.0

# Site shift risk lookup
_SHIFT_RISK = {
    "clean": 0.10,
    "noisy": 0.45,
    "low_contrast_bias": 0.60,
}


def _voxel_volume_ml(mask: np.ndarray, spacing_mm: List[float]) -> float:
    voxel_vol_mm3 = float(np.prod(spacing_mm))
    return float(mask.sum()) * voxel_vol_mm3 / 1000.0


def _boundary_uncertainty(mask: np.ndarray) -> float:
    """
    Proxy: ratio of boundary voxels to total mask voxels.
    High ratio → irregular boundary → higher uncertainty.
    """
    if mask.sum() == 0:
        return 1.0
    interior = binary_erosion(mask > 0, iterations=1)
    boundary = (mask > 0) & ~interior
    return float(boundary.sum()) / float(mask.sum())


def _num_connected_components(mask: np.ndarray) -> int:
    _, n = scipy_label(mask > 0)
    return int(n)


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 2.0 * inter / denom if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# QualityControlAgent
# ---------------------------------------------------------------------------

class QualityControlAgent:
    """Evaluate auto-labels and produce QC records."""

    def __init__(self, qc_dir: Path) -> None:
        self.qc_dir = qc_dir
        self.qc_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_case(
        self,
        case_id: str,
        site_id: str,
        gt_prostate: np.ndarray,
        gt_lesion: np.ndarray,
        pred_prostate: np.ndarray,
        pred_lesion: Optional[np.ndarray],
        metadata: Dict[str, Any],
    ) -> QCRecord:
        spacing: List[float] = metadata.get("spacing", [1.0, 1.0, 1.0])
        shift_type: str = metadata.get("synthetic_shift_type", "clean")

        # --- Metrics ---
        vol_ml = _voxel_volume_ml(pred_prostate, spacing)
        n_cc = _num_connected_components(pred_prostate)
        bnd_unc = _boundary_uncertainty(pred_prostate)
        shift_risk = _SHIFT_RISK.get(shift_type, 0.2)

        lesion_inside: Optional[bool] = None
        if pred_lesion is not None and pred_lesion.sum() > 0:
            overlap = float((pred_lesion & pred_prostate).sum()) / float(pred_lesion.sum() + 1e-8)
            lesion_inside = overlap > 0.8

        vol_plausible = _PROSTATE_MIN_ML <= vol_ml <= _PROSTATE_MAX_ML

        dice_val: Optional[float] = _dice(pred_prostate.astype(bool), gt_prostate.astype(bool))

        # --- Risk flags ---
        flags: List[str] = []
        if bnd_unc > 0.35:
            flags.append("high_boundary_uncertainty")
        if vol_ml < _PROSTATE_MIN_ML:
            flags.append("small_prostate_volume")
        if n_cc > 2:
            flags.append("fragmented_mask")
        if lesion_inside is False:
            flags.append("lesion_outside_prostate")
        if shift_risk >= 0.45:
            flags.append("high_site_shift_risk")

        # --- Status ---
        if len(flags) == 0:
            status = "auto_qc_pass"
        elif len(flags) >= 3 or not vol_plausible:
            status = "reject_or_redraw"
        else:
            status = "review_required"

        record = QCRecord(
            case_id=case_id,
            site_id=site_id,
            prostate_volume_ml=round(vol_ml, 2),
            num_connected_components=n_cc,
            boundary_uncertainty=round(bnd_unc, 4),
            lesion_inside_prostate=lesion_inside,
            volume_plausible=vol_plausible,
            site_shift_risk=shift_risk,
            status=status,
            risk_flags=flags,
            dice_vs_gt=round(dice_val, 4) if dice_val is not None else None,
        )

        self._save(record)
        return record

    def _save(self, record: QCRecord) -> None:
        out_dir = self.qc_dir / record.site_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{record.case_id}_qc.json"
        path.write_text(record.model_dump_json(indent=2))
