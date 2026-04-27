"""AutoLabelingAgent — heuristic prostate and lesion segmentation.

For the MVP, labels are created by:
  1. Thresholding + morphological ops on the volume (prostate).
  2. A degraded copy of the GT mask (adds realistic noise/erosion).

The interface mirrors what a MONAI Label or nnU-Net integration would expose,
making it straightforward to swap in a real model later.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    label as scipy_label,
)

from .schemas import LabelProvenance, LabelRecord

_MODEL_NAME = "HeuristicAutoLabeler"
_MODEL_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Heuristic segmentation helpers
# ---------------------------------------------------------------------------

def _heuristic_prostate_mask(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Rough prostate segmentation via Otsu-style threshold on smoothed volume.
    Returns a binary uint8 mask.
    """
    smooth = gaussian_filter(volume.astype(np.float32), sigma=2.0)
    threshold = float(np.percentile(smooth, 70))
    binary = smooth > threshold
    binary = binary_closing(binary, iterations=3)
    # Keep only largest connected component
    labeled, n = scipy_label(binary)
    if n == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    sizes = np.bincount(labeled.ravel())[1:]
    largest = int(np.argmax(sizes)) + 1
    return (labeled == largest).astype(np.uint8)


def _degrade_gt_mask(
    gt_mask: np.ndarray, rng: np.random.Generator, noise_level: float = 0.12
) -> np.ndarray:
    """
    Simulate an imperfect auto-label by:
    - randomly eroding or dilating the GT boundary
    - flipping a small fraction of voxels near the boundary
    """
    if gt_mask.sum() == 0:
        return gt_mask.copy()

    erode_iters = rng.integers(0, 3)
    dilate_iters = rng.integers(0, 3)
    mask = gt_mask.astype(bool)

    if erode_iters > 0:
        mask = binary_erosion(mask, iterations=int(erode_iters))
    if dilate_iters > 0:
        mask = binary_dilation(mask, iterations=int(dilate_iters))

    # Add voxel-level noise near the boundary
    boundary = binary_dilation(mask) ^ binary_erosion(mask)
    flip = rng.random(gt_mask.shape) < noise_level
    mask = mask ^ (boundary & flip)

    # Keep only the largest connected component to avoid satellite fragments
    labeled, n = scipy_label(mask)
    if n > 1:
        sizes = np.bincount(labeled.ravel())[1:]
        largest = int(np.argmax(sizes)) + 1
        mask = labeled == largest

    return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def _compute_confidence(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """Dice-based confidence (proxy for model certainty in the demo)."""
    intersection = float((pred_mask & gt_mask).sum())
    union = float(pred_mask.sum() + gt_mask.sum())
    if union == 0:
        return 1.0
    dice = 2.0 * intersection / union
    # Add small random jitter so per-case scores vary realistically
    jitter = random.gauss(0, 0.03)
    return float(np.clip(dice + jitter, 0.0, 1.0))


# ---------------------------------------------------------------------------
# AutoLabelingAgent
# ---------------------------------------------------------------------------

class AutoLabelingAgent:
    """Generate prostate and lesion labels with provenance metadata."""

    def __init__(self, labels_dir: Path, seed: int = 0) -> None:
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(seed)

    def label_case(
        self,
        case_id: str,
        site_id: str,
        volume: np.ndarray,
        gt_prostate: np.ndarray,
        gt_lesion: np.ndarray,
    ) -> Tuple[LabelRecord, LabelRecord | None]:
        """
        Run auto-labeling for one case.

        Returns (prostate_record, lesion_record_or_None).
        """
        now = datetime.now(timezone.utc)

        # --- Prostate ---
        pred_prostate = _degrade_gt_mask(gt_prostate, self._rng)
        conf_prostate = _compute_confidence(pred_prostate, gt_prostate)

        prov_p = LabelProvenance(
            model_name=_MODEL_NAME,
            model_version=_MODEL_VERSION,
            generated_at=now,
            input_case_id=case_id,
            label_type="prostate_gland",
            confidence_score=round(conf_prostate, 4),
        )
        p_path = self._save_label(case_id, site_id, "prostate", pred_prostate, prov_p)
        prostate_record = LabelRecord(
            case_id=case_id,
            site_id=site_id,
            provenance=prov_p,
            label_path=str(p_path),
        )

        # --- Lesion (only if GT lesion exists) ---
        lesion_record: Optional[LabelRecord] = None
        if gt_lesion.sum() > 0:
            pred_lesion = _degrade_gt_mask(gt_lesion, self._rng, noise_level=0.18)
            conf_lesion = _compute_confidence(pred_lesion, gt_lesion)

            prov_l = LabelProvenance(
                model_name=_MODEL_NAME,
                model_version=_MODEL_VERSION,
                generated_at=now,
                input_case_id=case_id,
                label_type="lesion_candidate",
                confidence_score=round(conf_lesion, 4),
            )
            l_path = self._save_label(case_id, site_id, "lesion", pred_lesion, prov_l)
            lesion_record = LabelRecord(
                case_id=case_id,
                site_id=site_id,
                provenance=prov_l,
                label_path=str(l_path),
            )

        return prostate_record, lesion_record

    def _save_label(
        self,
        case_id: str,
        site_id: str,
        label_type: str,
        mask: np.ndarray,
        provenance: LabelProvenance,
    ) -> Path:
        out_dir = self.labels_dir / site_id / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / f"{label_type}_pred.npz"
        np.savez_compressed(mask_path, mask=mask)
        prov_path = out_dir / f"{label_type}_provenance.json"
        prov_path.write_text(provenance.model_dump_json(indent=2))
        return mask_path
