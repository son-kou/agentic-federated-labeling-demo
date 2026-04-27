"""Per-site preprocessing and metadata profiling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def load_case(case_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load volume, gt_prostate, gt_lesion and metadata from *case_dir*."""
    data = np.load(case_dir / "volume.npz")
    meta = json.loads((case_dir / "metadata.json").read_text())
    return data["volume"], data["gt_prostate"], data["gt_lesion"], meta


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization clamped to [-3, 3] then rescaled to [0, 1]."""
    mean, std = float(volume.mean()), float(volume.std()) + 1e-8
    z = (volume - mean) / std
    z = np.clip(z, -3.0, 3.0)
    return ((z + 3.0) / 6.0).astype(np.float32)


def compute_site_profile(site_dir: Path) -> Dict[str, Any]:
    """Return a metadata profile dict for all cases in *site_dir*."""
    case_dirs = sorted(d for d in site_dir.iterdir() if d.is_dir())
    records: List[Dict[str, Any]] = []
    intensities: List[float] = []

    for cd in case_dirs:
        if not (cd / "metadata.json").exists():
            continue
        vol, _, _, meta = load_case(cd)
        records.append(meta)
        intensities.append(float(vol.mean()))

    if not records:
        return {"site_id": site_dir.name, "num_cases": 0}

    num_lesion = sum(r["has_lesion"] for r in records)
    return {
        "site_id": site_dir.name,
        "num_cases": len(records),
        "num_with_lesion": num_lesion,
        "mean_intensity": float(np.mean(intensities)),
        "std_intensity": float(np.std(intensities)),
        "shift_type": records[0].get("synthetic_shift_type", "unknown"),
        "modalities": records[0].get("modalities", []),
    }


def list_cases(site_dir: Path) -> List[Path]:
    return sorted(d for d in site_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists())
