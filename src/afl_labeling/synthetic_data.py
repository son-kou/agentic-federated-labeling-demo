"""Generate synthetic 3D prostate MRI volumes and masks per site."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Volume primitives
# ---------------------------------------------------------------------------

def _ellipsoid_mask(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
) -> np.ndarray:
    """Boolean mask of an axis-aligned ellipsoid."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    )
    return dist <= 1.0


def _sphere_mask(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    return _ellipsoid_mask(shape, center, (radius, radius, radius))


# ---------------------------------------------------------------------------
# Site-specific shift functions
# ---------------------------------------------------------------------------

def _apply_site_shift(
    volume: np.ndarray, shift_type: str, rng: np.random.Generator
) -> np.ndarray:
    if shift_type == "clean":
        return volume
    elif shift_type == "noisy":
        noise = rng.normal(0, 0.08, volume.shape).astype(np.float32)
        return np.clip(volume + noise, 0, 1)
    elif shift_type == "low_contrast_bias":
        # reduce contrast and add smooth bias field
        compressed = 0.4 + 0.4 * volume
        bias = gaussian_filter(
            rng.uniform(0.85, 1.15, volume.shape).astype(np.float32), sigma=15
        )
        return np.clip(compressed * bias, 0, 1)
    return volume


# ---------------------------------------------------------------------------
# Single-case generator
# ---------------------------------------------------------------------------

def _generate_case(
    case_id: str,
    site_id: str,
    shape: Tuple[int, int, int],
    shift_type: str,
    has_lesion: bool,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Return dict with volume, gt_prostate_mask, gt_lesion_mask, metadata."""
    D, H, W = shape
    center = (D / 2, H / 2, W / 2)

    # Prostate gland — ellipsoid roughly 25-40 % of FOV
    radii = (
        D * rng.uniform(0.20, 0.28),
        H * rng.uniform(0.22, 0.30),
        W * rng.uniform(0.22, 0.30),
    )
    gt_prostate = _ellipsoid_mask(shape, center, radii).astype(np.uint8)

    # MRI signal: bright inside prostate, darker outside
    volume = np.zeros(shape, dtype=np.float32)
    volume[gt_prostate == 1] = rng.uniform(0.65, 0.90)
    volume[gt_prostate == 0] = rng.uniform(0.10, 0.35, (gt_prostate == 0).sum())
    volume = gaussian_filter(volume, sigma=1.5)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    volume = _apply_site_shift(volume, shift_type, rng)

    # Lesion candidate — small sphere inside the prostate
    gt_lesion = np.zeros(shape, dtype=np.uint8)
    if has_lesion:
        # random position inside prostate bounding box
        inds = np.argwhere(gt_prostate)
        lc = inds[rng.integers(len(inds))]
        lr = min(radii) * rng.uniform(0.15, 0.30)
        lesion_mask = _sphere_mask(shape, tuple(lc), lr).astype(np.uint8)
        gt_lesion = (lesion_mask & gt_prostate).astype(np.uint8)
        # Lesion slightly brighter on T2W
        volume = np.where(gt_lesion == 1, np.clip(volume * 1.15, 0, 1), volume)

    metadata = {
        "case_id": case_id,
        "site_id": site_id,
        "spacing": [float(rng.uniform(0.5, 0.7))] * 3,
        "modalities": ["T2W", "ADC"],
        "has_lesion": bool(has_lesion),
        "synthetic_shift_type": shift_type,
    }

    return {
        "volume": volume,
        "gt_prostate": gt_prostate,
        "gt_lesion": gt_lesion,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SITE_SHIFTS = {
    "site_A": "clean",
    "site_B": "noisy",
    "site_C": "low_contrast_bias",
}


def generate_sites(data_dir: Path, config: Dict[str, Any]) -> None:
    """Generate all synthetic site data and write to *data_dir*/<site>/."""
    rng = np.random.default_rng(config.get("random_seed", 42))
    random.seed(config.get("random_seed", 42))

    shape: Tuple[int, int, int] = tuple(config.get("volume_shape", [96, 96, 64]))  # type: ignore[assignment]
    cases_per_site: int = config.get("cases_per_site", 12)

    for site_id in config.get("sites", ["site_A", "site_B", "site_C"]):
        site_dir = data_dir / site_id
        site_dir.mkdir(parents=True, exist_ok=True)
        shift_type = _SITE_SHIFTS.get(site_id, "clean")

        # ~40 % of cases have a lesion
        lesion_flags = [i < max(1, cases_per_site // 3) for i in range(cases_per_site)]
        random.shuffle(lesion_flags)

        for idx in range(cases_per_site):
            case_id = f"{site_id}_case_{idx:03d}"
            case = _generate_case(
                case_id, site_id, shape, shift_type, lesion_flags[idx], rng
            )
            case_dir = site_dir / case_id
            case_dir.mkdir(exist_ok=True)

            np.savez_compressed(
                case_dir / "volume.npz",
                volume=case["volume"],
                gt_prostate=case["gt_prostate"],
                gt_lesion=case["gt_lesion"],
            )
            (case_dir / "metadata.json").write_text(
                json.dumps(case["metadata"], indent=2)
            )

        print(f"  [{site_id}] generated {cases_per_site} cases (shift={shift_type})")
