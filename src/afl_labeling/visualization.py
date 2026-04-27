"""Matplotlib-based helpers for rendering MRI slices and overlays."""

from __future__ import annotations

from typing import Optional

import numpy as np


def get_middle_axial_slice(volume: np.ndarray) -> np.ndarray:
    """Return the middle axial (z-axis) slice as a 2-D float32 array."""
    mid = volume.shape[0] // 2
    return volume[mid].astype(np.float32)


def overlay_mask_on_slice(
    img_slice: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.35,
    color: tuple = (1.0, 0.2, 0.2),
) -> np.ndarray:
    """Return an RGB image with *mask_slice* painted over *img_slice*."""
    # Normalise grayscale slice to [0, 1]
    mn, mx = img_slice.min(), img_slice.max()
    norm = (img_slice - mn) / (mx - mn + 1e-8)
    # Stack to RGB
    rgb = np.stack([norm, norm, norm], axis=-1)
    # Paint mask
    for c, ch in enumerate(color):
        rgb[:, :, c] = np.where(mask_slice > 0, (1 - alpha) * rgb[:, :, c] + alpha * ch, rgb[:, :, c])
    return np.clip(rgb, 0, 1).astype(np.float32)


def slice_with_both_overlays(
    volume: np.ndarray,
    prostate_mask: np.ndarray,
    lesion_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Middle axial slice with prostate (green) and lesion (red) overlays."""
    mid = volume.shape[0] // 2
    img = volume[mid].astype(np.float32)
    pm = prostate_mask[mid].astype(np.uint8)

    mn, mx = img.min(), img.max()
    norm = (img - mn) / (mx - mn + 1e-8)
    rgb = np.stack([norm, norm, norm], axis=-1)

    # Prostate: semi-transparent green
    alpha_p = 0.25
    rgb[:, :, 1] = np.where(pm > 0, (1 - alpha_p) * rgb[:, :, 1] + alpha_p * 0.8, rgb[:, :, 1])
    rgb[:, :, 0] = np.where(pm > 0, rgb[:, :, 0] * (1 - alpha_p), rgb[:, :, 0])

    if lesion_mask is not None:
        lm = lesion_mask[mid].astype(np.uint8)
        alpha_l = 0.55
        rgb[:, :, 0] = np.where(lm > 0, (1 - alpha_l) * rgb[:, :, 0] + alpha_l, rgb[:, :, 0])
        rgb[:, :, 1] = np.where(lm > 0, rgb[:, :, 1] * (1 - alpha_l), rgb[:, :, 1])

    return np.clip(rgb, 0, 1).astype(np.float32)
