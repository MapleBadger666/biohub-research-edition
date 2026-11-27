"""Deterministic 3-D convolution interface and sparse association loss."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def default_kernel() -> np.ndarray:
    """Normalized Gaussian-shaped template for the synthetic demo."""
    grid = np.indices((3, 3, 3), dtype=float) - 1
    kernel = np.exp(-0.5 * np.square(grid / 0.65).sum(axis=0))
    return kernel / kernel.sum()


def load_kernel(path: Path) -> np.ndarray:
    return validate_kernel(np.load(path, allow_pickle=False))


def validate_kernel(kernel: np.ndarray) -> np.ndarray:
    if not isinstance(kernel, np.ndarray) or kernel.shape != (3, 3, 3):
        raise ValueError("expected a 3×3×3 NumPy kernel")
    if not np.issubdtype(kernel.dtype, np.number) or np.iscomplexobj(kernel) or not np.isfinite(kernel).all():
        raise ValueError("expected finite real kernel weights")
    return kernel.astype(float, copy=False)


def response(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Score an image with z/y/x axes, preserving its shape."""
    if volume.ndim != 3 or any(size == 0 for size in volume.shape) or not np.isfinite(volume).all():
        raise ValueError("expected a finite nonempty z/y/x image")
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(volume, 1), (3, 3, 3))
    return np.einsum("zyxijk,ijk->zyx", windows, validate_kernel(kernel))


def detect(volume: np.ndarray, kernel: np.ndarray, threshold: float) -> list[tuple[int, int, int]]:
    """Select local maxima with deterministic nonmaximum suppression in voxel units."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    scores = response(volume, kernel)
    peaks = np.argwhere(scores > threshold)
    ranked = sorted((tuple(int(v) for v in xyz) for xyz in peaks),
                    key=lambda xyz: (-scores[xyz], xyz))
    detections = []
    for xyz in ranked:
        if all(max(abs(a - b) for a, b in zip(xyz, other)) > 1 for other in detections):
            detections.append(xyz)
    return sorted(detections)


def sparse_focal_loss(logits: np.ndarray, target: np.ndarray,
                      division_positive_weight: float = 1.0) -> float:
    """Illustrate the historical annotated row/column mask and division weighting."""
    logits = np.asarray(logits, dtype=float)
    target = np.asarray(target, dtype=float)
    if logits.ndim != 2 or logits.shape != target.shape or not np.isin(target, [0, 1]).all():
        raise ValueError("expected equal-shaped 2-D logits and a binary target")
    if not np.isfinite(logits).all() or not np.isfinite(division_positive_weight) or division_positive_weight < 1:
        raise ValueError("expected finite logits and division weight >= 1")
    row_totals = target.sum(axis=1)
    mask = (row_totals[:, None] > 0) | (target.sum(axis=0)[None, :] > 0)
    if not mask.any():
        raise ValueError("target has no annotated positives")
    signed = logits * (2 * target - 1)
    loss = np.logaddexp(0, -signed) * np.exp(-2 * np.logaddexp(0, signed))
    weight = np.where((row_totals[:, None] > 1) & (target == 1), division_positive_weight, 1.0)
    return float((loss * weight)[mask].mean())
