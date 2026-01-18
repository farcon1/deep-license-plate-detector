from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def pr_curve_single_object_per_image(
    y_iou: np.ndarray,
    y_score: np.ndarray,
    iou_thr: float,
    points: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    assert y_iou.shape == y_score.shape
    n = int(y_iou.shape[0])
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    ts = np.linspace(1.0, 0.0, int(points))

    precisions = np.zeros_like(ts, dtype=np.float64)
    recalls = np.zeros_like(ts, dtype=np.float64)

    for i, t in enumerate(ts):
        pred_mask = y_score >= t
        tp = int(np.sum((y_iou >= iou_thr) & pred_mask))
        fp = int(np.sum((y_iou < iou_thr) & pred_mask))

        denom = tp + fp
        precisions[i] = (tp / denom) if denom > 0 else 0.0
        recalls[i] = tp / n

    return ts, precisions, recalls


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0

    order = np.argsort(recalls)
    r = recalls[order]
    p = precisions[order]

    p_env = p.copy()
    for i in range(p_env.size - 2, -1, -1):
        p_env[i] = max(p_env[i], p_env[i + 1])

    ap = 0.0
    r_prev = 0.0
    for ri, pi in zip(r, p_env):
        if ri > r_prev:
            ap += (ri - r_prev) * pi
            r_prev = ri
    return float(ap)


def summarize_metrics(y_iou: np.ndarray, y_score: np.ndarray, iou_thr: float, pr_points: int) -> Dict[str, float]:
    ts, ps, rs = pr_curve_single_object_per_image(y_iou, y_score, iou_thr=iou_thr, points=pr_points)
    ap = average_precision(rs, ps)

    den = ps + rs
    num = 2.0 * ps * rs
    f1 = np.zeros_like(den, dtype=np.float64)
    np.divide(num, den, out=f1, where=den > 0)
    f1 = np.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)

    best_idx = int(np.argmax(f1)) if f1.size else 0

    return {
        "n_images": float(y_iou.size),
        "iou_thr": float(iou_thr),
        "ap": float(ap),
        "best_f1": float(f1[best_idx]) if f1.size else 0.0,
        "best_precision": float(ps[best_idx]) if ps.size else 0.0,
        "best_recall": float(rs[best_idx]) if rs.size else 0.0,
        "best_score_threshold": float(ts[best_idx]) if ts.size else 0.0,
        "mean_iou": float(np.mean(y_iou)) if y_iou.size else 0.0,
        "median_iou": float(np.median(y_iou)) if y_iou.size else 0.0,
        "success_rate_iou_ge_thr": float(np.mean(y_iou >= iou_thr)) if y_iou.size else 0.0,
    }
