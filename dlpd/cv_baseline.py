from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .metrics import Box
from .utils import safe_div


@dataclass
class Detection:
    box: Optional[Box]
    score: float
    debug: dict


class CVPlateDetector:
    """
    Classical CV baseline
    """

    def __init__(
        self,
        max_side: int = 960,
        min_area_norm: float = 0.001,
        max_area_norm: float = 0.25,
        ar_min: float = 2.0,
        ar_max: float = 6.5,
    ):
        self.max_side = int(max_side)
        self.min_area_norm = float(min_area_norm)
        self.max_area_norm = float(max_area_norm)
        self.ar_min = float(ar_min)
        self.ar_max = float(ar_max)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, img_bgr: np.ndarray) -> Detection:
        h0, w0 = img_bgr.shape[:2]
        scale = 1.0
        if max(h0, w0) > self.max_side:
            scale = self.max_side / float(max(h0, w0))
            img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        else:
            img = img_bgr

        h, w = img.shape[:2]
        img_area = float(w * h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        sobelx = cv2.Sobel(gray_blur, cv2.CV_16S, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)

        _, thr = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0.0
        debug_best = {}

        edges = cv2.Canny(gray_blur, 80, 200)

        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            if ww <= 0 or hh <= 0:
                continue

            area_norm = (ww * hh) / img_area
            if area_norm < self.min_area_norm or area_norm > self.max_area_norm:
                continue

            ar = ww / float(hh)
            if ar < self.ar_min or ar > self.ar_max:
                continue

            cnt_area = cv2.contourArea(cnt)
            rect_area = float(ww * hh)
            rectangularity = safe_div(cnt_area, rect_area, default=0.0)
            roi = edges[y:y+hh, x:x+ww]
            edge_density = float(np.mean(roi > 0)) if roi.size else 0.0
            ratio_score = float(np.exp(-abs(np.log(ar / 3.5))))
            score = 0.50 * rectangularity + 0.30 * ratio_score + 0.20 * edge_density

            if score > best_score:
                best_score = score
                best = (x, y, x + ww, y + hh)
                debug_best = {
                    "rectangularity": rectangularity,
                    "ratio_score": ratio_score,
                    "edge_density": edge_density,
                    "area_norm": area_norm,
                    "aspect_ratio": ar,
                }
        if best is None:
            return Detection(box=None, score=0.0, debug={"reason": "no_candidates", "scale": scale})
        x1, y1, x2, y2 = best
        if scale != 1.0:
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale
        box = Box(float(x1), float(y1), float(x2), float(y2))
        score = float(np.clip(best_score, 0.0, 1.0))
        debug_best["scale"] = scale
        debug_best["w0"] = w0
        debug_best["h0"] = h0
        return Detection(box=box, score=score, debug=debug_best)