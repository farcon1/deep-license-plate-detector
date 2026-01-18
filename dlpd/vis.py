from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bbox_cv(img_bgr: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = img_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def put_text_cv(img_bgr: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
    out = img_bgr.copy()
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def save_image_bgr(path: str | Path, img_bgr: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img_bgr)


def make_montage(
    images_rgb: List[np.ndarray],
    cols: int,
    cell_size: Tuple[int, int] = (320, 200),
    pad: int = 6,
    bg: Tuple[int, int, int] = (18, 18, 18),
) -> Image.Image:
    w, h = cell_size
    rows = int(np.ceil(len(images_rgb) / cols))
    out_w = cols * w + (cols + 1) * pad
    out_h = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (out_w, out_h), bg)
    for i, arr in enumerate(images_rgb):
        r = i // cols
        c = i % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        im = Image.fromarray(arr)
        im = im.resize((w, h), Image.BILINEAR)
        canvas.paste(im, (x, y))
    return canvas
