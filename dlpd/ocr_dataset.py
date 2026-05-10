from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .ccpd import CCPDAnnotation, iter_ccpd_records
from .plate_text import decode_plate_indices, validate_plate_indices
from .utils import dump_json, ensure_dir, seed_everything


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(x1, max(0, w - 1))))
    y1 = int(max(0, min(y1, max(0, h - 1))))
    x2 = int(max(0, min(x2, max(0, w - 1))))
    y2 = int(max(0, min(y2, max(0, h - 1))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _safe_name_from_path(img_path: Path) -> str:
    parts = list(img_path.parts)
    if len(parts) > 3:
        tail = parts[-3:]
    else:
        tail = parts
    stem = "__".join(tail[:-1] + [img_path.stem])
    stem = stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return stem


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected points shape (4,2), got {pts.shape}")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def crop_plate_from_bbox(img_bgr: np.ndarray, ann: CCPDAnnotation, out_size: Tuple[int, int]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_box(ann.x1, ann.y1, ann.x2, ann.y2, w, h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox after clipping.")
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop from bbox.")
    out_w, out_h = int(out_size[0]), int(out_size[1])
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    return crop


def crop_plate_from_corners(img_bgr: np.ndarray, ann: CCPDAnnotation, out_size: Tuple[int, int]) -> np.ndarray:
    pts = np.array(ann.corners, dtype=np.float32)
    pts = _order_points(pts)

    out_w, out_h = int(out_size[0]), int(out_size[1])
    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC)
    if warped.size == 0:
        raise ValueError("Empty crop from corners.")
    return warped


def crop_plate(img_bgr: np.ndarray, ann: CCPDAnnotation, crop_mode: str, out_size: Tuple[int, int]) -> np.ndarray:
    mode = str(crop_mode).lower()
    if mode == "corners":
        try:
            return crop_plate_from_corners(img_bgr, ann, out_size=out_size)
        except Exception:
            return crop_plate_from_bbox(img_bgr, ann, out_size=out_size)
    return crop_plate_from_bbox(img_bgr, ann, out_size=out_size)


def export_ccpd_to_ocr(
    dataset_root: Path,
    split_dir: Path,
    image_exts: List[str],
    out_dir: Path,
    splits: List[str],
    max_images_per_split: int,
    crop_mode: str,
    crop_size: Tuple[int, int],
    overwrite: bool,
    seed: int,
) -> Path:
    seed_everything(seed)
    out_dir = Path(out_dir)

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    images_root = ensure_dir(out_dir / "images")
    labels_root = ensure_dir(out_dir / "labels")

    summary_rows: List[Dict] = []
    split_counts: Dict[str, int] = {}

    logging.info("OCR export: dataset_root=%s", dataset_root)
    logging.info("OCR export: out_dir=%s", out_dir)
    logging.info("OCR export: splits=%s", splits)
    logging.info("OCR export: crop_mode=%s crop_size=%s", crop_mode, crop_size)

    for split_name in splits:
        split_img_dir = ensure_dir(images_root / split_name)
        split_csv_path = labels_root / f"{split_name}.csv"

        rows: List[Dict] = []
        bad_images = 0

        it = iter_ccpd_records(dataset_root, split_dir, image_exts, split=split_name)
        for i, (img_path, ann, actual_split) in enumerate(tqdm(it, desc=f"OCR export {split_name}")):
            if max_images_per_split and i >= max_images_per_split:
                break

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                bad_images += 1
                continue

            try:
                indices = validate_plate_indices(ann.plate_indices)
                text = decode_plate_indices(indices)
                crop = crop_plate(img, ann, crop_mode=crop_mode, out_size=crop_size)
            except Exception:
                bad_images += 1
                continue

            export_stem = _safe_name_from_path(img_path)
            export_path = split_img_dir / f"{export_stem}.jpg"
            cv2.imwrite(str(export_path), crop)

            rows.append(
                {
                    "img_path": str(export_path),
                    "split_requested": split_name,
                    "split_actual": actual_split,
                    "source_img_path": str(img_path),
                    "text": text,
                    "p0": indices[0],
                    "p1": indices[1],
                    "p2": indices[2],
                    "p3": indices[3],
                    "p4": indices[4],
                    "p5": indices[5],
                    "p6": indices[6],
                    "crop_mode": crop_mode,
                    "crop_width": int(crop_size[0]),
                    "crop_height": int(crop_size[1]),
                    "x1": int(ann.x1),
                    "y1": int(ann.y1),
                    "x2": int(ann.x2),
                    "y2": int(ann.y2),
                    "brightness": int(ann.brightness),
                    "blurriness": int(ann.blurriness),
                    "tilt_h": int(ann.tilt_h),
                    "tilt_v": int(ann.tilt_v),
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(split_csv_path, index=False)

        split_counts[split_name] = int(len(df))
        summary_rows.append(
            {
                "split": split_name,
                "n_images": int(len(df)),
                "csv_path": str(split_csv_path),
                "images_dir": str(split_img_dir),
                "bad_images": int(bad_images),
            }
        )

    dump_json(
        {
            "dataset_root": str(Path(dataset_root).resolve() if Path(dataset_root).exists() else dataset_root),
            "split_dir": str(Path(split_dir).resolve() if Path(split_dir).exists() else split_dir),
            "counts": split_counts,
            "crop_mode": crop_mode,
            "crop_size": [int(crop_size[0]), int(crop_size[1])],
            "splits": summary_rows,
        },
        out_dir / "summary.json",
    )

    return out_dir


class OcrManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        image_size: Tuple[int, int],
        augment: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_width = int(image_size[0])
        self.image_height = int(image_size[1])
        self.augment = bool(augment)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        required_cols = ["img_path", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "text"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.df))

    def _apply_augment(self, img_rgb: np.ndarray) -> np.ndarray:
        if not self.augment:
            return img_rgb

        out = img_rgb.astype(np.float32)

        if np.random.rand() < 0.9:
            alpha = np.random.uniform(0.75, 1.25)
            beta = np.random.uniform(-25.0, 25.0)
            out = out * alpha + beta

        out = np.clip(out, 0, 255).astype(np.uint8)

        if np.random.rand() < 0.35:
            k = int(np.random.choice([3, 5]))
            out = cv2.GaussianBlur(out, (k, k), 0)

        if np.random.rand() < 0.25:
            dx = np.random.randint(-8, 9)
            dy = np.random.randint(-4, 5)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            out = cv2.warpAffine(
                out,
                M,
                (out.shape[1], out.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        if np.random.rand() < 0.25:
            h, w = out.shape[:2]
            src = np.float32([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ])
            jitter = np.float32([
                [np.random.uniform(-6, 6), np.random.uniform(-4, 4)],
                [np.random.uniform(-6, 6), np.random.uniform(-4, 4)],
                [np.random.uniform(-6, 6), np.random.uniform(-4, 4)],
                [np.random.uniform(-6, 6), np.random.uniform(-4, 4)],
            ])
            dst = src + jitter
            P = cv2.getPerspectiveTransform(src, dst)
            out = cv2.warpPerspective(
                out,
                P,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        if np.random.rand() < 0.25:
            scale = np.random.uniform(0.5, 0.9)
            h, w = out.shape[:2]
            sw = max(16, int(w * scale))
            sh = max(8, int(h * scale))
            out = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

        if np.random.rand() < 0.20:
            noise = np.random.normal(0, 8, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if np.random.rand() < 0.20:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.random.randint(35, 80))]
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), encode_param)
            if ok:
                out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        return out

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        img_rgb = self._apply_augment(img_rgb)
        x = img_rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x).float()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["img_path"])
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")

        x = self._preprocess(img)
        y = torch.tensor(
            [
                int(row["p0"]),
                int(row["p1"]),
                int(row["p2"]),
                int(row["p3"]),
                int(row["p4"]),
                int(row["p5"]),
                int(row["p6"]),
            ],
            dtype=torch.long,
        )
        text = str(row["text"])
        return x, y, text, str(img_path)