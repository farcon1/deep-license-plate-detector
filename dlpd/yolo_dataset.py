from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from .ccpd import CCPDAnnotation, iter_ccpd_records
from .utils import dump_json, ensure_dir, seed_everything


def _read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return int(im.width), int(im.height)


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


def _ann_to_yolo_line(ann: CCPDAnnotation, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = _clip_box(ann.x1, ann.y1, ann.x2, ann.y2, img_w, img_h)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"0 {cx / img_w:.8f} {cy / img_h:.8f} {bw / img_w:.8f} {bh / img_h:.8f}"


def _safe_name_from_path(img_path: Path) -> str:
    parts = list(img_path.parts)
    if len(parts) > 3:
        tail = parts[-3:]
    else:
        tail = parts
    stem = "__".join(tail[:-1] + [img_path.stem])
    stem = stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return stem


def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"

    tried: List[str]
    if mode == "auto":
        tried = ["hardlink", "copy"]
    else:
        tried = [mode]

    last_error: Exception | None = None
    for current in tried:
        try:
            if current == "hardlink":
                os.link(src, dst)
                return "hardlink"
            if current == "symlink":
                os.symlink(src, dst)
                return "symlink"
            if current == "copy":
                shutil.copy2(src, dst)
                return "copy"
            raise ValueError(f"Unsupported link mode: {current}")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to materialize image {src} -> {dst}: {last_error}")


def export_ccpd_to_yolo(
    dataset_root: Path,
    split_dir: Path,
    image_exts: List[str],
    out_dir: Path,
    dataset_name: str,
    splits: List[str],
    max_images_per_split: int,
    link_mode: str,
    overwrite: bool,
    seed: int,
) -> Path:
    seed_everything(seed)
    out_dir = Path(out_dir)

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    images_root = ensure_dir(out_dir / "images")
    labels_root = ensure_dir(out_dir / "labels")
    manifests_root = ensure_dir(out_dir / "manifests")

    summary_rows: List[Dict] = []
    split_counts: Dict[str, int] = {}
    split_modes: Dict[str, str] = {}

    logging.info("YOLO export: dataset_root=%s", dataset_root)
    logging.info("YOLO export: out_dir=%s", out_dir)
    logging.info("YOLO export: splits=%s", splits)

    for split_name in splits:
        logging.info("[export_ccpd_to_yolo] START split=%s", split_name)

        img_out = ensure_dir(images_root / split_name)
        lbl_out = ensure_dir(labels_root / split_name)
        rows: List[Dict] = []
        used_mode = "unknown"

        it = iter_ccpd_records(dataset_root, split_dir, image_exts, split=split_name)

        for i, (img_path, ann, actual_split) in enumerate(tqdm(it, desc=f"Export {split_name}")):
            if i == 0:
                logging.info(
                    "[export_ccpd_to_yolo] FIRST RECORD split=%s img=%s actual_split=%s",
                    split_name,
                    img_path,
                    actual_split,
                )

            if max_images_per_split and i >= max_images_per_split:
                logging.info(
                    "[export_ccpd_to_yolo] STOP by max_images_per_split split=%s i=%d",
                    split_name,
                    i,
                )
                break

            img_w, img_h = _read_image_size(img_path)
            export_stem = _safe_name_from_path(img_path)
            dst_img = img_out / f"{export_stem}{img_path.suffix.lower()}"
            dst_lbl = lbl_out / f"{export_stem}.txt"

            used_mode = _link_or_copy(img_path, dst_img, link_mode)

            if i == 0:
                logging.info(
                    "[export_ccpd_to_yolo] FIRST MATERIALIZED split=%s mode=%s dst_img=%s dst_lbl=%s",
                    split_name,
                    used_mode,
                    dst_img,
                    dst_lbl,
                )

            dst_lbl.write_text(_ann_to_yolo_line(ann, img_w, img_h) + "\n", encoding="utf-8")

            rows.append(
                {
                    "split_requested": split_name,
                    "split_actual": actual_split,
                    "src_img_path": str(img_path),
                    "dst_img_path": str(dst_img),
                    "dst_lbl_path": str(dst_lbl),
                    "img_w": int(img_w),
                    "img_h": int(img_h),
                    "x1": int(ann.x1),
                    "y1": int(ann.y1),
                    "x2": int(ann.x2),
                    "y2": int(ann.y2),
                    "area_ratio": float(ann.area_ratio),
                    "tilt_h": int(ann.tilt_h),
                    "tilt_v": int(ann.tilt_v),
                    "brightness": int(ann.brightness),
                    "blurriness": int(ann.blurriness),
                }
            )

            if (i + 1) % 5000 == 0:
                logging.info(
                    "[export_ccpd_to_yolo] PROGRESS split=%s exported=%d",
                    split_name,
                    i + 1,
                )

        if not rows:
            logging.warning("YOLO export: split=%s produced no rows.", split_name)
            continue

        df = pd.DataFrame(rows)
        df.to_csv(manifests_root / f"{split_name}.csv", index=False)

        split_counts[split_name] = int(len(df))
        split_modes[split_name] = used_mode
        summary_rows.append(
            {
                "split": split_name,
                "n_images": int(len(df)),
                "materialization": used_mode,
                "images_dir": str(img_out),
                "labels_dir": str(lbl_out),
            }
        )

        logging.info(
            "[export_ccpd_to_yolo] END split=%s n_images=%d materialization=%s",
            split_name,
            len(df),
            used_mode,
        )

    if not split_counts:
        raise RuntimeError("YOLO export produced 0 images. Check dataset.root and split configuration.")

    dataset_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "license_plate"},
    }
    (out_dir / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    dump_json(
        {
            "dataset_name": dataset_name,
            "dataset_root": str(Path(dataset_root).resolve() if Path(dataset_root).exists() else dataset_root),
            "split_dir": str(Path(split_dir).resolve() if Path(split_dir).exists() else split_dir),
            "counts": split_counts,
            "materialization": split_modes,
            "splits": summary_rows,
            "dataset_yaml": str((out_dir / "dataset.yaml").resolve()),
        },
        out_dir / "summary.json",
    )

    return out_dir / "dataset.yaml"