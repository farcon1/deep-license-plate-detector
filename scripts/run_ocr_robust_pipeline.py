from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dlpd.ccpd import CCPDAnnotation, iter_ccpd_records
from dlpd.config import load_config
from dlpd.ocr_dataset import crop_plate_from_bbox, crop_plate_from_corners
from dlpd.ocr_train import evaluate_ocr_model, train_ocr_model
from dlpd.plate_text import decode_plate_indices, validate_plate_indices
from dlpd.utils import dump_json, ensure_dir, seed_everything, setup_logging
from dlpd.yolo_baseline import _clip_box_to_image, _extract_best_prediction, _require_ultralytics


@dataclass
class RobustExportStats:
    total_seen: Dict[str, int] = field(default_factory=dict)
    total_exported: Dict[str, int] = field(default_factory=dict)
    bad_images: Dict[str, int] = field(default_factory=dict)
    bad_labels: Dict[str, int] = field(default_factory=dict)
    crop_errors: Dict[str, int] = field(default_factory=dict)
    yolo_misses: Dict[str, int] = field(default_factory=dict)
    yolo_errors: Dict[str, int] = field(default_factory=dict)

    def inc(self, bucket: str, key: str, value: int = 1) -> None:
        target = getattr(self, bucket)
        target[key] = int(target.get(key, 0)) + int(value)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "total_seen": self.total_seen,
            "total_exported": self.total_exported,
            "bad_images": self.bad_images,
            "bad_labels": self.bad_labels,
            "crop_errors": self.crop_errors,
            "yolo_misses": self.yolo_misses,
            "yolo_errors": self.yolo_errors,
        }


def _safe_name_from_path(img_path: Path) -> str:
    parts = list(Path(img_path).parts)
    tail = parts[-3:] if len(parts) > 3 else parts
    stem = "__".join(tail[:-1] + [Path(img_path).stem])
    stem = stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
    stem = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in stem)
    return stem


def _variant_token(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(".", "p")
        .replace("+", "plus")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def _resize_crop(crop_bgr: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = int(out_size[0]), int(out_size[1])
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty crop.")
    return cv2.resize(crop_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def _clip_xyxy_float(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    x1_i = int(round(max(0.0, min(float(x1), float(w - 1)))))
    y1_i = int(round(max(0.0, min(float(y1), float(h - 1)))))
    x2_i = int(round(max(0.0, min(float(x2), float(w)))))
    y2_i = int(round(max(0.0, min(float(y2), float(h)))))

    if x2_i < x1_i:
        x1_i, x2_i = x2_i, x1_i
    if y2_i < y1_i:
        y1_i, y2_i = y2_i, y1_i

    return x1_i, y1_i, x2_i, y2_i


def crop_plate_from_bbox_pad(
    img_bgr: np.ndarray,
    ann: CCPDAnnotation,
    out_size: Tuple[int, int],
    pad_ratio: float,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    x1 = float(ann.x1)
    y1 = float(ann.y1)
    x2 = float(ann.x2)
    y2 = float(ann.y2)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    px = bw * float(pad_ratio)
    py = bh * float(pad_ratio)

    cx1, cy1, cx2, cy2 = _clip_xyxy_float(
        x1 - px,
        y1 - py,
        x2 + px,
        y2 + py,
        w=w,
        h=h,
    )

    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("Invalid padded bbox crop.")

    crop = img_bgr[cy1:cy2, cx1:cx2]
    return _resize_crop(crop, out_size)


def crop_plate_from_bbox_jitter(
    img_bgr: np.ndarray,
    ann: CCPDAnnotation,
    out_size: Tuple[int, int],
    rng: np.random.Generator,
    max_shift_ratio: float = 0.08,
    max_scale_delta: float = 0.18,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    x1 = float(ann.x1)
    y1 = float(ann.y1)
    x2 = float(ann.x2)
    y2 = float(ann.y2)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    shift_x = float(rng.uniform(-max_shift_ratio, max_shift_ratio)) * bw
    shift_y = float(rng.uniform(-max_shift_ratio, max_shift_ratio)) * bh
    scale_x = 1.0 + float(rng.uniform(-max_scale_delta, max_scale_delta))
    scale_y = 1.0 + float(rng.uniform(-max_scale_delta, max_scale_delta))

    scale_x = max(0.72, scale_x)
    scale_y = max(0.72, scale_y)

    new_bw = bw * scale_x
    new_bh = bh * scale_y
    new_cx = cx + shift_x
    new_cy = cy + shift_y

    jx1 = new_cx - new_bw / 2.0
    jy1 = new_cy - new_bh / 2.0
    jx2 = new_cx + new_bw / 2.0
    jy2 = new_cy + new_bh / 2.0

    cx1, cy1, cx2, cy2 = _clip_xyxy_float(jx1, jy1, jx2, jy2, w=w, h=h)

    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("Invalid jitter bbox crop.")

    crop = img_bgr[cy1:cy2, cx1:cx2]
    return _resize_crop(crop, out_size)


def crop_plate_from_yolo_prediction(
    img_bgr: np.ndarray,
    detector: Any,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    out_size: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], float, Optional[List[int]]]:
    h, w = img_bgr.shape[:2]

    results = detector.predict(
        source=img_bgr,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        device=str(device),
        max_det=int(max_det),
        verbose=False,
    )

    pred = _extract_best_prediction(results[0])

    if pred.box is None or float(pred.score) < 0:
        return None, 0.0, None

    clipped = _clip_box_to_image(pred.box, w=w, h=h)
    x1, y1, x2, y2 = int(clipped.x1), int(clipped.y1), int(clipped.x2), int(clipped.y2)

    if x2 <= x1 or y2 <= y1:
        return None, float(pred.score), [x1, y1, x2, y2]

    crop = img_bgr[y1:y2, x1:x2].copy()
    crop = _resize_crop(crop, out_size)

    return crop, float(pred.score), [x1, y1, x2, y2]


def _make_row(
    export_path: Path,
    split_requested: str,
    split_actual: str,
    source_img_path: Path,
    text: str,
    indices: List[int],
    variant: str,
    crop_source: str,
    yolo_score: float = 0.0,
    yolo_bbox: Optional[List[int]] = None,
) -> Dict[str, Any]:
    return {
        "img_path": str(export_path),
        "split_requested": split_requested,
        "split_actual": split_actual,
        "source_img_path": str(source_img_path),
        "text": text,
        "p0": int(indices[0]),
        "p1": int(indices[1]),
        "p2": int(indices[2]),
        "p3": int(indices[3]),
        "p4": int(indices[4]),
        "p5": int(indices[5]),
        "p6": int(indices[6]),
        "variant": str(variant),
        "crop_source": str(crop_source),
        "yolo_score": float(yolo_score),
        "yolo_bbox": json.dumps(yolo_bbox, ensure_ascii=False) if yolo_bbox else "",
    }


def _write_crop(
    images_root: Path,
    manifest_name: str,
    img_path: Path,
    variant: str,
    crop_bgr: np.ndarray,
    split_requested: str,
    split_actual: str,
    text: str,
    indices: List[int],
    crop_source: str,
    yolo_score: float = 0.0,
    yolo_bbox: Optional[List[int]] = None,
) -> Dict[str, Any]:
    variant_safe = _variant_token(variant)
    safe_stem = _safe_name_from_path(img_path)
    export_dir = ensure_dir(images_root / manifest_name)
    export_path = export_dir / f"{safe_stem}__{variant_safe}.jpg"

    ok = cv2.imwrite(str(export_path), crop_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {export_path}")

    return _make_row(
        export_path=export_path,
        split_requested=split_requested,
        split_actual=split_actual,
        source_img_path=img_path,
        text=text,
        indices=indices,
        variant=variant,
        crop_source=crop_source,
        yolo_score=yolo_score,
        yolo_bbox=yolo_bbox,
    )


def _resolve_yolo_weights(cli_weights: str, cfg_weights: Path) -> Optional[Path]:
    candidates: List[Path] = []

    if cli_weights.strip():
        candidates.append(Path(cli_weights.strip()))

    candidates.extend(
        [
            Path(cfg_weights),
            Path("models/detector/best.pt"),
            Path("runs/detect/outputs/yolo_train/ccpd_base_yolo11n_960/weights/best.pt"),
            Path("runs/mlflow/166549915104436569/5cbca5312bdf417081637a7b9b81302b/artifacts/weights/best.pt"),
        ]
    )

    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)

        if p.exists() and p.is_file():
            return p

    return None


def _load_yolo_detector(weights: Optional[Path]) -> Optional[Any]:
    if weights is None:
        logging.warning("YOLO weights were not found. YOLO-crop test manifest will be skipped.")
        return None

    YOLO = _require_ultralytics()
    logging.info("Loading YOLO detector for robust OCR export: %s", weights)
    return YOLO(str(weights))


def _crop_by_variant(
    img_bgr: np.ndarray,
    ann: CCPDAnnotation,
    variant: str,
    out_size: Tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    if variant == "corners":
        return crop_plate_from_corners(img_bgr, ann, out_size=out_size)

    if variant == "bbox":
        return crop_plate_from_bbox(img_bgr, ann, out_size=out_size)

    if variant == "bbox_pad_0.08":
        return crop_plate_from_bbox_pad(img_bgr, ann, out_size=out_size, pad_ratio=0.08)

    if variant == "bbox_pad_0.12":
        return crop_plate_from_bbox_pad(img_bgr, ann, out_size=out_size, pad_ratio=0.12)

    if variant == "bbox_pad_0.15":
        return crop_plate_from_bbox_pad(img_bgr, ann, out_size=out_size, pad_ratio=0.15)

    if variant.startswith("bbox_jitter"):
        return crop_plate_from_bbox_jitter(
            img_bgr,
            ann,
            out_size=out_size,
            rng=rng,
            max_shift_ratio=0.08,
            max_scale_delta=0.18,
        )

    raise ValueError(f"Unknown crop variant: {variant}")


def export_robust_ocr_dataset(
    dataset_root: Path,
    split_dir: Path,
    image_exts: List[str],
    out_dir: Path,
    out_size: Tuple[int, int],
    splits: List[str],
    max_images_per_split: int,
    overwrite: bool,
    seed: int,
    yolo_detector: Optional[Any],
    yolo_device: str,
    yolo_imgsz: int,
    yolo_conf: float,
    yolo_iou: float,
    yolo_max_det: int,
) -> Dict[str, Path]:
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    out_dir = Path(out_dir)
    images_root = out_dir / "images"
    labels_root = out_dir / "labels"

    if overwrite and out_dir.exists():
        logging.warning("Removing existing robust OCR dataset directory: %s", out_dir)
        shutil.rmtree(out_dir)

    ensure_dir(images_root)
    ensure_dir(labels_root)

    manifests: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test_robust": [],
        "test_corners": [],
        "test_bbox_pad": [],
        "test_yolo_crop": [],
    }

    stats = RobustExportStats()

    train_variants = [
        "corners",
        "bbox",
        "bbox_pad_0.08",
        "bbox_pad_0.12",
        "bbox_pad_0.15",
        "bbox_jitter_01",
        "bbox_jitter_02",
    ]

    val_variants = [
        "corners",
        "bbox",
        "bbox_pad_0.12",
        "bbox_jitter_01",
    ]

    test_robust_variants = [
        "corners",
        "bbox",
        "bbox_pad_0.08",
        "bbox_pad_0.12",
        "bbox_pad_0.15",
        "bbox_jitter_01",
    ]

    logging.info("Robust OCR export started.")
    logging.info("dataset_root=%s", dataset_root)
    logging.info("split_dir=%s", split_dir)
    logging.info("out_dir=%s", out_dir)
    logging.info("out_size=%s", out_size)
    logging.info("splits=%s", splits)

    for split_name in splits:
        split_l = split_name.lower().strip()
        it = iter_ccpd_records(dataset_root, split_dir, image_exts, split=split_l)

        for i, (img_path, ann, actual_split) in enumerate(tqdm(it, desc=f"Robust OCR export {split_l}")):
            if max_images_per_split and i >= max_images_per_split:
                break

            stats.inc("total_seen", split_l)

            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                stats.inc("bad_images", split_l)
                continue

            try:
                indices = validate_plate_indices(ann.plate_indices)
                text = decode_plate_indices(indices)
            except Exception:
                stats.inc("bad_labels", split_l)
                continue

            if split_l == "train":
                target_variants = train_variants
                target_manifest = "train"
            elif split_l == "val":
                target_variants = val_variants
                target_manifest = "val"
            elif split_l == "test":
                target_variants = test_robust_variants
                target_manifest = "test_robust"
            else:
                target_variants = test_robust_variants
                target_manifest = f"{split_l}_robust"

            for variant in target_variants:
                try:
                    crop = _crop_by_variant(
                        img_bgr=img_bgr,
                        ann=ann,
                        variant=variant,
                        out_size=out_size,
                        rng=rng,
                    )
                    row = _write_crop(
                        images_root=images_root,
                        manifest_name=target_manifest,
                        img_path=img_path,
                        variant=variant,
                        crop_bgr=crop,
                        split_requested=split_l,
                        split_actual=actual_split,
                        text=text,
                        indices=indices,
                        crop_source="gt",
                    )
                    manifests.setdefault(target_manifest, []).append(row)
                    stats.inc("total_exported", target_manifest)
                except Exception:
                    stats.inc("crop_errors", f"{split_l}:{variant}")
                    continue

            if split_l == "test":
                try:
                    crop = _crop_by_variant(
                        img_bgr=img_bgr,
                        ann=ann,
                        variant="corners",
                        out_size=out_size,
                        rng=rng,
                    )
                    row = _write_crop(
                        images_root=images_root,
                        manifest_name="test_corners",
                        img_path=img_path,
                        variant="corners",
                        crop_bgr=crop,
                        split_requested=split_l,
                        split_actual=actual_split,
                        text=text,
                        indices=indices,
                        crop_source="gt_corners",
                    )
                    manifests["test_corners"].append(row)
                    stats.inc("total_exported", "test_corners")
                except Exception:
                    stats.inc("crop_errors", "test:corners_eval")

                try:
                    crop = _crop_by_variant(
                        img_bgr=img_bgr,
                        ann=ann,
                        variant="bbox_pad_0.12",
                        out_size=out_size,
                        rng=rng,
                    )
                    row = _write_crop(
                        images_root=images_root,
                        manifest_name="test_bbox_pad",
                        img_path=img_path,
                        variant="bbox_pad_0.12",
                        crop_bgr=crop,
                        split_requested=split_l,
                        split_actual=actual_split,
                        text=text,
                        indices=indices,
                        crop_source="gt_bbox_pad",
                    )
                    manifests["test_bbox_pad"].append(row)
                    stats.inc("total_exported", "test_bbox_pad")
                except Exception:
                    stats.inc("crop_errors", "test:bbox_pad_eval")

                if yolo_detector is not None:
                    try:
                        yolo_crop, yolo_score, yolo_bbox = crop_plate_from_yolo_prediction(
                            img_bgr=img_bgr,
                            detector=yolo_detector,
                            device=yolo_device,
                            imgsz=yolo_imgsz,
                            conf=yolo_conf,
                            iou=yolo_iou,
                            max_det=yolo_max_det,
                            out_size=out_size,
                        )

                        if yolo_crop is None:
                            stats.inc("yolo_misses", "test")
                        else:
                            row = _write_crop(
                                images_root=images_root,
                                manifest_name="test_yolo_crop",
                                img_path=img_path,
                                variant="yolo_crop",
                                crop_bgr=yolo_crop,
                                split_requested=split_l,
                                split_actual=actual_split,
                                text=text,
                                indices=indices,
                                crop_source="yolo_pred",
                                yolo_score=yolo_score,
                                yolo_bbox=yolo_bbox,
                            )
                            manifests["test_yolo_crop"].append(row)
                            stats.inc("total_exported", "test_yolo_crop")
                    except Exception as exc:
                        logging.debug("YOLO crop export error for %s: %s", img_path, exc)
                        stats.inc("yolo_errors", "test")

    manifest_paths: Dict[str, Path] = {}

    for manifest_name, rows in manifests.items():
        path = labels_root / f"{manifest_name}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        manifest_paths[manifest_name] = path
        logging.info("Manifest saved: %s rows=%d", path, len(df))

    summary = {
        "dataset_root": str(Path(dataset_root).resolve() if Path(dataset_root).exists() else dataset_root),
        "split_dir": str(Path(split_dir).resolve() if Path(split_dir).exists() else split_dir),
        "out_dir": str(out_dir.resolve()),
        "crop_size": [int(out_size[0]), int(out_size[1])],
        "splits": splits,
        "max_images_per_split": int(max_images_per_split),
        "manifest_paths": {k: str(v) for k, v in manifest_paths.items()},
        "stats": stats.to_dict(),
        "train_variants": train_variants,
        "val_variants": val_variants,
        "test_robust_variants": test_robust_variants,
        "yolo_test_enabled": yolo_detector is not None,
        "yolo_export_settings": {
            "device": str(yolo_device),
            "imgsz": int(yolo_imgsz),
            "conf": float(yolo_conf),
            "iou": float(yolo_iou),
            "max_det": int(yolo_max_det),
        },
    }

    dump_json(summary, out_dir / "summary.json")

    return manifest_paths


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_manifest_if_available(
    manifest_path: Path,
    weights: Path,
    out_dir: Path,
    image_size: Tuple[int, int],
    batch: int,
    device: str,
    num_workers: int,
    seed: int,
) -> Optional[Path]:
    if not manifest_path.exists():
        logging.warning("Evaluation manifest does not exist: %s", manifest_path)
        return None

    try:
        df = pd.read_csv(manifest_path)
    except Exception:
        logging.warning("Could not read manifest: %s", manifest_path)
        return None

    if df.empty:
        logging.warning("Evaluation manifest is empty, skipping: %s", manifest_path)
        return None

    return evaluate_ocr_model(
        manifest_test=manifest_path,
        weights=weights,
        out_dir=out_dir,
        image_size=image_size,
        batch=batch,
        device=device,
        num_workers=num_workers,
        seed=seed,
    )


def _write_final_report(
    eval_out_dir: Path,
    dataset_out_dir: Path,
    train_out_dir: Path,
    eval_dirs: Dict[str, Optional[Path]],
) -> None:
    rows: List[Dict[str, Any]] = []

    for name, path in eval_dirs.items():
        if path is None:
            rows.append(
                {
                    "eval": name,
                    "status": "skipped",
                    "n_samples": 0,
                    "loss": "",
                    "char_accuracy": "",
                    "full_plate_accuracy": "",
                    "mean_plate_confidence": "",
                    "summary_path": "",
                }
            )
            continue

        summary_path = path / "summary.json"
        summary = _read_json(summary_path)
        rows.append(
            {
                "eval": name,
                "status": "ok" if summary else "no_summary",
                "n_samples": summary.get("n_samples", 0),
                "loss": summary.get("loss", ""),
                "char_accuracy": summary.get("char_accuracy", ""),
                "full_plate_accuracy": summary.get("full_plate_accuracy", ""),
                "mean_plate_confidence": summary.get("mean_plate_confidence", ""),
                "summary_path": str(summary_path),
            }
        )

    ensure_dir(eval_out_dir)

    final_summary = {
        "dataset_out_dir": str(dataset_out_dir),
        "train_out_dir": str(train_out_dir),
        "eval_out_dir": str(eval_out_dir),
        "rows": rows,
    }

    dump_json(final_summary, eval_out_dir / "summary_all.json")

    df = pd.DataFrame(rows)
    df.to_csv(eval_out_dir / "summary_all.csv", index=False)

    markdown = "\n".join(
        [
            "# Robust OCR Pipeline Report",
            "",
            f"- dataset_out_dir: `{dataset_out_dir}`",
            f"- train_out_dir: `{train_out_dir}`",
            f"- eval_out_dir: `{eval_out_dir}`",
            "",
            "## Evaluation summary",
            "",
            df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- `corners`: качество OCR при идеальном перспективном crop по GT-углам.",
            "- `bbox_pad`: качество OCR при прямоугольном GT bbox с padding.",
            "- `yolo_crop`: реальное качество OCR на crop-ах, которые выдаёт YOLO-детектор.",
            "",
            "Главная метрика для web-инференса — `full_plate_accuracy` на `yolo_crop`.",
            "",
        ]
    )

    (eval_out_dir / "report_all.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export robust OCR dataset, train OCR model, and evaluate separately "
            "on corners, bbox_pad and YOLO-crop test manifests."
        )
    )

    parser.add_argument("--config", type=str, default="config.yaml")

    parser.add_argument("--out-dir", type=str, default="outputs/ocr_dataset_robust")
    parser.add_argument("--train-out-dir", type=str, default="outputs/ocr_train/ccpd_ocr_resnet34_robust_320x96")
    parser.add_argument("--eval-out-dir", type=str, default="outputs/ocr_eval_robust/ccpd_ocr_resnet34_robust_320x96")

    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=96)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-images-per-split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument("--yolo-weights", type=str, default="")
    parser.add_argument("--yolo-device", type=str, default="")
    parser.add_argument("--yolo-imgsz", type=int, default=960)
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--yolo-iou", type=float, default=0.70)
    parser.add_argument("--yolo-max-det", type=int, default=10)

    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    t0 = time.perf_counter()

    out_dir = Path(args.out_dir)
    train_out_dir = Path(args.train_out_dir)
    eval_out_dir = Path(args.eval_out_dir)
    image_size = (int(args.image_width), int(args.image_height))

    seed_everything(args.seed)

    yolo_weights = _resolve_yolo_weights(args.yolo_weights, cfg.yolo_eval.weights)
    yolo_detector = None

    if not args.skip_export:
        yolo_detector = _load_yolo_detector(yolo_weights)

    yolo_device = args.yolo_device.strip() if args.yolo_device.strip() else str(cfg.yolo_eval.device)

    manifest_paths = {
        "train": out_dir / "labels" / "train.csv",
        "val": out_dir / "labels" / "val.csv",
        "test_corners": out_dir / "labels" / "test_corners.csv",
        "test_bbox_pad": out_dir / "labels" / "test_bbox_pad.csv",
        "test_yolo_crop": out_dir / "labels" / "test_yolo_crop.csv",
    }

    if not args.skip_export:
        manifest_paths = export_robust_ocr_dataset(
            dataset_root=cfg.dataset.root,
            split_dir=cfg.dataset.split_dir,
            image_exts=cfg.dataset.image_exts,
            out_dir=out_dir,
            out_size=image_size,
            splits=["train", "val", "test"],
            max_images_per_split=int(args.max_images_per_split),
            overwrite=bool(args.overwrite),
            seed=int(args.seed),
            yolo_detector=yolo_detector,
            yolo_device=yolo_device,
            yolo_imgsz=int(args.yolo_imgsz),
            yolo_conf=float(args.yolo_conf),
            yolo_iou=float(args.yolo_iou),
            yolo_max_det=int(args.yolo_max_det),
        )

    if not args.skip_train:
        train_manifest = manifest_paths["train"]
        val_manifest = manifest_paths["val"]

        if not train_manifest.exists():
            raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
        if not val_manifest.exists():
            raise FileNotFoundError(f"Val manifest not found: {val_manifest}")

        logging.info("Starting robust OCR training.")
        logging.info("train_manifest=%s", train_manifest)
        logging.info("val_manifest=%s", val_manifest)
        logging.info("train_out_dir=%s", train_out_dir)
        logging.info("model=%s image_size=%s epochs=%d batch=%d", args.model, image_size, args.epochs, args.batch)

        train_ocr_model(
            manifest_train=train_manifest,
            manifest_val=val_manifest,
            out_dir=train_out_dir,
            model_name=str(args.model),
            image_size=image_size,
            epochs=int(args.epochs),
            batch=int(args.batch),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=str(args.device),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
        )

    best_weights = train_out_dir / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(
            f"OCR best weights not found: {best_weights}. "
            f"Run training first or check --train-out-dir."
        )

    eval_dirs: Dict[str, Optional[Path]] = {
        "corners": None,
        "bbox_pad": None,
        "yolo_crop": None,
    }

    if not args.skip_eval:
        logging.info("Starting separate robust OCR evaluations.")

        eval_dirs["corners"] = _evaluate_manifest_if_available(
            manifest_path=manifest_paths["test_corners"],
            weights=best_weights,
            out_dir=eval_out_dir / "corners",
            image_size=image_size,
            batch=int(args.batch),
            device=str(args.device),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
        )

        eval_dirs["bbox_pad"] = _evaluate_manifest_if_available(
            manifest_path=manifest_paths["test_bbox_pad"],
            weights=best_weights,
            out_dir=eval_out_dir / "bbox_pad",
            image_size=image_size,
            batch=int(args.batch),
            device=str(args.device),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
        )

        eval_dirs["yolo_crop"] = _evaluate_manifest_if_available(
            manifest_path=manifest_paths["test_yolo_crop"],
            weights=best_weights,
            out_dir=eval_out_dir / "yolo_crop",
            image_size=image_size,
            batch=int(args.batch),
            device=str(args.device),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
        )

        _write_final_report(
            eval_out_dir=eval_out_dir,
            dataset_out_dir=out_dir,
            train_out_dir=train_out_dir,
            eval_dirs=eval_dirs,
        )

    elapsed = time.perf_counter() - t0
    logging.info("Robust OCR pipeline finished in %.2f seconds.", elapsed)
    logging.info("Dataset: %s", out_dir)
    logging.info("Training output: %s", train_out_dir)
    logging.info("Evaluation output: %s", eval_out_dir)
    logging.info("Best weights: %s", best_weights)


if __name__ == "__main__":
    main()