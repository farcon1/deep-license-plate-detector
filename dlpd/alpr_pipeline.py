from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import Box
from .ocr_infer import load_ocr_model, recognize_plate_crop
from .utils import dump_json, ensure_dir
from .vis import draw_bbox_cv, put_text_cv, save_image_bgr
from .yolo_baseline import _clip_box_to_image, _extract_best_prediction, _require_ultralytics


def _collect_images(source: Path, exts: List[str]) -> List[Path]:
    source = Path(source)
    exts_l = {e.lower() for e in exts}
    if source.is_file():
        return [source] if source.suffix.lower() in exts_l else []
    if source.is_dir():
        out = []
        for p in source.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts_l:
                out.append(p)
        out.sort()
        return out
    return []


def run_alpr_inference(
    source: Path,
    detector_weights: Path,
    ocr_weights: Path,
    out_dir: Path,
    image_exts: List[str],
    detector_imgsz: int,
    detector_conf: float,
    detector_iou: float,
    detector_max_det: int,
    device: str,
) -> Path:
    out_dir = ensure_dir(out_dir)
    visuals_dir = ensure_dir(out_dir / "visuals")

    images = _collect_images(source, exts=image_exts)
    if not images:
        raise RuntimeError(f"No input images found in source: {source}")

    YOLO = _require_ultralytics()
    detector = YOLO(str(detector_weights))
    ocr_model, ocr_device, ocr_image_size = load_ocr_model(ocr_weights, device=device)

    rows: List[Dict] = []

    for img_path in tqdm(images, desc="ALPR infer"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            rows.append(
                {
                    "img_path": str(img_path),
                    "detected": 0,
                    "det_score": 0.0,
                    "pred_text": "",
                    "plate_confidence": 0.0,
                    "x1": 0,
                    "y1": 0,
                    "x2": 0,
                    "y2": 0,
                }
            )
            continue

        h, w = img.shape[:2]

        results = detector.predict(
            source=str(img_path),
            imgsz=int(detector_imgsz),
            conf=float(detector_conf),
            iou=float(detector_iou),
            device=str(device),
            max_det=int(detector_max_det),
            verbose=False,
        )
        pred = _extract_best_prediction(results[0])

        img_vis = img.copy()

        if pred.box is None or pred.score < 0:
            stem = img_path.stem
            save_image_bgr(visuals_dir / f"{stem}.jpg", img_vis)
            rows.append(
                {
                    "img_path": str(img_path),
                    "detected": 0,
                    "det_score": 0.0,
                    "pred_text": "",
                    "plate_confidence": 0.0,
                    "x1": 0,
                    "y1": 0,
                    "x2": 0,
                    "y2": 0,
                }
            )
            continue

        pb = _clip_box_to_image(pred.box, w=w, h=h)
        x1, y1, x2, y2 = int(pb.x1), int(pb.y1), int(pb.x2), int(pb.y2)

        if x2 <= x1 or y2 <= y1:
            stem = img_path.stem
            save_image_bgr(visuals_dir / f"{stem}.jpg", img_vis)
            rows.append(
                {
                    "img_path": str(img_path),
                    "detected": 0,
                    "det_score": float(pred.score),
                    "pred_text": "",
                    "plate_confidence": 0.0,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            continue

        crop = img[y1:y2, x1:x2]
        ocr_pred = recognize_plate_crop(crop, model=ocr_model, device=ocr_device, image_size=ocr_image_size)

        img_vis = draw_bbox_cv(img_vis, (x1, y1, x2, y2), color=(0, 255, 0), thickness=2)
        label = f"{ocr_pred.text} | det={pred.score:.3f} | ocr={ocr_pred.plate_confidence:.3f}"
        img_vis = put_text_cv(img_vis, label, org=(8, 24), color=(255, 255, 255))

        stem = img_path.stem
        save_image_bgr(visuals_dir / f"{stem}.jpg", img_vis)

        rows.append(
            {
                "img_path": str(img_path),
                "detected": 1,
                "det_score": float(pred.score),
                "pred_text": str(ocr_pred.text),
                "plate_confidence": float(ocr_pred.plate_confidence),
                "char_conf_0": float(ocr_pred.char_confidences[0]),
                "char_conf_1": float(ocr_pred.char_confidences[1]),
                "char_conf_2": float(ocr_pred.char_confidences[2]),
                "char_conf_3": float(ocr_pred.char_confidences[3]),
                "char_conf_4": float(ocr_pred.char_confidences[4]),
                "char_conf_5": float(ocr_pred.char_confidences[5]),
                "char_conf_6": float(ocr_pred.char_confidences[6]),
                "p0": int(ocr_pred.indices[0]),
                "p1": int(ocr_pred.indices[1]),
                "p2": int(ocr_pred.indices[2]),
                "p3": int(ocr_pred.indices[3]),
                "p4": int(ocr_pred.indices[4]),
                "p5": int(ocr_pred.indices[5]),
                "p6": int(ocr_pred.indices[6]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)

    summary = {
        "source": str(Path(source).resolve() if Path(source).exists() else source),
        "n_images": int(len(df)),
        "n_detected": int(df["detected"].sum()) if len(df) else 0,
        "detection_rate": float(df["detected"].mean()) if len(df) else 0.0,
        "mean_det_score": float(df.loc[df["detected"] == 1, "det_score"].mean()) if (df["detected"] == 1).any() else 0.0,
        "mean_plate_confidence": float(df.loc[df["detected"] == 1, "plate_confidence"].mean()) if (df["detected"] == 1).any() else 0.0,
        "detector_weights": str(Path(detector_weights).resolve()),
        "ocr_weights": str(Path(ocr_weights).resolve()),
    }
    dump_json(summary, out_dir / "summary.json")

    top_df = df[["img_path", "detected", "det_score", "pred_text", "plate_confidence"]].head(30)
    report_md = "\n".join(
        [
            "# ALPR Inference Report",
            "",
            f"- n_images: `{summary['n_images']}`",
            f"- n_detected: `{summary['n_detected']}`",
            f"- detection_rate: `{summary['detection_rate']:.6f}`",
            f"- mean_det_score: `{summary['mean_det_score']:.6f}`",
            f"- mean_plate_confidence: `{summary['mean_plate_confidence']:.6f}`",
            "",
            "## Sample predictions",
            "",
            top_df.to_markdown(index=False),
            "",
        ]
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    logging.info("ALPR inference done. Outputs at: %s", out_dir)
    return out_dir