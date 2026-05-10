from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dlpd.alpr_pipeline import run_alpr_inference
from dlpd.config import load_config
from dlpd.utils import setup_logging


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    if value:
        return Path(value)
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if value:
        return int(value)
    return int(default)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if value:
        return float(value)
    return float(default)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return str(default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end ALPR inference: YOLO detector + OCR recognizer."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("DLPD_CONFIG", "config.yaml"),
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/app/input",
        help="Input image file or directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/app/outputs/alpr_infer",
        help="Directory for inference outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "cpu"),
        help="Device: cpu, 0, cuda:0.",
    )
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=_env_float("DETECTOR_CONF", 0.25),
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--detector-iou",
        type=float,
        default=_env_float("DETECTOR_IOU", 0.70),
        help="YOLO NMS IoU threshold.",
    )
    parser.add_argument(
        "--detector-max-det",
        type=int,
        default=_env_int("DETECTOR_MAX_DET", 10),
        help="Maximum detections per image.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    detector_weights = _env_path("DETECTOR_WEIGHTS", cfg.yolo_eval.weights)
    ocr_weights = _env_path("OCR_WEIGHTS", cfg.ocr_eval.weights)
    detector_imgsz = _env_int("DETECTOR_IMGSZ", cfg.yolo_eval.imgsz)
    device = _env_str("DEVICE", args.device)

    logging.info("ALPR source: %s", args.source)
    logging.info("ALPR out_dir: %s", args.out_dir)
    logging.info("Detector weights: %s", detector_weights)
    logging.info("OCR weights: %s", ocr_weights)
    logging.info("Device: %s", device)

    run_alpr_inference(
        source=Path(args.source),
        detector_weights=detector_weights,
        ocr_weights=ocr_weights,
        out_dir=Path(args.out_dir),
        image_exts=cfg.dataset.image_exts,
        detector_imgsz=detector_imgsz,
        detector_conf=float(args.detector_conf),
        detector_iou=float(args.detector_iou),
        detector_max_det=int(args.detector_max_det),
        device=device,
    )


if __name__ == "__main__":
    main()