from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dlpd.config import load_config
from dlpd.utils import setup_logging
from dlpd.yolo_baseline import evaluate_yolo_detector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    out = evaluate_yolo_detector(
        dataset_root=cfg.dataset.root,
        split_dir=cfg.dataset.split_dir,
        image_exts=cfg.dataset.image_exts,
        weights=cfg.yolo_eval.weights,
        data_yaml=cfg.yolo_eval.dataset_yaml,
        out_dir=cfg.yolo_eval.out_dir,
        split=cfg.yolo_eval.split,
        max_images=cfg.yolo_eval.max_images,
        imgsz=cfg.yolo_eval.imgsz,
        batch=cfg.yolo_eval.batch,
        device=cfg.yolo_eval.device,
        conf=cfg.yolo_eval.conf,
        iou_nms=cfg.yolo_eval.iou,
        max_det=cfg.yolo_eval.max_det,
        iou_thresholds=cfg.yolo_eval.iou_thresholds,
        pr_points=cfg.yolo_eval.pr_points,
        save_visuals=cfg.yolo_eval.save_visuals,
        compare_to_cv_dir=cfg.yolo_eval.compare_to_cv_dir,
        seed=cfg.yolo_eval.seed,
    )
    logging.info("YOLO evaluation finished: %s", out)


if __name__ == "__main__":
    main()