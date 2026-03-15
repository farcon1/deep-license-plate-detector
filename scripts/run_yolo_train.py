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
from dlpd.yolo_baseline import train_yolo_detector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    save_dir = train_yolo_detector(
        data_yaml=cfg.yolo_train.dataset_yaml,
        out_dir=cfg.yolo_train.out_dir,
        model_name=cfg.yolo_train.model,
        epochs=cfg.yolo_train.epochs,
        imgsz=cfg.yolo_train.imgsz,
        batch=cfg.yolo_train.batch,
        device=cfg.yolo_train.device,
        workers=cfg.yolo_train.workers,
        patience=cfg.yolo_train.patience,
        optimizer=cfg.yolo_train.optimizer,
        lr0=cfg.yolo_train.lr0,
        lrf=cfg.yolo_train.lrf,
        weight_decay=cfg.yolo_train.weight_decay,
        cache=cfg.yolo_train.cache,
        pretrained=cfg.yolo_train.pretrained,
        amp=cfg.yolo_train.amp,
        hsv_h=cfg.yolo_train.hsv_h,
        hsv_s=cfg.yolo_train.hsv_s,
        hsv_v=cfg.yolo_train.hsv_v,
        degrees=cfg.yolo_train.degrees,
        translate=cfg.yolo_train.translate,
        scale=cfg.yolo_train.scale,
        shear=cfg.yolo_train.shear,
        perspective=cfg.yolo_train.perspective,
        fliplr=cfg.yolo_train.fliplr,
        mosaic=cfg.yolo_train.mosaic,
        mixup=cfg.yolo_train.mixup,
        copy_paste=cfg.yolo_train.copy_paste,
        close_mosaic=cfg.yolo_train.close_mosaic,
        seed=cfg.yolo_train.seed,
        resume=cfg.yolo_train.resume,
    )
    logging.info("YOLO training finished: %s", save_dir)


if __name__ == "__main__":
    main()