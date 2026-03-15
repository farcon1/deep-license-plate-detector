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
from dlpd.yolo_baseline import evaluate_yolo_detector, train_yolo_detector
from dlpd.yolo_dataset import export_ccpd_to_yolo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    dataset_yaml = cfg.yolo_train.dataset_yaml
    if not args.skip_export:
        dataset_yaml = export_ccpd_to_yolo(
            dataset_root=cfg.dataset.root,
            split_dir=cfg.dataset.split_dir,
            image_exts=cfg.dataset.image_exts,
            out_dir=cfg.yolo_export.out_dir,
            dataset_name=cfg.yolo_export.dataset_name,
            splits=cfg.yolo_export.splits,
            max_images_per_split=cfg.yolo_export.max_images_per_split,
            link_mode=cfg.yolo_export.link_mode,
            overwrite=cfg.yolo_export.overwrite,
            seed=cfg.yolo_export.seed,
        )
        logging.info("Export done: %s", dataset_yaml)

    weights_path = cfg.yolo_eval.weights
    if not args.skip_train:
        save_dir = train_yolo_detector(
            data_yaml=dataset_yaml,
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
        weights_path = save_dir / "weights" / "best.pt"
        logging.info("Train done: %s", save_dir)

    if not args.skip_eval:
        out = evaluate_yolo_detector(
            dataset_root=cfg.dataset.root,
            split_dir=cfg.dataset.split_dir,
            image_exts=cfg.dataset.image_exts,
            weights=weights_path,
            data_yaml=dataset_yaml,
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
        logging.info("Eval done: %s", out)


if __name__ == "__main__":
    main()