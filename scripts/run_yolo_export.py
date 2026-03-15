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
from dlpd.yolo_dataset import export_ccpd_to_yolo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

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
    logging.info("YOLO dataset export finished: %s", dataset_yaml)


if __name__ == "__main__":
    main()