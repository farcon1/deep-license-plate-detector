from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dlpd.config import load_config
from dlpd.ocr_dataset import export_ccpd_to_ocr
from dlpd.utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    out = export_ccpd_to_ocr(
        dataset_root=cfg.dataset.root,
        split_dir=cfg.dataset.split_dir,
        image_exts=cfg.dataset.image_exts,
        out_dir=cfg.ocr_export.out_dir,
        splits=cfg.ocr_export.splits,
        max_images_per_split=cfg.ocr_export.max_images_per_split,
        crop_mode=cfg.ocr_export.crop_mode,
        crop_size=cfg.ocr_export.crop_size,
        overwrite=cfg.ocr_export.overwrite,
        seed=cfg.ocr_export.seed,
    )
    logging.info("OCR dataset export finished: %s", out)


if __name__ == "__main__":
    main()