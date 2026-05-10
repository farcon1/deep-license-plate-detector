from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dlpd.config import load_config
from dlpd.ocr_train import train_ocr_model
from dlpd.utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    out = train_ocr_model(
        manifest_train=cfg.ocr_train.manifest_train,
        manifest_val=cfg.ocr_train.manifest_val,
        out_dir=cfg.ocr_train.out_dir,
        model_name=cfg.ocr_train.model,
        image_size=cfg.ocr_train.image_size,
        epochs=cfg.ocr_train.epochs,
        batch=cfg.ocr_train.batch,
        lr=cfg.ocr_train.lr,
        weight_decay=cfg.ocr_train.weight_decay,
        device=cfg.ocr_train.device,
        num_workers=cfg.ocr_train.num_workers,
        seed=cfg.ocr_train.seed,
    )
    logging.info("OCR training finished: %s", out)


if __name__ == "__main__":
    main()