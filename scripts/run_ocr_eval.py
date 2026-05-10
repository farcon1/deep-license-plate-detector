from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dlpd.config import load_config
from dlpd.ocr_train import evaluate_ocr_model
from dlpd.utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    out = evaluate_ocr_model(
        manifest_test=cfg.ocr_eval.manifest_test,
        weights=cfg.ocr_eval.weights,
        out_dir=cfg.ocr_eval.out_dir,
        image_size=cfg.ocr_eval.image_size,
        batch=cfg.ocr_eval.batch,
        device=cfg.ocr_eval.device,
        num_workers=cfg.ocr_eval.num_workers,
        seed=cfg.ocr_eval.seed,
    )
    logging.info("OCR evaluation finished: %s", out)


if __name__ == "__main__":
    main()