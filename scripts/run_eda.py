from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dlpd.config import load_config
from dlpd.eda import run_eda
from dlpd.utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    out = run_eda(
        dataset_root=cfg.dataset.root,
        split_dir=cfg.dataset.split_dir,
        image_exts=cfg.dataset.image_exts,
        split=cfg.eda.split,
        out_dir=cfg.eda.out_dir,
        max_images=cfg.eda.max_images,
        compute_image_metrics_max=cfg.eda.compute_image_metrics_max,
        gallery_size=cfg.eda.gallery_size,
        seed=cfg.eda.seed,
    )
    logging.info("EDA finished. See: %s", out)

if __name__ == "__main__":
    main()