from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass(frozen=True)
class DatasetCfg:
    root: Path
    split_dir: Path
    image_exts: List[str]


@dataclass(frozen=True)
class EdaCfg:
    out_dir: Path
    split: str
    max_images: int
    compute_image_metrics_max: int
    gallery_size: int
    seed: int


@dataclass(frozen=True)
class CvBaselineCfg:
    out_dir: Path
    split: str
    max_images: int
    iou_thresholds: List[float]
    pr_points: int
    save_visuals: int
    seed: int


@dataclass(frozen=True)
class LoggingCfg:
    level: str


@dataclass(frozen=True)
class AppCfg:
    dataset: DatasetCfg
    eda: EdaCfg
    cv_baseline: CvBaselineCfg
    logging: LoggingCfg


def load_config(path: str | Path) -> AppCfg:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    ds = raw["dataset"]
    eda = raw["eda"]
    cvb = raw["cv_baseline"]
    lg = raw.get("logging", {"level": "INFO"})

    return AppCfg(
        dataset=DatasetCfg(
            root=Path(ds["root"]),
            split_dir=Path(ds["split_dir"]),
            image_exts=list(ds.get("image_exts", [".jpg", ".jpeg", ".png"])),
        ),
        eda=EdaCfg(
            out_dir=Path(eda["out_dir"]),
            split=str(eda.get("split", "train")),
            max_images=int(eda.get("max_images", 50000)),
            compute_image_metrics_max=int(eda.get("compute_image_metrics_max", 20000)),
            gallery_size=int(eda.get("gallery_size", 36)),
            seed=int(eda.get("seed", 42)),
        ),
        cv_baseline=CvBaselineCfg(
            out_dir=Path(cvb["out_dir"]),
            split=str(cvb.get("split", "test")),
            max_images=int(cvb.get("max_images", 5000)),
            iou_thresholds=[float(x) for x in cvb.get("iou_thresholds", [0.5, 0.7])],
            pr_points=int(cvb.get("pr_points", 101)),
            save_visuals=int(cvb.get("save_visuals", 120)),
            seed=int(cvb.get("seed", 42)),
        ),
        logging=LoggingCfg(level=str(lg.get("level", "INFO"))),
    )
