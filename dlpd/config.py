from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
class YoloExportCfg:
    out_dir: Path
    dataset_name: str
    splits: List[str]
    max_images_per_split: int
    link_mode: str
    overwrite: bool
    seed: int


@dataclass(frozen=True)
class YoloTrainCfg:
    out_dir: Path
    dataset_yaml: Path
    model: str
    epochs: int
    imgsz: int
    batch: int | str
    device: str
    workers: int
    patience: int
    optimizer: str
    lr0: float
    lrf: float
    weight_decay: float
    cache: bool
    pretrained: bool
    amp: bool
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    fliplr: float
    mosaic: float
    mixup: float
    copy_paste: float
    close_mosaic: int
    seed: int
    resume: bool


@dataclass(frozen=True)
class YoloEvalCfg:
    out_dir: Path
    dataset_yaml: Path
    weights: Path
    split: str
    max_images: int
    imgsz: int
    batch: int
    device: str
    conf: float
    iou: float
    max_det: int
    iou_thresholds: List[float]
    pr_points: int
    save_visuals: int
    compare_to_cv_dir: Path
    seed: int


@dataclass(frozen=True)
class OcrExportCfg:
    out_dir: Path
    splits: List[str]
    max_images_per_split: int
    crop_mode: str
    crop_size: Tuple[int, int]
    overwrite: bool
    seed: int


@dataclass(frozen=True)
class OcrTrainCfg:
    manifest_train: Path
    manifest_val: Path
    out_dir: Path
    model: str
    image_size: Tuple[int, int]
    epochs: int
    batch: int
    lr: float
    weight_decay: float
    device: str
    num_workers: int
    seed: int


@dataclass(frozen=True)
class OcrEvalCfg:
    manifest_test: Path
    weights: Path
    out_dir: Path
    image_size: Tuple[int, int]
    batch: int
    device: str
    num_workers: int
    seed: int


@dataclass(frozen=True)
class LoggingCfg:
    level: str


@dataclass(frozen=True)
class AppCfg:
    dataset: DatasetCfg
    eda: EdaCfg
    cv_baseline: CvBaselineCfg
    yolo_export: YoloExportCfg
    yolo_train: YoloTrainCfg
    yolo_eval: YoloEvalCfg
    ocr_export: OcrExportCfg
    ocr_train: OcrTrainCfg
    ocr_eval: OcrEvalCfg
    logging: LoggingCfg


def _load_batch_value(value: int | str) -> int | str:
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "auto":
            return "auto"
        return int(s)
    return int(value)


def _load_size(value, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected image size as [width, height], got: {value}")


def load_config(path: str | Path) -> AppCfg:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    ds = raw["dataset"]
    eda = raw["eda"]
    cvb = raw["cv_baseline"]
    yexp = raw.get("yolo_export", {})
    ytr = raw.get("yolo_train", {})
    yev = raw.get("yolo_eval", {})
    oexp = raw.get("ocr_export", {})
    otr = raw.get("ocr_train", {})
    oev = raw.get("ocr_eval", {})
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
        yolo_export=YoloExportCfg(
            out_dir=Path(yexp.get("out_dir", "outputs/yolo_dataset")),
            dataset_name=str(yexp.get("dataset_name", "ccpd_base_yolo")),
            splits=[str(x).lower() for x in yexp.get("splits", ["train", "val", "test"])],
            max_images_per_split=int(yexp.get("max_images_per_split", 0)),
            link_mode=str(yexp.get("link_mode", "auto")).lower(),
            overwrite=bool(yexp.get("overwrite", False)),
            seed=int(yexp.get("seed", 42)),
        ),
        yolo_train=YoloTrainCfg(
            out_dir=Path(ytr.get("out_dir", "outputs/yolo_train/ccpd_base_yolo11n_960")),
            dataset_yaml=Path(ytr.get("dataset_yaml", "outputs/yolo_dataset/dataset.yaml")),
            model=str(ytr.get("model", "yolo11n.pt")),
            epochs=int(ytr.get("epochs", 50)),
            imgsz=int(ytr.get("imgsz", 960)),
            batch=_load_batch_value(ytr.get("batch", 16)),
            device=str(ytr.get("device", "0")),
            workers=int(ytr.get("workers", 8)),
            patience=int(ytr.get("patience", 20)),
            optimizer=str(ytr.get("optimizer", "auto")),
            lr0=float(ytr.get("lr0", 0.01)),
            lrf=float(ytr.get("lrf", 0.01)),
            weight_decay=float(ytr.get("weight_decay", 0.0005)),
            cache=bool(ytr.get("cache", False)),
            pretrained=bool(ytr.get("pretrained", True)),
            amp=bool(ytr.get("amp", True)),
            hsv_h=float(ytr.get("hsv_h", 0.015)),
            hsv_s=float(ytr.get("hsv_s", 0.7)),
            hsv_v=float(ytr.get("hsv_v", 0.4)),
            degrees=float(ytr.get("degrees", 5.0)),
            translate=float(ytr.get("translate", 0.08)),
            scale=float(ytr.get("scale", 0.35)),
            shear=float(ytr.get("shear", 2.0)),
            perspective=float(ytr.get("perspective", 0.0005)),
            fliplr=float(ytr.get("fliplr", 0.5)),
            mosaic=float(ytr.get("mosaic", 1.0)),
            mixup=float(ytr.get("mixup", 0.0)),
            copy_paste=float(ytr.get("copy_paste", 0.0)),
            close_mosaic=int(ytr.get("close_mosaic", 10)),
            seed=int(ytr.get("seed", 42)),
            resume=bool(ytr.get("resume", False)),
        ),
        yolo_eval=YoloEvalCfg(
            out_dir=Path(yev.get("out_dir", "outputs/yolo_eval")),
            dataset_yaml=Path(yev.get("dataset_yaml", "outputs/yolo_dataset/dataset.yaml")),
            weights=Path(yev.get("weights", "outputs/yolo_train/ccpd_base_yolo11n_960/weights/best.pt")),
            split=str(yev.get("split", "test")),
            max_images=int(yev.get("max_images", 0)),
            imgsz=int(yev.get("imgsz", 960)),
            batch=int(yev.get("batch", 16)),
            device=str(yev.get("device", "0")),
            conf=float(yev.get("conf", 0.001)),
            iou=float(yev.get("iou", 0.7)),
            max_det=int(yev.get("max_det", 10)),
            iou_thresholds=[float(x) for x in yev.get("iou_thresholds", [0.5, 0.7])],
            pr_points=int(yev.get("pr_points", 101)),
            save_visuals=int(yev.get("save_visuals", 200)),
            compare_to_cv_dir=Path(yev.get("compare_to_cv_dir", "outputs/cv_baseline")),
            seed=int(yev.get("seed", 42)),
        ),
        ocr_export=OcrExportCfg(
            out_dir=Path(oexp.get("out_dir", "outputs/ocr_dataset")),
            splits=[str(x).lower() for x in oexp.get("splits", ["train", "val", "test"])],
            max_images_per_split=int(oexp.get("max_images_per_split", 0)),
            crop_mode=str(oexp.get("crop_mode", "bbox")).lower(),
            crop_size=_load_size(oexp.get("crop_size", [224, 64]), default=(224, 64)),
            overwrite=bool(oexp.get("overwrite", True)),
            seed=int(oexp.get("seed", 42)),
        ),
        ocr_train=OcrTrainCfg(
            manifest_train=Path(otr.get("manifest_train", "outputs/ocr_dataset/labels/train.csv")),
            manifest_val=Path(otr.get("manifest_val", "outputs/ocr_dataset/labels/val.csv")),
            out_dir=Path(otr.get("out_dir", "outputs/ocr_train/ccpd_ocr_resnet18")),
            model=str(otr.get("model", "resnet18")),
            image_size=_load_size(otr.get("image_size", [224, 64]), default=(224, 64)),
            epochs=int(otr.get("epochs", 20)),
            batch=int(otr.get("batch", 128)),
            lr=float(otr.get("lr", 0.001)),
            weight_decay=float(otr.get("weight_decay", 0.0001)),
            device=str(otr.get("device", "0")),
            num_workers=int(otr.get("num_workers", 8)),
            seed=int(otr.get("seed", 42)),
        ),
        ocr_eval=OcrEvalCfg(
            manifest_test=Path(oev.get("manifest_test", "outputs/ocr_dataset/labels/test.csv")),
            weights=Path(oev.get("weights", "outputs/ocr_train/ccpd_ocr_resnet18/best.pt")),
            out_dir=Path(oev.get("out_dir", "outputs/ocr_eval")),
            image_size=_load_size(oev.get("image_size", [224, 64]), default=(224, 64)),
            batch=int(oev.get("batch", 256)),
            device=str(oev.get("device", "0")),
            num_workers=int(oev.get("num_workers", 8)),
            seed=int(oev.get("seed", 42)),
        ),
        logging=LoggingCfg(level=str(lg.get("level", "INFO"))),
    )