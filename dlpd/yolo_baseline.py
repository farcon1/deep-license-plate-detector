from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .ccpd import iter_ccpd_records
from .metrics import Box, average_precision, iou, pr_curve_single_object_per_image, summarize_metrics
from .utils import dump_json, ensure_dir, seed_everything
from .vis import draw_bbox_cv, put_text_cv, save_image_bgr


@dataclass
class Prediction:
    box: Optional[Box]
    score: float


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with: pip install -U ultralytics"
        ) from exc
    return YOLO


def _plot_pr(recalls: np.ndarray, precisions: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.plot(recalls, precisions)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_hist_iou(y_iou: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.hist(y_iou, bins=50)
    plt.title(title)
    plt.xlabel("IoU")
    plt.ylabel("count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _clip_box_to_image(box: Box, w: int, h: int) -> Box:
    x1 = float(np.clip(box.x1, 0.0, max(0.0, w - 1.0)))
    y1 = float(np.clip(box.y1, 0.0, max(0.0, h - 1.0)))
    x2 = float(np.clip(box.x2, 0.0, max(0.0, w - 1.0)))
    y2 = float(np.clip(box.y2, 0.0, max(0.0, h - 1.0)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return Box(x1, y1, x2, y2)


def _extract_best_prediction(result) -> Prediction:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return Prediction(box=None, score=-1.0)

    xyxy = _to_numpy(getattr(boxes, "xyxy", None))
    conf = _to_numpy(getattr(boxes, "conf", None))
    cls = _to_numpy(getattr(boxes, "cls", None))

    if xyxy is None or conf is None or len(xyxy) == 0:
        return Prediction(box=None, score=-1.0)

    xyxy = np.asarray(xyxy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    if cls is not None and len(cls) == len(conf):
        cls = np.asarray(cls).reshape(-1)
        keep = cls == 0
        if np.any(keep):
            xyxy = xyxy[keep]
            conf = conf[keep]

    if len(conf) == 0:
        return Prediction(box=None, score=-1.0)

    idx = int(np.argmax(conf))
    x1, y1, x2, y2 = [float(v) for v in xyxy[idx].tolist()]
    return Prediction(box=Box(x1, y1, x2, y2), score=float(conf[idx]))


def train_yolo_detector(
    data_yaml: Path,
    out_dir: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int | str,
    device: str,
    workers: int,
    patience: int,
    optimizer: str,
    lr0: float,
    lrf: float,
    weight_decay: float,
    cache: bool,
    pretrained: bool,
    amp: bool,
    hsv_h: float,
    hsv_s: float,
    hsv_v: float,
    degrees: float,
    translate: float,
    scale: float,
    shear: float,
    perspective: float,
    fliplr: float,
    mosaic: float,
    mixup: float,
    copy_paste: float,
    close_mosaic: int,
    seed: int,
    resume: bool,
) -> Path:
    YOLO = _require_ultralytics()
    seed_everything(seed)
    out_dir = Path(out_dir)
    project = out_dir.parent if out_dir.parent != Path("") else Path(".")
    name = out_dir.name
    ensure_dir(project)

    logging.info("YOLO train: data=%s", data_yaml)
    logging.info("YOLO train: model=%s", model_name)
    logging.info("YOLO train: out_dir=%s", out_dir)

    model = YOLO(model_name)
    kwargs = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": batch,
        "device": device,
        "workers": int(workers),
        "patience": int(patience),
        "optimizer": str(optimizer),
        "lr0": float(lr0),
        "lrf": float(lrf),
        "weight_decay": float(weight_decay),
        "cache": bool(cache),
        "pretrained": bool(pretrained),
        "amp": bool(amp),
        "hsv_h": float(hsv_h),
        "hsv_s": float(hsv_s),
        "hsv_v": float(hsv_v),
        "degrees": float(degrees),
        "translate": float(translate),
        "scale": float(scale),
        "shear": float(shear),
        "perspective": float(perspective),
        "fliplr": float(fliplr),
        "mosaic": float(mosaic),
        "mixup": float(mixup),
        "copy_paste": float(copy_paste),
        "close_mosaic": int(close_mosaic),
        "seed": int(seed),
        "resume": bool(resume),
        "single_cls": True,
        "save": True,
        "plots": True,
        "project": str(project),
        "name": str(name),
        "exist_ok": True,
        "verbose": True,
    }
    results = model.train(**kwargs)

    save_dir = out_dir
    for candidate in (
        getattr(results, "save_dir", None),
        getattr(getattr(model, "trainer", None), "save_dir", None),
    ):
        if candidate:
            save_dir = Path(str(candidate))
            break

    summary = {
        "data_yaml": str(Path(data_yaml).resolve()),
        "model_name": model_name,
        "save_dir": str(save_dir.resolve()),
        "best_weights": str((save_dir / "weights" / "best.pt").resolve()),
        "last_weights": str((save_dir / "weights" / "last.pt").resolve()),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": batch,
        "device": device,
        "seed": int(seed),
    }
    dump_json(summary, save_dir / "train_summary.json")
    return save_dir


def run_builtin_yolo_validation(
    weights: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou_thr: float,
    max_det: int,
    out_dir: Path,
) -> Dict[str, float]:
    YOLO = _require_ultralytics()
    out_dir = ensure_dir(out_dir)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=str(split),
        imgsz=int(imgsz),
        batch=int(batch),
        device=str(device),
        conf=float(conf),
        iou=float(iou_thr),
        max_det=int(max_det),
        plots=True,
        verbose=True,
    )

    out: Dict[str, float] = {}
    if hasattr(metrics, "results_dict"):
        try:
            for k, v in dict(metrics.results_dict).items():
                if isinstance(v, (int, float)):
                    out[str(k)] = float(v)
                elif hasattr(v, "item"):
                    out[str(k)] = float(v.item())
        except Exception:
            pass

    box = getattr(metrics, "box", None)
    if box is not None:
        for key_src, key_dst in (("map", "map50_95"), ("map50", "map50"), ("map75", "map75")):
            if hasattr(box, key_src):
                val = getattr(box, key_src)
                try:
                    out[key_dst] = float(val)
                except Exception:
                    try:
                        out[key_dst] = float(val.item())
                    except Exception:
                        pass

    dump_json(out, Path(out_dir) / "yolo_val_metrics.json")
    return out


def _build_slice_metrics(df: pd.DataFrame, iou_thresholds: List[float]) -> List[Dict[str, float]]:
    if df.empty:
        return []

    rows: List[Dict[str, float]] = []
    q_area_25 = float(df["area_ratio"].quantile(0.25))
    q_area_75 = float(df["area_ratio"].quantile(0.75))
    q_bright_25 = float(df["brightness"].quantile(0.25))
    q_bright_75 = float(df["brightness"].quantile(0.75))
    q_blur_75 = float(df["blurriness"].quantile(0.75))
    q_tilt_h_75 = float(df["tilt_h"].quantile(0.75))
    q_tilt_v_75 = float(df["tilt_v"].quantile(0.75))

    specs = [
        ("all", np.ones(len(df), dtype=bool)),
        ("small_area_q25", df["area_ratio"].values <= q_area_25),
        ("large_area_q75", df["area_ratio"].values >= q_area_75),
        ("dark_q25", df["brightness"].values <= q_bright_25),
        ("bright_q75", df["brightness"].values >= q_bright_75),
        ("blur_q75", df["blurriness"].values >= q_blur_75),
        ("tilt_h_q75", df["tilt_h"].values >= q_tilt_h_75),
        ("tilt_v_q75", df["tilt_v"].values >= q_tilt_v_75),
    ]

    for slice_name, mask in specs:
        part = df.loc[mask]
        if part.empty:
            continue

        row: Dict[str, float] = {
            "slice": slice_name,
            "n": float(len(part)),
            "mean_iou": float(part["iou"].mean()),
            "median_iou": float(part["iou"].median()),
            "mean_score": float(part["score"].mean()),
        }

        for thr in iou_thresholds:
            row[f"success_rate_iou_ge_{thr:.2f}"] = float((part["iou"] >= float(thr)).mean())

        rows.append(row)

    return rows


def _load_cv_comparison(cv_dir: Path, iou_thrs: List[float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    cv_dir = Path(cv_dir)
    if not cv_dir.exists():
        return out

    summary_path = cv_dir / "summary.json"
    if summary_path.exists():
        out["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    for thr in iou_thrs:
        p = cv_dir / f"metrics_iou_{thr:.2f}.json"
        if p.exists():
            out[f"metrics_{thr:.2f}"] = json.loads(p.read_text(encoding="utf-8"))
    return out


def evaluate_yolo_detector(
    dataset_root: Path,
    split_dir: Path,
    image_exts: List[str],
    weights: Path,
    data_yaml: Path,
    out_dir: Path,
    split: str,
    max_images: int,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou_nms: float,
    max_det: int,
    iou_thresholds: List[float],
    pr_points: int,
    save_visuals: int,
    compare_to_cv_dir: Path,
    seed: int,
) -> Path:
    YOLO = _require_ultralytics()
    seed_everything(seed)
    out_dir = ensure_dir(out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    visuals_dir = ensure_dir(out_dir / "visuals")

    builtin_metrics = run_builtin_yolo_validation(
        weights=weights,
        data_yaml=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou_thr=iou_nms,
        max_det=max_det,
        out_dir=out_dir,
    )

    model = YOLO(str(weights))

    gt_boxes: List[Box] = []
    pred_boxes: List[Box] = []
    pred_scores: List[float] = []
    img_paths: List[Path] = []
    split_names: List[str] = []
    meta_rows: List[Dict] = []
    n_read_fail = 0
    n_no_pred = 0

    it = iter_ccpd_records(dataset_root, split_dir, image_exts, split=split)
    for i, (img_path, ann, split_name) in enumerate(tqdm(it, desc="YOLO eval")):
        if max_images and i >= max_images:
            break
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            n_read_fail += 1
            continue
        h, w = img.shape[:2]
        gt = _clip_box_to_image(Box(float(ann.x1), float(ann.y1), float(ann.x2), float(ann.y2)), w=w, h=h)

        results = model.predict(
            source=str(img_path),
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou_nms),
            device=str(device),
            max_det=int(max_det),
            verbose=False,
        )
        pred = _extract_best_prediction(results[0])
        if pred.box is None:
            pb = Box(0.0, 0.0, 0.0, 0.0)
            sc = -1.0
            n_no_pred += 1
        else:
            pb = _clip_box_to_image(pred.box, w=w, h=h)
            sc = float(pred.score)

        gt_boxes.append(gt)
        pred_boxes.append(pb)
        pred_scores.append(sc)
        img_paths.append(Path(img_path))
        split_names.append(split_name)
        meta_rows.append(
            {
                "img_path": str(img_path),
                "split": split_name,
                "img_w": int(w),
                "img_h": int(h),
                "area_ratio": float(ann.area_ratio),
                "tilt_h": int(ann.tilt_h),
                "tilt_v": int(ann.tilt_v),
                "brightness": int(ann.brightness),
                "blurriness": int(ann.blurriness),
                "gt_x1": float(gt.x1),
                "gt_y1": float(gt.y1),
                "gt_x2": float(gt.x2),
                "gt_y2": float(gt.y2),
                "pred_x1": float(pb.x1),
                "pred_y1": float(pb.y1),
                "pred_x2": float(pb.x2),
                "pred_y2": float(pb.y2),
                "score": float(sc),
            }
        )

    if not gt_boxes:
        raise RuntimeError("YOLO evaluation processed 0 images. Check dataset.root and split configuration.")

    y_iou = np.array([iou(a, b) for a, b in zip(gt_boxes, pred_boxes)], dtype=np.float64)
    y_score = np.array(pred_scores, dtype=np.float64)

    df = pd.DataFrame(meta_rows)
    df["iou"] = y_iou
    df.to_csv(out_dir / "predictions.csv", index=False)

    for thr in iou_thresholds:
        ts, ps, rs = pr_curve_single_object_per_image(y_iou, y_score, iou_thr=float(thr), points=pr_points)
        apv = average_precision(rs, ps)
        summ = summarize_metrics(y_iou, y_score, iou_thr=float(thr), pr_points=pr_points)
        summ["ap_check"] = float(apv)
        dump_json(summ, out_dir / f"metrics_iou_{thr:.2f}.json")

        pr_df = pd.DataFrame({"score_thr": ts, "precision": ps, "recall": rs})
        pr_df.to_csv(out_dir / f"pr_iou_{thr:.2f}.csv", index=False)
        _plot_pr(rs, ps, plots_dir / f"pr_curve_iou_{thr:.2f}.png", title=f"YOLO PR curve (IoU >= {thr:.2f})")

    _plot_hist_iou(y_iou, plots_dir / "iou_hist_all.png", title="YOLO IoU histogram (all images)")

    pred_mask_any = y_score >= 0.0
    pred_rate = float(np.mean(pred_mask_any))
    mean_score_pred_only = float(np.mean(y_score[pred_mask_any])) if np.any(pred_mask_any) else 0.0

    strict_thr = max(iou_thresholds) if iou_thresholds else 0.7
    summary = {
        "n_images": int(len(gt_boxes)),
        "pred_rate": pred_rate,
        "mean_score_pred_only": mean_score_pred_only,
        "mean_iou": float(np.mean(y_iou)),
        "median_iou": float(np.median(y_iou)),
        "strict_thr": float(strict_thr),
        "strict_success_rate": float(np.mean(y_iou >= strict_thr)),
        "n_read_fail": int(n_read_fail),
        "n_no_predictions": int(n_no_pred),
        "weights": str(Path(weights).resolve()),
        "split": split,
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou_nms": float(iou_nms),
        "max_det": int(max_det),
        "builtin_val": builtin_metrics,
    }
    dump_json(summary, out_dir / "summary.json")

    slice_rows = _build_slice_metrics(df, iou_thresholds=iou_thresholds)
    pd.DataFrame(slice_rows).to_csv(out_dir / "slice_metrics.csv", index=False)
    dump_json({"rows": slice_rows}, out_dir / "slice_metrics.json")

    cv_metrics = _load_cv_comparison(compare_to_cv_dir, iou_thrs=iou_thresholds)
    if cv_metrics:
        compare: Dict[str, float] = {}
        cv_summary = cv_metrics.get("summary", {})
        if cv_summary:
            compare["delta_mean_iou_vs_cv"] = float(summary["mean_iou"] - float(cv_summary.get("mean_iou", 0.0)))
            compare["delta_strict_success_vs_cv"] = float(summary["strict_success_rate"] - float(cv_summary.get("strict_success_rate", 0.0)))
        for thr in iou_thresholds:
            this_path = out_dir / f"metrics_iou_{thr:.2f}.json"
            if this_path.exists() and f"metrics_{thr:.2f}" in cv_metrics:
                this_m = json.loads(this_path.read_text(encoding="utf-8"))
                cv_m = cv_metrics[f"metrics_{thr:.2f}"]
                compare[f"delta_success_rate_iou_{thr:.2f}"] = float(
                    float(this_m.get("success_rate_iou_ge_thr", 0.0)) - float(cv_m.get("success_rate_iou_ge_thr", 0.0))
                )
                compare[f"delta_ap_iou_{thr:.2f}"] = float(float(this_m.get("ap", 0.0)) - float(cv_m.get("ap", 0.0)))
        dump_json(compare, out_dir / "compare_vs_cv.json")

    save_n = int(save_visuals)
    rng = np.random.default_rng(seed)
    fail_idx = np.where(y_iou < strict_thr)[0]
    ok_idx = np.where(y_iou >= strict_thr)[0]

    n_fail = min(len(fail_idx), max(0, save_n // 2))
    n_ok = min(len(ok_idx), max(0, save_n - n_fail))
    fail_pick = rng.choice(fail_idx, size=n_fail, replace=False) if n_fail > 0 else np.array([], dtype=int)
    ok_pick = rng.choice(ok_idx, size=n_ok, replace=False) if n_ok > 0 else np.array([], dtype=int)
    picks = np.concatenate([fail_pick, ok_pick]) if (fail_pick.size + ok_pick.size) else np.array([], dtype=int)
    if picks.size > 0:
        rng.shuffle(picks)

    for k in tqdm(picks, desc="Save YOLO visuals"):
        p = img_paths[int(k)]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gt = gt_boxes[int(k)]
        pr = pred_boxes[int(k)]
        sc = y_score[int(k)]
        ii = y_iou[int(k)]
        sp = split_names[int(k)]

        img2 = draw_bbox_cv(img, (int(gt.x1), int(gt.y1), int(gt.x2), int(gt.y2)), color=(0, 255, 0), thickness=2)
        if sc >= 0:
            img2 = draw_bbox_cv(img2, (int(pr.x1), int(pr.y1), int(pr.x2), int(pr.y2)), color=(0, 165, 255), thickness=2)

        txt = f"split={sp} | iou={ii:.3f} | score={sc:.3f}"
        img2 = put_text_cv(img2, txt, org=(8, 24), color=(255, 255, 255))
        stem = f"{ii:.3f}_{sc:.3f}_{p.stem}"
        save_image_bgr(visuals_dir / f"{stem}.jpg", img2)

    logging.info("YOLO evaluation done. Outputs at: %s", out_dir)
    return out_dir