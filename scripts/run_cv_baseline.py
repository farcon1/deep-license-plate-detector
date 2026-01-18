from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dlpd.config import load_config
from dlpd.ccpd import iter_ccpd_records
from dlpd.cv_baseline import CVPlateDetector
from dlpd.metrics import Box, iou, pr_curve_single_object_per_image, average_precision, summarize_metrics
from dlpd.utils import ensure_dir, dump_json, seed_everything, setup_logging
from dlpd.vis import draw_bbox_cv, put_text_cv, save_image_bgr


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    seed_everything(cfg.cv_baseline.seed)

    out_dir = ensure_dir(cfg.cv_baseline.out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    visuals_dir = ensure_dir(out_dir / "visuals")

    detector = CVPlateDetector()

    split = cfg.cv_baseline.split
    max_images = cfg.cv_baseline.max_images
    iou_thrs = cfg.cv_baseline.iou_thresholds
    pr_points = cfg.cv_baseline.pr_points

    logging.info("CV baseline: dataset_root=%s", cfg.dataset.root)
    logging.info("CV baseline: split=%s, max_images=%s", split, max_images)

    gt_boxes: List[Box] = []
    pred_boxes: List[Box] = []
    pred_scores: List[float] = []
    img_paths: List[Path] = []
    split_names: List[str] = []

    n_read_fail = 0
    n_no_candidates = 0

    it = iter_ccpd_records(cfg.dataset.root, cfg.dataset.split_dir, cfg.dataset.image_exts, split=split)
    for i, (img_path, ann, split_name) in enumerate(tqdm(it, desc="CV baseline detect")):
        if max_images and i >= max_images:
            break
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            n_read_fail += 1
            continue
        h, w = img.shape[:2]
        gt = Box(float(ann.x1), float(ann.y1), float(ann.x2), float(ann.y2))
        gt = _clip_box_to_image(gt, w=w, h=h)
        det = detector.detect(img)
        if det.box is None:
            pb = Box(0.0, 0.0, 0.0, 0.0)
            sc = -1.0  # важно: исключаем из PR при t>=0
            n_no_candidates += 1
        else:
            pb = _clip_box_to_image(det.box, w=w, h=h)
            sc = float(det.score)

        gt_boxes.append(gt)
        pred_boxes.append(pb)
        pred_scores.append(sc)
        img_paths.append(Path(img_path))
        split_names.append(split_name)

    n = len(gt_boxes)
    if n == 0:
        raise RuntimeError("No images processed. Check dataset path/split.")
    if n_read_fail:
        logging.warning("cv2.imread failed for %d images (skipped).", n_read_fail)
    logging.info("No-candidate detections: %d / %d", n_no_candidates, n)

    y_iou = np.array([iou(a, b) for a, b in zip(gt_boxes, pred_boxes)], dtype=np.float64)
    y_score = np.array(pred_scores, dtype=np.float64)

    save_n = int(cfg.cv_baseline.save_visuals)
    rng = np.random.default_rng(cfg.cv_baseline.seed)

    strict_thr = max(iou_thrs) if iou_thrs else 0.7

    fail_idx = np.where(y_iou < strict_thr)[0]
    ok_idx = np.where(y_iou >= strict_thr)[0]

    n_fail = min(len(fail_idx), max(0, save_n // 2))
    n_ok = min(len(ok_idx), max(0, save_n - n_fail))

    fail_pick = rng.choice(fail_idx, size=n_fail, replace=False) if n_fail > 0 else np.array([], dtype=int)
    ok_pick = rng.choice(ok_idx, size=n_ok, replace=False) if n_ok > 0 else np.array([], dtype=int)

    picks = np.concatenate([fail_pick, ok_pick]) if (fail_pick.size + ok_pick.size) else np.array([], dtype=int)
    if picks.size > 0:
        rng.shuffle(picks)

    for k in tqdm(picks, desc="Save visuals"):
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
            img2 = draw_bbox_cv(img2, (int(pr.x1), int(pr.y1), int(pr.x2), int(pr.y2)), color=(0, 0, 255), thickness=2)
            txt = f"IoU={ii:.3f} score={sc:.3f} split={sp}"
        else:
            txt = f"IoU={ii:.3f} score=NO_DET split={sp}"

        img2 = put_text_cv(img2, txt, org=(8, 24), color=(255, 255, 255))

        tag = "FAIL" if ii < strict_thr else "OK"
        outp = visuals_dir / f"{tag}_{k:06d}_{p.name}"
        save_image_bgr(outp, img2)

    for thr in iou_thrs:
        ts, ps, rs = pr_curve_single_object_per_image(y_iou, y_score, iou_thr=float(thr), points=pr_points)
        apv = average_precision(rs, ps)
        summ = summarize_metrics(y_iou, y_score, iou_thr=float(thr), pr_points=pr_points)
        summ["ap_check"] = float(apv)

        dump_json(summ, out_dir / f"metrics_iou_{thr:.2f}.json")

        pr_df = pd.DataFrame({"score_thr": ts, "precision": ps, "recall": rs})
        pr_df.to_csv(out_dir / f"pr_iou_{thr:.2f}.csv", index=False)

        _plot_pr(rs, ps, plots_dir / f"pr_curve_iou_{thr:.2f}.png", title=f"PR curve (IoU >= {thr:.2f})")

    _plot_hist_iou(y_iou, plots_dir / "iou_hist_all.png", title="IoU histogram (all images)")

    pred_mask_any = y_score >= 0.0
    pred_rate = float(np.mean(pred_mask_any))
    mean_score_pred_only = float(np.mean(y_score[pred_mask_any])) if np.any(pred_mask_any) else 0.0

    dump_json(
        {
            "n_images": int(n),
            "pred_rate": pred_rate,
            "mean_score_pred_only": mean_score_pred_only,
            "mean_iou": float(np.mean(y_iou)),
            "median_iou": float(np.median(y_iou)),
            "strict_thr": float(strict_thr),
            "strict_success_rate": float(np.mean(y_iou >= strict_thr)),
            "n_read_fail": int(n_read_fail),
            "n_no_candidates": int(n_no_candidates),
        },
        out_dir/"summary.json",
    )
    logging.info("CV baseline done. Outputs at: %s", out_dir)

if __name__ == "__main__":
    main()
