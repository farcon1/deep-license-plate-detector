from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

from .ccpd import iter_ccpd_records
from .utils import ensure_dir, dump_json, seed_everything
from .vis import make_montage, draw_bbox_cv, put_text_cv


def _read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return w, h


def _compute_luminance_and_blur(path: Path) -> Tuple[float, float]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return float("nan"), float("nan")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lum = float(np.mean(gray))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())
    return lum, blur_var


def _plot_hist(series: pd.Series, title: str, xlabel: str, out_path: Path, bins: int = 50) -> None:
    plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    plt.figure()
    plt.scatter(x.values, y.values, s=4, alpha=0.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_heatmap_centers(df: pd.DataFrame, out_path: Path, bins: int = 40) -> None:
    x = df["cx_norm"].dropna().values
    y = df["cy_norm"].dropna().values
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

    plt.figure()
    plt.imshow(H.T, origin="lower", aspect="auto")
    plt.title("BBox centers heatmap (normalized)")
    plt.xlabel("cx_norm")
    plt.ylabel("cy_norm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _build_gallery(
    df: pd.DataFrame,
    rows: List[Dict],
    title: str,
    out_path: Path,
    cols: int = 6,
    cell_size: Tuple[int, int] = (360, 220),
) -> None:
    imgs_rgb: List[np.ndarray] = []
    for r in rows:
        p = Path(r["img_path"])
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gt = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
        img = draw_bbox_cv(img, gt, color=(0, 255, 0), thickness=2)
        txt = f"{p.parent.name} | br={r.get('brightness', 'na')} blurF={r.get('blurriness', 'na')}"
        img = put_text_cv(img, txt, org=(8, 24), color=(255, 255, 255))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs_rgb.append(img_rgb)

    if not imgs_rgb:
        return

    montage = make_montage(imgs_rgb, cols=cols, cell_size=cell_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path, quality=92)


def run_eda(
    dataset_root: Path,
    split_dir: Path,
    image_exts: List[str],
    split: str,
    out_dir: Path,
    max_images: int,
    compute_image_metrics_max: int,
    gallery_size: int,
    seed: int,
) -> Path:
    seed_everything(seed)
    out_dir = ensure_dir(out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    galleries_dir = ensure_dir(out_dir / "galleries")

    logging.info("EDA: dataset_root=%s", dataset_root)
    logging.info("EDA: split=%s, max_images=%s", split, max_images)

    rows: List[Dict] = []

    it = iter_ccpd_records(dataset_root, split_dir, image_exts, split=split)
    for i, (img_path, ann, split_name) in enumerate(tqdm(it, desc="Collect annotations")):
        if max_images and i >= max_images:
            break
        row = {
            "img_path": str(img_path),
            "subset": img_path.parent.name,
            "split": split_name,
            "area_ratio_token": ann.area_ratio,
            "tilt_h": ann.tilt_h,
            "tilt_v": ann.tilt_v,
            "x1": ann.x1,
            "y1": ann.y1,
            "x2": ann.x2,
            "y2": ann.y2,
            "brightness": ann.brightness,
            "blurriness": ann.blurriness,
            "plate_len": len(ann.plate_indices),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No images found. Check dataset.root and dataset structure.")
    df = pd.DataFrame(rows)
    logging.info("EDA: computing image sizes for %d rows", len(df))
    widths = []
    heights = []
    for p in tqdm(df["img_path"].values, desc="Read image sizes"):
        try:
            w, h = _read_image_size(Path(p))
        except Exception:
            w, h = (np.nan, np.nan)
        widths.append(w)
        heights.append(h)
    df["img_w"] = widths
    df["img_h"] = heights

    df["bbox_w"] = (df["x2"] - df["x1"]).clip(lower=0)
    df["bbox_h"] = (df["y2"] - df["y1"]).clip(lower=0)
    df["bbox_area_px"] = df["bbox_w"] * df["bbox_h"]
    df["img_area_px"] = df["img_w"] * df["img_h"]
    df["bbox_area_norm"] = df["bbox_area_px"] / df["img_area_px"]

    df["aspect_ratio"] = df["bbox_w"] / df["bbox_h"].replace(0, np.nan)
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    df["cx_norm"] = df["cx"] / df["img_w"]
    df["cy_norm"] = df["cy"] / df["img_h"]

    df["touches_border"] = (
        (df["x1"] <= 0) |
        (df["y1"] <= 0) |
        (df["x2"] >= df["img_w"]) |
        (df["y2"] >= df["img_h"])
    ).astype(int)

    m = min(int(compute_image_metrics_max), len(df)) if compute_image_metrics_max else len(df)
    df["luminance_mean"] = np.nan
    df["laplacian_var"] = np.nan

    logging.info("EDA: computing pixel metrics for %d images", m)
    idxs = np.random.choice(np.arange(len(df)), size=m, replace=False)
    for j in tqdm(idxs, desc="Compute luminance/blur"):
        p = Path(df.at[j, "img_path"])
        lum, blur_var = _compute_luminance_and_blur(p)
        df.at[j, "luminance_mean"] = lum
        df.at[j, "laplacian_var"] = blur_var
    df_path = out_dir / "data.csv"
    df.to_csv(df_path, index=False)

    stats: Dict[str, object] = {}
    stats["n_rows"] = int(len(df))
    stats["subsets"] = df["subset"].value_counts().to_dict()
    stats["splits"] = df["split"].value_counts().to_dict()

    def ssum(col: str) -> Dict[str, float]:
        x = df[col].dropna().astype(float).values
        if x.size == 0:
            return {"count": 0}
        return {
            "count": float(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p05": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    for col in ["bbox_area_norm", "aspect_ratio", "tilt_h", "tilt_v", "brightness", "blurriness", "luminance_mean", "laplacian_var"]:
        stats[col] = ssum(col)

    stats["touches_border_rate"] = float(df["touches_border"].mean())
    dump_json(stats, out_dir / "stats.json")

    _plot_hist(df["img_w"], "Image width distribution", "img_w", plots_dir / "img_width.png")
    _plot_hist(df["img_h"], "Image height distribution", "img_h", plots_dir / "img_height.png")
    _plot_hist(df["bbox_area_norm"], "BBox area / Image area", "bbox_area_norm", plots_dir / "bbox_area_norm.png")
    _plot_hist(df["aspect_ratio"], "BBox aspect ratio (w/h)", "aspect_ratio", plots_dir / "aspect_ratio.png")
    _plot_hist(df["tilt_h"], "Horizontal tilt (token)", "tilt_h", plots_dir / "tilt_h.png")
    _plot_hist(df["tilt_v"], "Vertical tilt (token)", "tilt_v", plots_dir / "tilt_v.png")
    _plot_hist(df["brightness"], "Filename brightness (LP region)", "brightness", plots_dir / "brightness.png")
    _plot_hist(df["blurriness"], "Filename blurriness (LP region)", "blurriness", plots_dir / "blurriness.png")

    if df["luminance_mean"].notna().any():
        _plot_hist(df["luminance_mean"], "Mean luminance (computed)", "luminance_mean", plots_dir / "luminance_mean.png")
    if df["laplacian_var"].notna().any():
        _plot_hist(df["laplacian_var"], "Laplacian variance (computed blur score)", "laplacian_var", plots_dir / "laplacian_var.png")

    _plot_heatmap_centers(df, plots_dir / "bbox_centers_heatmap.png", bins=44)
    _plot_scatter(df["bbox_area_norm"].fillna(0), df["aspect_ratio"].fillna(0),
                  "Area vs Aspect ratio", "bbox_area_norm", "aspect_ratio",
                  plots_dir / "area_vs_aspect.png")

    g = int(gallery_size)
    g = max(4, g)

    if df["luminance_mean"].notna().any():
        darkest = df.sort_values("luminance_mean", ascending=True).head(g).to_dict("records")
        _build_gallery(df, darkest, "darkest", galleries_dir / "gallery_darkest.jpg")
    else:
        darkest = df.sort_values("brightness", ascending=True).head(g).to_dict("records")
        _build_gallery(df, darkest, "darkest", galleries_dir / "gallery_darkest.jpg")

    if df["laplacian_var"].notna().any():
        blurriest = df.sort_values("laplacian_var", ascending=True).head(g).to_dict("records")
        _build_gallery(df, blurriest, "blurriest", galleries_dir / "gallery_blurriest.jpg")
    else:
        blurriest = df.sort_values("blurriness", ascending=False).head(g).to_dict("records")
        _build_gallery(df, blurriest, "blurriest", galleries_dir / "gallery_blurriest.jpg")

    smallest = df.sort_values("bbox_area_norm", ascending=True).head(g).to_dict("records")
    _build_gallery(df, smallest, "smallest", galleries_dir / "gallery_smallest.jpg")

    border = df[df["touches_border"] == 1].head(g).to_dict("records")
    if border:
        _build_gallery(df, border, "touches_border", galleries_dir / "gallery_touches_border.jpg")

    logging.info("EDA done. Outputs at: %s", out_dir)
    return out_dir
