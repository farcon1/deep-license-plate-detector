from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Optional, Iterable, Set

from .utils import read_text_lines


@dataclass(frozen=True)
class CCPDAnnotation:
    area_ratio: float
    tilt_h: int
    tilt_v: int
    x1: int
    y1: int
    x2: int
    y2: int
    corners: List[Tuple[int, int]]
    plate_indices: List[int]
    brightness: int
    blurriness: int


def _parse_xy(s: str) -> Tuple[int, int]:
    a, b = s.split("&")
    return int(a), int(b)


def parse_ccpd_filename(img_path: str | Path) -> CCPDAnnotation:
    """
    CCPD annotation is embedded in filename as 7 fields separated by '-':
    Area - Tilt - BBox - 4 points - Plate indices - Brightness - Blurriness

    Example:
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    """
    p = Path(img_path)
    stem = p.stem
    parts = stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"CCPD filename does not have 7 fields: {p.name}")

    area_s, tilt_s, bbox_s, corners_s, plate_s, bright_s, blur_s = parts[:7]
    area_ratio = float(int(area_s)) / 1000.0
    tilt_h_s, tilt_v_s = tilt_s.split("_")
    tilt_h, tilt_v = int(tilt_h_s), int(tilt_v_s)

    lu_s, rb_s = bbox_s.split("_")
    x1, y1 = _parse_xy(lu_s)
    x2, y2 = _parse_xy(rb_s)

    corner_tokens = corners_s.split("_")
    if len(corner_tokens) != 4:
        raise ValueError(f"CCPD corners must have 4 points: {p.name}")
    corners = [_parse_xy(t) for t in corner_tokens]

    plate_indices = [int(x) for x in plate_s.split("_")]
    brightness = int(bright_s)
    blurriness = int(blur_s)

    return CCPDAnnotation(area_ratio=area_ratio, tilt_h=tilt_h, tilt_v=tilt_v, x1=x1, y1=y1, x2=x2, y2=y2,
        corners=corners, plate_indices=plate_indices, brightness=brightness, blurriness=blurriness)


def _exts_set(exts: List[str]) -> Set[str]:
    return {e.lower() for e in exts}


def _quick_has_any_image(root: Path, exts: List[str]) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    exts_l = _exts_set(exts)
    for e in exts_l:
        for _ in root.rglob(f"*{e}"):
            return True
    return False


def _first_ccpd_like_image(root: Path, exts: List[str], limit: int = 5000) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    exts_l = _exts_set(exts)
    tried = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts_l:
            continue
        tried += 1
        try:
            _ = parse_ccpd_filename(p)
            return p
        except Exception:
            pass
        if tried >= limit:
            break
    return None


def resolve_ccpd_train_root(user_root: Path, exts: List[str]) -> Path:
    cwd = Path.cwd()
    anchors = []
    anchors.append(user_root)
    if not user_root.is_absolute():
        anchors.append(cwd / user_root)
    anchors.append(cwd / "data")
    anchors.append(cwd / "data" / "CCPD2019")
    anchors.append(cwd / "data" / "CCPD2019" / "CCPD2019")
    try:
        anchors.append(user_root.parent)
        anchors.append(user_root.parent.parent)
    except Exception:
        pass
    uniq_anchors: List[Path] = []
    seen: Set[str] = set()
    for a in anchors:
        key = str(a.resolve()) if a.exists() else str(a)
        if key not in seen:
            seen.add(key)
            uniq_anchors.append(a)
    preferred_rel = [
        Path("ccpd_base") / "train",
        Path("CCPD2019") / "ccpd_base" / "train",
        Path("CCPD2019") / "CCPD2019" / "ccpd_base" / "train",
    ]
    for base in uniq_anchors:
        for rel in preferred_rel:
            cand = base / rel
            if _quick_has_any_image(cand, exts):
                logging.info("Resolved CCPD root to: %s", cand)
                return cand
    search_roots = []
    if (cwd / "data").exists():
        search_roots.append(cwd / "data")
    if user_root.exists() and user_root.is_dir():
        search_roots.append(user_root)
    for sr in search_roots:
        for p in sr.rglob("train"):
            if p.is_dir() and p.parent.name.lower() == "ccpd_base":
                if _quick_has_any_image(p, exts) and _first_ccpd_like_image(p, exts, limit=200):
                    logging.info("Resolved CCPD root by search to: %s", p)
                    return p
    data_root = cwd / "data"
    if data_root.exists():
        any_img = _first_ccpd_like_image(data_root, exts, limit=20000)
        if any_img is not None:
            logging.info("Resolved CCPD root by CCPD-like filename to: %s", any_img.parent)
            return any_img.parent

    name = user_root.name.lower()
    parent = user_root.parent.name.lower() if user_root.parent else ""
    looks_specific = (
        name in {"train", "val", "test"}
        or "ccpd" in name
        or "ccpd" in parent
        or parent == "ccpd_base"
    )
    if looks_specific and _quick_has_any_image(user_root, exts) and _first_ccpd_like_image(user_root, exts, limit=500):
        logging.info("Resolved CCPD root to (user_root): %s", user_root)
        return user_root
    logging.warning("Could not resolve CCPD root automatically. Using: %s", user_root)
    return user_root



def find_images(root: Path, exts: List[str]) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    exts_l = _exts_set(exts)
    images: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_l:
            images.append(p)
    images.sort()
    return images


def load_splits(split_dir: Path) -> Dict[str, List[str]]:
    splits: Dict[str, List[str]] = {}
    if not split_dir.exists():
        return splits
    for txt in split_dir.glob("*.txt"):
        key = txt.stem.strip().lower()
        lines = read_text_lines(txt)
        splits[key] = lines
    return splits


def resolve_split_items(dataset_root: Path, items: List[str]) -> List[Path]:
    out: List[Path] = []
    for it in items:
        s = it.strip().lstrip("./")
        p = Path(s)
        if p.is_absolute() and p.exists():
            out.append(p)
            continue
        cand = dataset_root / p
        if cand.exists():
            out.append(cand)
            continue
        matches = list(dataset_root.rglob(p.name))
        if matches:
            out.append(matches[0])
            continue
    return out


def iter_ccpd_records(
    dataset_root: Path,
    split_dir: Path,
    exts: List[str],
    split: str = "auto",
) -> Iterator[Tuple[Path, CCPDAnnotation, str]]:

    resolved_root = resolve_ccpd_train_root(dataset_root, exts)
    dataset_root = resolved_root

    splits = load_splits(split_dir)
    split_l = split.lower()

    def safe_parse(img: Path) -> Optional[CCPDAnnotation]:
        try:
            return parse_ccpd_filename(img)
        except Exception:
            return None

    bad = 0
    bad_logged = 0

    def log_bad(img: Path) -> None:
        nonlocal bad_logged
        if bad_logged < 20:
            logging.warning("Skip non-CCPD image (bad filename format): %s", img.name)
            bad_logged += 1

    if split_l == "all" or not splits:
        for img in find_images(dataset_root, exts):
            ann = safe_parse(img)
            if ann is None:
                bad += 1
                log_bad(img)
                continue
            yield img, ann, "all"
        if bad:
            logging.info("Skipped %d non-CCPD images under %s", bad, dataset_root)
        return

    if split_l == "auto":
        preferred = [k for k in ("train", "val", "test") if k in splits]
        if not preferred:
            preferred = [next(iter(splits.keys()))]

        for k in preferred:
            paths = resolve_split_items(dataset_root, splits[k])
            for img in paths:
                ann = safe_parse(img)
                if ann is None:
                    bad += 1
                    log_bad(img)
                    continue
                yield img, ann, k

        if bad:
            logging.info("Skipped %d non-CCPD images under %s", bad, dataset_root)
        return

    if split_l not in splits:
        for img in find_images(dataset_root, exts):
            ann = safe_parse(img)
            if ann is None:
                bad += 1
                log_bad(img)
                continue
            yield img, ann, "all"
        if bad:
            logging.info("Skipped %d non-CCPD images under %s", bad, dataset_root)
        return

    paths = resolve_split_items(dataset_root, splits[split_l])
    for img in paths:
        ann = safe_parse(img)
        if ann is None:
            bad += 1
            log_bad(img)
            continue
        yield img, ann, split_l

    if bad:
        logging.info("Skipped %d non-CCPD images under %s", bad, dataset_root)

