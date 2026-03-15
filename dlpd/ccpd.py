from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple
import time
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

    return CCPDAnnotation(
        area_ratio=area_ratio,
        tilt_h=tilt_h,
        tilt_v=tilt_v,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        corners=corners,
        plate_indices=plate_indices,
        brightness=brightness,
        blurriness=blurriness,
    )


def _exts_set(exts: List[str]) -> Set[str]:
    return {e.lower() for e in exts}


def _quick_has_any_image(root: Path, exts: List[str]) -> bool:
    t0 = time.perf_counter()
    logging.info("[_quick_has_any_image] START root=%s", root)

    if not root.exists() or not root.is_dir():
        logging.info("[_quick_has_any_image] root missing or not dir: %s", root)
        return False

    exts_l = _exts_set(exts)
    scanned = 0

    for p in root.rglob("*"):
        scanned += 1
        if scanned % 50000 == 0:
            dt = time.perf_counter() - t0
            logging.info(
                "[_quick_has_any_image] progress root=%s scanned=%d elapsed=%.2fs",
                root, scanned, dt
            )

        if p.is_file() and p.suffix.lower() in exts_l:
            dt = time.perf_counter() - t0
            logging.info(
                "[_quick_has_any_image] FOUND root=%s file=%s scanned=%d elapsed=%.2fs",
                root, p, scanned, dt
            )
            return True

    dt = time.perf_counter() - t0
    logging.info(
        "[_quick_has_any_image] END root=%s no_images scanned=%d elapsed=%.2fs",
        root, scanned, dt
    )
    return False


def _first_ccpd_like_image(root: Path, exts: List[str], limit: int = 5000) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    exts_l = _exts_set(exts)
    tried = 0
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts_l:
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


def _candidate_anchors(user_root: Path) -> List[Path]:
    cwd = Path.cwd()
    anchors = [user_root]
    if not user_root.is_absolute():
        anchors.append(cwd / user_root)
    anchors.extend(
        [
            cwd / "data",
            cwd / "data" / "CCPD2019",
            cwd / "data" / "CCPD2019" / "CCPD2019",
        ]
    )
    for extra in (user_root.parent, user_root.parent.parent):
        if extra is not None:
            anchors.append(extra)
    uniq: List[Path] = []
    seen: Set[str] = set()
    for a in anchors:
        key = str(a.resolve()) if a.exists() else str(a)
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    return uniq


def resolve_ccpd_base_root(user_root: Path, exts: List[str]) -> Path:
    user_root = Path(user_root)
    logging.info("[resolve_ccpd_base_root] START user_root=%s", user_root)

    # Если пользователь уже указал точный ccpd_base - сразу принимаем его.
    # Никаких рекурсивных сканов тут делать не нужно.
    if user_root.exists() and user_root.is_dir() and user_root.name.lower() == "ccpd_base":
        logging.info("[resolve_ccpd_base_root] FAST RETURN exact ccpd_base=%s", user_root)
        return user_root

    # Если пользователь указал train/val/test внутри ccpd_base - возвращаем родителя.
    if (
        user_root.exists()
        and user_root.is_dir()
        and user_root.name.lower() in {"train", "val", "test"}
        and user_root.parent.name.lower() == "ccpd_base"
    ):
        logging.info("[resolve_ccpd_base_root] FAST RETURN from split dir=%s", user_root.parent)
        return user_root.parent

    preferred_rel = [
        Path("ccpd_base"),
        Path("CCPD2019") / "ccpd_base",
        Path("CCPD2019") / "CCPD2019" / "ccpd_base",
    ]

    for base in _candidate_anchors(user_root):
        logging.info("[resolve_ccpd_base_root] checking anchor=%s", base)
        for rel in preferred_rel:
            cand = base / rel
            logging.info("[resolve_ccpd_base_root] checking candidate=%s", cand)
            if cand.exists() and cand.is_dir():
                logging.info("[resolve_ccpd_base_root] resolved by candidate=%s", cand)
                return cand

    logging.warning("[resolve_ccpd_base_root] FALLBACK user_root=%s", user_root)
    return user_root


def resolve_ccpd_train_root(user_root: Path, exts: List[str]) -> Path:
    base_root = resolve_ccpd_base_root(user_root, exts)
    train_root = base_root / "train"
    if train_root.exists() and _quick_has_any_image(train_root, exts):
        logging.info("Resolved CCPD train root to: %s", train_root)
        return train_root
    return base_root


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


def iter_images(root: Path, exts: List[str]) -> Iterator[Path]:
    t0 = time.perf_counter()
    logging.info("[iter_images] START root=%s", root)

    if not root.exists() or not root.is_dir():
        logging.warning("[iter_images] root missing or not dir: %s", root)
        return

    exts_l = _exts_set(exts)
    yielded = 0
    scanned = 0

    for p in root.rglob("*"):
        scanned += 1

        if scanned % 50000 == 0:
            logging.info(
                "[iter_images] progress root=%s scanned=%d yielded=%d elapsed=%.2fs",
                root, scanned, yielded, time.perf_counter() - t0
            )

        if p.is_file() and p.suffix.lower() in exts_l:
            yielded += 1
            if yielded == 1:
                logging.info(
                    "[iter_images] FIRST IMAGE root=%s file=%s scanned=%d elapsed=%.2fs",
                    root, p, scanned, time.perf_counter() - t0
                )
            elif yielded % 10000 == 0:
                logging.info(
                    "[iter_images] yielded=%d last=%s elapsed=%.2fs",
                    yielded, p, time.perf_counter() - t0
                )
            yield p

    logging.info(
        "[iter_images] END root=%s scanned=%d yielded=%d elapsed=%.2fs",
        root, scanned, yielded, time.perf_counter() - t0
    )


def _dir_has_split_txt(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.glob("*.txt"))


def resolve_split_dir(user_split_dir: Path, dataset_root: Path) -> Path:
    user_split_dir = Path(user_split_dir)
    dataset_root = Path(dataset_root)
    cwd = Path.cwd()

    anchors: List[Path] = [user_split_dir]
    if not user_split_dir.is_absolute():
        anchors.append(cwd / user_split_dir)

    anchors.extend(
        [
            dataset_root,
            dataset_root / "splits",
            dataset_root.parent / "splits",
            dataset_root.parent.parent / "splits",
            cwd / "data",
            cwd / "data" / "splits",
            cwd / "data" / "CCPD2019" / "splits",
            cwd / "data" / "CCPD2019" / "CCPD2019" / "splits",
        ]
    )

    uniq: List[Path] = []
    seen: Set[str] = set()
    for a in anchors:
        key = str(a.resolve()) if a.exists() else str(a)
        if key not in seen:
            seen.add(key)
            uniq.append(a)

    for a in uniq:
        if _dir_has_split_txt(a):
            logging.info("Resolved CCPD split dir to: %s", a)
            return a

    search_roots = [dataset_root, dataset_root.parent, dataset_root.parent.parent, cwd / "data"]
    seen_search: Set[str] = set()
    for sr in search_roots:
        if sr is None or not sr.exists() or not sr.is_dir():
            continue
        sr_key = str(sr.resolve())
        if sr_key in seen_search:
            continue
        seen_search.add(sr_key)
        for p in sr.rglob("splits"):
            if _dir_has_split_txt(p):
                logging.info("Resolved CCPD split dir by search to: %s", p)
                return p

    logging.warning("Could not resolve CCPD split dir automatically. Using: %s", user_split_dir)
    return user_split_dir


def load_splits(split_dir: Path, dataset_root: Optional[Path] = None) -> Dict[str, List[str]]:
    splits: Dict[str, List[str]] = {}
    resolved = resolve_split_dir(split_dir, dataset_root or Path.cwd())
    if not resolved.exists():
        return splits
    for txt in resolved.glob("*.txt"):
        key = txt.stem.strip().lower()
        lines = read_text_lines(txt)
        if lines:
            splits[key] = lines
    return splits


def resolve_split_items(dataset_root: Path, items: List[str], extra_roots: Optional[List[Path]] = None) -> List[Path]:
    logging.info("[resolve_split_items] START dataset_root=%s n_items=%d", dataset_root, len(items))

    out: List[Path] = []
    roots = [dataset_root]
    if extra_roots:
        roots.extend(extra_roots)
    roots = [Path(r) for r in roots if r is not None]

    direct_hits = 0
    rglob_hits = 0
    misses = 0

    for idx, it in enumerate(items, start=1):
        if idx % 5000 == 0:
            logging.info(
                "[resolve_split_items] progress idx=%d/%d direct_hits=%d rglob_hits=%d misses=%d",
                idx, len(items), direct_hits, rglob_hits, misses
            )

        s = it.strip().lstrip("./")
        p = Path(s)

        if p.is_absolute() and p.exists():
            out.append(p)
            direct_hits += 1
            continue

        matched = False
        for root in roots:
            cand = root / p
            if cand.exists():
                out.append(cand)
                direct_hits += 1
                matched = True
                break
        if matched:
            continue

        for root in roots:
            matches = list(root.rglob(p.name))
            if matches:
                out.append(matches[0])
                rglob_hits += 1
                matched = True
                break

        if not matched:
            misses += 1
            if misses <= 20:
                logging.warning("[resolve_split_items] MISS item=%s", it)

    logging.info(
        "[resolve_split_items] END resolved=%d direct_hits=%d rglob_hits=%d misses=%d",
        len(out), direct_hits, rglob_hits, misses
    )
    return out

def _iter_from_dir(
    root: Path,
    exts: List[str],
    split_name: str,
    seen: Set[str],
) -> Iterator[Tuple[Path, CCPDAnnotation, str]]:
    bad = 0
    bad_logged = 0
    for img in iter_images(root, exts):
        key = str(img.resolve()) if img.exists() else str(img)
        if key in seen:
            continue
        seen.add(key)
        try:
            ann = parse_ccpd_filename(img)
        except Exception:
            bad += 1
            if bad_logged < 20:
                logging.warning("Skip non-CCPD image (bad filename format): %s", img.name)
                bad_logged += 1
            continue
        yield img, ann, split_name
    if bad:
        logging.info("Skipped %d non-CCPD images for split=%s", bad, split_name)

def _iter_from_paths(
    paths: List[Path],
    split_name: str,
    seen: Set[str],
) -> Iterator[Tuple[Path, CCPDAnnotation, str]]:
    bad = 0
    bad_logged = 0
    for img in paths:
        key = str(img.resolve()) if img.exists() else str(img)
        if key in seen:
            continue
        seen.add(key)
        try:
            ann = parse_ccpd_filename(img)
        except Exception:
            bad += 1
            if bad_logged < 20:
                logging.warning("Skip non-CCPD image (bad filename format): %s", img.name)
                bad_logged += 1
            continue
        yield img, ann, split_name
    if bad:
        logging.info("Skipped %d non-CCPD images for split=%s", bad, split_name)


def _available_split_dirs(base_root: Path, exts: List[str]) -> Dict[str, Path]:
    logging.info("[_available_split_dirs] START base_root=%s", base_root)

    out: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        p = base_root / split
        logging.info("[_available_split_dirs] checking split=%s path=%s exists=%s", split, p, p.exists())
        if p.exists() and p.is_dir():
            out[split] = p

    logging.info("[_available_split_dirs] END found=%s", list(out.keys()))
    return out


def iter_ccpd_records(
    dataset_root: Path,
    split_dir: Path,
    exts: List[str],
    split: str = "auto",
) -> Iterator[Tuple[Path, CCPDAnnotation, str]]:
    t0 = time.perf_counter()
    logging.info(
        "[iter_ccpd_records] START dataset_root=%s split_dir=%s split=%s",
        dataset_root, split_dir, split
    )

    base_root = resolve_ccpd_base_root(dataset_root, exts)
    logging.info("[iter_ccpd_records] base_root=%s", base_root)

    split_roots = _available_split_dirs(base_root, exts)
    logging.info("[iter_ccpd_records] split_roots=%s", {k: str(v) for k, v in split_roots.items()})

    split_l = str(split).lower()
    seen: Set[str] = set()

    extra_roots = [base_root, base_root.parent, base_root.parent.parent, Path(dataset_root)]

    if split_l in ("train", "val", "test") and split_l in split_roots:
        logging.info("[iter_ccpd_records] branch=direct_split split=%s path=%s", split_l, split_roots[split_l])
        yield from _iter_from_dir(split_roots[split_l], exts, split_l, seen)
        logging.info("[iter_ccpd_records] END branch=direct_split elapsed=%.2fs", time.perf_counter() - t0)
        return

    if split_l == "all":
        if split_roots:
            logging.info("[iter_ccpd_records] branch=all_direct_splits")
            for k in ("train", "val", "test"):
                if k in split_roots:
                    logging.info("[iter_ccpd_records] yielding split=%s path=%s", k, split_roots[k])
                    yield from _iter_from_dir(split_roots[k], exts, k, seen)
            logging.info("[iter_ccpd_records] END branch=all_direct_splits elapsed=%.2fs", time.perf_counter() - t0)
            return

        logging.info("[iter_ccpd_records] branch=all_base_root path=%s", base_root)
        yield from _iter_from_dir(base_root, exts, "all", seen)
        logging.info("[iter_ccpd_records] END branch=all_base_root elapsed=%.2fs", time.perf_counter() - t0)
        return

    logging.info("[iter_ccpd_records] loading splits txt")
    splits = load_splits(split_dir, dataset_root=base_root)
    logging.info("[iter_ccpd_records] loaded splits keys=%s", list(splits.keys()))

    if split_l == "auto":
        if split_roots:
            logging.info("[iter_ccpd_records] branch=auto_direct_splits")
            for k in ("train", "val", "test"):
                if k in split_roots:
                    logging.info("[iter_ccpd_records] yielding split=%s path=%s", k, split_roots[k])
                    yield from _iter_from_dir(split_roots[k], exts, k, seen)
            logging.info("[iter_ccpd_records] END branch=auto_direct_splits elapsed=%.2fs", time.perf_counter() - t0)
            return

        if splits:
            preferred = [k for k in ("train", "val", "test") if k in splits]
            if not preferred:
                preferred = [next(iter(splits.keys()))]
            logging.info("[iter_ccpd_records] branch=auto_txt_splits preferred=%s", preferred)

            for k in preferred:
                paths = resolve_split_items(base_root, splits[k], extra_roots=extra_roots)
                logging.info("[iter_ccpd_records] txt split=%s resolved_paths=%d", k, len(paths))
                yield from _iter_from_paths(paths, k, seen)

            logging.info("[iter_ccpd_records] END branch=auto_txt_splits elapsed=%.2fs", time.perf_counter() - t0)
            return

        logging.info("[iter_ccpd_records] branch=auto_base_root path=%s", base_root)
        yield from _iter_from_dir(base_root, exts, "all", seen)
        logging.info("[iter_ccpd_records] END branch=auto_base_root elapsed=%.2fs", time.perf_counter() - t0)
        return

    if split_l in splits:
        logging.info("[iter_ccpd_records] branch=explicit_txt_split split=%s", split_l)
        paths = resolve_split_items(base_root, splits[split_l], extra_roots=extra_roots)
        logging.info("[iter_ccpd_records] explicit txt split=%s resolved_paths=%d", split_l, len(paths))
        yield from _iter_from_paths(paths, split_l, seen)
        logging.info("[iter_ccpd_records] END branch=explicit_txt_split elapsed=%.2fs", time.perf_counter() - t0)
        return

    fallback_dir = base_root / split_l
    if fallback_dir.exists() and fallback_dir.is_dir():
        logging.info("[iter_ccpd_records] branch=fallback_dir split=%s path=%s", split_l, fallback_dir)
        yield from _iter_from_dir(fallback_dir, exts, split_l, seen)
        logging.info("[iter_ccpd_records] END branch=fallback_dir elapsed=%.2fs", time.perf_counter() - t0)
        return

    logging.info("[iter_ccpd_records] branch=final_base_root path=%s", base_root)
    yield from _iter_from_dir(base_root, exts, "all", seen)
    logging.info("[iter_ccpd_records] END branch=final_base_root elapsed=%.2fs", time.perf_counter() - t0)