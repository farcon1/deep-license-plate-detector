from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from .utils import read_text_lines


_FILE_INDEX_CACHE: Dict[str, Tuple[Dict[str, Path], Dict[str, Path]]] = {}


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

    if user_root.exists() and user_root.is_dir() and user_root.name.lower() == "ccpd_base":
        logging.info("[resolve_ccpd_base_root] FAST RETURN exact ccpd_base=%s", user_root)
        return user_root

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


def _normalize_split_item_key(value: str | Path) -> str:
    s = str(value).strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    while s.startswith("/"):
        s = s[1:]
    return s


def _root_cache_key(root: Path) -> str:
    try:
        return str(root.resolve())
    except Exception:
        return str(root)


def _dedupe_roots(roots: List[Path]) -> List[Path]:
    out: List[Path] = []
    seen: Set[str] = set()

    for root in roots:
        if root is None:
            continue
        root = Path(root)
        key = _root_cache_key(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)

    return out


def _build_root_file_index(root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    root = Path(root)

    relative_map: Dict[str, Path] = {}
    basename_map: Dict[str, Path] = {}

    logging.info("[_build_root_file_index] START root=%s", root)

    scanned = 0
    files = 0

    for p in root.rglob("*"):
        scanned += 1

        if scanned % 50000 == 0:
            logging.info(
                "[_build_root_file_index] progress root=%s scanned=%d files=%d",
                root, scanned, files
            )

        if not p.is_file():
            continue

        files += 1

        try:
            rel = p.relative_to(root)
            rel_key = _normalize_split_item_key(rel)
        except Exception:
            rel_key = _normalize_split_item_key(p.name)

        if rel_key not in relative_map:
            relative_map[rel_key] = p

        if p.name not in basename_map:
            basename_map[p.name] = p

    logging.info(
        "[_build_root_file_index] END root=%s scanned=%d files=%d relative_keys=%d basename_keys=%d",
        root, scanned, files, len(relative_map), len(basename_map)
    )

    return relative_map, basename_map


def _get_root_file_index(root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    key = _root_cache_key(root)
    if key not in _FILE_INDEX_CACHE:
        _FILE_INDEX_CACHE[key] = _build_root_file_index(root)
    return _FILE_INDEX_CACHE[key]


def resolve_split_items(dataset_root: Path, items: List[str], extra_roots: Optional[List[Path]] = None) -> List[Path]:
    """
    Быстрое разрешение путей из split-файлов CCPD.
    """
    logging.info("[resolve_split_items] START dataset_root=%s n_items=%d", dataset_root, len(items))

    dataset_root = Path(dataset_root)

    out: List[Optional[Path]] = [None] * len(items)
    unresolved: List[Tuple[int, str]] = []

    absolute_hits = 0
    direct_path_hits = 0
    basename_direct_hits = 0
    indexed_rel_hits = 0
    indexed_basename_hits = 0
    fallback_hits = 0

    # 1. FAST PATH: никаких rglob.
    for idx, it in enumerate(items):
        if idx > 0 and idx % 5000 == 0:
            logging.info(
                "[resolve_split_items] fast path progress idx=%d/%d absolute_hits=%d direct_path_hits=%d basename_direct_hits=%d unresolved=%d",
                idx,
                len(items),
                absolute_hits,
                direct_path_hits,
                basename_direct_hits,
                len(unresolved),
            )

        s = str(it).strip()
        p = Path(s)

        if not s:
            unresolved.append((idx, s))
            continue

        # 1.1 Абсолютный путь.
        if p.is_absolute() and p.exists():
            out[idx] = p
            absolute_hits += 1
            continue

        # 1.2 Относительный путь из split-файла.
        rel_key = _normalize_split_item_key(s)
        rel_candidate = dataset_root / Path(rel_key)

        if rel_candidate.exists() and rel_candidate.is_file():
            out[idx] = rel_candidate
            direct_path_hits += 1
            continue

        # 1.3 basename напрямую в ccpd_base.
        basename_candidate = dataset_root / p.name

        if basename_candidate.exists() and basename_candidate.is_file():
            out[idx] = basename_candidate
            basename_direct_hits += 1
            continue

        unresolved.append((idx, s))

    # Если всё найдено прямыми проверками — сразу возвращаемся.
    if not unresolved:
        resolved_out = [p for p in out if p is not None]
        logging.info(
            "[resolve_split_items] END fast resolved=%d absolute_hits=%d direct_path_hits=%d basename_direct_hits=%d misses=0",
            len(resolved_out),
            absolute_hits,
            direct_path_hits,
            basename_direct_hits,
        )
        return resolved_out

    logging.warning(
        "[resolve_split_items] fast path left unresolved=%d. Building base_root index only now.",
        len(unresolved),
    )

    # 2. SLOW PATH: строим индекс только если fast path не смог найти часть файлов.
    indexes: List[Tuple[Path, Dict[str, Path], Dict[str, Path]]] = []

    if dataset_root.exists() and dataset_root.is_dir():
        rel_map, base_map = _get_root_file_index(dataset_root)
        indexes.append((dataset_root, rel_map, base_map))
    else:
        logging.warning("[resolve_split_items] dataset_root missing or not dir: %s", dataset_root)

    still_unresolved: List[Tuple[int, str]] = []

    for pos, (idx, s) in enumerate(unresolved, start=1):
        if pos % 5000 == 0:
            logging.info(
                "[resolve_split_items] index fallback progress idx=%d/%d indexed_rel_hits=%d indexed_basename_hits=%d still_unresolved=%d",
                pos,
                len(unresolved),
                indexed_rel_hits,
                indexed_basename_hits,
                len(still_unresolved),
            )

        p = Path(s)
        rel_key = _normalize_split_item_key(s)
        basename = p.name

        matched = False

        for _, rel_map, _ in indexes:
            cand = rel_map.get(rel_key)
            if cand is not None:
                out[idx] = cand
                indexed_rel_hits += 1
                matched = True
                break

        if matched:
            continue

        for _, _, base_map in indexes:
            cand = base_map.get(basename)
            if cand is not None:
                out[idx] = cand
                indexed_basename_hits += 1
                matched = True
                break

        if not matched:
            still_unresolved.append((idx, s))

    if still_unresolved and extra_roots:
        roots = _dedupe_roots([Path(r) for r in extra_roots if r is not None and Path(r) != dataset_root])

        logging.warning(
            "[resolve_split_items] base_root still did not resolve %d items. Trying extra_roots=%s",
            len(still_unresolved),
            [str(r) for r in roots],
        )

        fallback_indexes: List[Tuple[Path, Dict[str, Path], Dict[str, Path]]] = []

        for root in roots:
            if root.exists() and root.is_dir():
                rel_map, base_map = _get_root_file_index(root)
                fallback_indexes.append((root, rel_map, base_map))

        next_unresolved: List[Tuple[int, str]] = []

        for pos, (idx, s) in enumerate(still_unresolved, start=1):
            if pos % 5000 == 0:
                logging.info(
                    "[resolve_split_items] extra fallback progress idx=%d/%d fallback_hits=%d still_misses=%d",
                    pos,
                    len(still_unresolved),
                    fallback_hits,
                    len(next_unresolved),
                )

            p = Path(s)
            rel_key = _normalize_split_item_key(s)
            basename = p.name

            matched = False

            for _, rel_map, _ in fallback_indexes:
                cand = rel_map.get(rel_key)
                if cand is not None:
                    out[idx] = cand
                    fallback_hits += 1
                    matched = True
                    break

            if matched:
                continue

            for _, _, base_map in fallback_indexes:
                cand = base_map.get(basename)
                if cand is not None:
                    out[idx] = cand
                    fallback_hits += 1
                    matched = True
                    break

            if not matched:
                next_unresolved.append((idx, s))

        still_unresolved = next_unresolved

    misses = len(still_unresolved)

    if misses:
        for _, missed_item in still_unresolved[:20]:
            logging.warning("[resolve_split_items] MISS item=%s", missed_item)

    resolved_out = [p for p in out if p is not None]

    logging.info(
        "[resolve_split_items] END resolved=%d absolute_hits=%d direct_path_hits=%d basename_direct_hits=%d indexed_rel_hits=%d indexed_basename_hits=%d fallback_hits=%d misses=%d",
        len(resolved_out),
        absolute_hits,
        direct_path_hits,
        basename_direct_hits,
        indexed_rel_hits,
        indexed_basename_hits,
        fallback_hits,
        misses,
    )

    return resolved_out

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

    extra_roots: List[Path] = []

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