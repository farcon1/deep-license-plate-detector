from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dlpd.ccpd import CCPDAnnotation, iter_ccpd_records
from dlpd.config import load_config
from dlpd.ocr_dataset import crop_plate
from dlpd.plate_text import decode_plate_indices, validate_plate_indices
from dlpd.utils import dump_json, ensure_dir, seed_everything, setup_logging


@dataclass
class ExternalOCRPrediction:
    text_raw: str
    text_norm: str
    confidence: float
    engine_error: str = ""


@dataclass
class ExternalOCREngine:
    name: str

    def recognize(self, img_bgr: np.ndarray) -> ExternalOCRPrediction:
        raise NotImplementedError


def normalize_ocr_text(value: Any) -> str:
    """
    Нормализация OCR-строки для честного сравнения:
    - Unicode NFKC;
    - uppercase для латиницы;
    - удаление пробелов, переносов строк, пунктуации;
    - оставляем латиницу, цифры и CJK-символы.

    Важно: здесь НЕ делаем агрессивных замен O<->0, I<->1 и т.п.,
    чтобы не завышать метрики внешних OCR.
    """
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.upper()
    chars: List[str] = []

    for ch in text:
        is_latin = "A" <= ch <= "Z"
        is_digit = "0" <= ch <= "9"
        is_cjk = "\u4e00" <= ch <= "\u9fff"

        if is_latin or is_digit or is_cjk:
            chars.append(ch)

    return "".join(chars)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return int(previous[-1])


def positional_char_accuracy(gt: str, pred: str) -> float:
    if not gt:
        return 0.0

    matches = 0
    for idx, ch in enumerate(gt):
        if idx < len(pred) and pred[idx] == ch:
            matches += 1

    return float(matches / len(gt))


def suffix_accuracy(gt: str, pred: str, start_idx: int = 2) -> float:
    """
    Для CCPD первая позиция — китайская провинция, вторая — латинская буква.
    С позиции 2 идут alphanumeric-символы. Эта метрика полезна, потому что
    внешние OCR часто плохо читают китайскую провинцию, но частично читают латиницу/цифры.
    """
    gt_suffix = gt[start_idx:]
    pred_suffix = pred[start_idx:]

    if not gt_suffix:
        return 0.0

    matches = 0
    for idx, ch in enumerate(gt_suffix):
        if idx < len(pred_suffix) and pred_suffix[idx] == ch:
            matches += 1

    return float(matches / len(gt_suffix))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def preprocess_for_tesseract(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()

    if mode == "none":
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    scale = 3
    gray = cv2.resize(
        gray,
        (gray.shape[1] * scale, gray.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )

    if mode == "gray":
        return gray

    if mode == "threshold":
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=45, sigmaSpace=45)
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        return thr

    raise ValueError(f"Unsupported tesseract preprocess mode: {mode}")


class TesseractOCREngine(ExternalOCREngine):
    def __init__(
        self,
        lang: str,
        psm: int,
        oem: int,
        preprocess_mode: str,
    ):
        super().__init__(name="tesseract")

        import pytesseract

        self.pytesseract = pytesseract
        self.lang = str(lang)
        self.psm = int(psm)
        self.oem = int(oem)
        self.preprocess_mode = str(preprocess_mode)

        if shutil.which("tesseract") is None:
            raise RuntimeError(
                "Tesseract binary not found in PATH. Install Tesseract OCR and add it to PATH."
            )

    def recognize(self, img_bgr: np.ndarray) -> ExternalOCRPrediction:
        prepared = preprocess_for_tesseract(img_bgr, mode=self.preprocess_mode)
        config = f"--oem {self.oem} --psm {self.psm}"

        text = self.pytesseract.image_to_string(
            prepared,
            lang=self.lang,
            config=config,
        )

        text_norm = normalize_ocr_text(text)

        return ExternalOCRPrediction(
            text_raw=str(text).strip(),
            text_norm=text_norm,
            confidence=0.0,
            engine_error="",
        )


class EasyOCREngine(ExternalOCREngine):
    def __init__(self, languages: Sequence[str], gpu: bool):
        super().__init__(name="easyocr")

        import easyocr

        self.languages = [str(x).strip() for x in languages if str(x).strip()]
        self.gpu = bool(gpu)
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu)

    def recognize(self, img_bgr: np.ndarray) -> ExternalOCRPrediction:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(
            img_rgb,
            detail=1,
            paragraph=False,
        )

        texts: List[str] = []
        confs: List[float] = []

        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            text = str(item[1])
            conf = safe_float(item[2], default=0.0)
            texts.append(text)
            confs.append(conf)

        raw = "".join(texts)
        norm = normalize_ocr_text(raw)
        confidence = float(np.mean(confs)) if confs else 0.0

        return ExternalOCRPrediction(
            text_raw=raw,
            text_norm=norm,
            confidence=confidence,
            engine_error="",
        )


def _flatten_paddle_result(obj: Any) -> List[Tuple[str, float]]:
    """
    Терпимый парсер результатов PaddleOCR.

    Поддерживает:
    - старый формат: [[box, (text, score)], ...]
    - вложенный старый формат: [[[box, (text, score)], ...]]
    - dict-формат: rec_texts / rec_scores
    - PaddleOCR 3.x result objects с json/dict-представлением
    """
    out: List[Tuple[str, float]] = []

    if obj is None:
        return out

    # PaddleOCR 3.x result object может иметь json/dict-представление.
    for attr_name in ("json", "dict"):
        if hasattr(obj, attr_name):
            try:
                attr = getattr(obj, attr_name)
                value = attr() if callable(attr) else attr
                if value is not obj:
                    out.extend(_flatten_paddle_result(value))
                    if out:
                        return out
            except Exception:
                pass

    # Некоторые result objects имеют res / result / data.
    for attr_name in ("res", "result", "data"):
        if hasattr(obj, attr_name):
            try:
                value = getattr(obj, attr_name)
                if value is not obj:
                    out.extend(_flatten_paddle_result(value))
                    if out:
                        return out
            except Exception:
                pass

    if isinstance(obj, dict):
        rec_texts = obj.get("rec_texts")
        rec_scores = obj.get("rec_scores")

        if isinstance(rec_texts, list):
            for idx, text in enumerate(rec_texts):
                score = 0.0
                if isinstance(rec_scores, list) and idx < len(rec_scores):
                    score = safe_float(rec_scores[idx], default=0.0)
                out.append((str(text), score))

        # Иногда ключи называются иначе.
        text_keys = ["text", "texts", "transcription", "label"]
        score_keys = ["score", "scores", "confidence", "conf"]

        for text_key in text_keys:
            if text_key in obj:
                value = obj.get(text_key)
                if isinstance(value, str):
                    score = 0.0
                    for score_key in score_keys:
                        if score_key in obj:
                            score = safe_float(obj.get(score_key), default=0.0)
                            break
                    out.append((value, score))
                elif isinstance(value, list):
                    scores = []
                    for score_key in score_keys:
                        if isinstance(obj.get(score_key), list):
                            scores = obj.get(score_key)
                            break
                    for idx, text in enumerate(value):
                        score = safe_float(scores[idx], default=0.0) if idx < len(scores) else 0.0
                        out.append((str(text), score))

        if out:
            return out

        for value in obj.values():
            if isinstance(value, (list, tuple, dict)) or hasattr(value, "__dict__"):
                out.extend(_flatten_paddle_result(value))

        return out

    if isinstance(obj, (list, tuple)):
        # Case: (text, score)
        if len(obj) >= 2:
            first = obj[0]
            second = obj[1]

            if isinstance(first, str) and isinstance(second, (float, int, np.floating, np.integer)):
                return [(str(first), safe_float(second, default=0.0))]

            # Case: [box, (text, score)]
            if isinstance(second, (list, tuple)) and len(second) >= 2:
                maybe_text = second[0]
                maybe_score = second[1]
                if isinstance(maybe_text, str):
                    return [(str(maybe_text), safe_float(maybe_score, default=0.0))]

        for item in obj:
            out.extend(_flatten_paddle_result(item))

    return out


class PaddleOCREngine(ExternalOCREngine):
    def __init__(self, lang: str, use_gpu: bool):
        super().__init__(name="paddleocr")

        # ВАЖНО:
        # Эта ошибка:
        # ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]
        # возникает внутри Paddle Runtime на CPU oneDNN/MKLDNN path.
        #
        # Даже если пользователь передал --paddleocr-gpu, установленный paddle
        # может быть CPU-only. Тогда PaddleOCR фактически идёт через CPU,
        # включает MKLDNN/oneDNN и падает.
        #
        # Поэтому для стабильного baseline-эксперимента отключаем MKLDNN.
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")

        from paddleocr import PaddleOCR

        self.lang = str(lang)
        self.use_gpu = bool(use_gpu)

        init_attempts = [
            {
                "lang": self.lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "device": "gpu:0" if self.use_gpu else "cpu",
            },
            {
                "lang": self.lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            },
            {
                "lang": self.lang,
                "use_angle_cls": False,
                "use_gpu": self.use_gpu,
                "show_log": False,
            },
            {
                "lang": self.lang,
                "use_angle_cls": False,
                "show_log": False,
            },
            {
                "lang": self.lang,
            },
        ]

        last_error: Optional[Exception] = None
        self.ocr = None
        self.init_kwargs: Dict[str, Any] = {}

        for kwargs in init_attempts:
            try:
                self.ocr = PaddleOCR(**kwargs)
                self.init_kwargs = dict(kwargs)
                last_error = None
                break
            except Exception as exc:
                last_error = exc

        if self.ocr is None:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {last_error}")

    def recognize(self, img_bgr: np.ndarray) -> ExternalOCRPrediction:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        result: Any = None
        errors: List[str] = []

        if hasattr(self.ocr, "predict"):
            try:
                result = self.ocr.predict(img_rgb)
            except Exception as exc:
                errors.append(f"predict_failed={type(exc).__name__}: {exc}")

        if result is None and hasattr(self.ocr, "ocr"):
            try:
                result = self.ocr.ocr(img_rgb, cls=False)
            except TypeError:
                try:
                    result = self.ocr.ocr(img_rgb)
                except Exception as exc:
                    errors.append(f"ocr_failed={type(exc).__name__}: {exc}")
            except Exception as exc:
                errors.append(f"ocr_failed={type(exc).__name__}: {exc}")

        if result is None:
            raise RuntimeError("; ".join(errors) if errors else "PaddleOCR returned no result")

        pairs = _flatten_paddle_result(result)

        texts = [text for text, _ in pairs]
        confs = [score for _, score in pairs]

        raw = "".join(texts)
        norm = normalize_ocr_text(raw)
        confidence = float(np.mean(confs)) if confs else 0.0

        return ExternalOCRPrediction(
            text_raw=raw,
            text_norm=norm,
            confidence=confidence,
            engine_error="",
        )


def _clip_bbox_with_padding(
    ann: CCPDAnnotation,
    img_w: int,
    img_h: int,
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    x1 = int(ann.x1)
    y1 = int(ann.y1)
    x2 = int(ann.x2)
    y2 = int(ann.y2)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad_x = int(round(bw * float(pad_ratio)))
    pad_y = int(round(bh * float(pad_ratio)))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(max(0, img_w - 1), x2 + pad_x)
    y2 = min(max(0, img_h - 1), y2 + pad_y)

    return x1, y1, x2, y2


def crop_plate_bbox_padded(
    img_bgr: np.ndarray,
    ann: CCPDAnnotation,
    out_size: Tuple[int, int],
    pad_ratio: float,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_bbox_with_padding(
        ann=ann,
        img_w=w,
        img_h=h,
        pad_ratio=pad_ratio,
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid padded bbox crop.")

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty padded bbox crop.")

    out_w = int(out_size[0])
    out_h = int(out_size[1])

    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def make_normalized_crop(
    img_bgr: np.ndarray,
    ann: CCPDAnnotation,
    crop_mode: str,
    out_size: Tuple[int, int],
    pad_ratio: float,
) -> np.ndarray:
    mode = str(crop_mode).strip().lower()

    if mode in {"bbox", "corners"}:
        return crop_plate(
            img_bgr=img_bgr,
            ann=ann,
            crop_mode=mode,
            out_size=out_size,
        )

    if mode == "bbox_pad":
        return crop_plate_bbox_padded(
            img_bgr=img_bgr,
            ann=ann,
            out_size=out_size,
            pad_ratio=pad_ratio,
        )

    raise ValueError(f"Unsupported crop mode: {crop_mode}")


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def build_engines(args: argparse.Namespace) -> Tuple[List[ExternalOCREngine], List[Dict[str, str]]]:
    requested = [x.lower() for x in parse_csv_arg(args.engines)]
    engines: List[ExternalOCREngine] = []
    unavailable: List[Dict[str, str]] = []

    for engine_name in requested:
        try:
            if engine_name == "tesseract":
                engines.append(
                    TesseractOCREngine(
                        lang=args.tesseract_lang,
                        psm=args.tesseract_psm,
                        oem=args.tesseract_oem,
                        preprocess_mode=args.tesseract_preprocess,
                    )
                )
            elif engine_name == "easyocr":
                engines.append(
                    EasyOCREngine(
                        languages=parse_csv_arg(args.easyocr_langs),
                        gpu=bool(args.easyocr_gpu),
                    )
                )
            elif engine_name == "paddleocr":
                engines.append(
                    PaddleOCREngine(
                        lang=args.paddleocr_lang,
                        use_gpu=bool(args.paddleocr_gpu),
                    )
                )
            else:
                unavailable.append(
                    {
                        "engine": engine_name,
                        "reason": "unknown_engine",
                        "error": f"Unsupported engine name: {engine_name}",
                    }
                )
        except Exception as exc:
            unavailable.append(
                {
                    "engine": engine_name,
                    "reason": "initialization_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return engines, unavailable


def evaluate_prediction(gt_text: str, pred_text: str) -> Dict[str, Any]:
    gt_norm = normalize_ocr_text(gt_text)
    pred_norm = normalize_ocr_text(pred_text)

    edit_distance = levenshtein_distance(gt_norm, pred_norm)
    denom = max(1, len(gt_norm))

    province_match = int(len(gt_norm) >= 1 and len(pred_norm) >= 1 and gt_norm[0] == pred_norm[0])
    region_letter_match = int(len(gt_norm) >= 2 and len(pred_norm) >= 2 and gt_norm[1] == pred_norm[1])

    return {
        "gt_text_norm": gt_norm,
        "pred_text_norm": pred_norm,
        "exact_match": int(gt_norm == pred_norm),
        "length_match": int(len(gt_norm) == len(pred_norm)),
        "edit_distance": int(edit_distance),
        "cer": float(edit_distance / denom),
        "char_accuracy_positional": float(positional_char_accuracy(gt_norm, pred_norm)),
        "suffix_accuracy_positional": float(suffix_accuracy(gt_norm, pred_norm, start_idx=2)),
        "province_match": province_match,
        "region_letter_match": region_letter_match,
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def build_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит итоговую таблицу по OCR engine и crop_mode.

    Важное изменение:
    - движки с ошибками НЕ исчезают из summary;
    - метрики качества считаются только по строкам без engine_error;
    - отдельно выводятся n_images, n_ok, n_errors, error_rate.

    Это нужно, чтобы PaddleOCR не пропадал из отчёта, если он падает на predict/ocr.
    """
    if pred_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    grouped = pred_df.groupby(["engine", "crop_mode"], dropna=False)

    for (engine, crop_mode), part in grouped:
        part = part.copy()

        error_mask = part["engine_error"].fillna("") != ""
        ok_part = part.loc[~error_mask].copy()

        n_images = int(len(part))
        n_errors = int(error_mask.sum())
        n_ok = int(len(ok_part))
        error_rate = float(n_errors / n_images) if n_images else 0.0

        row: Dict[str, Any] = {
            "engine": str(engine),
            "crop_mode": str(crop_mode),
            "n_images": n_images,
            "n_ok": n_ok,
            "n_errors": n_errors,
            "error_rate": error_rate,
        }

        if n_ok > 0:
            row.update(
                {
                    "exact_match_rate": float(ok_part["exact_match"].mean()),
                    "char_accuracy": float(ok_part["char_accuracy_positional"].mean()),
                    "suffix_accuracy": float(ok_part["suffix_accuracy_positional"].mean()),
                    "length_match_rate": float(ok_part["length_match"].mean()),
                    "province_match_rate": float(ok_part["province_match"].mean()),
                    "region_letter_match_rate": float(ok_part["region_letter_match"].mean()),
                    "mean_edit_distance": float(ok_part["edit_distance"].mean()),
                    "mean_cer": float(ok_part["cer"].mean()),
                    "mean_confidence": float(ok_part["confidence"].mean()),
                    "mean_latency_ms": float(ok_part["latency_ms"].mean()),
                    "top_engine_error": "",
                }
            )
        else:
            top_error = ""
            if n_errors > 0:
                top_error = str(part["engine_error"].fillna("").value_counts().index[0])

            row.update(
                {
                    "exact_match_rate": 0.0,
                    "char_accuracy": 0.0,
                    "suffix_accuracy": 0.0,
                    "length_match_rate": 0.0,
                    "province_match_rate": 0.0,
                    "region_letter_match_rate": 0.0,
                    "mean_edit_distance": 0.0,
                    "mean_cer": 1.0,
                    "mean_confidence": 0.0,
                    "mean_latency_ms": float(part["latency_ms"].mean()) if "latency_ms" in part.columns else 0.0,
                    "top_engine_error": top_error,
                }
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    numeric_cols = [
        "error_rate",
        "exact_match_rate",
        "char_accuracy",
        "suffix_accuracy",
        "length_match_rate",
        "province_match_rate",
        "region_letter_match_rate",
        "mean_edit_distance",
        "mean_cer",
        "mean_confidence",
        "mean_latency_ms",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].astype(float).round(6)

    out = out.sort_values(
        by=["exact_match_rate", "char_accuracy", "suffix_accuracy", "error_rate"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return out


def build_normalization_delta(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Сравнение эффекта нормализации bbox.

    Метрики считаются только для engine/crop_mode, где есть n_ok > 0.
    Если у engine всё упало, он всё равно будет виден в summary.csv,
    но в normalization_delta.csv попадать не обязан.
    """
    if summary_df.empty:
        return pd.DataFrame()

    usable = summary_df.copy()

    if "n_ok" in usable.columns:
        usable = usable[usable["n_ok"] > 0].copy()

    if usable.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    metrics = [
        "exact_match_rate",
        "char_accuracy",
        "suffix_accuracy",
        "mean_cer",
    ]

    for engine in sorted(usable["engine"].dropna().unique()):
        part = usable[usable["engine"] == engine].copy()

        row: Dict[str, Any] = {"engine": engine}

        for metric in metrics:
            for crop_mode in sorted(part["crop_mode"].dropna().unique()):
                value = part.loc[part["crop_mode"] == crop_mode, metric]
                if len(value):
                    row[f"{metric}__{crop_mode}"] = float(value.iloc[0])

        if "exact_match_rate__corners" in row and "exact_match_rate__bbox" in row:
            row["delta_exact_corners_minus_bbox"] = float(
                row["exact_match_rate__corners"] - row["exact_match_rate__bbox"]
            )

        if "char_accuracy__corners" in row and "char_accuracy__bbox" in row:
            row["delta_char_acc_corners_minus_bbox"] = float(
                row["char_accuracy__corners"] - row["char_accuracy__bbox"]
            )

        if "mean_cer__corners" in row and "mean_cer__bbox" in row:
            row["delta_cer_corners_minus_bbox"] = float(
                row["mean_cer__corners"] - row["mean_cer__bbox"]
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    for col in out.columns:
        if col != "engine":
            out[col] = out[col].astype(float).round(6)

    return out


def maybe_save_crop(
    img_bgr: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_bgr)


def safe_image_stem(img_path: Path, index: int) -> str:
    stem = img_path.stem
    stem = re.sub(r"[^A-Za-z0-9А-Яа-яЁё_\-]+", "_", stem)
    stem = stem[:120]
    return f"{index:07d}_{stem}"


def run_external_ocr_baselines(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    seed_everything(int(args.seed))

    out_dir = ensure_dir(args.out_dir)
    crops_dir = ensure_dir(out_dir / "crops")
    error_crops_dir = ensure_dir(out_dir / "error_crops")

    crop_modes = [x.lower() for x in parse_csv_arg(args.crop_modes)]
    crop_size = (int(args.crop_width), int(args.crop_height))

    engines, unavailable_engines = build_engines(args)

    if not engines:
        dump_json(
            {
                "error": "No external OCR engines are available.",
                "unavailable_engines": unavailable_engines,
            },
            out_dir / "summary.json",
        )
        raise RuntimeError(
            "No external OCR engines are available. Run scripts/check_ocr_external_dependencies.py."
        )

    logging.info("External OCR engines: %s", [e.name for e in engines])
    logging.info("Unavailable engines: %s", unavailable_engines)
    logging.info("Crop modes: %s", crop_modes)
    logging.info("Crop size: %s", crop_size)
    logging.info("Split: %s", args.split)

    rows: List[Dict[str, Any]] = []
    crop_failures: List[Dict[str, Any]] = []

    records = iter_ccpd_records(
        dataset_root=cfg.dataset.root,
        split_dir=cfg.dataset.split_dir,
        exts=cfg.dataset.image_exts,
        split=args.split,
    )

    processed_images = 0

    for idx, (img_path, ann, actual_split) in enumerate(tqdm(records, desc="External OCR baselines")):
        if int(args.max_images) > 0 and processed_images >= int(args.max_images):
            break

        img_path = Path(img_path)

        try:
            indices = validate_plate_indices(ann.plate_indices)
            gt_text = decode_plate_indices(indices)
        except Exception as exc:
            crop_failures.append(
                {
                    "img_path": str(img_path),
                    "split": actual_split,
                    "stage": "decode_gt",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            crop_failures.append(
                {
                    "img_path": str(img_path),
                    "split": actual_split,
                    "stage": "read_image",
                    "error": "cv2.imread returned None",
                }
            )
            continue

        processed_images += 1

        img_stem = safe_image_stem(img_path, processed_images)

        for crop_mode in crop_modes:
            try:
                crop_bgr = make_normalized_crop(
                    img_bgr=img_bgr,
                    ann=ann,
                    crop_mode=crop_mode,
                    out_size=crop_size,
                    pad_ratio=float(args.pad_ratio),
                )
            except Exception as exc:
                crop_failures.append(
                    {
                        "img_path": str(img_path),
                        "split": actual_split,
                        "stage": "make_crop",
                        "crop_mode": crop_mode,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            crop_rel_path = Path("")

            if bool(args.save_all_crops):
                crop_path = crops_dir / crop_mode / f"{img_stem}.jpg"
                maybe_save_crop(crop_bgr, crop_path)
                crop_rel_path = crop_path.relative_to(out_dir)

            for engine in engines:
                started = time.perf_counter()
                pred = ExternalOCRPrediction(
                    text_raw="",
                    text_norm="",
                    confidence=0.0,
                    engine_error="",
                )

                try:
                    pred = engine.recognize(crop_bgr)
                except Exception as exc:
                    pred = ExternalOCRPrediction(
                        text_raw="",
                        text_norm="",
                        confidence=0.0,
                        engine_error=f"{type(exc).__name__}: {exc}",
                    )

                latency_ms = float((time.perf_counter() - started) * 1000.0)

                metrics = evaluate_prediction(
                    gt_text=gt_text,
                    pred_text=pred.text_norm,
                )

                should_save_error_crop = (
                    bool(args.save_error_crops)
                    and pred.engine_error == ""
                    and int(metrics["exact_match"]) == 0
                )

                error_crop_rel_path = Path("")

                if should_save_error_crop:
                    error_crop_path = (
                        error_crops_dir
                        / engine.name
                        / crop_mode
                        / f"{img_stem}__gt_{metrics['gt_text_norm']}__pred_{metrics['pred_text_norm'] or 'EMPTY'}.jpg"
                    )
                    maybe_save_crop(crop_bgr, error_crop_path)
                    error_crop_rel_path = error_crop_path.relative_to(out_dir)

                row: Dict[str, Any] = {
                    "img_path": str(img_path),
                    "split_requested": str(args.split),
                    "split_actual": str(actual_split),
                    "engine": engine.name,
                    "crop_mode": crop_mode,
                    "crop_width": int(crop_size[0]),
                    "crop_height": int(crop_size[1]),
                    "pad_ratio": float(args.pad_ratio),
                    "gt_text": gt_text,
                    "pred_text_raw": pred.text_raw,
                    "pred_text_norm": pred.text_norm,
                    "confidence": float(pred.confidence),
                    "latency_ms": latency_ms,
                    "engine_error": pred.engine_error,
                    "crop_rel_path": str(crop_rel_path),
                    "error_crop_rel_path": str(error_crop_rel_path),
                    "area_ratio": float(ann.area_ratio),
                    "tilt_h": int(ann.tilt_h),
                    "tilt_v": int(ann.tilt_v),
                    "brightness": int(ann.brightness),
                    "blurriness": int(ann.blurriness),
                    "x1": int(ann.x1),
                    "y1": int(ann.y1),
                    "x2": int(ann.x2),
                    "y2": int(ann.y2),
                }
                row.update(metrics)
                rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    engine_errors_path = out_dir / "engine_errors.csv"

    if not pred_df.empty and "engine_error" in pred_df.columns:
        engine_errors_df = (
            pred_df[pred_df["engine_error"].fillna("") != ""]
            .groupby(["engine", "crop_mode", "engine_error"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["engine", "crop_mode", "count"], ascending=[True, True, False])
        )
    else:
        engine_errors_df = pd.DataFrame(columns=["engine", "crop_mode", "engine_error", "count"])

    engine_errors_df.to_csv(engine_errors_path, index=False, encoding="utf-8-sig")

    crop_failures_df = pd.DataFrame(crop_failures)
    crop_failures_path = out_dir / "crop_failures.csv"
    crop_failures_df.to_csv(crop_failures_path, index=False, encoding="utf-8-sig")

    summary_df = build_summary(pred_df)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    normalization_delta_df = build_normalization_delta(summary_df)
    normalization_delta_path = out_dir / "normalization_delta.csv"
    normalization_delta_df.to_csv(normalization_delta_path, index=False, encoding="utf-8-sig")

    errors_df = pred_df[
        (pred_df["engine_error"].fillna("") == "")
        & (pred_df["exact_match"] == 0)
    ].copy()

    if not errors_df.empty:
        errors_df = errors_df.sort_values(
            by=["engine", "crop_mode", "cer", "char_accuracy_positional"],
            ascending=[True, True, False, True],
        )

    errors_path = out_dir / "recognition_errors.csv"
    errors_df.to_csv(errors_path, index=False, encoding="utf-8-sig")

    summary_json = {
        "config": str(Path(args.config).resolve()),
        "out_dir": str(out_dir.resolve()),
        "split": str(args.split),
        "max_images": int(args.max_images),
        "processed_images": int(processed_images),
        "n_prediction_rows": int(len(pred_df)),
        "crop_modes": crop_modes,
        "crop_size": [int(crop_size[0]), int(crop_size[1])],
        "pad_ratio": float(args.pad_ratio),
        "engines": [engine.name for engine in engines],
        "unavailable_engines": unavailable_engines,
        "files": {
            "predictions_csv": str(pred_path),
            "summary_csv": str(summary_path),
            "normalization_delta_csv": str(normalization_delta_path),
            "recognition_errors_csv": str(errors_path),
            "crop_failures_csv": str(crop_failures_path),
            "engine_errors_csv": str(engine_errors_path),
        },
    }

    dump_json(summary_json, out_dir / "summary.json")

    report_lines = [
        "# External OCR Baselines Report",
        "",
        "## Experiment settings",
        "",
        f"- split: `{args.split}`",
        f"- processed_images: `{processed_images}`",
        f"- engines: `{', '.join([engine.name for engine in engines])}`",
        f"- unavailable_engines: `{len(unavailable_engines)}`",
        f"- crop_modes: `{', '.join(crop_modes)}`",
        f"- crop_size: `{crop_size[0]}x{crop_size[1]}`",
        f"- pad_ratio: `{float(args.pad_ratio):.4f}`",
        "",
        "## Main summary",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Bounding-box normalization effect",
        "",
        dataframe_to_markdown(normalization_delta_df),
        "",
        "## Output files",
        "",
        f"- predictions: `{pred_path}`",
        f"- summary: `{summary_path}`",
        f"- normalization_delta: `{normalization_delta_path}`",
        f"- recognition_errors: `{errors_path}`",
        f"- crop_failures: `{crop_failures_path}`",
        f"- engine_errors: `{engine_errors_path}`",
        "",
    ]

    if unavailable_engines:
        unavailable_df = pd.DataFrame(unavailable_engines)
        report_lines.extend(
            [
                "## Unavailable engines",
                "",
                dataframe_to_markdown(unavailable_df),
                "",
            ]
        )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    logging.info("External OCR baseline evaluation finished.")
    logging.info("Predictions: %s", pred_path)
    logging.info("Summary: %s", summary_path)
    logging.info("Report: %s", report_path)

    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare external OCR engines on CCPD plate crops: "
            "Tesseract, EasyOCR, PaddleOCR. "
            "Also compares bbox normalization modes: bbox, corners, bbox_pad."
        )
    )

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-dir", type=str, default="outputs/ocr_external_baselines")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--engines",
        type=str,
        default="tesseract,easyocr,paddleocr",
        help="Comma-separated list: tesseract,easyocr,paddleocr",
    )

    parser.add_argument(
        "--crop-modes",
        type=str,
        default="bbox,corners,bbox_pad",
        help="Comma-separated list: bbox,corners,bbox_pad",
    )
    parser.add_argument("--crop-width", type=int, default=224)
    parser.add_argument("--crop-height", type=int, default=64)
    parser.add_argument("--pad-ratio", type=float, default=0.08)

    parser.add_argument("--tesseract-lang", type=str, default="chi_sim+eng")
    parser.add_argument("--tesseract-psm", type=int, default=7)
    parser.add_argument("--tesseract-oem", type=int, default=3)
    parser.add_argument(
        "--tesseract-preprocess",
        type=str,
        default="gray",
        choices=["none", "gray", "threshold"],
    )

    parser.add_argument("--easyocr-langs", type=str, default="ch_sim,en")
    parser.add_argument("--easyocr-gpu", action="store_true")

    parser.add_argument("--paddleocr-lang", type=str, default="ch")
    parser.add_argument("--paddleocr-gpu", action="store_true")

    parser.add_argument("--save-all-crops", action="store_true")
    parser.add_argument("--save-error-crops", dest="save_error_crops", action="store_true", default=True)
    parser.add_argument("--no-save-error-crops", dest="save_error_crops", action="store_false")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_external_ocr_baselines(args)


if __name__ == "__main__":
    main()