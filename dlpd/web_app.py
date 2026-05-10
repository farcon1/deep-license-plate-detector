from __future__ import annotations

import base64
import html
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from dlpd.config import load_config
from dlpd.ocr_infer import load_ocr_model, recognize_plate_crop
from dlpd.yolo_baseline import _clip_box_to_image, _extract_best_prediction, _require_ultralytics
from dlpd.utils import setup_logging
from dlpd.metrics import Box


@dataclass
class LoadedRuntime:
    detector: Any
    ocr_model: Any
    ocr_device: Any
    ocr_image_size: Tuple[int, int]
    detector_weights: Path
    ocr_weights: Path
    device: str


_runtime: Optional[LoadedRuntime] = None
_runtime_lock = threading.RLock()


def _get_env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    if value:
        return Path(value)
    return Path(default)


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return str(default)


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if value:
        return int(value)
    return int(default)


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if value:
        return float(value)
    return float(default)


def _load_runtime() -> LoadedRuntime:
    global _runtime

    with _runtime_lock:
        if _runtime is not None:
            return _runtime

        config_path = Path(os.getenv("DLPD_CONFIG", "config.yaml"))
        cfg = load_config(config_path)
        setup_logging(cfg.logging.level)

        detector_weights = _get_env_path("DETECTOR_WEIGHTS", cfg.yolo_eval.weights)
        ocr_weights = _get_env_path("OCR_WEIGHTS", cfg.ocr_eval.weights)
        device = _get_env_str("DEVICE", cfg.yolo_eval.device)

        if not detector_weights.exists():
            raise RuntimeError(f"Detector weights not found: {detector_weights}")

        if not ocr_weights.exists():
            raise RuntimeError(f"OCR weights not found: {ocr_weights}")

        YOLO = _require_ultralytics()

        logging.info("Loading YOLO detector: %s", detector_weights)
        detector = YOLO(str(detector_weights))

        logging.info("Loading OCR model: %s", ocr_weights)
        ocr_model, ocr_device, ocr_image_size = load_ocr_model(ocr_weights, device=device)

        _runtime = LoadedRuntime(
            detector=detector,
            ocr_model=ocr_model,
            ocr_device=ocr_device,
            ocr_image_size=ocr_image_size,
            detector_weights=detector_weights,
            ocr_weights=ocr_weights,
            device=device,
        )
        return _runtime


def _decode_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Не удалось прочитать изображение. Загрузите JPG, JPEG или PNG.",
        )

    return img


def _encode_bgr_to_data_url(img_bgr: np.ndarray, quality: int = 92) -> str:
    ok, encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG.")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _draw_prediction(
    img_bgr: np.ndarray,
    box: Optional[Tuple[int, int, int, int]],
    title: str,
) -> np.ndarray:
    out = img_bgr.copy()

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 80), 3)

        label = title.strip()
        if label:
            safe_label = label.encode("ascii", errors="ignore").decode("ascii").strip()
            if not safe_label:
                safe_label = "license plate"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            thickness = 2
            text_size, baseline = cv2.getTextSize(safe_label, font, font_scale, thickness)

            tx = max(0, x1)
            ty = max(28, y1 - 10)
            bg_x2 = min(out.shape[1] - 1, tx + text_size[0] + 16)
            bg_y1 = max(0, ty - text_size[1] - baseline - 12)
            bg_y2 = min(out.shape[0] - 1, ty + baseline)

            overlay = out.copy()
            cv2.rectangle(overlay, (tx, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
            out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
            cv2.putText(out, safe_label, (tx + 8, ty - 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def _safe_float(value: float, min_value: float, max_value: float, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return max(min_value, min(max_value, v))


def _safe_int(value: int, min_value: int, max_value: int, default: int) -> int:
    try:
        v = int(value)
    except Exception:
        return default
    return max(min_value, min(max_value, v))


def _run_detector(
    runtime: LoadedRuntime,
    img_bgr: np.ndarray,
    detector_imgsz: int,
    detector_conf: float,
    detector_iou: float,
    detector_max_det: int,
) -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]

    with _runtime_lock:
        results = runtime.detector.predict(
            source=img_bgr,
            imgsz=int(detector_imgsz),
            conf=float(detector_conf),
            iou=float(detector_iou),
            device=str(runtime.device),
            max_det=int(detector_max_det),
            verbose=False,
        )

    pred = _extract_best_prediction(results[0])

    if pred.box is None or float(pred.score) < 0:
        return {
            "detected": False,
            "det_score": 0.0,
            "bbox": None,
            "crop_bgr": None,
        }

    clipped: Box = _clip_box_to_image(pred.box, w=w, h=h)
    x1, y1, x2, y2 = int(clipped.x1), int(clipped.y1), int(clipped.x2), int(clipped.y2)

    if x2 <= x1 or y2 <= y1:
        return {
            "detected": False,
            "det_score": float(pred.score),
            "bbox": [x1, y1, x2, y2],
            "crop_bgr": None,
        }

    crop_bgr = img_bgr[y1:y2, x1:x2].copy()

    return {
        "detected": True,
        "det_score": float(pred.score),
        "bbox": [x1, y1, x2, y2],
        "crop_bgr": crop_bgr,
    }


def _run_ocr(runtime: LoadedRuntime, crop_bgr: np.ndarray) -> Dict[str, Any]:
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "text": "",
            "plate_confidence": 0.0,
            "char_confidences": [],
            "indices": [],
        }

    with _runtime_lock:
        pred = recognize_plate_crop(
            crop_bgr,
            model=runtime.ocr_model,
            device=runtime.ocr_device,
            image_size=runtime.ocr_image_size,
        )

    return {
        "text": str(pred.text),
        "plate_confidence": float(pred.plate_confidence),
        "char_confidences": [float(x) for x in pred.char_confidences],
        "indices": [int(x) for x in pred.indices],
    }


app = FastAPI(
    title="Deep License Plate Detector",
    description="Web UI for YOLO + OCR ALPR inference.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    try:
        _load_runtime()
        logging.info("DLPD web inference app started successfully.")
    except Exception as exc:
        logging.exception("DLPD runtime preload failed: %s", exc)


@app.get("/health")
def health() -> Dict[str, Any]:
    runtime = _load_runtime()
    return {
        "status": "ok",
        "detector_weights": str(runtime.detector_weights),
        "ocr_weights": str(runtime.ocr_weights),
        "device": str(runtime.device),
        "ocr_image_size": list(runtime.ocr_image_size),
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form("full"),
    detector_conf: float = Form(0.25),
    detector_iou: float = Form(0.70),
    detector_imgsz: int = Form(960),
    detector_max_det: int = Form(10),
) -> JSONResponse:
    t0 = time.perf_counter()

    mode = str(mode).strip().lower()
    allowed_modes = {"full", "detector", "ocr_crop"}
    if mode not in allowed_modes:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    detector_conf = _safe_float(detector_conf, 0.001, 0.999, 0.25)
    detector_iou = _safe_float(detector_iou, 0.10, 0.95, 0.70)
    detector_imgsz = _safe_int(detector_imgsz, 320, 1920, 960)
    detector_max_det = _safe_int(detector_max_det, 1, 100, 10)

    max_upload_mb = _get_env_int("MAX_UPLOAD_MB", 20)
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой.")

    if len(file_bytes) > max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой. Максимальный размер: {max_upload_mb} MB.",
        )

    img_bgr = _decode_upload_to_bgr(file_bytes)
    runtime = _load_runtime()

    original_h, original_w = img_bgr.shape[:2]

    detector_result: Dict[str, Any] = {
        "detected": False,
        "det_score": 0.0,
        "bbox": None,
        "crop_bgr": None,
    }
    ocr_result: Dict[str, Any] = {
        "text": "",
        "plate_confidence": 0.0,
        "char_confidences": [],
        "indices": [],
    }

    if mode in {"full", "detector"}:
        detector_result = _run_detector(
            runtime=runtime,
            img_bgr=img_bgr,
            detector_imgsz=detector_imgsz,
            detector_conf=detector_conf,
            detector_iou=detector_iou,
            detector_max_det=detector_max_det,
        )

    if mode == "full" and detector_result["detected"] and detector_result["crop_bgr"] is not None:
        ocr_result = _run_ocr(runtime, detector_result["crop_bgr"])

    if mode == "ocr_crop":
        ocr_result = _run_ocr(runtime, img_bgr)

    label_parts = []
    if ocr_result["text"]:
        label_parts.append(f"plate={ocr_result['text']}")
    if detector_result["detected"]:
        label_parts.append(f"det={detector_result['det_score']:.3f}")
    if ocr_result["plate_confidence"]:
        label_parts.append(f"ocr={ocr_result['plate_confidence']:.3f}")

    annotated_bgr = _draw_prediction(
        img_bgr=img_bgr,
        box=tuple(detector_result["bbox"]) if detector_result["bbox"] else None,
        title=" | ".join(label_parts),
    )

    crop_data_url = ""
    if detector_result["crop_bgr"] is not None:
        crop_data_url = _encode_bgr_to_data_url(detector_result["crop_bgr"])

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    payload: Dict[str, Any] = {
        "ok": True,
        "filename": file.filename,
        "mode": mode,
        "image": {
            "width": int(original_w),
            "height": int(original_h),
        },
        "settings": {
            "detector_conf": float(detector_conf),
            "detector_iou": float(detector_iou),
            "detector_imgsz": int(detector_imgsz),
            "detector_max_det": int(detector_max_det),
            "device": str(runtime.device),
        },
        "detection": {
            "detected": bool(detector_result["detected"]),
            "score": float(detector_result["det_score"]),
            "bbox": detector_result["bbox"],
        },
        "ocr": {
            "text": str(ocr_result["text"]),
            "plate_confidence": float(ocr_result["plate_confidence"]),
            "char_confidences": ocr_result["char_confidences"],
            "indices": ocr_result["indices"],
        },
        "images": {
            "original": _encode_bgr_to_data_url(img_bgr),
            "annotated": _encode_bgr_to_data_url(annotated_bgr),
            "crop": crop_data_url,
        },
        "timing": {
            "elapsed_ms": elapsed_ms,
        },
    }

    return JSONResponse(payload)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Deep License Plate Detector</title>
  <style>
    :root {
      --bg: #07111f;
      --bg-2: #0d1d33;
      --card: rgba(255, 255, 255, 0.085);
      --card-strong: rgba(255, 255, 255, 0.13);
      --text: #f4f7fb;
      --muted: #a9b7c9;
      --border: rgba(255, 255, 255, 0.16);
      --accent: #67e8f9;
      --accent-2: #a78bfa;
      --good: #86efac;
      --warn: #fde68a;
      --bad: #fca5a5;
      --shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
      --radius: 24px;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(103, 232, 249, 0.22), transparent 34rem),
        radial-gradient(circle at top right, rgba(167, 139, 250, 0.20), transparent 34rem),
        linear-gradient(135deg, var(--bg), var(--bg-2));
    }

    .page {
      width: min(1360px, calc(100% - 32px));
      margin: 0 auto;
      padding: 34px 0 44px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 22px;
      align-items: stretch;
      margin-bottom: 22px;
    }

    .hero-card,
    .panel,
    .result-card {
      border: 1px solid var(--border);
      background: linear-gradient(180deg, var(--card-strong), var(--card));
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
      border-radius: var(--radius);
    }

    .hero-card {
      padding: 34px;
      overflow: hidden;
      position: relative;
    }

    .hero-card::after {
      content: "";
      position: absolute;
      inset: auto -120px -150px auto;
      width: 330px;
      height: 330px;
      background: radial-gradient(circle, rgba(103, 232, 249, 0.23), transparent 68%);
      pointer-events: none;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 11px;
      border: 1px solid rgba(103, 232, 249, 0.38);
      border-radius: 999px;
      color: #cffafe;
      background: rgba(103, 232, 249, 0.08);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.02em;
      margin-bottom: 18px;
    }

    .pulse {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--good);
      box-shadow: 0 0 20px rgba(134, 239, 172, 0.95);
    }

    h1 {
      margin: 0;
      font-size: clamp(34px, 4vw, 62px);
      line-height: 0.95;
      letter-spacing: -0.055em;
    }

    .hero-text {
      margin: 18px 0 0;
      color: var(--muted);
      max-width: 760px;
      font-size: 17px;
      line-height: 1.65;
    }

    .hero-metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 28px;
    }

    .metric {
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.06);
      border-radius: 18px;
      padding: 15px;
    }

    .metric strong {
      display: block;
      font-size: 22px;
      letter-spacing: -0.03em;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }

    .status-card {
      padding: 24px;
    }

    .status-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
    }

    .status-title h2 {
      margin: 0;
      font-size: 20px;
      letter-spacing: -0.025em;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.07);
      border-radius: 999px;
      color: #dbeafe;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }

    .status-grid {
      display: grid;
      gap: 10px;
    }

    .status-row {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 10px;
      padding: 10px 0;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      color: var(--muted);
      font-size: 14px;
    }

    .status-row:last-child {
      border-bottom: none;
    }

    .status-row b {
      color: var(--text);
      overflow-wrap: anywhere;
    }

    .main-grid {
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 22px;
      align-items: start;
    }

    .panel {
      padding: 22px;
    }

    .panel h2 {
      margin: 0 0 16px;
      font-size: 22px;
      letter-spacing: -0.035em;
    }

    .upload-box {
      position: relative;
      display: grid;
      place-items: center;
      min-height: 250px;
      border: 1.5px dashed rgba(103, 232, 249, 0.48);
      border-radius: 22px;
      background:
        linear-gradient(135deg, rgba(103, 232, 249, 0.10), rgba(167, 139, 250, 0.08)),
        rgba(255, 255, 255, 0.04);
      cursor: pointer;
      transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
      overflow: hidden;
    }

    .upload-box:hover,
    .upload-box.dragover {
      transform: translateY(-2px);
      border-color: rgba(103, 232, 249, 0.92);
      background:
        linear-gradient(135deg, rgba(103, 232, 249, 0.16), rgba(167, 139, 250, 0.13)),
        rgba(255, 255, 255, 0.05);
    }

    .upload-box input {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }

    .upload-inner {
      text-align: center;
      padding: 20px;
    }

    .upload-icon {
      width: 74px;
      height: 74px;
      display: inline-grid;
      place-items: center;
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.10);
      border: 1px solid var(--border);
      font-size: 34px;
      margin-bottom: 16px;
    }

    .upload-title {
      font-size: 18px;
      font-weight: 800;
      margin-bottom: 7px;
    }

    .upload-hint {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }

    .file-name {
      margin-top: 12px;
      color: #cffafe;
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    .controls {
      display: grid;
      gap: 16px;
      margin-top: 18px;
    }

    .field {
      display: grid;
      gap: 8px;
    }

    .field label {
      font-size: 13px;
      color: #dbeafe;
      font-weight: 800;
      letter-spacing: 0.01em;
    }

    .field small {
      color: var(--muted);
      line-height: 1.45;
    }

    select,
    input[type="number"],
    input[type="range"] {
      width: 100%;
    }

    select,
    input[type="number"] {
      border: 1px solid var(--border);
      background: rgba(4, 12, 24, 0.78);
      color: var(--text);
      border-radius: 14px;
      padding: 12px 13px;
      outline: none;
      font-size: 14px;
    }

    input[type="range"] {
      accent-color: var(--accent);
    }

    .range-row {
      display: grid;
      grid-template-columns: 1fr 72px;
      gap: 10px;
      align-items: center;
    }

    .button {
      width: 100%;
      margin-top: 8px;
      border: none;
      border-radius: 18px;
      padding: 15px 18px;
      color: #04111f;
      font-weight: 900;
      font-size: 15px;
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      box-shadow: 0 16px 40px rgba(103, 232, 249, 0.22);
      transition: transform 0.16s ease, filter 0.16s ease;
    }

    .button:hover {
      transform: translateY(-1px);
      filter: brightness(1.05);
    }

    .button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
      transform: none;
    }

    .result-empty {
      min-height: 560px;
      display: grid;
      place-items: center;
      text-align: center;
      color: var(--muted);
      padding: 32px;
    }

    .result-empty .big {
      font-size: 56px;
      margin-bottom: 14px;
    }

    .result-card {
      padding: 22px;
      display: none;
    }

    .result-top {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: start;
      margin-bottom: 18px;
    }

    .plate-value {
      display: inline-flex;
      align-items: center;
      min-height: 58px;
      padding: 12px 18px;
      border-radius: 18px;
      border: 1px solid rgba(134, 239, 172, 0.36);
      background: rgba(134, 239, 172, 0.10);
      font-size: clamp(28px, 4vw, 46px);
      font-weight: 950;
      letter-spacing: 0.02em;
      color: #dcfce7;
      overflow-wrap: anywhere;
    }

    .plate-muted {
      border-color: rgba(253, 230, 138, 0.38);
      background: rgba(253, 230, 138, 0.10);
      color: #fef3c7;
      font-size: 22px;
    }

    .result-meta {
      display: flex;
      gap: 9px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .meta-pill {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 11px;
      background: rgba(255,255,255,0.07);
      color: #dbeafe;
      font-size: 13px;
      font-weight: 800;
      white-space: nowrap;
    }

    .image-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 18px;
    }

    .image-card {
      border: 1px solid var(--border);
      border-radius: 20px;
      overflow: hidden;
      background: rgba(0,0,0,0.25);
    }

    .image-card h3 {
      margin: 0;
      padding: 13px 15px;
      font-size: 14px;
      color: #e0f2fe;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.055);
    }

    .image-card img {
      display: block;
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      background: rgba(0,0,0,0.38);
    }

    .crop-card {
      margin-top: 16px;
    }

    .crop-card img {
      max-height: 180px;
      object-fit: contain;
    }

    .details {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }

    .detail {
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.06);
      border-radius: 16px;
      padding: 12px;
    }

    .detail span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }

    .detail b {
      display: block;
      color: var(--text);
      font-size: 15px;
      overflow-wrap: anywhere;
    }

    .error {
      display: none;
      margin-top: 14px;
      padding: 13px 14px;
      border-radius: 16px;
      border: 1px solid rgba(252, 165, 165, 0.36);
      background: rgba(252, 165, 165, 0.10);
      color: #fee2e2;
      line-height: 1.5;
    }

    .loader {
      display: none;
      margin-top: 14px;
      padding: 13px 14px;
      border-radius: 16px;
      border: 1px solid rgba(103, 232, 249, 0.30);
      background: rgba(103, 232, 249, 0.09);
      color: #cffafe;
      line-height: 1.5;
    }

    @media (max-width: 1050px) {
      .hero,
      .main-grid {
        grid-template-columns: 1fr;
      }

      .image-grid,
      .details {
        grid-template-columns: 1fr;
      }

      .result-top {
        grid-template-columns: 1fr;
      }

      .result-meta {
        justify-content: flex-start;
      }
    }

    @media (max-width: 640px) {
      .page {
        width: min(100% - 20px, 1360px);
        padding-top: 16px;
      }

      .hero-card,
      .panel,
      .result-card,
      .status-card {
        padding: 18px;
        border-radius: 20px;
      }

      .hero-metrics {
        grid-template-columns: 1fr;
      }

      .range-row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow"><span class="pulse"></span> ALPR inference web interface</div>
        <h1>Deep License Plate Detector</h1>
        <p class="hero-text">
          Загрузите изображение автомобиля, выберите режим работы и получите результат:
          найденная область номерного знака, OCR-расшифровка, confidence и визуализация bbox.
        </p>
        <div class="hero-metrics">
          <div class="metric">
            <strong>YOLO</strong>
            <span>детекция номера</span>
          </div>
          <div class="metric">
            <strong>OCR</strong>
            <span>распознавание crop</span>
          </div>
          <div class="metric">
            <strong>Docker</strong>
            <span>готовый infer-контур</span>
          </div>
        </div>
      </div>

      <div class="hero-card status-card">
        <div class="status-title">
          <h2>Runtime</h2>
          <span class="badge">● online</span>
        </div>
        <div class="status-grid" id="runtimeStatus">
          <div class="status-row"><span>Статус</span><b>Загрузка...</b></div>
        </div>
      </div>
    </section>

    <section class="main-grid">
      <aside class="panel">
        <h2>Настройки инференса</h2>

        <form id="predictForm">
          <label class="upload-box" id="dropZone">
            <input id="fileInput" type="file" name="file" accept="image/png,image/jpeg,image/jpg" required>
            <div class="upload-inner">
              <div class="upload-icon">↥</div>
              <div class="upload-title">Загрузите изображение</div>
              <div class="upload-hint">JPG, JPEG или PNG. Можно перетащить файл сюда.</div>
              <div class="file-name" id="fileName"></div>
            </div>
          </label>

          <div class="controls">
            <div class="field">
              <label for="mode">Компоненты</label>
              <select id="mode" name="mode">
                <option value="full" selected>Детекция + OCR</option>
                <option value="detector">Только детекция номера</option>
                <option value="ocr_crop">Только OCR по crop номеру</option>
              </select>
              <small>OCR-only используйте, если загружаете уже вырезанный номерной знак.</small>
            </div>

            <div class="field">
              <label for="detectorConf">YOLO confidence threshold</label>
              <div class="range-row">
                <input id="detectorConf" name="detector_conf" type="range" min="0.001" max="0.999" step="0.001" value="0.25">
                <input id="detectorConfValue" type="number" min="0.001" max="0.999" step="0.001" value="0.25">
              </div>
              <small>Чем выше значение, тем строже детектор.</small>
            </div>

            <div class="field">
              <label for="detectorIou">YOLO NMS IoU</label>
              <div class="range-row">
                <input id="detectorIou" name="detector_iou" type="range" min="0.10" max="0.95" step="0.01" value="0.70">
                <input id="detectorIouValue" type="number" min="0.10" max="0.95" step="0.01" value="0.70">
              </div>
              <small>Порог подавления пересекающихся bbox.</small>
            </div>

            <div class="field">
              <label for="detectorImgsz">YOLO image size</label>
              <select id="detectorImgsz" name="detector_imgsz">
                <option value="640">640</option>
                <option value="768">768</option>
                <option value="960" selected>960</option>
                <option value="1280">1280</option>
              </select>
            </div>

            <div class="field">
              <label for="detectorMaxDet">Max detections</label>
              <input id="detectorMaxDet" name="detector_max_det" type="number" min="1" max="100" step="1" value="10">
            </div>

            <button class="button" id="submitButton" type="submit">Запустить распознавание</button>
          </div>

          <div class="loader" id="loader">Выполняю инференс. Модель ищет номер и запускает OCR...</div>
          <div class="error" id="errorBox"></div>
        </form>
      </aside>

      <section>
        <div class="panel result-empty" id="emptyState">
          <div>
            <div class="big">▣</div>
            <h2>Результат появится здесь</h2>
            <p>После загрузки изображения система покажет найденный номер, bbox, crop и confidence.</p>
          </div>
        </div>

        <div class="result-card" id="resultCard">
          <div class="result-top">
            <div>
              <div class="plate-value" id="plateValue">—</div>
            </div>
            <div class="result-meta">
              <span class="meta-pill" id="modePill">mode</span>
              <span class="meta-pill" id="timePill">0 ms</span>
              <span class="meta-pill" id="sizePill">0×0</span>
            </div>
          </div>

          <div class="details">
            <div class="detail">
              <span>Детекция</span>
              <b id="detectedValue">—</b>
            </div>
            <div class="detail">
              <span>YOLO score</span>
              <b id="detScoreValue">—</b>
            </div>
            <div class="detail">
              <span>OCR confidence</span>
              <b id="ocrConfidenceValue">—</b>
            </div>
            <div class="detail">
              <span>BBox</span>
              <b id="bboxValue">—</b>
            </div>
          </div>

          <div class="image-grid">
            <div class="image-card">
              <h3>Исходное изображение</h3>
              <img id="originalImage" alt="Исходное изображение">
            </div>
            <div class="image-card">
              <h3>Результат с выделенным номером</h3>
              <img id="annotatedImage" alt="Результат">
            </div>
          </div>

          <div class="image-card crop-card" id="cropCard">
            <h3>Вырезанный номерной знак</h3>
            <img id="cropImage" alt="Crop номерного знака">
          </div>
        </div>
      </section>
    </section>
  </main>

  <script>
    const form = document.getElementById("predictForm");
    const fileInput = document.getElementById("fileInput");
    const fileName = document.getElementById("fileName");
    const dropZone = document.getElementById("dropZone");
    const submitButton = document.getElementById("submitButton");
    const loader = document.getElementById("loader");
    const errorBox = document.getElementById("errorBox");
    const emptyState = document.getElementById("emptyState");
    const resultCard = document.getElementById("resultCard");

    const detectorConf = document.getElementById("detectorConf");
    const detectorConfValue = document.getElementById("detectorConfValue");
    const detectorIou = document.getElementById("detectorIou");
    const detectorIouValue = document.getElementById("detectorIouValue");

    const plateValue = document.getElementById("plateValue");
    const modePill = document.getElementById("modePill");
    const timePill = document.getElementById("timePill");
    const sizePill = document.getElementById("sizePill");
    const detectedValue = document.getElementById("detectedValue");
    const detScoreValue = document.getElementById("detScoreValue");
    const ocrConfidenceValue = document.getElementById("ocrConfidenceValue");
    const bboxValue = document.getElementById("bboxValue");
    const originalImage = document.getElementById("originalImage");
    const annotatedImage = document.getElementById("annotatedImage");
    const cropCard = document.getElementById("cropCard");
    const cropImage = document.getElementById("cropImage");
    const runtimeStatus = document.getElementById("runtimeStatus");

    function syncRange(range, number) {
      range.addEventListener("input", () => {
        number.value = range.value;
      });
      number.addEventListener("input", () => {
        range.value = number.value;
      });
    }

    syncRange(detectorConf, detectorConfValue);
    syncRange(detectorIou, detectorIouValue);

    function setError(message) {
      if (!message) {
        errorBox.style.display = "none";
        errorBox.textContent = "";
        return;
      }
      errorBox.style.display = "block";
      errorBox.textContent = message;
    }

    function setLoading(value) {
      loader.style.display = value ? "block" : "none";
      submitButton.disabled = value;
    }

    function formatNumber(value, digits = 4) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "—";
      }
      return Number(value).toFixed(digits);
    }

    function modeLabel(mode) {
      if (mode === "full") return "Детекция + OCR";
      if (mode === "detector") return "Только детекция";
      if (mode === "ocr_crop") return "Только OCR crop";
      return mode;
    }

    function updateFileName() {
      const file = fileInput.files && fileInput.files[0];
      fileName.textContent = file ? file.name : "";
    }

    fileInput.addEventListener("change", updateFileName);

    ["dragenter", "dragover"].forEach(eventName => {
      dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", event => {
      const files = event.dataTransfer.files;
      if (files && files.length > 0) {
        fileInput.files = files;
        updateFileName();
      }
    });

    async function loadHealth() {
      try {
        const response = await fetch("/health");
        const data = await response.json();

        runtimeStatus.innerHTML = `
          <div class="status-row"><span>Статус</span><b>${data.status}</b></div>
          <div class="status-row"><span>Device</span><b>${data.device}</b></div>
          <div class="status-row"><span>Detector</span><b>${data.detector_weights}</b></div>
          <div class="status-row"><span>OCR</span><b>${data.ocr_weights}</b></div>
          <div class="status-row"><span>OCR size</span><b>${data.ocr_image_size.join("×")}</b></div>
        `;
      } catch (error) {
        runtimeStatus.innerHTML = `
          <div class="status-row"><span>Статус</span><b>Ошибка runtime</b></div>
          <div class="status-row"><span>Детали</span><b>${String(error)}</b></div>
        `;
      }
    }

    form.addEventListener("submit", async event => {
      event.preventDefault();
      setError("");

      if (!fileInput.files || !fileInput.files[0]) {
        setError("Сначала загрузите изображение.");
        return;
      }

      const formData = new FormData(form);

      setLoading(true);

      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (!response.ok || !data.ok) {
          throw new Error(data.detail || "Ошибка инференса.");
        }

        emptyState.style.display = "none";
        resultCard.style.display = "block";

        const text = data.ocr && data.ocr.text ? data.ocr.text : "";
        if (text) {
          plateValue.textContent = text;
          plateValue.classList.remove("plate-muted");
        } else if (data.detection.detected) {
          plateValue.textContent = "Номер найден, OCR не запускался или не дал текст";
          plateValue.classList.add("plate-muted");
        } else {
          plateValue.textContent = "Номер не найден";
          plateValue.classList.add("plate-muted");
        }

        modePill.textContent = modeLabel(data.mode);
        timePill.textContent = `${data.timing.elapsed_ms} ms`;
        sizePill.textContent = `${data.image.width}×${data.image.height}`;

        detectedValue.textContent = data.detection.detected ? "Да" : "Нет";
        detScoreValue.textContent = formatNumber(data.detection.score, 4);
        ocrConfidenceValue.textContent = formatNumber(data.ocr.plate_confidence, 4);
        bboxValue.textContent = data.detection.bbox ? `[${data.detection.bbox.join(", ")}]` : "—";

        originalImage.src = data.images.original;
        annotatedImage.src = data.images.annotated;

        if (data.images.crop) {
          cropCard.style.display = "block";
          cropImage.src = data.images.crop;
        } else {
          cropCard.style.display = "none";
          cropImage.removeAttribute("src");
        }
      } catch (error) {
        setError(String(error.message || error));
      } finally {
        setLoading(false);
      }
    });

    loadHealth();
  </script>
</body>
</html>
        """
    )