from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from .ocr_model import build_ocr_model
from .plate_text import decode_plate_indices


@dataclass
class OcrPrediction:
    indices: List[int]
    text: str
    char_confidences: List[float]
    plate_confidence: float


def resolve_torch_device(device: str) -> torch.device:
    d = str(device).strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(d)
        return torch.device("cpu")
    if d.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{d}")
        return torch.device("cpu")
    return torch.device("cpu")


def load_ocr_model(weights_path: Path | str, device: str):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"OCR weights not found: {weights_path}")

    dev = resolve_torch_device(device)
    ckpt = torch.load(str(weights_path), map_location=dev)

    model_name = str(ckpt.get("model_name", "resnet18"))
    image_width = int(ckpt.get("image_width", 224))
    image_height = int(ckpt.get("image_height", 64))

    model = build_ocr_model(backbone_name=model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    return model, dev, (image_width, image_height)


def _preprocess_crop(img_bgr: np.ndarray, image_size: Tuple[int, int]) -> torch.Tensor:
    image_width = int(image_size[0])
    image_height = int(image_size[1])

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (image_width, image_height), interpolation=cv2.INTER_CUBIC)

    x = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).float().unsqueeze(0)
    return x


@torch.no_grad()
def recognize_plate_crop(
    img_bgr: np.ndarray,
    model,
    device: torch.device,
    image_size: Tuple[int, int],
) -> OcrPrediction:
    x = _preprocess_crop(img_bgr, image_size=image_size).to(device)
    logits = model(x)

    probs = [torch.softmax(z, dim=1) for z in logits]
    indices = [int(torch.argmax(p, dim=1).item()) for p in probs]
    char_confidences = [float(torch.max(p, dim=1).values.item()) for p in probs]
    plate_confidence = float(np.mean(char_confidences))
    text = decode_plate_indices(indices)

    return OcrPrediction(
        indices=indices,
        text=text,
        char_confidences=char_confidences,
        plate_confidence=plate_confidence,
    )