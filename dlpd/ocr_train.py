from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ocr_dataset import OcrManifestDataset
from .ocr_infer import load_ocr_model, resolve_torch_device
from .ocr_model import build_ocr_model
from .plate_text import decode_plate_indices
from .utils import dump_json, ensure_dir, seed_everything


def _compute_metrics(targets: torch.Tensor, preds: torch.Tensor) -> Dict[str, float]:
    targets_np = targets.cpu().numpy()
    preds_np = preds.cpu().numpy()

    char_acc = float((targets_np == preds_np).mean())
    full_acc = float(np.all(targets_np == preds_np, axis=1).mean())
    return {
        "char_accuracy": char_acc,
        "full_plate_accuracy": full_acc,
    }


def _forward_batch(model, xb: torch.Tensor):
    logits = model(xb)
    pred_idx = torch.stack([torch.argmax(z, dim=1) for z in logits], dim=1)
    return logits, pred_idx


def _loss_multihead(logits: List[torch.Tensor], yb: torch.Tensor, criterion) -> torch.Tensor:
    loss = 0.0
    for i in range(7):
        loss = loss + criterion(logits[i], yb[:, i])
    return loss


def train_ocr_model(
    manifest_train: Path,
    manifest_val: Path,
    out_dir: Path,
    model_name: str,
    image_size: Tuple[int, int],
    epochs: int,
    batch: int,
    lr: float,
    weight_decay: float,
    device: str,
    num_workers: int,
    seed: int,
) -> Path:
    seed_everything(seed)
    out_dir = ensure_dir(out_dir)

    dev = resolve_torch_device(device)

    ds_train = OcrManifestDataset(manifest_train, image_size=image_size, augment=True)
    ds_val = OcrManifestDataset(manifest_val, image_size=image_size, augment=False)

    dl_train = DataLoader(
        ds_train,
        batch_size=int(batch),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=dev.type == "cuda",
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(batch),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=dev.type == "cuda",
        drop_last=False,
    )

    model = build_ocr_model(backbone_name=model_name).to(dev)
    optimizer = AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.CrossEntropyLoss()

    best_full_acc = -1.0
    history_rows: List[Dict] = []

    image_width = int(image_size[0])
    image_height = int(image_size[1])

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses: List[float] = []
        train_targets: List[torch.Tensor] = []
        train_preds: List[torch.Tensor] = []

        for xb, yb, _, _ in tqdm(dl_train, desc=f"OCR train epoch {epoch}/{epochs}"):
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, pred_idx = _forward_batch(model, xb)
            loss = _loss_multihead(logits, yb, criterion)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_targets.append(yb.detach().cpu())
            train_preds.append(pred_idx.detach().cpu())

        model.eval()
        val_losses: List[float] = []
        val_targets: List[torch.Tensor] = []
        val_preds: List[torch.Tensor] = []

        with torch.no_grad():
            for xb, yb, _, _ in tqdm(dl_val, desc=f"OCR val epoch {epoch}/{epochs}"):
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)

                logits, pred_idx = _forward_batch(model, xb)
                loss = _loss_multihead(logits, yb, criterion)

                val_losses.append(float(loss.item()))
                val_targets.append(yb.detach().cpu())
                val_preds.append(pred_idx.detach().cpu())

        train_targets_t = torch.cat(train_targets, dim=0)
        train_preds_t = torch.cat(train_preds, dim=0)
        val_targets_t = torch.cat(val_targets, dim=0)
        val_preds_t = torch.cat(val_preds, dim=0)

        train_metrics = _compute_metrics(train_targets_t, train_preds_t)
        val_metrics = _compute_metrics(val_targets_t, val_preds_t)

        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "train_char_accuracy": float(train_metrics["char_accuracy"]),
            "train_full_plate_accuracy": float(train_metrics["full_plate_accuracy"]),
            "val_char_accuracy": float(val_metrics["char_accuracy"]),
            "val_full_plate_accuracy": float(val_metrics["full_plate_accuracy"]),
        }
        history_rows.append(row)

        logging.info(
            "OCR epoch=%d train_loss=%.6f val_loss=%.6f train_char=%.4f train_full=%.4f val_char=%.4f val_full=%.4f",
            row["epoch"],
            row["train_loss"],
            row["val_loss"],
            row["train_char_accuracy"],
            row["train_full_plate_accuracy"],
            row["val_char_accuracy"],
            row["val_full_plate_accuracy"],
        )

        ckpt = {
            "model_name": str(model_name),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_full_plate_accuracy": float(row["val_full_plate_accuracy"]),
            "val_char_accuracy": float(row["val_char_accuracy"]),
        }
        torch.save(ckpt, out_dir / "last.pt")

        if row["val_full_plate_accuracy"] > best_full_acc:
            best_full_acc = float(row["val_full_plate_accuracy"])
            torch.save(ckpt, out_dir / "best.pt")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "history.csv", index=False)

    summary = {
        "manifest_train": str(Path(manifest_train).resolve()),
        "manifest_val": str(Path(manifest_val).resolve()),
        "out_dir": str(Path(out_dir).resolve()),
        "model_name": str(model_name),
        "image_size": [int(image_width), int(image_height)],
        "epochs": int(epochs),
        "batch": int(batch),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "device": str(dev),
        "best_val_full_plate_accuracy": float(best_full_acc),
        "final_epoch": int(epochs),
        "best_weights": str((out_dir / "best.pt").resolve()),
        "last_weights": str((out_dir / "last.pt").resolve()),
    }
    dump_json(summary, out_dir / "summary.json")

    report_md = "\n".join(
        [
            "# OCR Training Report",
            "",
            f"- model: `{model_name}`",
            f"- image_size: `{image_width}x{image_height}`",
            f"- epochs: `{epochs}`",
            f"- batch: `{batch}`",
            f"- lr: `{lr}`",
            f"- weight_decay: `{weight_decay}`",
            f"- best_val_full_plate_accuracy: `{best_full_acc:.6f}`",
            "",
            "## Last epochs",
            "",
            history_df.tail(10).to_markdown(index=False),
            "",
        ]
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    return out_dir


def evaluate_ocr_model(
    manifest_test: Path,
    weights: Path,
    out_dir: Path,
    image_size: Tuple[int, int],
    batch: int,
    device: str,
    num_workers: int,
    seed: int,
) -> Path:
    seed_everything(seed)
    out_dir = ensure_dir(out_dir)

    model, dev, ckpt_image_size = load_ocr_model(weights, device=device)

    requested_w = int(image_size[0])
    requested_h = int(image_size[1])
    ckpt_w = int(ckpt_image_size[0])
    ckpt_h = int(ckpt_image_size[1])

    if (requested_w, requested_h) != (ckpt_w, ckpt_h):
        logging.warning(
            "OCR eval image_size from config (%dx%d) differs from checkpoint (%dx%d). "
            "Checkpoint size will be used.",
            requested_w, requested_h, ckpt_w, ckpt_h
        )

    ds_test = OcrManifestDataset(manifest_test, image_size=(ckpt_w, ckpt_h), augment=False)
    dl_test = DataLoader(
        ds_test,
        batch_size=int(batch),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=dev.type == "cuda",
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()

    all_rows: List[Dict] = []
    losses: List[float] = []
    all_targets: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for xb, yb, texts, paths in tqdm(dl_test, desc="OCR eval"):
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)

            logits, pred_idx = _forward_batch(model, xb)
            loss = _loss_multihead(logits, yb, criterion)
            losses.append(float(loss.item()))

            probs = [torch.softmax(z, dim=1) for z in logits]
            confs = torch.stack([torch.max(p, dim=1).values for p in probs], dim=1)

            all_targets.append(yb.detach().cpu())
            all_preds.append(pred_idx.detach().cpu())

            yb_cpu = yb.detach().cpu().numpy()
            pred_cpu = pred_idx.detach().cpu().numpy()
            conf_cpu = confs.detach().cpu().numpy()

            for i in range(len(texts)):
                gt_indices = [int(v) for v in yb_cpu[i].tolist()]
                pred_indices = [int(v) for v in pred_cpu[i].tolist()]
                gt_text = decode_plate_indices(gt_indices)
                pred_text = decode_plate_indices(pred_indices)

                row = {
                    "img_path": str(paths[i]),
                    "gt_text": gt_text,
                    "pred_text": pred_text,
                    "full_match": int(gt_indices == pred_indices),
                    "plate_confidence": float(np.mean(conf_cpu[i])),
                    "c0": float(conf_cpu[i][0]),
                    "c1": float(conf_cpu[i][1]),
                    "c2": float(conf_cpu[i][2]),
                    "c3": float(conf_cpu[i][3]),
                    "c4": float(conf_cpu[i][4]),
                    "c5": float(conf_cpu[i][5]),
                    "c6": float(conf_cpu[i][6]),
                    "gt_p0": gt_indices[0],
                    "gt_p1": gt_indices[1],
                    "gt_p2": gt_indices[2],
                    "gt_p3": gt_indices[3],
                    "gt_p4": gt_indices[4],
                    "gt_p5": gt_indices[5],
                    "gt_p6": gt_indices[6],
                    "pred_p0": pred_indices[0],
                    "pred_p1": pred_indices[1],
                    "pred_p2": pred_indices[2],
                    "pred_p3": pred_indices[3],
                    "pred_p4": pred_indices[4],
                    "pred_p5": pred_indices[5],
                    "pred_p6": pred_indices[6],
                }
                all_rows.append(row)

    targets_t = torch.cat(all_targets, dim=0)
    preds_t = torch.cat(all_preds, dim=0)
    metrics = _compute_metrics(targets_t, preds_t)

    pred_df = pd.DataFrame(all_rows)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    summary = {
        "manifest_test": str(Path(manifest_test).resolve()),
        "weights": str(Path(weights).resolve()),
        "n_samples": int(len(pred_df)),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "char_accuracy": float(metrics["char_accuracy"]),
        "full_plate_accuracy": float(metrics["full_plate_accuracy"]),
        "mean_plate_confidence": float(pred_df["plate_confidence"].mean()) if len(pred_df) else 0.0,
    }
    dump_json(summary, out_dir / "summary.json")

    head_rows = pred_df.head(30)[["img_path", "gt_text", "pred_text", "full_match", "plate_confidence"]]
    report_md = "\n".join(
        [
            "# OCR Evaluation Report",
            "",
            f"- n_samples: `{summary['n_samples']}`",
            f"- loss: `{summary['loss']:.6f}`",
            f"- char_accuracy: `{summary['char_accuracy']:.6f}`",
            f"- full_plate_accuracy: `{summary['full_plate_accuracy']:.6f}`",
            f"- mean_plate_confidence: `{summary['mean_plate_confidence']:.6f}`",
            "",
            "## Sample predictions",
            "",
            head_rows.to_markdown(index=False),
            "",
        ]
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    return out_dir