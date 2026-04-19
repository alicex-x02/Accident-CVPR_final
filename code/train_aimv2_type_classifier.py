from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.aimv2_type_classifier import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_ENCODER_PATH,
    TYPE_LABELS,
    AIMv2TypeClassifier,
    AccidentTypeDataset,
    TypeBatchCollator,
    build_aimv2_encoder,
    load_records_from_dataframe,
    pick_best_cuda_device,
    save_checkpoint,
)


DEFAULT_LABELS_CSV = "/root/Desktop/workspace/woo/ACCIDENT@CVPR/data/raw/accident/sim_dataset/labels.csv"
DEFAULT_VIDEO_ROOT = "/root/Desktop/workspace/woo/ACCIDENT@CVPR/data/raw/accident/sim_dataset"


def resolve_device(name: str) -> torch.device:
    normalized = (name or "auto").strip().lower()
    if normalized in {"auto", "cuda"}:
        return pick_best_cuda_device()
    return torch.device(normalized)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def stratified_split_indices(labels: Sequence[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=np.int64)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_idx in range(len(TYPE_LABELS)):
        class_indices = np.flatnonzero(labels_arr == class_idx)
        if class_indices.size == 0:
            continue
        rng.shuffle(class_indices)
        if class_indices.size == 1:
            val_count = 0
        else:
            val_count = int(round(class_indices.size * val_ratio))
            val_count = max(1, min(val_count, class_indices.size - 1))
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def compute_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict[str, object]:
    targets_arr = np.asarray(targets, dtype=np.int64)
    preds_arr = np.asarray(predictions, dtype=np.int64)
    num_classes = len(TYPE_LABELS)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(targets_arr, preds_arr):
        if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
            confusion[true_idx, pred_idx] += 1

    per_class = []
    for class_idx, label in enumerate(TYPE_LABELS):
        tp = int(confusion[class_idx, class_idx])
        fp = int(confusion[:, class_idx].sum() - tp)
        fn = int(confusion[class_idx, :].sum() - tp)
        support = int(confusion[class_idx, :].sum())
        precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
        recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
        f1 = float((2.0 * precision * recall) / (precision + recall)) if precision + recall > 0 else 0.0
        per_class.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    total = int(confusion.sum())
    accuracy = float(confusion.trace() / total) if total > 0 else 0.0
    macro_precision = float(np.mean([row["precision"] for row in per_class])) if per_class else 0.0
    macro_recall = float(np.mean([row["recall"] for row in per_class])) if per_class else 0.0
    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "num_samples": total,
    }


def log_metrics(title: str, metrics: dict[str, object]) -> None:
    print(f"\n{title}")
    print(
        "  accuracy={accuracy:.4f} | macro_precision={macro_precision:.4f} | macro_recall={macro_recall:.4f} | macro_f1={macro_f1:.4f}".format(
            accuracy=float(metrics["accuracy"]),
            macro_precision=float(metrics["macro_precision"]),
            macro_recall=float(metrics["macro_recall"]),
            macro_f1=float(metrics["macro_f1"]),
        )
    )
    per_class_df = pd.DataFrame(metrics["per_class"])
    print(per_class_df.to_string(index=False))
    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=TYPE_LABELS, columns=TYPE_LABELS)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())


def compute_class_weight_tensor(labels: Sequence[int]) -> tuple[torch.Tensor, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=len(TYPE_LABELS)).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(TYPE_LABELS) * safe_counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts


def run_epoch(
    model: AIMv2TypeClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> tuple[float, dict[str, object]]:
    if train and optimizer is None:
        raise ValueError("optimizer is required when train=True")

    if train:
        model.train()
    else:
        model.eval()
    model.freeze_encoder()

    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in iterator:
        pixel_values = batch["pixel_values"].to(device=device, dtype=model.encoder_dtype, non_blocking=True)
        labels = batch["labels"].to(device=device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(pixel_values)
                loss = criterion(logits, labels)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        batch_predictions = torch.argmax(logits, dim=-1)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(batch_predictions.detach().cpu().tolist())
        iterator.set_postfix(loss=float(loss.item()))

    average_loss = total_loss / max(len(targets), 1)
    metrics = compute_metrics(targets, predictions)
    metrics["loss"] = float(average_loss)
    return average_loss, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an AIMv2 5-frame accident type classifier")
    parser.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--encoder-path", default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--jitter-sec", type=float, default=1.0)
    parser.add_argument("--max-decode-side", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    seed_everything(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Labels CSV: %s", args.labels_csv)
    logging.info("Video root: %s", args.video_root)
    logging.info("Encoder path: %s", args.encoder_path)
    logging.info("Output dir: %s", args.output_dir)
    logging.info("Device: %s", device)

    labels_df = pd.read_csv(args.labels_csv)
    records = load_records_from_dataframe(labels_df, args.video_root)
    labels = [record.label for record in records]
    train_indices, val_indices = stratified_split_indices(labels, args.val_ratio, args.seed)

    train_records = [records[idx] for idx in train_indices]
    val_records = [records[idx] for idx in val_indices]
    train_dataset = AccidentTypeDataset(train_records)
    val_dataset = AccidentTypeDataset(val_records)

    logging.info("Dataset size: %d | train=%d | val=%d", len(records), len(train_dataset), len(val_dataset))
    train_weight_tensor, train_counts = compute_class_weight_tensor([record.label for record in train_records])
    logging.info("Train class counts: %s", {label: int(count) for label, count in zip(TYPE_LABELS, train_counts.tolist())})
    logging.info("Class weights: %s", {label: float(weight) for label, weight in zip(TYPE_LABELS, train_weight_tensor.tolist())})

    encoder, processor, _ = build_aimv2_encoder(args.encoder_path, device=device, freeze=True)
    model = AIMv2TypeClassifier(
        encoder=encoder,
        hidden_size=int(encoder.config.hidden_size),
        num_classes=len(TYPE_LABELS),
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        freeze_encoder=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=train_weight_tensor.to(device))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=TypeBatchCollator(
            processor=processor,
            training=True,
            jitter_sec=args.jitter_sec,
            max_decode_side=args.max_decode_side,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        collate_fn=TypeBatchCollator(
            processor=processor,
            training=False,
            jitter_sec=0.0,
            max_decode_side=args.max_decode_side,
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / args.checkpoint_name
    last_checkpoint_path = output_dir / f"last_{args.checkpoint_name}"

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d/%d", epoch, args.epochs)
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            train=True,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            train=False,
        )

        log_metrics(f"[Epoch {epoch}] Train", train_metrics)
        log_metrics(f"[Epoch {epoch}] Val", val_metrics)
        logging.info(
            "Epoch %d summary | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f",
            epoch,
            train_loss,
            val_loss,
            float(val_metrics["macro_f1"]),
        )

        save_checkpoint(
            checkpoint_path=str(last_checkpoint_path),
            model=model,
            encoder_path=args.encoder_path,
            label_names=TYPE_LABELS,
            head_hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            epoch=epoch,
            best_macro_f1=best_macro_f1,
            val_metrics=val_metrics,
        )

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=str(best_checkpoint_path),
                model=model,
                encoder_path=args.encoder_path,
                label_names=TYPE_LABELS,
                head_hidden_dim=args.head_hidden_dim,
                dropout=args.dropout,
                epoch=epoch,
                best_macro_f1=best_macro_f1,
                val_metrics=val_metrics,
            )
            logging.info("New best checkpoint saved to %s", best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            logging.info("Early stopping after %d epochs without improvement", args.patience)
            break

    logging.info("Best epoch: %d | best macro-F1: %.4f", best_epoch, best_macro_f1)
    print(f"Best checkpoint saved to {best_checkpoint_path}")
    print(f"Last checkpoint saved to {last_checkpoint_path}")


if __name__ == "__main__":
    main()
