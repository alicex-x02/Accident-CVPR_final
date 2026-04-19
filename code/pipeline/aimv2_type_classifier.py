from __future__ import annotations

import json
import os
import random
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn
try:
    from transformers import AutoImageProcessor
except ImportError:  # pragma: no cover - fallback for older transformer builds
    from transformers import AutoProcessor as AutoImageProcessor
from transformers import AutoModel
try:
    from transformers import Aimv2VisionModel
except ImportError:  # pragma: no cover - fallback if the installed build lacks native AIMv2 support
    Aimv2VisionModel = None

from .optical_flow import VideoInfo, get_video_info


TYPE_LABELS = ("rear-end", "head-on", "sideswipe", "t-bone", "single")
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(TYPE_LABELS)}
FRAME_OFFSETS_SEC = (-0.75, -0.25, 0.0, 0.25, 0.75)
DEFAULT_ENCODER_PATH = "/root/Desktop/workspace/yujin/models/apple/aimv2-large-patch14-224"
DEFAULT_CHECKPOINT_DIR = "/root/Desktop/workspace/yujin/models/aimv2_type_classifier"
DEFAULT_CHECKPOINT_FILE = "best.pt"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def pick_best_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        visible_map = None
        if visible_env:
            visible_map = [item.strip() for item in visible_env.split(",") if item.strip()]

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        best_physical_idx = 0
        best_free_mib = -1
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            physical_idx = int(parts[0])
            free_mib = int(parts[1])
            if visible_map is not None and str(physical_idx) not in visible_map:
                continue
            if free_mib > best_free_mib:
                best_free_mib = free_mib
                best_physical_idx = physical_idx

        if visible_map is not None:
            visible_index = visible_map.index(str(best_physical_idx))
            return torch.device(f"cuda:{visible_index}")
        return torch.device(f"cuda:{best_physical_idx}")
    except Exception:
        return torch.device("cuda:0")


def normalize_type_label(label: str) -> str:
    normalized = (label or "").strip().lower().replace("_", "-")
    aliases = {
        "rear end": "rear-end",
        "rear-end": "rear-end",
        "head on": "head-on",
        "head-on": "head-on",
        "side swipe": "sideswipe",
        "side-swipe": "sideswipe",
        "sideswipe": "sideswipe",
        "t bone": "t-bone",
        "t-bone": "t-bone",
        "single": "single",
        "other": "single",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown accident type label: {label!r}")
    return aliases[normalized]


def resolve_video_path(path: str, video_root: str | None = None) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if video_root:
        candidate = os.path.join(video_root, path)
        if os.path.exists(candidate):
            return candidate
    return path


def build_video_info_from_row(row: Mapping[str, Any]) -> VideoInfo:
    no_frames = int(float(row.get("no_frames", 0) or 0))
    duration_sec = float(row.get("duration", 0.0) or 0.0)
    width = int(float(row.get("width", 0) or 0))
    height = int(float(row.get("height", 0) or 0))
    fps = float(no_frames / duration_sec) if duration_sec > 0 and no_frames > 0 else 30.0
    return VideoInfo(
        fps=fps,
        frame_count=no_frames,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def build_frame_times(center_time_sec: float, duration_sec: float, jitter_sec: float = 0.0, training: bool = False) -> list[float]:
    sample_center = float(center_time_sec)
    if training and jitter_sec > 0:
        sample_center += random.uniform(-jitter_sec, jitter_sec)

    upper_bound = duration_sec if duration_sec > 0 else max(sample_center + FRAME_OFFSETS_SEC[-1], 0.0)
    return [clamp(sample_center + offset, 0.0, upper_bound) for offset in FRAME_OFFSETS_SEC]


def _resize_frame_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    if max(height, width) <= max_side:
        return frame
    scale = max_side / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _black_frame(info: VideoInfo | None = None) -> np.ndarray:
    height = int(info.height) if info and info.height > 0 else 224
    width = int(info.width) if info and info.width > 0 else 224
    return np.zeros((height, width, 3), dtype=np.uint8)


def _read_frame_from_capture(
    cap: cv2.VideoCapture,
    time_sec: float,
    info: VideoInfo,
    max_decode_side: int = 0,
) -> np.ndarray | None:
    safe_duration = max(float(info.duration_sec), 0.0)
    safe_time = clamp(float(time_sec), 0.0, max(safe_duration - 1e-3, 0.0)) if safe_duration > 0 else max(0.0, float(time_sec))
    target_frame = int(round(safe_time * max(float(info.fps), 1e-6)))
    if info.frame_count > 0:
        target_frame = min(max(target_frame, 0), info.frame_count - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_MSEC, safe_time * 1000.0)
        ret, frame = cap.read()
    if not ret or frame is None:
        return None
    if max_decode_side > 0:
        frame = _resize_frame_max_side(frame, max_decode_side)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_frames_at_times(
    video_path: str,
    times_sec: Sequence[float],
    info: VideoInfo | None = None,
    max_decode_side: int = 0,
) -> list[np.ndarray]:
    if info is None:
        info = get_video_info(video_path)

    if not os.path.exists(video_path) or info.frame_count <= 0:
        return [_black_frame(info) for _ in times_sec]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [_black_frame(info) for _ in times_sec]

    frames: list[np.ndarray] = []
    for time_sec in times_sec:
        frame = _read_frame_from_capture(cap, time_sec, info, max_decode_side=max_decode_side)
        frames.append(frame if frame is not None else _black_frame(info))

    cap.release()
    return frames


def build_image_processor(encoder_path: str = DEFAULT_ENCODER_PATH):
    return AutoImageProcessor.from_pretrained(encoder_path, trust_remote_code=True)


def build_aimv2_encoder(
    encoder_path: str = DEFAULT_ENCODER_PATH,
    device: torch.device | str | None = None,
    freeze: bool = True,
):
    if device is None:
        target_device = pick_best_cuda_device()
    else:
        target_device = torch.device(device)
    encoder_dtype = torch.float16 if target_device.type == "cuda" else torch.float32

    if Aimv2VisionModel is not None:
        try:
            encoder = Aimv2VisionModel.from_pretrained(
                encoder_path,
                dtype=encoder_dtype,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            encoder = Aimv2VisionModel.from_pretrained(
                encoder_path,
                torch_dtype=encoder_dtype,
                low_cpu_mem_usage=True,
            )
    else:
        encoder = AutoModel.from_pretrained(
            encoder_path,
            trust_remote_code=True,
            torch_dtype=encoder_dtype,
        )
    encoder.to(target_device)
    if freeze:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
    return encoder, build_image_processor(encoder_path), encoder_dtype


class AIMv2TypeClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_classes: int = len(TYPE_LABELS),
        head_hidden_dim: int = 512,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.head_hidden_dim = int(head_hidden_dim)
        self.dropout = float(dropout)
        self.encoder_frozen = bool(freeze_encoder)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, self.num_classes),
        )
        if freeze_encoder:
            self.freeze_encoder()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def encoder_device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def encoder_dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    def freeze_encoder(self) -> None:
        self.encoder_frozen = True
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        self.encoder_frozen = False
        self.encoder.train(self.training)
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.dim() != 5:
            raise ValueError(f"Expected pixel_values with shape [B, 5, C, H, W], got {tuple(pixel_values.shape)}")

        batch_size, num_frames = pixel_values.shape[:2]
        flat_pixel_values = pixel_values.reshape(batch_size * num_frames, *pixel_values.shape[2:])
        flat_pixel_values = flat_pixel_values.to(device=self.encoder_device, dtype=self.encoder_dtype, non_blocking=True)

        context = torch.no_grad() if self.encoder_frozen else nullcontext()
        with context:
            outputs = self.encoder(pixel_values=flat_pixel_values, return_dict=True)
            frame_features = outputs.last_hidden_state.mean(dim=1)

        frame_features = frame_features.reshape(batch_size, num_frames, -1)
        pooled_features = frame_features.mean(dim=1)
        logits = self.head(pooled_features.float())
        return logits


@dataclass(frozen=True)
class AccidentRecord:
    rgb_path: str
    video_path: str
    label: int
    accident_time: float
    duration_sec: float
    video_info: VideoInfo


class AccidentTypeDataset(torch.utils.data.Dataset):
    def __init__(self, records: Sequence[AccidentRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AccidentRecord:
        return self.records[index]


class TypeBatchCollator:
    def __init__(
        self,
        processor,
        training: bool,
        jitter_sec: float = 1.0,
        max_decode_side: int = 0,
    ):
        self.processor = processor
        self.training = bool(training)
        self.jitter_sec = float(jitter_sec)
        self.max_decode_side = int(max_decode_side)

    def __call__(self, batch: Sequence[AccidentRecord]) -> dict[str, Any]:
        flat_images: list[np.ndarray] = []
        labels: list[int] = []
        centers: list[float] = []

        for record in batch:
            frame_times = build_frame_times(
                center_time_sec=record.accident_time,
                duration_sec=record.video_info.duration_sec,
                jitter_sec=self.jitter_sec,
                training=self.training,
            )
            frames = extract_frames_at_times(
                video_path=record.video_path,
                times_sec=frame_times,
                info=record.video_info,
                max_decode_side=self.max_decode_side,
            )
            flat_images.extend(frames)
            labels.append(record.label)
            centers.append(float(frame_times[2]))

        processed = self.processor(images=flat_images, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        pixel_values = pixel_values.reshape(len(batch), len(FRAME_OFFSETS_SEC), *pixel_values.shape[1:])

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
            "paths": [record.video_path for record in batch],
            "centers": torch.tensor(centers, dtype=torch.float32),
        }


class AIMv2TypePredictor:
    def __init__(
        self,
        model: AIMv2TypeClassifier,
        processor,
        label_names: Sequence[str] = TYPE_LABELS,
        video_root: str | None = None,
    ):
        self.model = model
        self.processor = processor
        self.label_names = tuple(label_names)
        self.video_root = video_root
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def encoder_dtype(self) -> torch.dtype:
        return self.model.encoder_dtype

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device | str | None = None,
        video_root: str | None = None,
    ) -> "AIMv2TypePredictor":
        payload = torch.load(checkpoint_path, map_location="cpu")
        encoder_path = str(payload["encoder_path"])
        label_names = tuple(payload.get("label_names", TYPE_LABELS))
        hidden_size = int(payload["hidden_size"])
        head_hidden_dim = int(payload["head_hidden_dim"])
        dropout = float(payload["dropout"])

        model_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder, processor, _ = build_aimv2_encoder(encoder_path=encoder_path, device=model_device, freeze=True)
        model = AIMv2TypeClassifier(
            encoder=encoder,
            hidden_size=hidden_size,
            num_classes=len(label_names),
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            freeze_encoder=True,
        )
        model.head.load_state_dict(payload["head_state_dict"])
        model.to(model_device)
        model.eval()
        return cls(model=model, processor=processor, label_names=label_names, video_root=video_root)

    def resolve_video_path(self, path: str) -> str:
        return resolve_video_path(path, self.video_root)

    def _build_pixel_values(self, video_path: str, accident_time_sec: float) -> torch.Tensor:
        info = get_video_info(video_path)
        frame_times = build_frame_times(
            center_time_sec=accident_time_sec,
            duration_sec=info.duration_sec,
            jitter_sec=0.0,
            training=False,
        )
        frames = extract_frames_at_times(
            video_path=video_path,
            times_sec=frame_times,
            info=info,
            max_decode_side=0,
        )
        processed = self.processor(images=frames, return_tensors="pt")
        pixel_values = processed["pixel_values"].unsqueeze(0)
        return pixel_values.to(device=self.device, dtype=self.encoder_dtype, non_blocking=True)

    @torch.inference_mode()
    def predict_logits(self, video_path: str, accident_time_sec: float) -> torch.Tensor:
        pixel_values = self._build_pixel_values(video_path, accident_time_sec)
        logits = self.model(pixel_values)
        return logits.squeeze(0)

    @torch.inference_mode()
    def predict_type(self, video_path: str, accident_time_sec: float) -> str:
        logits = self.predict_logits(video_path, accident_time_sec)
        pred_idx = int(torch.argmax(logits, dim=-1).item())
        return self.label_names[pred_idx]

    @torch.inference_mode()
    def predict_submission_type(self, row: Mapping[str, Any]) -> str:
        path_value = row.get("path") or row.get("rgb_path") or row.get("video_path")
        if path_value is None:
            raise KeyError("row must contain one of: path, rgb_path, video_path")
        if "accident_time" not in row:
            raise KeyError("row must contain accident_time")
        video_path = self.resolve_video_path(str(path_value))
        accident_time_sec = float(row["accident_time"])
        return self.predict_type(video_path, accident_time_sec)


def predict_submission_type(row: Mapping[str, Any], predictor: AIMv2TypePredictor) -> str:
    return predictor.predict_submission_type(row)


def predict_type_from_video(
    video_path: str,
    accident_time_sec: float,
    predictor: AIMv2TypePredictor,
) -> str:
    return predictor.predict_type(video_path, accident_time_sec)


def save_checkpoint(
    checkpoint_path: str,
    model: AIMv2TypeClassifier,
    encoder_path: str,
    label_names: Sequence[str],
    head_hidden_dim: int,
    dropout: float,
    epoch: int,
    best_macro_f1: float,
    val_metrics: Mapping[str, Any],
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "encoder_path": encoder_path,
        "hidden_size": int(model.hidden_size),
        "head_hidden_dim": int(head_hidden_dim),
        "dropout": float(dropout),
        "label_names": list(label_names),
        "frame_offsets_sec": list(FRAME_OFFSETS_SEC),
        "epoch": int(epoch),
        "best_macro_f1": float(best_macro_f1),
        "val_metrics": val_metrics,
        "head_state_dict": model.head.state_dict(),
    }
    torch.save(payload, checkpoint_path)

    metadata_path = os.path.splitext(checkpoint_path)[0] + ".json"
    metadata = {key: value for key, value in payload.items() if key != "head_state_dict"}
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)


def load_records_from_dataframe(
    dataframe,
    video_root: str,
) -> list[AccidentRecord]:
    records: list[AccidentRecord] = []
    for _, row in dataframe.iterrows():
        label_name = normalize_type_label(str(row["type"]))
        video_rel_path = str(row["rgb_path"])
        video_path = resolve_video_path(video_rel_path, video_root)
        video_info = build_video_info_from_row(row)
        records.append(
            AccidentRecord(
                rgb_path=video_rel_path,
                video_path=video_path,
                label=LABEL_TO_INDEX[label_name],
                accident_time=float(row["accident_time"]),
                duration_sec=float(video_info.duration_sec),
                video_info=video_info,
            )
        )
    return records


def build_submission_prediction_frame(
    dataframe,
    predictor: AIMv2TypePredictor,
):
    predicted = dataframe.copy()
    predicted["type"] = [predictor.predict_submission_type(row) for _, row in predicted.iterrows()]
    return predicted
