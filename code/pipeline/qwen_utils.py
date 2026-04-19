from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as _AutoQwenModel
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoQwenModel


DEFAULT_MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"


def resolve_model_path(model_path: str) -> str:
    snapshots_dir = os.path.join(model_path, "snapshots")
    refs_main = os.path.join(model_path, "refs", "main")
    if os.path.isdir(snapshots_dir) and os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            snapshot_name = f.read().strip()
        snapshot_path = os.path.join(snapshots_dir, snapshot_name)
        if os.path.isdir(snapshot_path):
            return snapshot_path
    return model_path


def move_inputs_to_device(processed: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in processed.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    decoder = json.JSONDecoder()

    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start_idx:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def write_video_clip(
    video_path: str,
    center_time_sec: float,
    output_path: str,
    window_sec: float = 1.0,
    clip_fps: float = 8.0,
    max_side: int = 448,
) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / source_fps if frame_count > 0 else 0.0

    start_time = max(0.0, center_time_sec - window_sec)
    end_time = min(duration, center_time_sec + window_sec)
    start_frame = max(0, int(start_time * source_fps))
    end_frame = max(start_frame + 1, int(end_time * source_fps))
    frame_step = max(int(round(source_fps / max(clip_fps, 1e-6))), 1)

    if width <= 0 or height <= 0:
        width, height = 448, 448
    scale = min(1.0, max_side / float(max(width, height)))
    out_width = max(2, int(round(width * scale)))
    out_height = max(2, int(round(height * scale)))
    if out_width % 2 == 1:
        out_width += 1
    if out_height % 2 == 1:
        out_height += 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, clip_fps, (out_width, out_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create clip writer: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (current_frame - start_frame) % frame_step == 0:
            resized = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            writer.write(resized)
        current_frame += 1

    writer.release()
    cap.release()
    return output_path


class QwenVideoReasoner:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        resolved_path = resolve_model_path(model_path)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        self.model = _AutoQwenModel.from_pretrained(
            resolved_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def infer_json_from_video(self, video_path: str, prompt: str, max_new_tokens: int = 128) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Respond with JSON only. No reasoning. /no_think"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        processed = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        processed = move_inputs_to_device(processed, self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **processed,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        prompt_len = processed["input_ids"].shape[-1]
        generated_ids = generated[:, prompt_len:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        parsed = extract_first_json_object(text)
        if parsed is None:
            raise ValueError(f"Failed to parse JSON from Qwen output: {text}")
        return parsed

    def infer_json_from_temp_clip(
        self,
        video_path: str,
        center_time_sec: float,
        prompt: str,
        temp_dir: str,
        window_sec: float = 1.0,
        clip_fps: float = 8.0,
        clip_prefix: str = "clip",
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        os.makedirs(temp_dir, exist_ok=True)
        safe_time = f"{center_time_sec:.3f}".replace(".", "_")
        with tempfile.NamedTemporaryFile(
            prefix=f"{clip_prefix}_{safe_time}_",
            suffix=".mp4",
            dir=temp_dir,
            delete=False,
        ) as fp:
            clip_path = fp.name

        write_video_clip(
            video_path=video_path,
            center_time_sec=center_time_sec,
            output_path=clip_path,
            window_sec=window_sec,
            clip_fps=clip_fps,
        )

        try:
            return self.infer_json_from_video(clip_path, prompt=prompt, max_new_tokens=max_new_tokens)
        finally:
            if os.path.exists(clip_path):
                os.remove(clip_path)
