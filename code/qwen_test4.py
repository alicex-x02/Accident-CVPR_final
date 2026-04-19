from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tempfile
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from pipeline.optical_flow import compute_motion_curve, get_video_info, moving_average, select_top_k_peaks
from pipeline.qwen_utils import DEFAULT_MODEL_PATH, QwenVideoReasoner, clamp, extract_first_json_object, move_inputs_to_device, write_video_clip


VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_ROOT = os.path.join(WORKSPACE_ROOT, "woo", "ACCIDENT@CVPR", "data", "raw", "accident")
METADATA_PATH = os.path.join(DATASET_ROOT, "test_metadata.csv")
VIDEOS_DIR = os.path.join(DATASET_ROOT, "videos")
LOG_DIR = os.path.join(BASE_DIR, "log")
RESULT_DIR = os.path.join(BASE_DIR, "result")

TIME_SAMPLE_FPS = 5.0
TIME_SMOOTH_WINDOW = 5
TIME_TOP_K = 3
TIME_MIN_SEPARATION_SEC = 1.5
TIME_VERIFY_WINDOW_SEC = 1.0
TIME_VERIFY_CLIP_FPS = 8.0

LOCATION_HALF_WINDOW_SEC = 0.5
LOCATION_SAMPLE_FPS = 8.0
LOCATION_RESIZE_WIDTH = 320
LOCATION_HOTSPOT_QUANTILE = 0.95

TYPE_FRAME_OFFSETS = (-0.75, 0.0, 0.75)
TYPE_CENTER_WEIGHT = 2.0
TYPE_SIDE_WEIGHT = 1.0
TYPE_FALLBACK_LABEL = "single"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9


# Prompt builders
def build_time_verify_prompt(meta: Dict[str, str], candidate_time: float) -> str:
    region = meta.get("region", "")
    scene_layout = meta.get("scene_layout", "")
    weather = meta.get("weather", "")
    day_time = meta.get("day_time", "")
    quality = meta.get("quality", "")
    duration = meta.get("duration", "")
    no_frames = meta.get("no_frames", "")
    height = meta.get("height", "")
    width = meta.get("width", "")

    prompt = f"""
You are an expert traffic accident analyst looking at a short CCTV clip centered near {candidate_time:.3f} seconds.

Video metadata:
- region: {region}
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}
- quality (before enhancement): {quality}
- duration (seconds): {duration}
- no_frames: {no_frames}
- frame_height: {height}
- frame_width: {width}

Your task is to verify whether the first clear traffic accident begins in this clip.

Instructions:
1. Focus only on the first collision moment or the first unavoidable impact.
2. Return `contains_accident=true` only if the onset is clearly visible in this clip.
3. If the clip shows only pre-accident motion or only aftermath, return false.
4. Ignore the exact location and the accident type in this step.
5. Use the clip center as a hint, but judge the actual visible collision.

Critical output rules:
- Output JSON only.
- No reasoning.
- No analysis.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- The JSON must contain exactly these keys:
  "contains_accident", "confidence"

Output format:
{{
  "contains_accident": true,
  "confidence": 0.0
}}
"""
    return prompt.strip()


def build_type_prompt(meta: Dict[str, str], accident_time: float) -> str:
    region = meta.get("region", "")
    scene_layout = meta.get("scene_layout", "")
    weather = meta.get("weather", "")
    day_time = meta.get("day_time", "")
    quality = meta.get("quality", "")
    duration = meta.get("duration", "")
    no_frames = meta.get("no_frames", "")
    height = meta.get("height", "")
    width = meta.get("width", "")

    prompt = f"""
You are an expert traffic accident analyst looking at a single key frame from CCTV footage.

This image corresponds to the FIRST clear moment of a traffic accident in the video
at approximately accident_time = {accident_time:.3f} seconds.

Video metadata:
- region: {region}
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}
- quality (before enhancement): {quality}
- duration (seconds): {duration}
- no_frames: {no_frames}
- frame_height: {height}
- frame_width: {width}

Definitions of accident types (choose exactly one):
- rear-end: One vehicle crashes into the back of another vehicle traveling in the same direction.
- head-on: Two vehicles traveling in opposite directions collide front-to-front.
- sideswipe: Two vehicles moving in roughly the same direction make side-to-side contact while overlapping partially.
- t-bone: The front of one vehicle crashes into the side of another vehicle, forming a "T" shape.
- single: An accident involving only one vehicle (e.g., hitting a pole, barrier, guardrail, or going off the road) with no other vehicle collision.

Your task is to classify the accident type in this frame.

Instructions:
1. Carefully analyze the visible interaction between vehicles and/or objects.
2. Choose exactly one type from:
   ["rear-end", "head-on", "sideswipe", "t-bone", "single"].
3. If uncertain, choose the single best guess.

Critical output rules:
- Output JSON only.
- No reasoning.
- No analysis.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- The JSON must contain exactly this key:
  "type"

Output format:
{{
  "type": "<one of: rear-end, head-on, sideswipe, t-bone, single>"
}}
"""
    return prompt.strip()


def normalize_metadata(row: Dict[str, str]) -> Dict[str, str]:
    row = dict(row)
    if "scene_layout" not in row and "scene_layoutm" in row:
        row["scene_layout"] = row["scene_layoutm"]
    return row


def build_location_prompt(meta: Dict[str, str], accident_time: float) -> str:
    region = meta.get("region", "")
    scene_layout = meta.get("scene_layout", "")
    weather = meta.get("weather", "")
    day_time = meta.get("day_time", "")
    quality = meta.get("quality", "")
    duration = meta.get("duration", "")
    no_frames = meta.get("no_frames", "")
    height = meta.get("height", "")
    width = meta.get("width", "")

    prompt = f"""
You are an expert traffic accident analyst looking at a single key frame from CCTV footage.

This image corresponds to the FIRST clear moment of a traffic accident in the video
at approximately accident_time = {accident_time:.3f} seconds.

Video metadata:
- region: {region}
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}
- quality (before enhancement): {quality}
- duration (seconds): {duration}
- no_frames: {no_frames}
- frame_height: {height}
- frame_width: {width}

Your task is to precisely localize the primary collision point in this frame.

Instructions:
1. Focus on the main collision area where vehicles or objects are physically impacting.
2. Output normalized coordinates of the center of this collision region:
   - center_x: from left (0.0) to right (1.0)
   - center_y: from top (0.0) to bottom (1.0)
3. The coordinates must indicate the center of the actual contact region, not the center of the whole vehicle.
4. Ignore accident type classification in this step.
5. If uncertain, choose the single best estimate.

Critical output rules:
- Output JSON only.
- No reasoning.
- No analysis.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- The JSON must contain exactly these keys:
  "center_x", "center_y"

Output format:
{{
  "center_x": <float>,
  "center_y": <float>
}}
"""
    return prompt.strip()


# Metadata and path helpers
def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shardable accident pipeline with flow + Qwen ensemble")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--metadata-csv", default=METADATA_PATH)
    parser.add_argument("--videos-dir", default=VIDEOS_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--time-sample-fps", type=float, default=TIME_SAMPLE_FPS)
    parser.add_argument("--time-smooth-window", type=int, default=TIME_SMOOTH_WINDOW)
    parser.add_argument("--time-top-k", type=int, default=TIME_TOP_K)
    parser.add_argument("--time-min-separation-sec", type=float, default=TIME_MIN_SEPARATION_SEC)
    parser.add_argument("--time-verify-window-sec", type=float, default=TIME_VERIFY_WINDOW_SEC)
    parser.add_argument("--time-verify-clip-fps", type=float, default=TIME_VERIFY_CLIP_FPS)
    parser.add_argument("--location-half-window-sec", type=float, default=LOCATION_HALF_WINDOW_SEC)
    parser.add_argument("--location-sample-fps", type=float, default=LOCATION_SAMPLE_FPS)
    parser.add_argument("--location-resize-width", type=int, default=LOCATION_RESIZE_WIDTH)
    parser.add_argument("--location-hotspot-quantile", type=float, default=LOCATION_HOTSPOT_QUANTILE)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def get_run_paths(args: argparse.Namespace) -> Dict[str, str]:
    shard_tag = f"shard{args.shard_id:02d}_of_{args.num_shards:02d}"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return {
        "shard_tag": shard_tag,
        "output_csv": os.path.join(args.result_dir, f"submission_{shard_tag}.csv"),
        "raw_log": os.path.join(args.log_dir, f"raw_outputs_{shard_tag}.jsonl"),
        "run_log": os.path.join(args.log_dir, f"run_{shard_tag}_{timestamp}.log"),
        "temp_dir": os.path.join(args.log_dir, f"temp_{shard_tag}"),
    }


def setup_logging(run_log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(run_log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def append_jsonl(path: str, payload: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_predictions_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["path", "accident_time", "center_x", "center_y", "type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_results(path: str) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "path" in df.columns:
        key_col = "path"
    elif "video_id" in df.columns:
        key_col = "video_id"
    else:
        return {}
    return {str(row[key_col]): row.to_dict() for _, row in df.iterrows()}


def filter_shard_rows(df: pd.DataFrame, shard_id: int, num_shards: int, limit: int) -> pd.DataFrame:
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")

    shard_df = df.iloc[shard_id::num_shards].copy()
    shard_df.reset_index(drop=True, inplace=True)
    if limit > 0:
        shard_df = shard_df.iloc[:limit].copy()
        shard_df.reset_index(drop=True, inplace=True)
    return shard_df


def resolve_video_entry(row: pd.Series, dataset_root: str, videos_dir: str) -> Tuple[str, str]:
    if "path" in row.index and pd.notna(row["path"]):
        rel_path = str(row["path"]).strip()
        if os.path.isabs(rel_path):
            return os.path.basename(rel_path), rel_path

        candidate = os.path.join(dataset_root, rel_path)
        if os.path.exists(candidate):
            return rel_path, candidate

        fallback = os.path.join(dataset_root, os.path.basename(rel_path))
        return rel_path, fallback

    if "video_id" in row.index and pd.notna(row["video_id"]):
        video_id = str(row["video_id"]).strip()
        rel_path = f"videos/{video_id}.mp4"
        return rel_path, os.path.join(videos_dir, f"{video_id}.mp4")

    raise KeyError("Expected either 'path' or 'video_id' in metadata row")



# Media helpers
def _generate_json_from_media(
    qwen: QwenVideoReasoner,
    media_type: str,
    media_path: str,
    prompt: str,
    max_new_tokens: int = 128,
) -> Tuple[Dict[str, Any], str]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Respond with JSON only. No reasoning. /no_think"}],
        },
        {
            "role": "user",
            "content": [
                {"type": media_type, "path": media_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    processed = qwen.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    processed = move_inputs_to_device(processed, qwen.device)

    with torch.no_grad():
        generated = qwen.model.generate(
            **processed,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=qwen.processor.tokenizer.eos_token_id,
        )

    prompt_len = processed["input_ids"].shape[-1]
    generated_ids = generated[:, prompt_len:]
    raw_text = qwen.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    parsed = extract_first_json_object(raw_text)
    if parsed is None:
        raise ValueError(f"Failed to parse JSON from Qwen output: {raw_text}")
    return parsed, raw_text


def infer_json_from_temp_clip(
    qwen: QwenVideoReasoner,
    video_path: str,
    center_time_sec: float,
    prompt: str,
    temp_dir: str,
    window_sec: float = 1.0,
    clip_fps: float = 8.0,
    clip_prefix: str = "clip",
    max_new_tokens: int = 128,
) -> Tuple[Dict[str, Any], str, str]:
    ensure_dirs(temp_dir)
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
        parsed, raw_text = _generate_json_from_media(
            qwen=qwen,
            media_type="video",
            media_path=clip_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        return parsed, raw_text, clip_path
    finally:
        if os.path.exists(clip_path):
            os.remove(clip_path)


def save_frame_to_temp(frame: np.ndarray, temp_dir: str, prefix: str) -> Optional[str]:
    ensure_dirs(temp_dir)
    with tempfile.NamedTemporaryFile(prefix=f"{prefix}_", suffix=".jpg", dir=temp_dir, delete=False) as fp:
        frame_path = fp.name
    if not cv2.imwrite(frame_path, frame):
        if os.path.exists(frame_path):
            os.remove(frame_path)
        return None
    return frame_path


def extract_frame_at_time(
    video_path: str,
    accident_time: float,
    meta: Dict[str, str],
    fps_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    try:
        info = get_video_info(video_path)
    except Exception:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = fps_override if fps_override is not None else info.fps
    if fps <= 0 and meta.get("duration") and meta.get("no_frames"):
        try:
            fps = float(meta["no_frames"]) / float(meta["duration"])
        except Exception:
            fps = 0.0

    if fps <= 0:
        cap.release()
        return None

    total_frames = info.frame_count
    frame_index = int(accident_time * fps)
    frame_index = max(0, min(frame_index, max(0, total_frames - 1)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}

    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, accident_time) * 1000.0)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}

    for delta in range(1, 8):
        for candidate in (frame_index - delta, frame_index + delta):
            if candidate < 0 or candidate >= max(1, total_frames):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(candidate))
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return {"frame": frame, "fps": fps, "frame_index": int(candidate)}

    cap.release()
    return None


def _clamp_sample_time(accident_time: float, offset: float, meta: Dict[str, str], duration_fallback: float = 0.0) -> float:
    try:
        duration = float(meta.get("duration", duration_fallback))
    except Exception:
        duration = duration_fallback
    return clamp(accident_time + offset, 0.0, duration if duration > 0 else max(0.0, accident_time + abs(offset)))


# Time module
def detect_candidate_times(
    video_path: str,
    sample_fps: float,
    smooth_window: int,
    top_k: int,
    min_separation_sec: float,
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    times, motion_values = compute_motion_curve(video_path=video_path, sample_fps=sample_fps)
    if motion_values.size == 0:
        return [], times, motion_values

    smoothed = moving_average(motion_values, window_size=smooth_window)
    candidate_times = select_top_k_peaks(
        times=times,
        values=smoothed,
        top_k=top_k,
        min_separation_sec=min_separation_sec,
    )
    return candidate_times, times, smoothed


def verify_accident_time(
    video_path: str,
    candidate_times: List[float],
    meta: Dict[str, str],
    qwen: QwenVideoReasoner,
    temp_dir: str,
    verify_window_sec: float,
    verify_clip_fps: float,
    raw_log_path: str,
) -> Tuple[float, List[Dict[str, float]]]:
    if not candidate_times:
        candidate_times = [0.0]

    scored_candidates: List[Dict[str, float]] = []
    for candidate_time in candidate_times:
        prompt = build_time_verify_prompt(meta, candidate_time)
        try:
            parsed, raw_text, _ = infer_json_from_temp_clip(
                qwen=qwen,
                video_path=video_path,
                center_time_sec=candidate_time,
                prompt=prompt,
                temp_dir=temp_dir,
                window_sec=verify_window_sec,
                clip_fps=verify_clip_fps,
                clip_prefix="verify",
                max_new_tokens=128,
            )
            contains_accident = bool(parsed.get("contains_accident", False))
            confidence = clamp(float(parsed.get("confidence", 0.0)), 0.0, 1.0)
            append_jsonl(
                raw_log_path,
                {
                    "stage": "time_verification",
                    "candidate_time": float(candidate_time),
                    "raw_output": raw_text,
                    "parsed": parsed,
                },
            )
        except Exception as exc:
            contains_accident = False
            confidence = 0.0
            append_jsonl(
                raw_log_path,
                {
                    "stage": "time_verification",
                    "candidate_time": float(candidate_time),
                    "error": str(exc),
                },
            )

        scored_candidates.append(
            {
                "candidate_time": float(candidate_time),
                "contains_accident": float(1.0 if contains_accident else 0.0),
                "confidence": float(confidence),
            }
        )

    positive_candidates = [item for item in scored_candidates if item["contains_accident"] > 0.5]
    if positive_candidates:
        positive_candidates.sort(key=lambda item: (-item["confidence"], item["candidate_time"]))
        best_time = float(positive_candidates[0]["candidate_time"])
    else:
        scored_candidates.sort(key=lambda item: (-item["confidence"], item["candidate_time"]))
        best_time = float(scored_candidates[0]["candidate_time"])

    return best_time, scored_candidates


# Location module
def estimate_motion_hotspot(
    video_path: str,
    accident_time_sec: float,
    half_window_sec: float,
    sample_fps: float,
    resize_width: int,
    hotspot_quantile: float,
) -> Tuple[float, float, Dict[str, Any]]:
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    start_time = max(0.0, accident_time_sec - half_window_sec)
    end_time = min(info.duration_sec, accident_time_sec + half_window_sec)
    start_frame = max(0, int(start_time * info.fps))
    end_frame = max(start_frame + 1, int(end_time * info.fps))
    frame_step = max(int(round(info.fps / max(sample_fps, 1e-6))), 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_gray = None
    accumulated_motion = None
    current_frame = start_frame
    out_shape = None

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (current_frame - start_frame) % frame_step != 0:
            current_frame += 1
            continue

        if resize_width > 0 and frame.shape[1] > resize_width:
            scale = resize_width / float(frame.shape[1])
            resize_height = max(1, int(round(frame.shape[0] * scale)))
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_shape = gray.shape

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
            if accumulated_motion is None:
                accumulated_motion = magnitude.astype(np.float32)
            else:
                accumulated_motion += magnitude.astype(np.float32)

        prev_gray = gray
        current_frame += 1

    cap.release()

    if accumulated_motion is None or out_shape is None or float(np.max(accumulated_motion)) <= 0.0:
        diagnostics = {
            "confident": False,
            "hotspot_fraction": 0.0,
            "peak_motion": 0.0,
            "mean_motion": 0.0,
            "concentration": 0.0,
        }
        return 0.5, 0.5, diagnostics

    threshold = float(np.quantile(accumulated_motion, hotspot_quantile))
    hotspot_mask = accumulated_motion >= threshold
    if not np.any(hotspot_mask):
        hotspot_mask = accumulated_motion > 0

    ys, xs = np.where(hotspot_mask)
    weights = accumulated_motion[ys, xs]
    if xs.size == 0 or ys.size == 0 or float(weights.sum()) <= 0.0:
        center_x = float(out_shape[1] / 2.0)
        center_y = float(out_shape[0] / 2.0)
    else:
        center_x = float(np.average(xs, weights=weights))
        center_y = float(np.average(ys, weights=weights))

    height, width = out_shape
    norm_x = clamp(center_x / max(width - 1, 1), 0.0, 1.0)
    norm_y = clamp(center_y / max(height - 1, 1), 0.0, 1.0)

    hotspot_fraction = float(np.mean(hotspot_mask))
    peak_motion = float(np.max(accumulated_motion))
    mean_motion = float(np.mean(accumulated_motion))
    concentration = float(peak_motion / max(mean_motion, 1e-6))
    confident = bool(
        xs.size > 0
        and ys.size > 0
        and hotspot_fraction >= 0.002
        and hotspot_fraction <= 0.20
        and concentration >= 1.10
    )

    diagnostics = {
        "confident": confident,
        "hotspot_fraction": hotspot_fraction,
        "peak_motion": peak_motion,
        "mean_motion": mean_motion,
        "concentration": concentration,
    }
    return norm_x, norm_y, diagnostics


def validate_location_prediction(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        center_x = float(result["center_x"])
        center_y = float(result["center_y"])
    except (KeyError, TypeError, ValueError):
        return None

    center_x = clamp(center_x, 0.0, 1.0)
    center_y = clamp(center_y, 0.0, 1.0)
    return {"center_x": center_x, "center_y": center_y}


def estimate_location_with_sanity(
    video_path: str,
    accident_time_sec: float,
    meta: Dict[str, str],
    qwen: QwenVideoReasoner,
    temp_dir: str,
    raw_log_path: str,
    half_window_sec: float,
    sample_fps: float,
    resize_width: int,
    hotspot_quantile: float,
) -> Tuple[float, float, Dict[str, Any]]:
    flow_x, flow_y, diagnostics = estimate_motion_hotspot(
        video_path=video_path,
        accident_time_sec=accident_time_sec,
        half_window_sec=half_window_sec,
        sample_fps=sample_fps,
        resize_width=resize_width,
        hotspot_quantile=hotspot_quantile,
    )

    result: Dict[str, Any] = {
        "flow_center_x": flow_x,
        "flow_center_y": flow_y,
        "flow_diagnostics": diagnostics,
        "used_sanity_check": False,
        "sanity_center_x": None,
        "sanity_center_y": None,
    }

    needs_sanity = (not diagnostics["confident"]) or diagnostics["hotspot_fraction"] < 0.004 or diagnostics["hotspot_fraction"] > 0.15
    if needs_sanity:
        sanity: Optional[Dict[str, float]] = None
        extracted = extract_frame_at_time(video_path=video_path, accident_time=accident_time_sec, meta=meta)
        if extracted is not None:
            frame_path = save_frame_to_temp(
                extracted["frame"],
                temp_dir=temp_dir,
                prefix=f"location_{accident_time_sec:.3f}".replace(".", "_"),
            )
            if frame_path is not None:
                try:
                    parsed, raw_text = _generate_json_from_media(
                        qwen=qwen,
                        media_type="image",
                        media_path=frame_path,
                        prompt=build_location_prompt(meta, accident_time_sec),
                        max_new_tokens=128,
                    )
                    sanity = validate_location_prediction(parsed)
                    append_jsonl(
                        raw_log_path,
                        {
                            "stage": "location_sanity",
                            "accident_time": float(accident_time_sec),
                            "raw_output": raw_text,
                            "parsed": parsed,
                            "frame_path": frame_path,
                        },
                    )
                except Exception as exc:
                    append_jsonl(
                        raw_log_path,
                        {
                            "stage": "location_sanity",
                            "accident_time": float(accident_time_sec),
                            "error": str(exc),
                            "frame_path": frame_path,
                        },
                    )
                finally:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)

        if sanity is None:
            try:
                parsed, raw_text, _ = infer_json_from_temp_clip(
                    qwen=qwen,
                    video_path=video_path,
                    center_time_sec=accident_time_sec,
                    prompt=build_location_prompt(meta, accident_time_sec),
                    temp_dir=temp_dir,
                    window_sec=1.0,
                    clip_fps=8.0,
                    clip_prefix="location",
                    max_new_tokens=128,
                )
                sanity = validate_location_prediction(parsed)
                append_jsonl(
                    raw_log_path,
                    {
                        "stage": "location_sanity_clip",
                        "accident_time": float(accident_time_sec),
                        "raw_output": raw_text,
                        "parsed": parsed,
                    },
                )
            except Exception as exc:
                append_jsonl(
                    raw_log_path,
                    {
                        "stage": "location_sanity_clip",
                        "accident_time": float(accident_time_sec),
                        "error": str(exc),
                    },
                )

        if sanity is not None:
            result["used_sanity_check"] = True
            result["sanity_center_x"] = sanity["center_x"]
            result["sanity_center_y"] = sanity["center_y"]

            if diagnostics["confident"]:
                dist = abs(flow_x - sanity["center_x"]) + abs(flow_y - sanity["center_y"])
                if dist <= 0.25:
                    flow_x = float(0.6 * flow_x + 0.4 * sanity["center_x"])
                    flow_y = float(0.6 * flow_y + 0.4 * sanity["center_y"])
                result["blend_distance"] = dist
            else:
                flow_x = sanity["center_x"]
                flow_y = sanity["center_y"]
        else:
            append_jsonl(
                raw_log_path,
                {
                    "stage": "location_sanity",
                    "accident_time": float(accident_time_sec),
                    "note": "no_valid_sanity_result",
                },
            )

    result["final_center_x"] = flow_x
    result["final_center_y"] = flow_y
    append_jsonl(
        raw_log_path,
        {
            "stage": "location_final",
            "accident_time": float(accident_time_sec),
            **result,
        },
    )
    return flow_x, flow_y, result


# Type module
TYPE_ALIASES = {
    "rear end": "rear-end",
    "rear-end": "rear-end",
    "head on": "head-on",
    "head-on": "head-on",
    "sideswipe": "sideswipe",
    "side swipe": "sideswipe",
    "side-swipe": "sideswipe",
    "t bone": "t-bone",
    "t-bone": "t-bone",
    "single": "single",
    "other": "single",
}


def normalize_type_prediction(result: Dict[str, Any]) -> Optional[str]:
    value = result.get("type", result.get("accident_type", ""))
    try:
        label = str(value).strip().lower()
    except Exception:
        return None

    label = TYPE_ALIASES.get(label, label)
    return label if label in VALID_TYPES else None


def classify_accident_type_ensemble(
    video_path: str,
    accident_time_sec: float,
    meta: Dict[str, str],
    qwen: QwenVideoReasoner,
    temp_dir: str,
    raw_log_path: str,
    frame_offsets: Sequence[float] = TYPE_FRAME_OFFSETS,
) -> Tuple[str, List[Dict[str, Any]]]:
    prompt = build_type_prompt(meta, accident_time_sec)
    votes: List[Dict[str, Any]] = []
    weighted_counts: Counter[str] = Counter()
    central_label: Optional[str] = None

    for offset in frame_offsets:
        sample_time = _clamp_sample_time(accident_time_sec, float(offset), meta)
        sample = {
            "offset": float(offset),
            "sample_time": float(sample_time),
            "media_type": None,
            "label": None,
            "raw_output": None,
            "parsed": None,
        }

        extracted = extract_frame_at_time(video_path=video_path, accident_time=sample_time, meta=meta)
        try:
            if extracted is not None:
                frame_path = save_frame_to_temp(
                    extracted["frame"],
                    temp_dir=temp_dir,
                    prefix=f"type_{sample_time:.3f}".replace(".", "_"),
                )
                if frame_path is not None:
                    try:
                        parsed, raw_text = _generate_json_from_media(
                            qwen=qwen,
                            media_type="image",
                            media_path=frame_path,
                            prompt=prompt,
                            max_new_tokens=MAX_NEW_TOKENS,
                        )
                        sample["media_type"] = "image"
                        sample["raw_output"] = raw_text
                        sample["parsed"] = parsed
                        label = normalize_type_prediction(parsed)
                        sample["label"] = label
                    except Exception as exc:
                        sample["error"] = str(exc)
                        parsed, raw_text, clip_path = infer_json_from_temp_clip(
                            qwen=qwen,
                            video_path=video_path,
                            center_time_sec=sample_time,
                            prompt=prompt,
                            temp_dir=temp_dir,
                            window_sec=1.0,
                            clip_fps=8.0,
                            clip_prefix="type",
                            max_new_tokens=MAX_NEW_TOKENS,
                        )
                        sample["media_type"] = "video"
                        sample["raw_output"] = raw_text
                        sample["parsed"] = parsed
                        label = normalize_type_prediction(parsed)
                        sample["label"] = label
                        sample["clip_path"] = clip_path
                    finally:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                else:
                    parsed, raw_text, clip_path = infer_json_from_temp_clip(
                        qwen=qwen,
                        video_path=video_path,
                        center_time_sec=sample_time,
                        prompt=prompt,
                        temp_dir=temp_dir,
                        window_sec=1.0,
                        clip_fps=8.0,
                        clip_prefix="type",
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    sample["media_type"] = "video"
                    sample["raw_output"] = raw_text
                    sample["parsed"] = parsed
                    label = normalize_type_prediction(parsed)
                    sample["label"] = label
                    sample["clip_path"] = clip_path
            else:
                parsed, raw_text, clip_path = infer_json_from_temp_clip(
                    qwen=qwen,
                    video_path=video_path,
                    center_time_sec=sample_time,
                    prompt=prompt,
                    temp_dir=temp_dir,
                    window_sec=1.0,
                    clip_fps=8.0,
                    clip_prefix="type",
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                sample["media_type"] = "video"
                sample["raw_output"] = raw_text
                sample["parsed"] = parsed
                label = normalize_type_prediction(parsed)
                sample["label"] = label
                sample["clip_path"] = clip_path
        except Exception as exc:
            sample["error"] = str(exc)
            label = None

        if abs(offset) < 1e-9:
            if label is not None:
                central_label = label
            sample["weight"] = TYPE_CENTER_WEIGHT
        else:
            sample["weight"] = TYPE_SIDE_WEIGHT

        if label is not None:
            weighted_counts[label] += float(sample["weight"])

        votes.append(sample)

    if weighted_counts:
        best_weight = max(weighted_counts.values())
        best_labels = [label for label, weight in weighted_counts.items() if weight == best_weight]
        if len(best_labels) == 1:
            selected = best_labels[0]
        elif central_label in best_labels:
            selected = central_label or TYPE_FALLBACK_LABEL
        else:
            selected = sorted(best_labels)[0]
    else:
        selected = central_label or TYPE_FALLBACK_LABEL

    append_jsonl(
        raw_log_path,
        {
            "stage": "type_ensemble",
            "accident_time": float(accident_time_sec),
            "selected": selected,
            "votes": votes,
        },
    )
    return selected, votes


# Pipeline
def process_video(
    path_id: str,
    video_path: str,
    meta: Dict[str, str],
    qwen: QwenVideoReasoner,
    temp_dir: str,
    raw_log_path: str,
    time_sample_fps: float,
    time_smooth_window: int,
    time_top_k: int,
    time_min_separation_sec: float,
    time_verify_window_sec: float,
    time_verify_clip_fps: float,
    location_half_window_sec: float,
    location_sample_fps: float,
    location_resize_width: int,
    location_hotspot_quantile: float,
) -> Dict[str, object]:
    candidate_times, _, _ = detect_candidate_times(
        video_path=video_path,
        sample_fps=time_sample_fps,
        smooth_window=time_smooth_window,
        top_k=time_top_k,
        min_separation_sec=time_min_separation_sec,
    )
    if not candidate_times:
        candidate_times = [0.0]

    append_jsonl(
        raw_log_path,
        {
            "stage": "time_candidates",
            "path": path_id,
            "candidate_times": candidate_times,
        },
    )

    accident_time, scored_candidates = verify_accident_time(
        video_path=video_path,
        candidate_times=candidate_times,
        meta=meta,
        qwen=qwen,
        temp_dir=temp_dir,
        verify_window_sec=time_verify_window_sec,
        verify_clip_fps=time_verify_clip_fps,
        raw_log_path=raw_log_path,
    )

    center_x, center_y, location_info = estimate_location_with_sanity(
        video_path=video_path,
        accident_time_sec=accident_time,
        meta=meta,
        qwen=qwen,
        temp_dir=temp_dir,
        raw_log_path=raw_log_path,
        half_window_sec=location_half_window_sec,
        sample_fps=location_sample_fps,
        resize_width=location_resize_width,
        hotspot_quantile=location_hotspot_quantile,
    )

    accident_type, type_votes = classify_accident_type_ensemble(
        video_path=video_path,
        accident_time_sec=accident_time,
        meta=meta,
        qwen=qwen,
        temp_dir=temp_dir,
        raw_log_path=raw_log_path,
    )

    result = {
        "path": path_id,
        "accident_time": float(round(accident_time, 3)),
        "center_x": float(round(center_x, 6)),
        "center_y": float(round(center_y, 6)),
        "type": accident_type,
    }
    append_jsonl(
        raw_log_path,
        {
            "stage": "final_prediction",
            "path": path_id,
            "accident_time": result["accident_time"],
            "center_x": result["center_x"],
            "center_y": result["center_y"],
            "type": accident_type,
            "time_candidates": candidate_times,
            "scored_candidates": scored_candidates,
            "location_info": location_info,
            "type_votes": type_votes,
        },
    )
    return result


def main() -> None:
    args = parse_args()
    ensure_dirs(args.log_dir, args.result_dir)
    run_paths = get_run_paths(args)
    ensure_dirs(run_paths["temp_dir"])
    setup_logging(run_paths["run_log"])

    if os.path.exists(run_paths["raw_log"]):
        os.remove(run_paths["raw_log"])

    logging.info("Dataset root: %s", args.dataset_root)
    logging.info("Metadata CSV: %s", args.metadata_csv)
    logging.info("Videos dir: %s", args.videos_dir)
    logging.info("Model path: %s", args.model_path)
    logging.info("Shard: %d / %d", args.shard_id, args.num_shards)
    logging.info("Output CSV: %s", run_paths["output_csv"])
    logging.info("Raw log: %s", run_paths["raw_log"])
    logging.info("Temp dir: %s", run_paths["temp_dir"])

    metadata_df = pd.read_csv(args.metadata_csv)
    metadata_df = pd.DataFrame([normalize_metadata(row) for _, row in metadata_df.iterrows()])

    shard_df = filter_shard_rows(metadata_df, args.shard_id, args.num_shards, args.limit)
    logging.info("Rows assigned to this shard: %d", len(shard_df))

    existing_results = load_existing_results(run_paths["output_csv"]) if args.resume else {}
    predictions: List[Dict[str, object]] = list(existing_results.values())
    write_predictions_csv(run_paths["output_csv"], predictions)

    qwen = QwenVideoReasoner(model_path=args.model_path)

    for idx, row in shard_df.iterrows():
        try:
            path_id, video_path = resolve_video_entry(row, args.dataset_root, args.videos_dir)
            if path_id in existing_results:
                logging.info("[%d/%d] Skip existing: %s", idx + 1, len(shard_df), path_id)
                continue

            if not os.path.exists(video_path):
                logging.warning("[%d/%d] Missing video: %s", idx + 1, len(shard_df), video_path)
                result = {
                    "path": path_id,
                    "accident_time": 0.0,
                    "center_x": 0.5,
                    "center_y": 0.5,
                    "type": TYPE_FALLBACK_LABEL,
                }
            else:
                logging.info("[%d/%d] Processing %s", idx + 1, len(shard_df), path_id)
                result = process_video(
                    path_id=path_id,
                    video_path=video_path,
                    meta=row.to_dict(),
                    qwen=qwen,
                    temp_dir=run_paths["temp_dir"],
                    raw_log_path=run_paths["raw_log"],
                    time_sample_fps=args.time_sample_fps,
                    time_smooth_window=args.time_smooth_window,
                    time_top_k=args.time_top_k,
                    time_min_separation_sec=args.time_min_separation_sec,
                    time_verify_window_sec=args.time_verify_window_sec,
                    time_verify_clip_fps=args.time_verify_clip_fps,
                    location_half_window_sec=args.location_half_window_sec,
                    location_sample_fps=args.location_sample_fps,
                    location_resize_width=args.location_resize_width,
                    location_hotspot_quantile=args.location_hotspot_quantile,
                )

            predictions.append(result)
            write_predictions_csv(run_paths["output_csv"], predictions)
            logging.info(
                "[%d/%d] Updated CSV | %s | t=%.3f xy=(%.6f, %.6f) type=%s",
                idx + 1,
                len(shard_df),
                result["path"],
                result["accident_time"],
                result["center_x"],
                result["center_y"],
                result["type"],
            )
        except Exception as exc:
            logging.exception("[%d/%d] Failed row, falling back: %s", idx + 1, len(shard_df), exc)
            fallback_path = str(row.get("path", row.get("video_id", f"row_{idx}")))
            fallback_result = {
                "path": fallback_path,
                "accident_time": 0.0,
                "center_x": 0.5,
                "center_y": 0.5,
                "type": TYPE_FALLBACK_LABEL,
            }
            predictions.append(fallback_result)
            write_predictions_csv(run_paths["output_csv"], predictions)

    logging.info("Finished shard run. Saved CSV: %s", run_paths["output_csv"])


if __name__ == "__main__":
    main()
