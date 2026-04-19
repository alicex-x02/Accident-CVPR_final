import argparse
import csv
import json
import os
import re
import shutil
import tempfile
import time
import subprocess
import sys
from statistics import median
from threading import Thread
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from pipeline.optical_flow import compute_motion_curve, moving_average, select_top_k_peaks
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCIDENT_DIR = os.path.join(BASE_DIR, "accident")
RESULT_DIR = os.path.join(os.path.dirname(BASE_DIR), "result")
METADATA_PATH = os.path.join(ACCIDENT_DIR, "test_metadata.csv")
PREDICTION_PATH = os.path.join(RESULT_DIR, "qwen_test10_hybrid.csv")
RAW_LOG_PATH = os.path.join(ACCIDENT_DIR, "raw_outputs.jsonl")
PART_PREDICTION_TEMPLATE = os.path.join(RESULT_DIR, "qwen_test10_hybrid_{part}.csv")
PART_RAW_LOG_TEMPLATE = os.path.join(ACCIDENT_DIR, "qwen_test10_hybrid_{part}_raw_outputs.jsonl")
PART_RUN_LOG_TEMPLATE = os.path.join(os.path.dirname(BASE_DIR), "log", "qwen_test10_hybrid_{part}_gpu{gpu}.out")

MODEL_NAME = "Qwen/Qwen3.5-9B"
VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}
VALID_MULTI_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone"}

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9

# Hybrid time selector for qwen_test10: Qwen baseline + OF rule fallback
TIME_FLOW_SAMPLE_FPS = 5.0
TIME_FLOW_SMOOTH_WINDOW = 5
TIME_FLOW_TOP_K = 5
TIME_FLOW_MIN_SEPARATION_SEC = 1.0
TIME_FLOW_DELTA_WEIGHT = 0.7
TIME_FLOW_MIN_VALID_SEC = 0.50
TIME_FLOW_END_MARGIN_SEC = 1.00
TIME_FLOW_MIN_SCORE_Z = 0.00
TIME_FLOW_LATE_ABSOLUTE_SEC = 18.00
TIME_FLOW_LATE_FRACTION = 0.75

# Candidate clip code is kept for debugging, but Qwen selector is no longer used.
TIME_CANDIDATE_CLIP_WINDOW_SEC = 0.75
TIME_CANDIDATE_CLIP_FPS = 6.0

# Hybrid switch rule:
# - Trust Qwen for normal predictions.
# - Use OF rule when Qwen collapses to the start or jumps too late.
HYBRID_QWEN_ZERO_THRESHOLD_SEC = 0.30
HYBRID_QWEN_LATE_THRESHOLD_SEC = 15.00

LOCATION_FRAME_OFFSETS = (-0.20, 0.0, 0.20)
LOCATION_CROP_SCALES = (0.40, 0.26)
TYPE_WINDOW_SEC = 1.2
TYPE_CLIP_FPS = 8.0


def resolve_output_paths(part_name: Optional[str] = None) -> Tuple[str, str]:
    if not part_name:
        return PREDICTION_PATH, RAW_LOG_PATH
    safe_part = str(part_name).strip()
    return (
        PART_PREDICTION_TEMPLATE.format(part=safe_part),
        PART_RAW_LOG_TEMPLATE.format(part=safe_part),
    )


def count_metadata_rows(csv_path: str) -> int:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def build_worker_command(
    start_row: int,
    end_row: int,
    part_name: str,
    gpu_id: int,
) -> List[str]:
    return [
        sys.executable,
        os.path.abspath(__file__),
        "--start-row",
        str(start_row),
        "--end-row",
        str(end_row),
        "--part-name",
        part_name,
        "--gpu-id",
        str(gpu_id),
    ]


def launch_two_gpu_shards(total_rows: int) -> int:
    os.makedirs(os.path.dirname(PART_RUN_LOG_TEMPLATE), exist_ok=True)
    mid = total_rows // 2
    shards = [
        {"start_row": 1, "end_row": mid, "part_name": "part0", "gpu_id": 0},
        {"start_row": mid + 1, "end_row": total_rows, "part_name": "part1", "gpu_id": 1},
    ]

    processes: List[Tuple[str, Any, Any]] = []
    try:
        for shard in shards:
            log_path = PART_RUN_LOG_TEMPLATE.format(part=shard["part_name"], gpu=shard["gpu_id"])
            log_file = open(log_path, "w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(shard["gpu_id"])
            cmd = build_worker_command(
                start_row=shard["start_row"],
                end_row=shard["end_row"],
                part_name=shard["part_name"],
                gpu_id=shard["gpu_id"],
            )
            print(
                f"Launching {shard['part_name']} on GPU {shard['gpu_id']} "
                f"for rows {shard['start_row']}..{shard['end_row']} -> {log_path}"
            )
            proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((log_path, proc, log_file))

        exit_code = 0
        for log_path, proc, log_file in processes:
            ret = proc.wait()
            log_file.close()
            if ret != 0 and exit_code == 0:
                exit_code = ret
            print(f"Finished {log_path} with exit code {ret}")
        return exit_code
    finally:
        for _, proc, log_file in processes:
            if proc.poll() is None:
                proc.terminate()
            if not log_file.closed:
                log_file.close()


def enhance_video(video_path: str) -> str:
    return video_path


def normalize_metadata(row: Dict[str, str]) -> Dict[str, str]:
    row = dict(row)
    if "scene_layout" not in row and "scene_layoutm" in row:
        row["scene_layout"] = row["scene_layoutm"]
    return row


def _meta_block(metadata: Dict[str, str]) -> str:
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")
    return f"""
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
""".strip()


def build_time_prompt(metadata: Dict[str, str]) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at CCTV footage.

Your task is to detect the first clear traffic accident in the video and return ONLY the accident start time in seconds.

{_meta_block(metadata)}

Instructions:
1. Carefully analyze the ENTIRE video.
2. Find the earliest accident_time (in seconds) when a traffic accident CLEARLY BEGINS.
3. accident_time must correspond to the earliest collision moment:
   - the first frame where physical contact begins, or
   - the first frame where collision is clearly unavoidable and immediate.
4. Ignore the exact location and the accident type in this step.
5. Focus only on accurately detecting the first accident_time.

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
  "accident_time"

Output format:
{{
  "accident_time": <float>
}}
"""
    return prompt.strip()


def build_time_candidate_selector_prompt(metadata: Dict[str, str], num_candidates: int) -> str:
    prompt = f"""
You are an expert traffic accident analyst.

You will watch ONE video made by concatenating {num_candidates} short candidate clips.
Each clip is labeled visually as CANDIDATE 1, CANDIDATE 2, ..., CANDIDATE {num_candidates}.
These candidates were generated by optical flow peaks.

{_meta_block(metadata)}

Your task:
Choose exactly ONE candidate clip that is closest to the FIRST physical collision/contact moment.

Important rules:
1. Choose the candidate where the first physical contact begins, not the most dramatic aftermath.
2. If several clips show aftermath, choose the earliest candidate where contact begins.
3. You MUST choose one integer from 1 to {num_candidates}.
4. Do NOT invent a new time.
5. Do NOT output 0.0.
6. Do NOT say none/unknown. Even if uncertain, choose the best candidate.

Critical output rules:
- Output JSON only.
- No reasoning.
- No analysis.
- No markdown.
- No text before JSON.
- No text after JSON.
- The JSON must contain exactly this key:
  "selected_candidate_index"

Output format:
{{
  "selected_candidate_index": <integer from 1 to {num_candidates}>
}}
"""
    return prompt.strip()


def build_location_prompt(metadata: Dict[str, str], accident_time: float, frame_offset: float = 0.0) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at ONE frame from CCTV footage.

This image is from the traffic accident video at approximately:
- accident_time = {accident_time:.3f} seconds
- frame_offset_from_accident = {frame_offset:+.3f} seconds

{_meta_block(metadata)}

Your task is to localize the PRIMARY collision contact point in this frame.

Instructions:
1. Focus on the exact contact region where the collision occurs or is visually beginning.
2. Output normalized coordinates of the center of the contact region:
   - center_x: left=0.0, right=1.0
   - center_y: top=0.0, bottom=1.0
3. Return the center of the CONTACT POINT, not the center of an entire vehicle.
4. If the frame is slightly before or after the first contact, still estimate the same collision point.
5. Ignore accident type classification.
6. If uncertain, choose the single best estimate.

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


def build_location_crop_refine_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at a CROPPED zoom-in image around a predicted collision area.

This crop comes from the traffic accident video at approximately accident_time = {accident_time:.3f} seconds.

{_meta_block(metadata)}

Your task is to refine the collision contact point inside THIS CROP ONLY.

Instructions:
1. The accident contact point is expected to be inside or near the center of this crop.
2. Output normalized coordinates relative to THIS CROP ONLY:
   - center_x: left=0.0, right=1.0
   - center_y: top=0.0, bottom=1.0
3. Return the center of the actual contact region between vehicles/objects.
4. Do not return coordinates for the original full frame.
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


def build_type_binary_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at a SHORT CCTV clip around the first traffic collision.

The clip is centered near accident_time = {accident_time:.3f} seconds.

{_meta_block(metadata)}

Your task is step 1 of accident classification.
Determine whether the FIRST impact involves MULTIPLE vehicles physically colliding with each other.

Definitions:
- true: the first impact is a vehicle-to-vehicle collision involving two or more vehicles.
- false: the first impact involves only one vehicle and a non-vehicle object (for example a pole, barrier, guardrail, ditch, curb, wall, or roadside object), with no direct vehicle-to-vehicle contact.

Instructions:
1. Focus on the FIRST physical impact only, not the aftermath.
2. Watch the short motion in the clip, not just one frame.
3. Return true if another vehicle is directly struck, is struck by the target vehicle, or is clearly part of the first impact.
4. Return false only if the first impact is truly vehicle-to-object or single-vehicle with NO direct vehicle-to-vehicle contact.
5. If another vehicle is plausibly part of the first impact, prefer true rather than false.
6. Do not classify by severity. Use only who physically collides in the first impact.

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
  "involves_multiple_vehicles", "confidence"

Output format:
{{
  "involves_multiple_vehicles": true,
  "confidence": <float>
}}
"""
    return prompt.strip()


def build_type_headon_verifier_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at a SHORT CCTV clip around the first traffic collision.

The clip is centered near accident_time = {accident_time:.3f} seconds.

{_meta_block(metadata)}

The collision is already assumed to involve MULTIPLE vehicles.

Your task:
Decide whether the FIRST impact is a HEAD-ON collision.

Definition:
- head-on: two vehicles traveling in roughly opposite directions collide front-to-front at the first impact.

Instructions:
1. Focus on the FIRST physical impact only.
2. Use vehicle approach directions before contact.
3. Return true only if the first impact is clearly front-to-front between vehicles moving in opposite directions.
4. Return false if the impact is front-to-side, rear-to-front, side-to-side, vehicle-to-object, or unclear.
5. If the vehicles are moving in the same general direction, return false.

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
  "is_head_on", "confidence"

Output format:
{{
  "is_head_on": true,
  "confidence": <float>
}}
"""
    return prompt.strip()


def build_type_tbone_verifier_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at a SHORT CCTV clip around the first traffic collision.

The clip is centered near accident_time = {accident_time:.3f} seconds.

{_meta_block(metadata)}

The collision is already assumed to involve MULTIPLE vehicles and is NOT head-on.

Your task:
Decide whether the FIRST impact is a T-BONE collision.

Definition:
- t-bone: the front of one vehicle hits the side of another vehicle at the first impact.

Instructions:
1. Focus on the FIRST physical impact only.
2. Identify the main contact surfaces.
3. Return true only when one vehicle's front hits another vehicle's side.
4. Return false if the contact is side-to-side overlap, front-to-front, rear-to-front, vehicle-to-object, or unclear.
5. If the geometry is not clearly front-to-side, return false.

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
  "is_t_bone", "confidence"

Output format:
{{
  "is_t_bone": true,
  "confidence": <float>
}}
"""
    return prompt.strip()


def build_type_rearend_vs_sideswipe_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at a SHORT CCTV clip around the first traffic collision.

The clip is centered near accident_time = {accident_time:.3f} seconds.

{_meta_block(metadata)}

The collision is already assumed to involve MULTIPLE vehicles.
It is also assumed to be neither head-on nor t-bone.

Choose exactly one label:
- rear-end: one vehicle's front hits the rear of another vehicle moving in roughly the same direction.
- sideswipe: two vehicles moving in roughly the same direction make side-to-side contact with lateral overlap.

Instructions:
1. Focus on the FIRST physical impact only.
2. Use both motion direction and contact surfaces.
3. If the main impact is front-to-rear, choose "rear-end".
4. If the main impact is side-to-side overlap, choose "sideswipe".
5. Do not output head-on, t-bone, or single.

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
  "type": "<one of: rear-end, sideswipe>"
}}
"""
    return prompt.strip()


def build_type_fallback_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at a SHORT CCTV clip around the first traffic collision.

The clip is centered near accident_time = {accident_time:.3f} seconds.

{_meta_block(metadata)}

Classify the FIRST impact into exactly one of these labels:
["rear-end", "head-on", "sideswipe", "t-bone", "single"]

Definitions:
- rear-end: one vehicle's front hits the rear of another vehicle moving in roughly the same direction.
- head-on: two vehicles moving in roughly opposite directions collide front-to-front.
- sideswipe: two vehicles moving in roughly the same direction make side-to-side contact with lateral overlap.
- t-bone: the front of one vehicle hits the side of another vehicle.
- single: one vehicle first collides with a non-vehicle object, with no direct vehicle-to-vehicle impact.

Instructions:
1. Focus on the FIRST physical impact only, not the aftermath.
2. Use both approach direction and contact surfaces.
3. Prefer a multi-vehicle label if another vehicle is clearly part of the first impact.
4. Return "single" only when the first impact is truly vehicle-to-object.
5. Return "head-on" only for front-to-front opposite-direction impact.
6. Return "t-bone" only for front-to-side impact.
7. Return "rear-end" only for front-to-rear same-direction impact.
8. Return "sideswipe" only for side-to-side overlap contact.

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


def strip_thinking_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def try_parse_single_json(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = strip_thinking_text(text)
    direct = try_parse_single_json(text)
    if direct is not None:
        return direct
    brace_positions = [i for i, ch in enumerate(text) if ch == "{"]
    for start in brace_positions:
        depth = 0
        for end in range(start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    parsed = try_parse_single_json(candidate)
                    if parsed is not None:
                        return parsed
                    break
    return None


def validate_candidate_selection(result: Dict[str, Any], num_candidates: int) -> Optional[int]:
    try:
        idx = int(result["selected_candidate_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if 1 <= idx <= num_candidates:
        return idx
    return None


def validate_time_prediction(result: Dict[str, Any], meta: Dict[str, str]) -> Optional[float]:
    try:
        accident_time = float(result["accident_time"])
    except (KeyError, TypeError, ValueError):
        return None
    duration_str = meta.get("duration", "")
    try:
        duration = float(duration_str)
        accident_time = min(max(accident_time, 0.0), duration)
    except (TypeError, ValueError):
        accident_time = max(accident_time, 0.0)
    return accident_time


def clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def validate_location_prediction(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        center_x = float(result["center_x"])
        center_y = float(result["center_y"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"center_x": clamp01(center_x), "center_y": clamp01(center_y)}


def validate_type_prediction(result: Dict[str, Any], allow_single: bool = True) -> Optional[str]:
    try:
        accident_type = str(result["type"]).strip().lower()
    except (KeyError, TypeError, ValueError):
        return None
    valid = VALID_TYPES if allow_single else VALID_MULTI_TYPES
    if accident_type not in valid:
        return None
    return accident_type


def validate_multi_binary_prediction(result: Dict[str, Any]) -> Optional[Tuple[bool, float]]:
    try:
        value = bool(result["involves_multiple_vehicles"])
    except Exception:
        return None
    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    return value, confidence


def validate_bool_confidence_prediction(result: Dict[str, Any], key: str) -> Optional[Tuple[bool, float]]:
    try:
        value = bool(result[key])
    except Exception:
        return None
    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    return value, confidence


def move_inputs_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if hasattr(v, "to") else v
    return moved


def append_raw_log(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_duration(meta: Dict[str, str]) -> Optional[float]:
    try:
        return float(meta.get("duration", ""))
    except Exception:
        return None


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        if value is None:
            return "None"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def extract_frame_at_time(
    video_path: str,
    accident_time: float,
    meta: Dict[str, str],
    fps_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    def _open():
        cap_ = cv