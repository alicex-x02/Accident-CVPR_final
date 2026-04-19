import csv
import json
import math
import os
import re
import shutil
import tempfile
import time
import sys
from statistics import median
from threading import Thread
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = BASE_DIR
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
from pipeline.aimv2_type_classifier import AIMv2TypePredictor, DEFAULT_CHECKPOINT_DIR, pick_best_cuda_device
from pipeline.optical_flow import compute_motion_curve, moving_average, select_top_k_peaks

ACCIDENT_DIR = os.path.join(BASE_DIR, "accident")
RESULT_DIR = os.path.join(os.path.dirname(BASE_DIR), "result")
METADATA_PATH = os.path.join(ACCIDENT_DIR, "test_metadata.csv")
# Keep the output filename specific to this pipeline so it is easy to tell apart
# from other experiment runs in the shared `result` directory.
PREDICTION_PATH = os.path.join(RESULT_DIR, "test6_vision_encoder_predictions.csv")
RAW_LOG_PATH = os.path.join(ACCIDENT_DIR, "test6_vision_encoder_raw_outputs.jsonl")

MODEL_NAME = "Qwen/Qwen3.5-9B"
VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}
VALID_MULTI_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone"}

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9

LOCATION_FRAME_OFFSETS = (-0.20, 0.0, 0.20)
LOCATION_CROP_SCALES = (0.40, 0.26)
TYPE_WINDOW_SEC = 1.2
TYPE_CLIP_FPS = 8.0
TYPE_CHECKPOINT_PATH = os.path.join(DEFAULT_CHECKPOINT_DIR, "best.pt")
TIME_FLOW_SAMPLE_FPS = 5.0
TIME_FLOW_SMOOTH_WINDOW = 5
TIME_FLOW_TOP_K = 5
TIME_FLOW_MIN_SEPARATION_SEC = 1.5
TIME_FLOW_DELTA_WEIGHT = 0.7
TIME_REFINE_PRE_SEC = 1.5
TIME_REFINE_POST_SEC = 1.0
TIME_REFINE_CLIP_FPS = 8.0
TIME_REFINE_CONFIDENCE_MARGIN = 0.15


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
Determine whether the collision involves MULTIPLE vehicles physically colliding with each other.

Definitions:
- true: two or more vehicles are physically involved in the impact with each other.
- false: only one vehicle is involved in the accident impact (for example hitting a pole, barrier, guardrail, ditch, or roadside object), with no direct vehicle-to-vehicle collision.

Instructions:
1. Watch the short motion in the clip, not just one frame.
2. Decide whether another vehicle is clearly part of the actual impact.
3. If another vehicle is clearly struck or strikes the target vehicle, return true.
4. Only return false when the accident is truly single-vehicle.

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


def build_type_multi_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at a SHORT CCTV clip around the first traffic collision.

The clip is centered near accident_time = {accident_time:.3f} seconds.

{_meta_block(metadata)}

The collision is already assumed to involve MULTIPLE vehicles.
Classify the multi-vehicle collision into exactly one of these four types:
- rear-end: one vehicle crashes into the back of another vehicle traveling in the same direction.
- head-on: two vehicles traveling in opposite directions collide front-to-front.
- sideswipe: two vehicles moving in roughly the same direction make side-to-side contact while overlapping partially.
- t-bone: the front of one vehicle crashes into the side of another vehicle, forming a T shape.

Instructions:
1. Use the motion in the clip, not just one frame.
2. Compare the relative approach directions and the contact surfaces.
3. Choose exactly one label from ["rear-end", "head-on", "sideswipe", "t-bone"].
4. Do not output "single" in this step.

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
  "type": "<one of: rear-end, head-on, sideswipe, t-bone>"
}}
"""
    return prompt.strip()


def build_type_fallback_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at a SHORT CCTV clip around the first traffic collision.

The clip is centered near accident_time = {accident_time:.3f} seconds.

{_meta_block(metadata)}

Classify the accident type into exactly one of these labels:
["rear-end", "head-on", "sideswipe", "t-bone", "single"]

Definitions:
- rear-end: one vehicle crashes into the back of another vehicle traveling in the same direction.
- head-on: two vehicles traveling in opposite directions collide front-to-front.
- sideswipe: two vehicles moving in roughly the same direction make side-to-side contact while overlapping partially.
- t-bone: the front of one vehicle crashes into the side of another vehicle, forming a T shape.
- single: only one vehicle is involved in the crash, with no direct vehicle-to-vehicle collision.

Instructions:
1. Use motion over the clip, not just one frame.
2. Prefer a non-single label when another vehicle is clearly part of the impact.
3. Return only the best single label.

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


def build_aimv2_type_predictor() -> AIMv2TypePredictor:
    device_name = os.environ.get("AIMV2_TYPE_DEVICE", "cpu").strip().lower()
    if device_name in {"auto", "cuda"} and torch.cuda.is_available():
        predictor_device: torch.device | str = pick_best_cuda_device()
    else:
        predictor_device = torch.device("cpu")

    return AIMv2TypePredictor.from_checkpoint(
        checkpoint_path=TYPE_CHECKPOINT_PATH,
        device=predictor_device,
        video_root=ACCIDENT_DIR,
    )


def predict_type_with_aimv2(
    type_predictor: AIMv2TypePredictor,
    abs_video_path: str,
    rel_path: str,
    accident_time: float,
) -> Tuple[Optional[str], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "checkpoint": TYPE_CHECKPOINT_PATH,
        "device": str(type_predictor.device),
        "accident_time": float(accident_time),
    }
    try:
        accident_type = type_predictor.predict_type(abs_video_path, accident_time)
        diagnostics["prediction"] = accident_type
        diagnostics["path"] = rel_path
        return accident_type, diagnostics
    except Exception as exc:
        diagnostics["error"] = str(exc)
        print(f"  [WARN] AIMv2 type prediction failed, falling back to single: {exc}")
        return "single", diagnostics


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


def move_inputs_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if hasattr(v, "to") else v
    return moved


def append_raw_log(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def extract_frame_at_time(
    video_path: str,
    accident_time: float,
    meta: Dict[str, str],
    fps_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    def _open():
        cap_ = cv2.VideoCapture(video_path)
        return cap_ if cap_.isOpened() else None

    cap = _open()
    if cap is None:
        return None

    fps = fps_override if fps_override is not None else cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 and meta.get("duration") and meta.get("no_frames"):
        try:
            fps = float(meta["no_frames"]) / float(meta["duration"])
        except Exception:
            fps = 0

    if fps <= 0:
        cap.release()
        return None

    frame_index = int(accident_time * fps)
    frame_index = max(0, min(frame_index, max(0, total_frames - 1)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}

    cap.release()
    cap = _open()
    if cap is None:
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

    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, frame = cap.read()
        if ret and frame is not None:
            cap.release()
            return {"frame": frame, "fps": fps, "frame_index": max(0, total_frames - 1)}

    cap.release()
    return None


def save_temp_frame(frame: Any, prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".jpg", dir=ACCIDENT_DIR)
    os.close(fd)
    if not cv2.imwrite(path, frame):
        raise RuntimeError(f"Failed to write temp frame: {path}")
    return path


def create_center_crop(frame: Any, center_x: float, center_y: float, scale: float) -> Tuple[Any, Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    crop_w = max(32, int(round(w * scale)))
    crop_h = max(32, int(round(h * scale)))
    cx = int(round(center_x * (w - 1)))
    cy = int(round(center_y * (h - 1)))
    x1 = max(0, min(w - crop_w, cx - crop_w // 2))
    y1 = max(0, min(h - crop_h, cy - crop_h // 2))
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    crop = frame[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


def remap_crop_point_to_global(local_xy: Dict[str, float], bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    px = x1 + clamp01(local_xy["center_x"]) * (crop_w - 1)
    py = y1 + clamp01(local_xy["center_y"]) * (crop_h - 1)
    return {
        "center_x": clamp01(px / max(w - 1, 1)),
        "center_y": clamp01(py / max(h - 1, 1)),
    }


def median_xy(points: Sequence[Dict[str, float]]) -> Dict[str, float]:
    xs = [p["center_x"] for p in points]
    ys = [p["center_y"] for p in points]
    return {"center_x": clamp01(float(median(xs))), "center_y": clamp01(float(median(ys)))}


def copy_to_persistent_frame_dir(src_path: str, rel_path: str, suffix: str) -> str:
    frame_dir = os.path.join(ACCIDENT_DIR, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    dst = os.path.join(frame_dir, f"{stem}_{suffix}.jpg")
    shutil.copyfile(src_path, dst)
    return dst


def write_video_clip(video_path: str, center_time_sec: float, output_path: str, window_sec: float, clip_fps: float) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if src_fps <= 0:
        src_fps = clip_fps if clip_fps > 0 else 8.0

    start_time = max(0.0, center_time_sec - window_sec)
    end_time = max(start_time + 0.05, center_time_sec + window_sec)

    start_frame = max(0, int(round(start_time * src_fps)))
    end_frame = min(max(start_frame + 1, frame_count), int(round(end_time * src_fps)))
    step = max(1, int(round(src_fps / max(clip_fps, 1e-6))))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, clip_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while idx < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if (idx - start_frame) % step == 0:
            writer.write(frame)
        idx += 1

    writer.release()
    cap.release()


def save_temp_clip(video_path: str, accident_time: float, window_sec: float, clip_fps: float, prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".mp4", dir=ACCIDENT_DIR)
    os.close(fd)
    write_video_clip(video_path, accident_time, path, window_sec, clip_fps)
    return path


def call_qwen_for_media(
    model,
    processor,
    media_type: str,
    media_path: str,
    prompt: str,
    rel_path: str,
    stage: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Respond with JSON only. No reasoning. No explanation. /no_think"}],
        },
        {
            "role": "user",
            "content": [
                {"type": media_type, "path": media_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            processed = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            processed = move_inputs_to_device(processed, model.device)
            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **processed,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                streamer=streamer,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            print("  -> model output: ", end="", flush=True)
            collected_text = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                collected_text += new_text
            thread.join()
            print()
            append_raw_log(RAW_LOG_PATH, {"path": rel_path, "stage": stage, "attempt": attempt, "raw_output": collected_text})
            parsed = extract_first_json_object(collected_text)
            if parsed is not None:
                return parsed
            last_error = f"JSON parse failed on attempt {attempt}. Raw output: {collected_text[:500]}"
        except Exception as e:
            last_error = str(e)
        time.sleep(1.0)
    print(f"    [ERROR] Qwen request failed: {last_error}")
    return None


def read_metadata(csv_path: str):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield normalize_metadata(row)


def predict_location_with_multiframe_refine(
    model,
    processor,
    abs_video_path: str,
    rel_path: str,
    meta: Dict[str, str],
    accident_time: float,
) -> Tuple[Optional[Dict[str, float]], Optional[str], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"coarse_points": [], "refined_points": []}
    extracted_frames: List[Tuple[float, Any, str]] = []

    for offset in LOCATION_FRAME_OFFSETS:
        sample_time = max(0.0, accident_time + offset)
        extracted = extract_frame_at_time(abs_video_path, sample_time, meta)
        if extracted is None:
            continue
        frame = extracted["frame"]
        frame_path = save_temp_frame(frame, prefix=f"loc_{offset:+.2f}_".replace(".", "_"))
        extracted_frames.append((offset, frame, frame_path))
        raw_loc = call_qwen_for_media(
            model=model,
            processor=processor,
            media_type="image",
            media_path=os.path.abspath(frame_path),
            prompt=build_location_prompt(meta, accident_time, frame_offset=offset),
            rel_path=rel_path,
            stage=f"location_coarse_{offset:+.2f}",
        )
        loc = validate_location_prediction(raw_loc) if raw_loc is not None else None
        if loc is not None:
            diagnostics["coarse_points"].append({"offset": offset, **loc})

    if not diagnostics["coarse_points"]:
        for _, _, frame_path in extracted_frames:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        return None, None, diagnostics

    coarse = median_xy(diagnostics["coarse_points"])
    diagnostics["coarse_median"] = coarse

    central_entry = None
    for offset, frame, frame_path in extracted_frames:
        if abs(offset) < 1e-6:
            central_entry = (offset, frame, frame_path)
            break
    if central_entry is None:
        central_entry = extracted_frames[len(extracted_frames) // 2]

    _, central_frame, central_frame_path = central_entry
    persistent_frame_path = copy_to_persistent_frame_dir(central_frame_path, rel_path, f"t{accident_time:.3f}")

    refined_candidates: List[Dict[str, float]] = [coarse]
    for scale in LOCATION_CROP_SCALES:
        crop, bbox = create_center_crop(central_frame, coarse["center_x"], coarse["center_y"], scale)
        crop_path = save_temp_frame(crop, prefix=f"loc_crop_{int(scale * 100):02d}_")
        raw_crop = call_qwen_for_media(
            model=model,
            processor=processor,
            media_type="image",
            media_path=os.path.abspath(crop_path),
            prompt=build_location_crop_refine_prompt(meta, accident_time),
            rel_path=rel_path,
            stage=f"location_crop_refine_{int(scale * 100):02d}",
        )
        crop_loc = validate_location_prediction(raw_crop) if raw_crop is not None else None
        if crop_loc is not None:
            global_loc = remap_crop_point_to_global(crop_loc, bbox, central_frame.shape)
            diagnostics["refined_points"].append({"scale": scale, **global_loc})
            refined_candidates.append(global_loc)
        if os.path.exists(crop_path):
            os.remove(crop_path)

    final_loc = median_xy(refined_candidates)
    diagnostics["final_location"] = final_loc

    for _, _, frame_path in extracted_frames:
        if os.path.exists(frame_path):
            os.remove(frame_path)

    return final_loc, persistent_frame_path, diagnostics


def predict_type_with_clip_two_stage(
    model,
    processor,
    abs_video_path: str,
    rel_path: str,
    meta: Dict[str, str],
    accident_time: float,
) -> Tuple[Optional[str], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}
    clip_path = save_temp_clip(abs_video_path, accident_time, window_sec=TYPE_WINDOW_SEC, clip_fps=TYPE_CLIP_FPS, prefix="type_clip_")
    try:
        raw_binary = call_qwen_for_media(
            model=model,
            processor=processor,
            media_type="video",
            media_path=os.path.abspath(clip_path),
            prompt=build_type_binary_prompt(meta, accident_time),
            rel_path=rel_path,
            stage="type_binary",
        )
        binary = validate_multi_binary_prediction(raw_binary) if raw_binary is not None else None
        diagnostics["binary"] = raw_binary

        if binary is not None:
            is_multi, confidence = binary
            diagnostics["binary_decision"] = {"is_multi": is_multi, "confidence": confidence}
            if not is_multi and confidence >= 0.45:
                return "single", diagnostics

            raw_multi = call_qwen_for_media(
                model=model,
                processor=processor,
                media_type="video",
                media_path=os.path.abspath(clip_path),
                prompt=build_type_multi_prompt(meta, accident_time),
                rel_path=rel_path,
                stage="type_multi",
            )
            multi_type = validate_type_prediction(raw_multi, allow_single=False) if raw_multi is not None else None
            diagnostics["multi"] = raw_multi
            if multi_type is not None:
                return multi_type, diagnostics

            if not is_multi:
                return "single", diagnostics

        raw_fallback = call_qwen_for_media(
            model=model,
            processor=processor,
            media_type="video",
            media_path=os.path.abspath(clip_path),
            prompt=build_type_fallback_prompt(meta, accident_time),
            rel_path=rel_path,
            stage="type_fallback",
        )
        diagnostics["fallback"] = raw_fallback
        fallback_type = validate_type_prediction(raw_fallback, allow_single=True) if raw_fallback is not None else None
        return fallback_type, diagnostics
    finally:
        if os.path.exists(clip_path):
            os.remove(clip_path)


def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")

    os.makedirs(ACCIDENT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    if os.path.exists(RAW_LOG_PATH):
        os.remove(RAW_LOG_PATH)
    if os.path.exists(PREDICTION_PATH):
        os.remove(PREDICTION_PATH)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")
    type_predictor = build_aimv2_type_predictor()
    print(f"Using AIMv2 type classifier: {TYPE_CHECKPOINT_PATH} on {type_predictor.device}")

    predictions: List[Dict[str, Any]] = []
    fieldnames = ["path", "accident_time", "center_x", "center_y", "type"]

    with open(PREDICTION_PATH, "w", newline="", encoding="utf-8") as prediction_file:
        writer = csv.DictWriter(prediction_file, fieldnames=fieldnames)
        writer.writeheader()
        prediction_file.flush()
        os.fsync(prediction_file.fileno())

        for idx, meta in enumerate(read_metadata(METADATA_PATH), start=1):
            rel_path = meta.get("path")
            if not rel_path:
                print(f"[WARN] Row {idx}: missing path column, skipping.")
                continue

            video_path = os.path.join(ACCIDENT_DIR, rel_path)
            if not os.path.exists(video_path):
                print(f"[WARN] Video file not found: {video_path}")
                continue

            enhanced_video_path = enhance_video(video_path)
            abs_video_path = os.path.abspath(enhanced_video_path)
            print(f"\n[{idx}] Processing: {rel_path}")

            raw_time = call_qwen_for_media(
                model=model,
                processor=processor,
                media_type="video",
                media_path=abs_video_path,
                prompt=build_time_prompt(meta),
                rel_path=rel_path,
                stage="time",
            )
            if raw_time is None:
                print("  -> Failed to get valid JSON response for accident_time")
                continue

            accident_time = validate_time_prediction(raw_time, meta)
            if accident_time is None:
                print(f"  -> Invalid time prediction schema: {raw_time}")
                continue
            print(f"  -> predicted accident_time={accident_time:.4f}")

            location, persistent_frame_path, location_diag = predict_location_with_multiframe_refine(
                model=model,
                processor=processor,
                abs_video_path=abs_video_path,
                rel_path=rel_path,
                meta=meta,
                accident_time=accident_time,
            )
            append_raw_log(RAW_LOG_PATH, {"path": rel_path, "stage": "location_summary", **location_diag})
            if location is None:
                print("  -> Failed to get valid location prediction")
                continue
            center_x, center_y = location["center_x"], location["center_y"]
            print(f"  -> predicted location: center_x={center_x:.4f}, center_y={center_y:.4f}")
            if persistent_frame_path:
                print(f"  -> saved representative frame: {persistent_frame_path}")

            accident_type, type_diag = predict_type_with_aimv2(
                type_predictor=type_predictor,
                abs_video_path=abs_video_path,
                rel_path=rel_path,
                accident_time=accident_time,
            )
            append_raw_log(RAW_LOG_PATH, {"path": rel_path, "stage": "type_aimv2", **type_diag})
            if accident_type is None:
                print("  -> Failed to predict accident type")
                continue

            print(
                f"  -> final parsed result: accident_time={accident_time:.4f}, "
                f"center_x={center_x:.4f}, center_y={center_y:.4f}, type={accident_type}"
            )

            row = {
                "path": rel_path,
                "accident_time": accident_time,
                "center_x": center_x,
                "center_y": center_y,
                "type": accident_type,
            }
            predictions.append(row)
            writer.writerow(row)
            prediction_file.flush()
            os.fsync(prediction_file.fileno())
            print(f"  -> CSV updated: {PREDICTION_PATH} ({len(predictions)} rows saved)")

    if predictions:
        print(f"\nSaved predictions incrementally to: {PREDICTION_PATH}")
    else:
        print("\nNo predictions generated.")
    print(f"Raw outputs saved to: {RAW_LOG_PATH}")


if __name__ == "__main__":
    main()
