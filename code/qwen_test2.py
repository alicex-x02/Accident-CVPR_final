import csv
import json
import os
import re
import tempfile
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCIDENT_DIR = os.path.join(BASE_DIR, "accident")
METADATA_PATH = os.path.join(ACCIDENT_DIR, "test_metadata.csv")
PREDICTION_PATH = os.path.join(ACCIDENT_DIR, "predictions.csv")
RAW_LOG_PATH = os.path.join(ACCIDENT_DIR, "raw_outputs.jsonl")
TEMP_DIR = os.path.join(ACCIDENT_DIR, "temp_clips")

MODEL_NAME = "Qwen/Qwen3.5-9B"
VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single", "other"}

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9

# Hybrid pipeline parameters
TIME_SAMPLE_FPS = 5.0
TIME_SMOOTH_WINDOW = 5
TIME_TOP_K = 3
TIME_MIN_SEPARATION_SEC = 1.5
VERIFY_WINDOW_SEC = 1.0
VERIFY_CLIP_FPS = 8.0

LOCATION_HALF_WINDOW_SEC = 0.5
LOCATION_SAMPLE_FPS = 8.0
LOCATION_HOTSPOT_QUANTILE = 0.95

TYPE_WINDOW_SEC = 1.0
TYPE_CLIP_FPS = 8.0
CLIP_MAX_SIDE = 448
FLOW_RESIZE_WIDTH = 320


def enhance_video(video_path: str) -> str:
    return video_path


def normalize_metadata(row: Dict[str, str]) -> Dict[str, str]:
    row = dict(row)
    if "scene_layout" not in row and "scene_layoutm" in row:
        row["scene_layout"] = row["scene_layoutm"]
    return row


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
                    candidate = text[start:end + 1]
                    parsed = try_parse_single_json(candidate)
                    if parsed is not None:
                        return parsed
                    break
    return None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def move_inputs_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def append_raw_log(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_predictions_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["path", "accident_time", "center_x", "center_y", "type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_metadata(csv_path: str):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield normalize_metadata(row)


# ---------------------------
# Prompt builders
# ---------------------------
def build_verify_time_prompt(candidate_time: float, metadata: Dict[str, str]) -> str:
    return f"""
You are analyzing a short CCTV traffic clip centered near {candidate_time:.2f} seconds.

Question:
Does this clip contain the moment when the traffic accident FIRST clearly begins?

Important rules:
1. Focus on the first actual collision or the first unavoidable impact moment.
2. Do NOT confuse the following with accident onset:
   - normal driving
   - large but non-collision motion
   - passing behind a tree, pole, barrier, or other occluder
   - near-miss events
   - aftermath only
3. If the clip contains only approach motion or only post-impact aftermath, confidence should be low.
4. If uncertain, be conservative.

Return JSON only:
{{
  "contains_accident": true,
  "confidence": 0.0
}}
""".strip()


def build_type_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")

    return f"""
You are an expert traffic accident analyst looking at a short CCTV clip.

This clip is centered around the first clear accident moment at approximately {accident_time:.3f} seconds.

Metadata:
- region: {region}
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}

Choose exactly one accident type:
- rear-end: one vehicle hits the back of another moving in the same direction
- head-on: two vehicles moving in opposite directions collide front-to-front
- sideswipe: two vehicles moving in roughly the same direction make side contact while overlapping
- t-bone: the front of one vehicle hits the side of another vehicle, forming a T shape
- single: only one vehicle is involved, such as hitting a pole, barrier, guardrail, tree, or going off road
- other: accident exists but none of the above fits well

Return JSON only:
{{
  "type": "rear-end"
}}
""".strip()


# ---------------------------
# Video / Flow helpers
# ---------------------------
def get_video_info(video_path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    fps = fps if fps > 0 else 30.0
    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def _resize_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    if resize_width <= 0 or frame.shape[1] <= resize_width:
        return frame
    scale = resize_width / float(frame.shape[1])
    resize_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)


def moving_average(values: Sequence[float], window_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or window_size <= 1:
        return arr
    window_size = min(window_size, arr.size)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(arr, kernel, mode="same")


def compute_motion_curve(
    video_path: str,
    sample_fps: float = TIME_SAMPLE_FPS,
    resize_width: int = FLOW_RESIZE_WIDTH,
) -> Tuple[np.ndarray, np.ndarray]:
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_step = max(int(round(info["fps"] / max(sample_fps, 1e-6))), 1)
    times: List[float] = []
    motion_values: List[float] = []
    prev_gray = None
    current_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if current_frame_idx % frame_step != 0:
            current_frame_idx += 1
            continue

        resized = _resize_frame(frame, resize_width=resize_width)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

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
            motion_values.append(float(np.mean(magnitude)))
            times.append(current_frame_idx / info["fps"])

        prev_gray = gray
        current_frame_idx += 1

    cap.release()
    return np.asarray(times, dtype=np.float32), np.asarray(motion_values, dtype=np.float32)


def select_top_k_peaks(
    times: Sequence[float],
    values: Sequence[float],
    top_k: int = TIME_TOP_K,
    min_separation_sec: float = TIME_MIN_SEPARATION_SEC,
) -> List[float]:
    times_arr = np.asarray(times, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)

    if times_arr.size == 0 or values_arr.size == 0:
        return []

    order = np.argsort(values_arr)[::-1]
    selected: List[float] = []
    for idx in order:
        candidate_time = float(times_arr[idx])
        if all(abs(candidate_time - chosen) >= min_separation_sec for chosen in selected):
            selected.append(candidate_time)
        if len(selected) >= top_k:
            break

    if not selected:
        selected.append(float(times_arr[int(np.argmax(values_arr))]))
    return sorted(selected)


def detect_candidate_times(video_path: str) -> Tuple[List[float], np.ndarray, np.ndarray]:
    times, motion_values = compute_motion_curve(video_path=video_path, sample_fps=TIME_SAMPLE_FPS)
    if motion_values.size == 0:
        return [], times, motion_values
    smoothed = moving_average(motion_values, TIME_SMOOTH_WINDOW)
    candidates = select_top_k_peaks(times, smoothed, TIME_TOP_K, TIME_MIN_SEPARATION_SEC)
    return candidates, times, smoothed


def write_video_clip(
    video_path: str,
    center_time_sec: float,
    output_path: str,
    window_sec: float = 1.0,
    clip_fps: float = 8.0,
    max_side: int = CLIP_MAX_SIDE,
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
        width, height = CLIP_MAX_SIDE, CLIP_MAX_SIDE
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


# ---------------------------
# Qwen wrappers
# ---------------------------
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
            "content": [
                {"type": "text", "text": "Respond with JSON only. No reasoning. No explanation. /no_think"}
            ],
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

            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

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

            append_raw_log(
                RAW_LOG_PATH,
                {
                    "path": rel_path,
                    "stage": stage,
                    "attempt": attempt,
                    "raw_output": collected_text,
                },
            )

            parsed = extract_first_json_object(collected_text)
            if parsed is not None:
                return parsed

            last_error = f"JSON parse failed on attempt {attempt}. Raw output: {collected_text[:500]}"
        except Exception as e:
            last_error = str(e)
        time.sleep(1.0)

    print(f"    [ERROR] Qwen request failed: {last_error}")
    return None


def call_qwen_for_temp_clip(
    model,
    processor,
    video_path: str,
    center_time_sec: float,
    prompt: str,
    rel_path: str,
    stage: str,
    window_sec: float,
    clip_fps: float,
) -> Optional[Dict[str, Any]]:
    os.makedirs(TEMP_DIR, exist_ok=True)
    safe_time = f"{center_time_sec:.3f}".replace(".", "_")
    with tempfile.NamedTemporaryFile(
        prefix=f"{stage}_{safe_time}_",
        suffix=".mp4",
        dir=TEMP_DIR,
        delete=False,
    ) as fp:
        clip_path = fp.name

    try:
        write_video_clip(
            video_path=video_path,
            center_time_sec=center_time_sec,
            output_path=clip_path,
            window_sec=window_sec,
            clip_fps=clip_fps,
        )
        return call_qwen_for_media(
            model=model,
            processor=processor,
            media_type="video",
            media_path=os.path.abspath(clip_path),
            prompt=prompt,
            rel_path=rel_path,
            stage=stage,
        )
    finally:
        if os.path.exists(clip_path):
            os.remove(clip_path)


# ---------------------------
# Validation
# ---------------------------
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


def validate_type_prediction(result: Dict[str, Any]) -> Optional[str]:
    try:
        accident_type = str(result["type"]).strip().lower()
    except (KeyError, TypeError, ValueError):
        return None

    aliases = {
        "rear-end": "rear-end",
        "rear end": "rear-end",
        "head-on": "head-on",
        "head on": "head-on",
        "sideswipe": "sideswipe",
        "side-swipe": "sideswipe",
        "side swipe": "sideswipe",
        "t-bone": "t-bone",
        "t bone": "t-bone",
        "single": "single",
        "other": "other",
    }
    accident_type = aliases.get(accident_type, accident_type)
    if accident_type not in VALID_TYPES:
        return None
    return accident_type


# ---------------------------
# Hybrid modules
# ---------------------------
def verify_accident_time(
    video_path: str,
    candidate_times: List[float],
    meta: Dict[str, str],
    model,
    processor,
    rel_path: str,
) -> Tuple[float, List[Dict[str, float]]]:
    if not candidate_times:
        return 0.0, []

    scored_candidates: List[Dict[str, float]] = []
    for candidate_time in candidate_times:
        prompt = build_verify_time_prompt(candidate_time, meta)
        parsed = call_qwen_for_temp_clip(
            model=model,
            processor=processor,
            video_path=video_path,
            center_time_sec=candidate_time,
            prompt=prompt,
            rel_path=rel_path,
            stage="time_verify",
            window_sec=VERIFY_WINDOW_SEC,
            clip_fps=VERIFY_CLIP_FPS,
        )
        if parsed is None:
            contains_accident = False
            confidence = 0.0
        else:
            contains_accident = bool(parsed.get("contains_accident", False))
            try:
                confidence = clamp(float(parsed.get("confidence", 0.0)), 0.0, 1.0)
            except Exception:
                confidence = 0.0

        scored_candidates.append(
            {
                "candidate_time": float(candidate_time),
                "contains_accident": float(1.0 if contains_accident else 0.0),
                "confidence": float(confidence),
            }
        )

    scored_candidates.sort(
        key=lambda item: (item["contains_accident"], item["confidence"]),
        reverse=True,
    )
    return float(scored_candidates[0]["candidate_time"]), scored_candidates


def estimate_location(
    video_path: str,
    accident_time_sec: float,
    half_window_sec: float = LOCATION_HALF_WINDOW_SEC,
    sample_fps: float = LOCATION_SAMPLE_FPS,
    resize_width: int = FLOW_RESIZE_WIDTH,
    hotspot_quantile: float = LOCATION_HOTSPOT_QUANTILE,
) -> Tuple[float, float]:
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    start_time = max(0.0, accident_time_sec - half_window_sec)
    end_time = min(info["duration_sec"], accident_time_sec + half_window_sec)
    start_frame = max(0, int(start_time * info["fps"]))
    end_frame = max(start_frame + 1, int(end_time * info["fps"]))
    frame_step = max(int(round(info["fps"] / max(sample_fps, 1e-6))), 1)

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
        return 0.5, 0.5

    threshold = float(np.quantile(accumulated_motion, hotspot_quantile))
    hotspot_mask = accumulated_motion >= threshold
    if not np.any(hotspot_mask):
        hotspot_mask = accumulated_motion > 0

    ys, xs = np.where(hotspot_mask)
    weights = accumulated_motion[ys, xs]
    if weights.sum() <= 0:
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))
    else:
        center_x = float(np.average(xs, weights=weights))
        center_y = float(np.average(ys, weights=weights))

    height, width = out_shape
    norm_x = clamp(center_x / max(width - 1, 1), 0.0, 1.0)
    norm_y = clamp(center_y / max(height - 1, 1), 0.0, 1.0)
    return norm_x, norm_y


def classify_accident_type(
    video_path: str,
    accident_time_sec: float,
    meta: Dict[str, str],
    model,
    processor,
    rel_path: str,
) -> Optional[str]:
    parsed = call_qwen_for_temp_clip(
        model=model,
        processor=processor,
        video_path=video_path,
        center_time_sec=accident_time_sec,
        prompt=build_type_prompt(meta, accident_time_sec),
        rel_path=rel_path,
        stage="type",
        window_sec=TYPE_WINDOW_SEC,
        clip_fps=TYPE_CLIP_FPS,
    )
    if parsed is None:
        return None
    return validate_type_prediction(parsed)


# ---------------------------
# Path helpers
# ---------------------------
def resolve_video_path(meta: Dict[str, str]) -> Optional[str]:
    rel_path = meta.get("path")
    if rel_path:
        candidate = os.path.join(ACCIDENT_DIR, rel_path)
        if os.path.exists(candidate):
            return candidate
        # handle path without base dir nested mismatch
        candidate2 = os.path.join(ACCIDENT_DIR, os.path.basename(rel_path))
        if os.path.exists(candidate2):
            return candidate2

    video_id = meta.get("video_id")
    if video_id:
        candidates = [
            os.path.join(ACCIDENT_DIR, f"{video_id}.mp4"),
            os.path.join(ACCIDENT_DIR, "videos", f"{video_id}.mp4"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

    return None


def output_identifier(meta: Dict[str, str], video_path: str) -> str:
    if meta.get("path"):
        return meta["path"]
    if meta.get("video_id"):
        return str(meta["video_id"])
    return os.path.basename(video_path)


# ---------------------------
# Main
# ---------------------------
def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")

    os.makedirs(ACCIDENT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    if os.path.exists(RAW_LOG_PATH):
        os.remove(RAW_LOG_PATH)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    predictions = []
    write_predictions_csv(PREDICTION_PATH, predictions)
    print(f"Initialized predictions CSV: {PREDICTION_PATH}")

    for idx, meta in enumerate(read_metadata(METADATA_PATH), start=1):
        video_path = resolve_video_path(meta)
        rel_path = output_identifier(meta, video_path or "")

        if not video_path:
            print(f"[WARN] Row {idx}: video not found for metadata row: {meta}")
            continue

        enhanced_video_path = enhance_video(video_path)
        abs_video_path = os.path.abspath(enhanced_video_path)

        print(f"\n[{idx}] Processing: {rel_path}")

        # 1) time: optical flow candidate generation
        candidate_times, _, _ = detect_candidate_times(abs_video_path)
        if not candidate_times:
            candidate_times = [0.0]
        print(f"  -> candidate_times={candidate_times}")

        # 2) time: Qwen verification over candidate clips
        accident_time, scored_candidates = verify_accident_time(
            video_path=abs_video_path,
            candidate_times=candidate_times,
            meta=meta,
            model=model,
            processor=processor,
            rel_path=rel_path,
        )
        print(f"  -> verified accident_time={accident_time:.4f}")
        append_raw_log(
            RAW_LOG_PATH,
            {
                "path": rel_path,
                "stage": "time_candidates",
                "candidate_times": candidate_times,
                "scored_candidates": scored_candidates,
            },
        )

        # 3) location: flow hotspot centroid around predicted time
        center_x, center_y = estimate_location(abs_video_path, accident_time)
        print(f"  -> predicted location: center_x={center_x:.4f}, center_y={center_y:.4f}")

        # 4) type: Qwen clip classification around accident time
        accident_type = classify_accident_type(
            video_path=abs_video_path,
            accident_time_sec=accident_time,
            meta=meta,
            model=model,
            processor=processor,
            rel_path=rel_path,
        )
        if accident_type is None:
            accident_type = "other"
        print(f"  -> predicted type={accident_type}")

        predictions.append(
            {
                "path": rel_path,
                "accident_time": float(round(accident_time, 4)),
                "center_x": float(round(center_x, 6)),
                "center_y": float(round(center_y, 6)),
                "type": accident_type,
            }
        )
        write_predictions_csv(PREDICTION_PATH, predictions)
        print(f"  -> updated predictions CSV ({len(predictions)} rows)")

    if predictions:
        print(f"\nSaved predictions to: {PREDICTION_PATH}")
    else:
        print("\nNo predictions generated.")

    print(f"Raw outputs saved to: {RAW_LOG_PATH}")


if __name__ == "__main__":
    main()
