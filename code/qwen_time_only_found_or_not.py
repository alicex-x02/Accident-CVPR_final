import csv
import json
import os
import re
import time
from threading import Thread
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCIDENT_DIR = os.path.join(BASE_DIR, "accident")
RESULT_DIR = os.path.join(os.path.dirname(BASE_DIR), "result")
METADATA_PATH = os.path.join(ACCIDENT_DIR, "test_metadata.csv")
PREDICTION_PATH = os.path.join(RESULT_DIR, "qwen_time_only_predictions.csv")
RAW_LOG_PATH = os.path.join(ACCIDENT_DIR, "qwen_time_only_raw_outputs.jsonl")

MODEL_NAME = "Qwen/Qwen3.5-9B"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9

# True면 Qwen이 0초 근처를 찍은 경우도 "못찾음"으로 기록한다.
# Qwen의 0초 fallback을 증명/분석하려는 실험이면 True 추천.
NEAR_ZERO_IS_NOT_FOUND = True
NEAR_ZERO_THRESHOLD_SEC = 0.3


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

Your task is to detect the first clear traffic accident in the video.

{_meta_block(metadata)}

Instructions:
1. Carefully analyze the ENTIRE video.
2. Find the earliest accident_time in seconds when a traffic accident CLEARLY BEGINS.
3. accident_time must correspond to the earliest collision moment:
   - the first frame where physical contact begins, or
   - the first frame where collision is clearly unavoidable and immediate.
4. Ignore the exact location and the accident type in this step.
5. If you cannot clearly identify the accident start time, do NOT guess.
6. If the accident is not clearly visible or the timing is uncertain, return found=false.

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
  "found", "accident_time", "confidence"

If you find the accident time, output:
{{
  "found": true,
  "accident_time": <float>,
  "confidence": <float from 0.0 to 1.0>
}}

If you cannot find the accident time, output:
{{
  "found": false,
  "accident_time": null,
  "confidence": 0.0
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


def move_inputs_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if hasattr(v, "to") else v
    return moved


def append_raw_log(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def call_qwen_for_video_time(
    model,
    processor,
    video_path: str,
    prompt: str,
    rel_path: str,
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
                {"type": "video", "path": video_path},
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

            parsed = extract_first_json_object(collected_text)
            append_raw_log(
                RAW_LOG_PATH,
                {
                    "path": rel_path,
                    "stage": "time",
                    "attempt": attempt,
                    "raw_output": collected_text,
                    "parsed": parsed,
                },
            )

            if parsed is not None:
                return parsed
            last_error = f"JSON parse failed on attempt {attempt}. Raw output: {collected_text[:500]}"
        except Exception as e:
            last_error = str(e)
        time.sleep(1.0)

    print(f"    [ERROR] Qwen request failed: {last_error}")
    append_raw_log(
        RAW_LOG_PATH,
        {"path": rel_path, "stage": "time_error", "error": last_error},
    )
    return None


def validate_time_result(result: Optional[Dict[str, Any]], meta: Dict[str, str]) -> Dict[str, Any]:
    if result is None:
        return {
            "time_status": "못찾음",
            "accident_time": "",
            "confidence": 0.0,
            "reason": "no_valid_json_response",
        }

    found = result.get("found", True)
    if isinstance(found, str):
        found = found.strip().lower() in {"true", "yes", "1"}
    else:
        found = bool(found)

    try:
        confidence = float(result.get("confidence", 1.0 if found else 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    if not found:
        return {
            "time_status": "못찾음",
            "accident_time": "",
            "confidence": confidence,
            "reason": "qwen_returned_found_false",
        }

    raw_time = result.get("accident_time", None)
    if raw_time is None:
        return {
            "time_status": "못찾음",
            "accident_time": "",
            "confidence": confidence,
            "reason": "accident_time_is_null",
        }

    try:
        accident_time = float(raw_time)
    except (TypeError, ValueError):
        return {
            "time_status": "못찾음",
            "accident_time": "",
            "confidence": confidence,
            "reason": "invalid_accident_time_schema",
        }

    duration_str = meta.get("duration", "")
    try:
        duration = float(duration_str)
        accident_time = min(max(accident_time, 0.0), duration)
    except (TypeError, ValueError):
        accident_time = max(accident_time, 0.0)

    if NEAR_ZERO_IS_NOT_FOUND and accident_time <= NEAR_ZERO_THRESHOLD_SEC:
        return {
            "time_status": "못찾음",
            "accident_time": "",
            "confidence": confidence,
            "reason": f"near_zero_time<={NEAR_ZERO_THRESHOLD_SEC}",
            "raw_accident_time": accident_time,
        }

    return {
        "time_status": "찾음",
        "accident_time": accident_time,
        "confidence": confidence,
        "reason": "ok",
    }


def read_metadata(csv_path: str):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield normalize_metadata(row)


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
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    fieldnames = ["path", "time_status", "accident_time", "confidence", "reason", "raw_accident_time"]
    saved = 0

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

            abs_video_path = os.path.abspath(video_path)
            print(f"\n[{idx}] Processing: {rel_path}")

            raw_time = call_qwen_for_video_time(
                model=model,
                processor=processor,
                video_path=abs_video_path,
                prompt=build_time_prompt(meta),
                rel_path=rel_path,
            )

            time_result = validate_time_result(raw_time, meta)
            row = {
                "path": rel_path,
                "time_status": time_result.get("time_status", "못찾음"),
                "accident_time": time_result.get("accident_time", ""),
                "confidence": time_result.get("confidence", 0.0),
                "reason": time_result.get("reason", ""),
                "raw_accident_time": time_result.get("raw_accident_time", ""),
            }

            if row["time_status"] == "찾음":
                print(f"  -> 찾음: accident_time={float(row['accident_time']):.4f}, confidence={float(row['confidence']):.3f}")
            else:
                print(f"  -> 못찾음: reason={row['reason']}, raw_time={raw_time}")

            writer.writerow(row)
            prediction_file.flush()
            os.fsync(prediction_file.fileno())
            saved += 1
            print(f"  -> CSV updated: {PREDICTION_PATH} ({saved} rows saved)")

    print(f"\nSaved time-only predictions to: {PREDICTION_PATH}")
    print(f"Raw outputs saved to: {RAW_LOG_PATH}")


if __name__ == "__main__":
    main()
