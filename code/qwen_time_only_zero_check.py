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
PREDICTION_PATH = os.path.join(RESULT_DIR, "qwen_time_only_zero_check_predictions.csv")
RAW_LOG_PATH = os.path.join(ACCIDENT_DIR, "qwen_time_only_zero_check_raw_outputs.jsonl")

MODEL_NAME = "Qwen/Qwen3.5-9B"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9
ZERO_EPS = 1e-6


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


# 1차 질문: 원래 qwen_test5.py의 시간 프롬프트 그대로 사용.
# 즉 원래 찾던 애들은 최대한 그대로 찾게 둔다.
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


# 2차 질문: 1차에서 정확히 0.0을 냈을 때만 사용.
# 여기서만 "진짜 0초 사고인지 / 못 찾은 건지" 판단하게 한다.
def build_zero_check_prompt(metadata: Dict[str, str]) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at CCTV footage.

The previous answer for accident_time was exactly 0.0 seconds.
Your task is NOT to find a new accident time.
Your task is ONLY to decide whether the accident is truly visible at the very beginning of the video, or whether the accident start time could not be clearly found.

{_meta_block(metadata)}

Definitions:
- found_at_zero = true: the first clear accident really begins at or extremely near 0.0 seconds.
- found_at_zero = false: 0.0 was used because the accident start time was unclear, not visible, or not confidently detected.

Instructions:
1. Watch the beginning of the video carefully.
2. Return true only if the accident is already clearly happening at the start of the video.
3. If the accident is not clearly visible at 0.0 seconds, return false.
4. Do not guess a different accident time.
5. Do not output any location or accident type.

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
  "found_at_zero", "confidence"

Output format:
{{
  "found_at_zero": <true or false>,
  "confidence": <float from 0.0 to 1.0>
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


def validate_zero_check(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None

    if "found_at_zero" not in result:
        return None

    value = result["found_at_zero"]
    if isinstance(value, bool):
        found_at_zero = value
    elif isinstance(value, str):
        found_at_zero = value.strip().lower() in {"true", "yes", "1"}
    else:
        found_at_zero = bool(value)

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    return {"found_at_zero": found_at_zero, "confidence": confidence}


def move_inputs_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if hasattr(v, "to") else v
    return moved


def append_raw_log(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def is_exact_zero_time(accident_time: float) -> bool:
    return abs(float(accident_time)) <= ZERO_EPS


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

    fieldnames = [
        "path",
        "time_status",
        "accident_time",
        "zero_check_confidence",
        "reason",
        "raw_qwen_time",
        "raw_zero_check",
    ]

    saved = 0
    found_count = 0
    not_found_count = 0
    zero_checked_count = 0

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

            raw_time = call_qwen_for_media(
                model=model,
                processor=processor,
                media_type="video",
                media_path=abs_video_path,
                prompt=build_time_prompt(meta),
                rel_path=rel_path,
                stage="time_original_prompt",
            )

            if raw_time is None:
                print("  -> 못찾음: failed_to_get_valid_json")
                row = {
                    "path": rel_path,
                    "time_status": "못찾음",
                    "accident_time": "",
                    "zero_check_confidence": "",
                    "reason": "failed_to_get_valid_json",
                    "raw_qwen_time": "",
                    "raw_zero_check": "",
                }
                not_found_count += 1
            else:
                accident_time = validate_time_prediction(raw_time, meta)
                if accident_time is None:
                    print(f"  -> 못찾음: invalid_time_schema: {raw_time}")
                    row = {
                        "path": rel_path,
                        "time_status": "못찾음",
                        "accident_time": "",
                        "zero_check_confidence": "",
                        "reason": "invalid_time_schema",
                        "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                        "raw_zero_check": "",
                    }
                    not_found_count += 1
                elif not is_exact_zero_time(accident_time):
                    # 핵심: 0이 아니면 원래 Qwen 예측을 그대로 사용. 추가 판단 안 함.
                    print(f"  -> 찾음: accident_time={accident_time:.4f} (original qwen time, no extra check)")
                    row = {
                        "path": rel_path,
                        "time_status": "찾음",
                        "accident_time": accident_time,
                        "zero_check_confidence": "",
                        "reason": "nonzero_original_qwen_time",
                        "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                        "raw_zero_check": "",
                    }
                    found_count += 1
                else:
                    # Qwen이 정확히 0.0을 냈을 때만 2차 판단.
                    zero_checked_count += 1
                    print("  -> Qwen predicted 0.0, running zero-only check...")
                    raw_zero_check = call_qwen_for_media(
                        model=model,
                        processor=processor,
                        media_type="video",
                        media_path=abs_video_path,
                        prompt=build_zero_check_prompt(meta),
                        rel_path=rel_path,
                        stage="zero_check_only",
                    )
                    zero_check = validate_zero_check(raw_zero_check) if raw_zero_check is not None else None

                    if zero_check is not None and zero_check["found_at_zero"]:
                        print(
                            f"  -> 찾음: accident_time=0.0000 "
                            f"(zero check true, confidence={zero_check['confidence']:.3f})"
                        )
                        row = {
                            "path": rel_path,
                            "time_status": "찾음",
                            "accident_time": 0.0,
                            "zero_check_confidence": zero_check["confidence"],
                            "reason": "zero_check_found_at_zero_true",
                            "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                            "raw_zero_check": json.dumps(raw_zero_check, ensure_ascii=False),
                        }
                        found_count += 1
                    else:
                        reason = "zero_check_found_at_zero_false" if zero_check is not None else "zero_check_invalid_or_failed"
                        conf = zero_check["confidence"] if zero_check is not None else ""
                        print(f"  -> 못찾음: {reason}")
                        row = {
                            "path": rel_path,
                            "time_status": "못찾음",
                            "accident_time": "",
                            "zero_check_confidence": conf,
                            "reason": reason,
                            "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                            "raw_zero_check": "" if raw_zero_check is None else json.dumps(raw_zero_check, ensure_ascii=False),
                        }
                        not_found_count += 1

            writer.writerow(row)
            prediction_file.flush()
            os.fsync(prediction_file.fileno())
            saved += 1
            print(f"  -> CSV updated: {PREDICTION_PATH} ({saved} rows saved)")

    print("\nDone.")
    print(f"Saved predictions to: {PREDICTION_PATH}")
    print(f"Raw outputs saved to: {RAW_LOG_PATH}")
    print(f"Found: {found_count}")
    print(f"Not found: {not_found_count}")
    print(f"Zero-check cases: {zero_checked_count}")


if __name__ == "__main__":
    main()
