import csv
import importlib.util
import json
import os
from typing import Any, Dict, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_SCRIPT_PATH = os.path.join(BASE_DIR, "qwen_time_only_threshold check.py")

if not os.path.exists(ORIGINAL_SCRIPT_PATH):
    raise FileNotFoundError(f"Original script not found: {ORIGINAL_SCRIPT_PATH}")

_spec = importlib.util.spec_from_file_location("_qwen_time_only_threshold_original", ORIGINAL_SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load original script: {ORIGINAL_SCRIPT_PATH}")

_orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig)


ACCIDENT_DIR = _orig.ACCIDENT_DIR
RESULT_DIR = _orig.RESULT_DIR
METADATA_PATH = _orig.METADATA_PATH
MODEL_NAME = _orig.MODEL_NAME
MAX_NEW_TOKENS = _orig.MAX_NEW_TOKENS
TEMPERATURE = _orig.TEMPERATURE
TOP_P = _orig.TOP_P
torch = _orig.torch
AutoProcessor = _orig.AutoProcessor
AutoModelForImageTextToText = _orig.AutoModelForImageTextToText

normalize_metadata = _orig.normalize_metadata
_meta_block = _orig._meta_block
build_time_prompt = _orig.build_time_prompt
strip_thinking_text = _orig.strip_thinking_text
try_parse_single_json = _orig.try_parse_single_json
extract_first_json_object = _orig.extract_first_json_object
validate_time_prediction = _orig.validate_time_prediction
get_metadata_duration = _orig.get_metadata_duration
move_inputs_to_device = _orig.move_inputs_to_device
append_raw_log = _orig.append_raw_log
call_qwen_for_media = _orig.call_qwen_for_media
read_metadata = _orig.read_metadata
split_metadata_rows = _orig.split_metadata_rows


SCRIPT_STEM = os.path.splitext(os.path.basename(__file__))[0]
PART_INDEX = int(os.environ.get("PART_INDEX", "0"))
PART_COUNT = max(1, int(os.environ.get("PART_COUNT", "1")))
RUN_SUFFIX = os.environ.get("RUN_SUFFIX", "")
if not RUN_SUFFIX and PART_COUNT > 1:
    RUN_SUFFIX = f"_part{PART_INDEX}"
OUTPUT_STEM = f"{SCRIPT_STEM}{RUN_SUFFIX}"
PREDICTION_PATH = os.path.join(RESULT_DIR, f"{OUTPUT_STEM}_predictions.csv")
RAW_LOG_PATH = os.path.join(ACCIDENT_DIR, f"{OUTPUT_STEM}_raw_outputs.jsonl")


_orig.RAW_LOG_PATH = RAW_LOG_PATH
_orig.PREDICTION_PATH = PREDICTION_PATH


LOW_THRESHOLD = 0.05
HIGH_THRESHOLD = 0.95


def _validate_boundary_check(result: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None

    if key not in result:
        return None

    value = result[key]
    if isinstance(value, bool):
        found_at_boundary = value
    elif isinstance(value, str):
        found_at_boundary = value.strip().lower() in {"true", "yes", "1"}
    else:
        found_at_boundary = bool(value)

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    return {key: found_at_boundary, "confidence": confidence}


def validate_low_check(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _validate_boundary_check(result, "found_at_5_percent")


def validate_high_check(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _validate_boundary_check(result, "found_at_95_percent")


def build_low_check_prompt(metadata: Dict[str, str]) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at CCTV footage.

The previous answer for accident_time was at or within the first 5% of the video's total duration.
Your task is NOT to find a new accident time.
Your task is ONLY to decide whether the accident is truly visible near the beginning of the video, or whether the accident start time could not be clearly found.

{_meta_block(metadata)}

Definitions:
- found_at_5_percent = true: the first clear accident really begins at or within the first 5% of the total video duration.
- found_at_5_percent = false: a very early timestamp was used because the accident start time was unclear, not visible, or not confidently detected.

Instructions:
1. Watch the beginning of the video carefully.
2. Return true only if the accident is already clearly happening at or within the first 5% of the video duration.
3. If the accident is not clearly visible near the start, return false.
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
  "found_at_5_percent", "confidence"

Output format:
{{
  "found_at_5_percent": <true or false>,
  "confidence": <float from 0.0 to 1.0>
}}
"""
    return prompt.strip()


def build_high_check_prompt(metadata: Dict[str, str]) -> str:
    prompt = f"""
You are an expert traffic accident analyst looking at CCTV footage.

The previous answer for accident_time was at or after 95% of the video's total duration.
Your task is NOT to find a new accident time.
Your task is ONLY to decide whether the accident is truly visible near the end of the video, or whether the accident start time could not be clearly found.

{_meta_block(metadata)}

Definitions:
- found_at_95_percent = true: the first clear accident really begins at or after 95% of the total video duration.
- found_at_95_percent = false: a very late timestamp was used because the accident start time was unclear, not visible, or not confidently detected.

Instructions:
1. Watch the end of the video carefully.
2. Return true only if the accident is already clearly happening at or after 95% of the video duration.
3. If the accident is not clearly visible near the end, return false.
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
  "found_at_95_percent", "confidence"

Output format:
{{
  "found_at_95_percent": <true or false>,
  "confidence": <float from 0.0 to 1.0>
}}
"""
    return prompt.strip()


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
    low_checked_count = 0
    high_checked_count = 0

    all_metadata = list(read_metadata(METADATA_PATH))
    metadata_rows, row_start, row_end = split_metadata_rows(all_metadata, PART_INDEX, PART_COUNT)
    print(
        f"Shard setup: part {PART_INDEX + 1}/{PART_COUNT}, "
        f"rows {row_start + 1}-{row_end} of {len(all_metadata)}"
    )

    with open(PREDICTION_PATH, "w", newline="", encoding="utf-8") as prediction_file:
        writer = csv.DictWriter(prediction_file, fieldnames=fieldnames)
        writer.writeheader()
        prediction_file.flush()
        os.fsync(prediction_file.fileno())

        for idx, meta in enumerate(metadata_rows, start=row_start + 1):
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

            duration = get_metadata_duration(meta)

            if raw_time is None:
                print("  -> not found: failed_to_get_valid_json")
                row = {
                    "path": rel_path,
                    "time_status": "not found",
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
                    print(f"  -> not found: invalid_time_schema: {raw_time}")
                    row = {
                        "path": rel_path,
                        "time_status": "not found",
                        "accident_time": "",
                        "zero_check_confidence": "",
                        "reason": "invalid_time_schema",
                        "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                        "raw_zero_check": "",
                    }
                    not_found_count += 1
                elif duration is not None and accident_time <= duration * LOW_THRESHOLD:
                    low_checked_count += 1
                    print("  -> Qwen predicted very early time, running 5% check...")
                    raw_low_check = call_qwen_for_media(
                        model=model,
                        processor=processor,
                        media_type="video",
                        media_path=abs_video_path,
                        prompt=build_low_check_prompt(meta),
                        rel_path=rel_path,
                        stage="five_percent_check_only",
                    )
                    low_check = validate_low_check(raw_low_check) if raw_low_check is not None else None

                    if low_check is not None and low_check["found_at_5_percent"]:
                        print(
                            f"  -> found: accident_time={accident_time:.4f} "
                            f"(5% check true, confidence={low_check['confidence']:.3f})"
                        )
                        row = {
                            "path": rel_path,
                            "time_status": "found",
                            "accident_time": accident_time,
                            "zero_check_confidence": low_check["confidence"],
                            "reason": "five_percent_check_found_at_5_percent_true",
                            "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                            "raw_zero_check": json.dumps(raw_low_check, ensure_ascii=False),
                        }
                        found_count += 1
                    else:
                        reason = "five_percent_check_found_at_5_percent_false" if low_check is not None else "five_percent_check_invalid_or_failed"
                        conf = low_check["confidence"] if low_check is not None else ""
                        print(f"  -> not found 5%: {reason}")
                        row = {
                            "path": rel_path,
                            "time_status": "not found 5%",
                            "accident_time": "",
                            "zero_check_confidence": conf,
                            "reason": reason,
                            "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                            "raw_zero_check": "" if raw_low_check is None else json.dumps(raw_low_check, ensure_ascii=False),
                        }
                        not_found_count += 1
                elif duration is not None and accident_time >= duration * HIGH_THRESHOLD:
                    high_checked_count += 1
                    print("  -> Qwen predicted late time, running 95% check...")
                    raw_high_check = call_qwen_for_media(
                        model=model,
                        processor=processor,
                        media_type="video",
                        media_path=abs_video_path,
                        prompt=build_high_check_prompt(meta),
                        rel_path=rel_path,
                        stage="ninety_five_percent_check_only",
                    )
                    high_check = validate_high_check(raw_high_check) if raw_high_check is not None else None

                    if high_check is not None and high_check["found_at_95_percent"]:
                        print(
                            f"  -> found: accident_time={accident_time:.4f} "
                            f"(95% check true, confidence={high_check['confidence']:.3f})"
                        )
                        row = {
                            "path": rel_path,
                            "time_status": "found",
                            "accident_time": accident_time,
                            "zero_check_confidence": "",
                            "reason": "ninety_five_percent_check_found_at_95_percent_true",
                            "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                            "raw_zero_check": "",
                        }
                        found_count += 1
                    else:
                        reason = "ninety_five_percent_check_found_at_95_percent_false" if high_check is not None else "ninety_five_percent_check_invalid_or_failed"
                        print(f"  -> not found 95%: {reason}")
                        row = {
                            "path": rel_path,
                            "time_status": "not found 95%",
                            "accident_time": "",
                            "zero_check_confidence": "",
                            "reason": reason,
                            "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                            "raw_zero_check": "",
                        }
                        not_found_count += 1
                else:
                    print(f"  -> found: accident_time={accident_time:.4f} (original qwen time, no extra check)")
                    row = {
                        "path": rel_path,
                        "time_status": "found",
                        "accident_time": accident_time,
                        "zero_check_confidence": "",
                        "reason": "nonboundary_original_qwen_time",
                        "raw_qwen_time": json.dumps(raw_time, ensure_ascii=False),
                        "raw_zero_check": "",
                    }
                    found_count += 1

            writer.writerow(row)
            prediction_file.flush()
            os.fsync(prediction_file.fileno())
            saved += 1
            print(f"  -> CSV updated: {PREDICTION_PATH} ({saved} rows saved)")

    print("\nDone.")
    print(f"Saved predictions to: {PREDICTION_PATH}")
    print(f"Raw outputs saved to: {RAW_LOG_PATH}")
    print(f"Shard: part {PART_INDEX + 1}/{PART_COUNT} (rows {row_start + 1}-{row_end})")
    print(f"Found: {found_count}")
    print(f"Not found: {not_found_count}")
    print(f"5% check cases: {low_checked_count}")
    print(f"95% check cases: {high_checked_count}")


if __name__ == "__main__":
    main()
