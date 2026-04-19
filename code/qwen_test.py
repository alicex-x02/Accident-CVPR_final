import csv
import json
import os
import re
import time
from threading import Thread
from typing import Dict, Any, Optional

import cv2
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCIDENT_DIR = os.path.join(BASE_DIR, "accident")
RESULT_DIR = os.path.join(os.path.dirname(BASE_DIR), "result")
METADATA_PATH = os.path.join(ACCIDENT_DIR, "test_metadata.csv")
PREDICTION_PATH = os.path.join(RESULT_DIR, "predictions.csv")
RAW_LOG_PATH = os.path.join(ACCIDENT_DIR, "raw_outputs.jsonl")

MODEL_NAME = "Qwen/Qwen3.5-9B"
VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9


def enhance_video(video_path: str) -> str:
    return video_path


def normalize_metadata(row: Dict[str, str]) -> Dict[str, str]:
    row = dict(row)
    if "scene_layout" not in row and "scene_layoutm" in row:
        row["scene_layout"] = row["scene_layoutm"]
    return row


def build_time_prompt(metadata: Dict[str, str]) -> str:
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")

    prompt = f"""
You are an expert traffic accident analyst looking at CCTV footage.

Your task is to detect the first clear traffic accident in the video and return ONLY the accident start time in seconds.

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


def build_location_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")

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


def build_type_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")

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
    """
    문자열 전체에서 첫 번째로 파싱 가능한 JSON object를 찾는다.
    reasoning이 앞뒤에 섞여 있어도 최대한 복구한다.
    """
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


def validate_location_prediction(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        center_x = float(result["center_x"])
        center_y = float(result["center_y"])
    except (KeyError, TypeError, ValueError):
        return None

    center_x = min(max(center_x, 0.0), 1.0)
    center_y = min(max(center_y, 0.0), 1.0)

    return {"center_x": center_x, "center_y": center_y}


def validate_type_prediction(result: Dict[str, Any]) -> Optional[str]:
    try:
        accident_type = str(result["type"]).strip()
    except (KeyError, TypeError, ValueError):
        return None

    if accident_type not in VALID_TYPES:
        return None

    return accident_type


def move_inputs_to_device(batch, device):
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


def extract_frame_at_time(
    video_path: str,
    accident_time: float,
    meta: Dict[str, str],
    fps_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    OpenCV 시킹이 코덱/키프레임에 따라 실패하는 경우가 있어,
    프레임 인덱스/밀리초 시킹/주변 프레임 fallback으로 최대한 복구한다.
    반환: {"frame": np.ndarray, "fps": float, "frame_index": int, "frame_path_hint": str}
    """

    try:
        import cv2  # noqa: F401
    except ImportError:
        return None

    import cv2

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

    # 1) 프레임 인덱스 시킹
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}

    # 2) 밀리초 시킹으로 재시도 (cap을 새로 열어야 성공률이 올라감)
    cap.release()
    cap = _open()
    if cap is None:
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, accident_time) * 1000.0)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}

    # 3) 주변 프레임 fallback (끝부분 시킹 실패 대비)
    for delta in range(1, 8):
        for candidate in (frame_index - delta, frame_index + delta):
            if candidate < 0 or candidate >= max(1, total_frames):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(candidate))
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return {"frame": frame, "fps": fps, "frame_index": int(candidate)}

    # 4) 마지막 프레임 시도
    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, frame = cap.read()
        if ret and frame is not None:
            cap.release()
            return {"frame": frame, "fps": fps, "frame_index": max(0, total_frames - 1)}

    cap.release()
    return None


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
                {
                    "type": "text",
                    "text": "Respond with JSON only. No reasoning. No explanation. /no_think"
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": media_type, "path": media_path},
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
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

    predictions = []
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

            # 1) 사고 시점 예측
            time_prompt = build_time_prompt(meta)
            raw_time = call_qwen_for_media(
                model=model,
                processor=processor,
                media_type="video",
                media_path=abs_video_path,
                prompt=time_prompt,
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

            # 2) 사고 시점 프레임 추출 (이미지) - 시킹 실패 대비 fallback 포함
            extracted = extract_frame_at_time(abs_video_path, accident_time, meta)
            if extracted is None:
                print("  -> Failed to extract frame at predicted accident_time (seek/read failed).")
                continue

            frame = extracted["frame"]
            frame_index = extracted["frame_index"]

            frame_dir = os.path.join(ACCIDENT_DIR, "frames")
            os.makedirs(frame_dir, exist_ok=True)
            frame_filename = os.path.splitext(os.path.basename(rel_path))[0]
            frame_path = os.path.join(frame_dir, f"{frame_filename}_t{accident_time:.3f}.jpg")

            cv2.imwrite(frame_path, frame)
            abs_frame_path = os.path.abspath(frame_path)

            # 3) 사고 위치 예측
            location_prompt = build_location_prompt(meta, accident_time)
            raw_loc = call_qwen_for_media(
                model=model,
                processor=processor,
                media_type="image",
                media_path=abs_frame_path,
                prompt=location_prompt,
                rel_path=rel_path,
                stage="location",
            )
            if raw_loc is None:
                print("  -> Failed to get valid JSON response for location")
                continue

            loc = validate_location_prediction(raw_loc)
            if loc is None:
                print(f"  -> Invalid location prediction schema: {raw_loc}")
                continue

            center_x = loc["center_x"]
            center_y = loc["center_y"]

            print(f"  -> predicted location: center_x={center_x:.4f}, center_y={center_y:.4f}")

            # 4) 사고 유형 예측
            type_prompt = build_type_prompt(meta, accident_time)
            raw_type = call_qwen_for_media(
                model=model,
                processor=processor,
                media_type="image",
                media_path=abs_frame_path,
                prompt=type_prompt,
                rel_path=rel_path,
                stage="type",
            )
            if raw_type is None:
                print("  -> Failed to get valid JSON response for type")
                continue

            accident_type = validate_type_prediction(raw_type)
            if accident_type is None:
                print(f"  -> Invalid type prediction schema: {raw_type}")
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
