import argparse
import csv
import json
import math
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
SCRIPT_STEM = os.path.splitext(os.path.basename(__file__))[0]
PREDICTION_PATH = os.path.join(RESULT_DIR, f"{SCRIPT_STEM}.csv")
RAW_LOG_PATH = os.path.join(ACCIDENT_DIR, f"{SCRIPT_STEM}_raw_outputs.jsonl")
PART_PREDICTION_TEMPLATE = os.path.join(RESULT_DIR, f"{SCRIPT_STEM}_{{part}}.csv")
PART_RAW_LOG_TEMPLATE = os.path.join(ACCIDENT_DIR, f"{SCRIPT_STEM}_{{part}}_raw_outputs.jsonl")
PART_RUN_LOG_TEMPLATE = os.path.join(os.path.dirname(BASE_DIR), "log", f"{SCRIPT_STEM}_{{part}}_gpu{{gpu}}.out")

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
# - Use OF rule when Qwen collapses to the first 5% of the clip or predicts beyond 95% of the clip duration.
HYBRID_QWEN_FRONT_RATIO = 0.05
HYBRID_QWEN_LATE_RATIO = 0.95

LOCATION_FRAME_OFFSETS = (-0.20, 0.0, 0.20)
LOCATION_CROP_SCALES = (0.40, 0.26)
# Accident type clips are centered on accident_time, so this is applied to both
# sides of the center time. 2.0 means 2 seconds before and 2 seconds after.
TYPE_WINDOW_SEC = 2.0
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


def validate_candidate_selection(result: Dict[str, Any], num_candidates: int) -> Optional[int]:
    try:
        idx = int(result["selected_candidate_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if 1 <= idx <= num_candidates:
        return idx
    return None


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


def detect_flow_time_candidates(video_path: str, meta: Dict[str, str]) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {"candidate_rows": []}
    duration = _safe_duration(meta)
    try:
        times, motion_values = compute_motion_curve(video_path=video_path, sample_fps=TIME_FLOW_SAMPLE_FPS)
        diagnostics["num_motion_points"] = int(motion_values.size)
        if motion_values.size == 0:
            return diagnostics

        smoothed_motion = moving_average(motion_values, window_size=TIME_FLOW_SMOOTH_WINDOW).astype(np.float32, copy=False)
        positive_delta = np.zeros_like(smoothed_motion, dtype=np.float32)
        if smoothed_motion.size > 1:
            positive_delta[1:] = np.maximum(np.diff(smoothed_motion), 0.0)
        score = smoothed_motion + (TIME_FLOW_DELTA_WEIGHT * positive_delta)

        score_mean = float(np.mean(score))
        score_std = float(np.std(score))
        diagnostics["motion_stats"] = {
            "min": float(np.min(motion_values)),
            "max": float(np.max(motion_values)),
            "mean": float(np.mean(motion_values)),
            "std": float(np.std(motion_values)),
        }
        diagnostics["score_stats"] = {
            "min": float(np.min(score)),
            "max": float(np.max(score)),
            "mean": score_mean,
            "std": score_std,
        }

        candidate_times = select_top_k_peaks(
            times=times,
            values=score,
            top_k=TIME_FLOW_TOP_K,
            min_separation_sec=TIME_FLOW_MIN_SEPARATION_SEC,
        )

        candidate_rows: List[Dict[str, Any]] = []
        for candidate_time in candidate_times:
            idx = int(np.argmin(np.abs(times - candidate_time)))
            candidate_score = float(score[idx])
            z_score = (candidate_score - score_mean) / (score_std + 1e-6)
            reject_reason = None
            candidate_time = float(candidate_time)
            is_late = False
            late_thresholds = []
            if TIME_FLOW_LATE_ABSOLUTE_SEC is not None:
                late_thresholds.append(float(TIME_FLOW_LATE_ABSOLUTE_SEC))
            if duration is not None:
                late_thresholds.append(float(duration) * float(TIME_FLOW_LATE_FRACTION))
            if late_thresholds and candidate_time >= min(late_thresholds):
                is_late = True

            if candidate_time <= TIME_FLOW_MIN_VALID_SEC:
                reject_reason = f"too_close_to_start<={TIME_FLOW_MIN_VALID_SEC}"
            elif duration is not None and candidate_time >= max(0.0, duration - TIME_FLOW_END_MARGIN_SEC):
                reject_reason = f"too_close_to_end(duration-{TIME_FLOW_END_MARGIN_SEC})"
            elif float(z_score) < TIME_FLOW_MIN_SCORE_Z:
                reject_reason = f"low_score_z<{TIME_FLOW_MIN_SCORE_Z}"
            candidate_rows.append(
                {
                    "candidate_time": candidate_time,
                    "score": candidate_score,
                    "score_z": float(z_score),
                    "smoothed_motion": float(smoothed_motion[idx]),
                    "positive_delta": float(positive_delta[idx]),
                    "motion_time_index": int(idx),
                    "is_late": bool(is_late),
                    "keep": reject_reason is None,
                    "reject_reason": reject_reason,
                }
            )

        candidate_rows.sort(key=lambda item: (-float(item["score"]), float(item["candidate_time"])))
        for rank, row in enumerate(candidate_rows, start=1):
            row["rank"] = int(rank)
        diagnostics["candidate_rows"] = candidate_rows
        return diagnostics
    except Exception as exc:
        diagnostics["error"] = str(exc)
        return diagnostics


def print_flow_candidates(candidate_rows: Sequence[Dict[str, Any]]) -> None:
    print("  -> optical-flow candidates:", flush=True)
    if not candidate_rows:
        print("     none", flush=True)
        return
    for row in candidate_rows:
        status = "KEEP" if row.get("keep") else f"REJECT({row.get('reject_reason')})"
        print(
            "     "
            f"rank={row.get('rank', '?')} "
            f"t={_fmt_float(row.get('candidate_time'), 3)}s "
            f"score={_fmt_float(row.get('score'), 6)} "
            f"z={_fmt_float(row.get('score_z'), 3)} "
            f"motion={_fmt_float(row.get('smoothed_motion'), 6)} "
            f"delta={_fmt_float(row.get('positive_delta'), 6)} "
            f"idx={row.get('motion_time_index', '?')} "
            f"late={row.get('is_late', False)} "
            f"{status}",
            flush=True,
        )


def write_labeled_candidate_sequence_video(
    video_path: str,
    candidate_rows: Sequence[Dict[str, Any]],
    output_path: str,
    window_sec: float,
    clip_fps: float,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if src_fps <= 0:
        src_fps = clip_fps if clip_fps > 0 else 6.0
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, clip_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    duration = frame_count / src_fps
    step = max(1, int(round(src_fps / max(clip_fps, 1e-6))))
    title_frames = max(1, int(round(0.35 * clip_fps)))

    for display_idx, row in enumerate(candidate_rows, start=1):
        candidate_time = float(row["candidate_time"])
        title = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(title, f"CANDIDATE {display_idx}", (max(20, width // 12), height // 2), cv2.FONT_HERSHEY_SIMPLEX, max(1.0, width / 800.0), (255, 255, 255), 2, cv2.LINE_AA)
        for _ in range(title_frames):
            writer.write(title)

        start_time = max(0.0, candidate_time - window_sec)
        end_time = min(duration, candidate_time + window_sec)
        start_frame = max(0, int(round(start_time * src_fps)))
        end_frame = min(frame_count, max(start_frame + 1, int(round(end_time * src_fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        while idx < end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if (idx - start_frame) % step == 0:
                frame = frame.copy()
                cv2.rectangle(frame, (0, 0), (width, max(42, height // 12)), (0, 0, 0), -1)
                cv2.putText(frame, f"CANDIDATE {display_idx}", (12, max(30, height // 16)), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, width / 1000.0), (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame)
            idx += 1

    writer.release()
    cap.release()


def save_temp_candidate_sequence(video_path: str, candidate_rows: Sequence[Dict[str, Any]], prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".mp4", dir=ACCIDENT_DIR)
    os.close(fd)
    write_labeled_candidate_sequence_video(
        video_path=video_path,
        candidate_rows=candidate_rows,
        output_path=path,
        window_sec=TIME_CANDIDATE_CLIP_WINDOW_SEC,
        clip_fps=TIME_CANDIDATE_CLIP_FPS,
    )
    return path


def select_flow_candidate_by_rule(candidate_rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Select an OF candidate without asking Qwen.

    Rule:
    1. Use candidates that survived hard filters: start/end/z-score.
    2. If at least one non-late candidate exists, prefer non-late candidates.
    3. Among remaining candidates, choose highest OF rank / score.
    4. If all candidates were rejected, return None so caller can fallback to Qwen.
    """
    kept = [row for row in candidate_rows if row.get("keep")]
    non_late = [row for row in kept if not row.get("is_late")]
    selection_pool = non_late if non_late else kept

    selection_info = {
        "num_candidates": len(candidate_rows),
        "num_kept": len(kept),
        "num_non_late_kept": len(non_late),
        "rule": "hard_filter_start_end_z_then_prefer_non_late_then_best_rank",
        "used_late_pool": bool(kept and not non_late),
    }

    if not selection_pool:
        return None, selection_info

    selected = sorted(
        selection_pool,
        key=lambda row: (
            int(row.get("rank", 10**9)),
            -float(row.get("score", 0.0)),
            float(row.get("candidate_time", float("inf"))),
        ),
    )[0]
    return selected, selection_info


def predict_time_with_hybrid_qwen_of(
    model,
    processor,
    abs_video_path: str,
    rel_path: str,
    meta: Dict[str, str],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Hybrid time selector.

    Strategy:
    1. Ask Qwen for the full-video accident_time.
    2. If Qwen is normal, use Qwen time.
    3. If Qwen collapses to the first 5% of the clip or jumps later than 95% of the clip duration,
       run OF rule selector.
    4. If OF produces a valid candidate, use OF time; otherwise fallback to Qwen time.

    Location/type code remains unchanged and receives the selected accident_time.
    """
    diagnostics: Dict[str, Any] = {
        "selector_mode": "hybrid_qwen_then_of_rule_on_qwen_outlier",
        "qwen_time": None,
        "qwen_raw": None,
        "qwen_suspicious": False,
        "qwen_suspicious_reason": None,
        "flow_candidates": [],
        "kept_candidates": [],
        "selected_candidate": None,
        "selected_time": None,
        "final_source": None,
        "of_attempted": False,
        "of_failed_reason": None,
    }

    print("  -> hybrid time selector: running full-video Qwen first", flush=True)
    raw_qwen_time = call_qwen_for_media(
        model=model,
        processor=processor,
        media_type="video",
        media_path=abs_video_path,
        prompt=build_time_prompt(meta),
        rel_path=rel_path,
        stage="time_qwen_initial_hybrid",
    )
    diagnostics["qwen_raw"] = raw_qwen_time
    qwen_time = validate_time_prediction(raw_qwen_time, meta) if raw_qwen_time is not None else None
    diagnostics["qwen_time"] = qwen_time
    print(f"  -> initial Qwen time: {_fmt_float(qwen_time, 4)}s", flush=True)

    duration = _safe_duration(meta)
    front_threshold = duration * HYBRID_QWEN_FRONT_RATIO if duration is not None else None
    late_threshold = duration * HYBRID_QWEN_LATE_RATIO if duration is not None else None

    suspicious_reason = None
    if qwen_time is None:
        suspicious_reason = "qwen_time_none"
    elif front_threshold is not None and float(qwen_time) <= front_threshold:
        suspicious_reason = f"qwen_time<={HYBRID_QWEN_FRONT_RATIO:.2f}*duration"
    elif late_threshold is not None and float(qwen_time) > late_threshold:
        suspicious_reason = f"qwen_time>{HYBRID_QWEN_LATE_RATIO:.2f}*duration"

    diagnostics["qwen_suspicious"] = suspicious_reason is not None
    diagnostics["qwen_suspicious_reason"] = suspicious_reason

    if suspicious_reason is None:
        diagnostics["selected_time"] = qwen_time
        diagnostics["final_source"] = "qwen"
        print(
            "  -> Qwen time accepted by hybrid rule; "
            f"final_time={_fmt_float(qwen_time, 4)}s",
            flush=True,
        )
        return qwen_time, diagnostics

    print(
        "  -> Qwen time is suspicious; running OF rule selector "
        f"(reason={suspicious_reason})",
        flush=True,
    )
    diagnostics["of_attempted"] = True

    flow_diag = detect_flow_time_candidates(abs_video_path, meta)
    diagnostics["flow_diagnostics"] = flow_diag
    candidate_rows = list(flow_diag.get("candidate_rows", []))
    diagnostics["flow_candidates"] = candidate_rows

    if flow_diag.get("error"):
        diagnostics["flow_error"] = flow_diag["error"]
        print(f"  -> optical-flow error: {flow_diag['error']}", flush=True)
    print(
        "  -> optical-flow stats: "
        f"points={flow_diag.get('num_motion_points', 0)} "
        f"score_max={_fmt_float(flow_diag.get('score_stats', {}).get('max'), 6)} "
        f"score_mean={_fmt_float(flow_diag.get('score_stats', {}).get('mean'), 6)} "
        f"score_std={_fmt_float(flow_diag.get('score_stats', {}).get('std'), 6)}",
        flush=True,
    )
    print_flow_candidates(candidate_rows)

    kept = [row for row in candidate_rows if row.get("keep")]
    diagnostics["kept_candidates"] = kept

    print("  -> rule-based OF selector: Qwen selector is disabled", flush=True)
    selected_row, selection_info = select_flow_candidate_by_rule(candidate_rows)
    diagnostics["rule_selection_info"] = selection_info

    if selected_row is not None:
        selected_time = float(selected_row["candidate_time"])
        diagnostics["selected_candidate"] = dict(selected_row)
        diagnostics["selected_time"] = selected_time
        diagnostics["final_source"] = "optical_flow_rule"
        print(
            "  -> selected OF candidate by hybrid rule: "
            f"rank={selected_row.get('rank')} "
            f"time={selected_time:.4f}s "
            f"score={_fmt_float(selected_row.get('score'), 6)} "
            f"z={_fmt_float(selected_row.get('score_z'), 3)} "
            f"late={selected_row.get('is_late')} "
            f"rule={selection_info.get('rule')}",
            flush=True,
        )
        print(
            f"  -> final hybrid time: {selected_time:.4f}s "
            f"(source=optical_flow_rule, qwen_time={_fmt_float(qwen_time, 4)}s)",
            flush=True,
        )
        return selected_time, diagnostics

    diagnostics["of_failed_reason"] = "no_usable_of_candidate_after_rule_filters"
    diagnostics["selected_time"] = qwen_time
    diagnostics["final_source"] = "qwen_fallback_after_of_failure"
    print(
        "  -> no usable OF candidates after rule filters; falling back to suspicious Qwen time "
        f"{_fmt_float(qwen_time, 4)}s",
        flush=True,
    )
    return qwen_time, diagnostics

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


def main(
    start_row: int = 1,
    end_row: Optional[int] = None,
    part_name: Optional[str] = None,
    gpu_id: Optional[int] = None,
):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")

    os.makedirs(ACCIDENT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    prediction_path, raw_log_path = resolve_output_paths(part_name)

    if os.path.exists(raw_log_path):
        os.remove(raw_log_path)
    if os.path.exists(prediction_path):
        os.remove(prediction_path)

    if end_row is not None and end_row < start_row:
        raise ValueError(f"end_row must be >= start_row, got {start_row}..{end_row}")

    run_label = part_name or "full"

    print(f"Loading model: {MODEL_NAME} [{run_label}]")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Processing rows: {start_row}..{end_row if end_row is not None else 'end'}")
    print(f"CSV output: {prediction_path}")
    print(f"Raw log: {raw_log_path}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

    predictions: List[Dict[str, Any]] = []
    fieldnames = ["path", "accident_time", "center_x", "center_y", "type"]

    with open(prediction_path, "w", newline="", encoding="utf-8") as prediction_file:
        writer = csv.DictWriter(prediction_file, fieldnames=fieldnames)
        writer.writeheader()
        prediction_file.flush()
        os.fsync(prediction_file.fileno())

        for idx, meta in enumerate(read_metadata(METADATA_PATH), start=1):
            if idx < start_row:
                continue
            if end_row is not None and idx > end_row:
                break

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

            accident_time, time_diag = predict_time_with_hybrid_qwen_of(
                model=model,
                processor=processor,
                abs_video_path=abs_video_path,
                rel_path=rel_path,
                meta=meta,
            )
            append_raw_log(raw_log_path, {"path": rel_path, "stage": "time_hybrid_summary", **time_diag})
            if accident_time is None:
                print("  -> Failed to resolve accident_time from hybrid Qwen/OF selector")
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
            append_raw_log(raw_log_path, {"path": rel_path, "stage": "location_summary", **location_diag})
            if location is None:
                print("  -> Failed to get valid location prediction")
                continue
            center_x, center_y = location["center_x"], location["center_y"]
            print(f"  -> predicted location: center_x={center_x:.4f}, center_y={center_y:.4f}")
            if persistent_frame_path:
                print(f"  -> saved representative frame: {persistent_frame_path}")

            accident_type, type_diag = predict_type_with_clip_two_stage(
                model=model,
                processor=processor,
                abs_video_path=abs_video_path,
                rel_path=rel_path,
                meta=meta,
                accident_time=accident_time,
            )
            append_raw_log(raw_log_path, {"path": rel_path, "stage": "type_summary", **type_diag})
            if accident_type is None:
                print("  -> Failed to get valid JSON response for type")
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
            print(f"  -> CSV updated: {prediction_path} ({len(predictions)} rows saved)")

    if predictions:
        print(f"\nSaved predictions incrementally to: {prediction_path}")
    else:
        print("\nNo predictions generated.")
    print(f"Raw outputs saved to: {raw_log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the qwen_test10 hybrid accident pipeline")
    parser.add_argument("--start-row", type=int, default=1, help="1-based inclusive start row from test_metadata.csv")
    parser.add_argument("--end-row", type=int, default=None, help="1-based inclusive end row from test_metadata.csv")
    parser.add_argument("--part-name", default=None, help="Part name used in output filenames, e.g. part0")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id to expose to this process via CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--launch-two-gpus",
        action="store_true",
        help="Launch two worker subprocesses on GPU 0 and GPU 1 using a half-and-half row split",
    )
    return parser.parse_args()


def run_cli() -> int:
    args = parse_args()

    if args.launch_two_gpus:
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")
        total_rows = count_metadata_rows(METADATA_PATH)
        return launch_two_gpu_shards(total_rows)

    main(
        start_row=args.start_row,
        end_row=args.end_row,
        part_name=args.part_name,
        gpu_id=args.gpu_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
