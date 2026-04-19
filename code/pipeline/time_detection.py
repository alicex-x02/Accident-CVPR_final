from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from pipeline.optical_flow import compute_motion_curve, moving_average, select_top_k_peaks
from pipeline.qwen_utils import QwenVideoReasoner, clamp


def detect_candidate_times(
    video_path: str,
    sample_fps: float = 5.0,
    smooth_window: int = 5,
    top_k: int = 3,
    min_separation_sec: float = 1.5,
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


def _build_verification_prompt(candidate_time: float) -> str:
    return f"""
You are analyzing a short traffic video clip centered near {candidate_time:.2f} seconds.

Question:
Does this clip contain the moment when the accident first happens?

Instructions:
- Focus on the first actual collision or first unavoidable impact moment.
- Return higher confidence only if the accident onset is clearly visible in this clip.
- If the clip shows only pre-accident motion or only aftermath, confidence should be low.

Return JSON only:
{{
  "contains_accident": true,
  "confidence": 0.0
}}
""".strip()


def verify_accident_time(
    video_path: str,
    candidate_times: List[float],
    qwen: QwenVideoReasoner,
    temp_dir: str,
) -> Tuple[float, List[Dict[str, float]]]:
    if not candidate_times:
        raise ValueError("candidate_times is empty")

    scored_candidates: List[Dict[str, float]] = []

    for candidate_time in candidate_times:
        prompt = _build_verification_prompt(candidate_time)
        parsed = qwen.infer_json_from_temp_clip(
            video_path=video_path,
            center_time_sec=candidate_time,
            prompt=prompt,
            temp_dir=temp_dir,
            window_sec=1.0,
            clip_fps=8.0,
            clip_prefix="verify",
        )
        contains_accident = bool(parsed.get("contains_accident", False))
        confidence = clamp(float(parsed.get("confidence", 0.0)), 0.0, 1.0)
        scored_candidates.append(
            {
                "candidate_time": float(candidate_time),
                "contains_accident": float(1.0 if contains_accident else 0.0),
                "confidence": confidence,
            }
        )

    scored_candidates.sort(
        key=lambda item: (item["contains_accident"], item["confidence"]),
        reverse=True,
    )
    best_time = float(scored_candidates[0]["candidate_time"])
    return best_time, scored_candidates

