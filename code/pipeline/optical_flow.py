from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def get_video_info(video_path: str) -> VideoInfo:
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
    return VideoInfo(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def moving_average(values: Sequence[float], window_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or window_size <= 1:
        return arr
    window_size = min(window_size, arr.size)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(arr, kernel, mode="same")


def _resize_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    if resize_width <= 0 or frame.shape[1] <= resize_width:
        return frame
    scale = resize_width / float(frame.shape[1])
    resize_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)


def compute_motion_curve(
    video_path: str,
    sample_fps: float = 5.0,
    resize_width: int = 320,
    start_time_sec: float | None = None,
    end_time_sec: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_step = max(int(round(info.fps / max(sample_fps, 1e-6))), 1)
    start_frame = 0 if start_time_sec is None else max(0, int(start_time_sec * info.fps))
    end_frame = info.frame_count if end_time_sec is None else min(info.frame_count, int(end_time_sec * info.fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    times: List[float] = []
    motion_values: List[float] = []
    current_frame_idx = start_frame
    prev_gray = None

    while current_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (current_frame_idx - start_frame) % frame_step != 0:
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
            times.append(current_frame_idx / info.fps)

        prev_gray = gray
        current_frame_idx += 1

    cap.release()
    return np.asarray(times, dtype=np.float32), np.asarray(motion_values, dtype=np.float32)


def select_top_k_peaks(
    times: Sequence[float],
    values: Sequence[float],
    top_k: int = 3,
    min_separation_sec: float = 1.5,
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
        fallback_idx = int(np.argmax(values_arr))
        selected.append(float(times_arr[fallback_idx]))

    return sorted(selected)

