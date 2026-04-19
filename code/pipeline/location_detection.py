from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from pipeline.optical_flow import get_video_info
from pipeline.qwen_utils import clamp


def estimate_location(
    video_path: str,
    accident_time_sec: float,
    half_window_sec: float = 0.5,
    sample_fps: float = 8.0,
    resize_width: int = 320,
    hotspot_quantile: float = 0.95,
) -> Tuple[float, float]:
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    start_time = max(0.0, accident_time_sec - half_window_sec)
    end_time = min(info.duration_sec, accident_time_sec + half_window_sec)
    start_frame = max(0, int(start_time * info.fps))
    end_frame = max(start_frame + 1, int(end_time * info.fps))
    frame_step = max(int(round(info.fps / max(sample_fps, 1e-6))), 1)

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

