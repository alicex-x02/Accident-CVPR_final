from __future__ import annotations

from typing import List

from pipeline.qwen_utils import QwenVideoReasoner


VALID_TYPES: List[str] = ["rear-end", "side-swipe", "t-bone", "head-on", "other"]


def _build_type_prompt() -> str:
    return """
Identify the traffic accident type shown in this clip.
Choose the single best label among:
- rear-end
- side-swipe
- t-bone
- head-on
- other

Return JSON only:
{
  "accident_type": "rear-end",
  "confidence": 0.0
}
""".strip()


def normalize_accident_type(label: str) -> str:
    normalized = (label or "").strip().lower()
    aliases = {
        "rear end": "rear-end",
        "rear-end": "rear-end",
        "sideswipe": "side-swipe",
        "side swipe": "side-swipe",
        "side-swipe": "side-swipe",
        "t bone": "t-bone",
        "t-bone": "t-bone",
        "head on": "head-on",
        "head-on": "head-on",
        "other": "other",
    }
    return aliases.get(normalized, "other")


def classify_accident_type(
    video_path: str,
    accident_time_sec: float,
    qwen: QwenVideoReasoner,
    temp_dir: str,
) -> str:
    parsed = qwen.infer_json_from_temp_clip(
        video_path=video_path,
        center_time_sec=accident_time_sec,
        prompt=_build_type_prompt(),
        temp_dir=temp_dir,
        window_sec=1.0,
        clip_fps=8.0,
        clip_prefix="type",
    )
    return normalize_accident_type(str(parsed.get("accident_type", "other")))

