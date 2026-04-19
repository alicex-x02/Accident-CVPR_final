from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from pipeline.location_detection import estimate_location
from pipeline.qwen_utils import DEFAULT_MODEL_PATH, QwenVideoReasoner
from pipeline.time_detection import detect_candidate_times, verify_accident_time
from pipeline.type_classification import classify_accident_type


DEFAULT_DATASET_ROOT = "/root/Desktop/workspace/woo/ACCIDENT@CVPR/data/raw/accident"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "log")
DEFAULT_RESULT_DIR = os.path.join(PROJECT_ROOT, "result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic accident video understanding pipeline")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--videos-dir", default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> Dict[str, str]:
    metadata_csv = args.metadata_csv or os.path.join(args.dataset_root, "test_metadata.csv")
    videos_dir = args.videos_dir or os.path.join(args.dataset_root, "videos")
    return {
        "metadata_csv": metadata_csv,
        "videos_dir": videos_dir,
    }


def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_pipeline_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def load_existing_results(output_csv: str) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(output_csv):
        return {}
    df = pd.read_csv(output_csv)
    if "video_id" not in df.columns:
        return {}
    return {str(row["video_id"]): row.to_dict() for _, row in df.iterrows()}


def process_video(
    video_id: str,
    video_path: str,
    qwen: QwenVideoReasoner,
    temp_dir: str,
    sample_fps: float,
    top_k: int,
) -> Dict[str, object]:
    candidate_times, _, _ = detect_candidate_times(
        video_path=video_path,
        sample_fps=sample_fps,
        top_k=top_k,
    )

    if not candidate_times:
        candidate_times = [0.0]

    accident_time, _ = verify_accident_time(
        video_path=video_path,
        candidate_times=candidate_times,
        qwen=qwen,
        temp_dir=temp_dir,
    )
    center_x, center_y = estimate_location(video_path=video_path, accident_time_sec=accident_time)
    accident_type = classify_accident_type(
        video_path=video_path,
        accident_time_sec=accident_time,
        qwen=qwen,
        temp_dir=temp_dir,
    )

    return {
        "video_id": video_id,
        "accident_time": float(round(accident_time, 3)),
        "center_x": float(round(center_x, 6)),
        "center_y": float(round(center_y, 6)),
        "accident_type": accident_type,
    }


def main() -> None:
    args = parse_args()
    paths = build_paths(args)
    output_csv = args.output_csv or os.path.join(DEFAULT_RESULT_DIR, "submission.csv")
    temp_dir = args.temp_dir or os.path.join(args.log_dir, "pipeline_temp")

    os.makedirs(DEFAULT_RESULT_DIR, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    log_path = setup_logging(args.log_dir)
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Dataset root: %s", args.dataset_root)
    logging.info("Videos dir: %s", paths["videos_dir"])
    logging.info("Metadata CSV: %s", paths["metadata_csv"])
    logging.info("Output CSV: %s", output_csv)
    logging.info("Temp dir: %s", temp_dir)
    logging.info("Log file: %s", log_path)

    metadata_df = pd.read_csv(paths["metadata_csv"])
    existing_results = load_existing_results(output_csv) if args.resume else {}

    qwen = QwenVideoReasoner(model_path=args.model_path)

    results: List[Dict[str, object]] = []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing videos"):
        video_id = str(row["video_id"])
        if video_id in existing_results:
            results.append(existing_results[video_id])
            continue

        video_path = os.path.join(paths["videos_dir"], f"{video_id}.mp4")
        if not os.path.exists(video_path):
            logging.warning("Missing video file: %s", video_path)
            results.append(
                {
                    "video_id": video_id,
                    "accident_time": 0.0,
                    "center_x": 0.5,
                    "center_y": 0.5,
                    "accident_type": "other",
                }
            )
            continue

        result = process_video(
            video_id=video_id,
            video_path=video_path,
            qwen=qwen,
            temp_dir=temp_dir,
            sample_fps=args.sample_fps,
            top_k=args.top_k,
        )
        results.append(result)
        logging.info(
            "Processed %s | accident_time=%.3f center=(%.6f, %.6f) type=%s",
            video_id,
            result["accident_time"],
            result["center_x"],
            result["center_y"],
            result["accident_type"],
        )

        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)
            logging.info("Checkpoint saved: %s (%d rows)", output_csv, len(results))

    pd.DataFrame(results).to_csv(output_csv, index=False)
    logging.info("Saved submission to %s", output_csv)
    print(f"Saved submission to {output_csv}")


if __name__ == "__main__":
    main()
