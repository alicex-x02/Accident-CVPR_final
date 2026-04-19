from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from pipeline.location_detection import estimate_location
from pipeline.qwen_utils import DEFAULT_MODEL_PATH, QwenVideoReasoner
from pipeline.time_detection import detect_candidate_times, verify_accident_time
from pipeline.type_classification import classify_accident_type


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = "/root/Desktop/workspace/woo/ACCIDENT@CVPR/data/raw/accident"
METADATA_PATH = os.path.join(DATASET_ROOT, "test_metadata.csv")
VIDEOS_DIR = os.path.join(DATASET_ROOT, "videos")
LOG_DIR = os.path.join(BASE_DIR, "log")
RESULT_DIR = os.path.join(BASE_DIR, "result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shardable traffic accident pipeline runner")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--metadata-csv", default=METADATA_PATH)
    parser.add_argument("--videos-dir", default=VIDEOS_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_run_paths(args: argparse.Namespace) -> Dict[str, str]:
    shard_tag = f"shard{args.shard_id:02d}_of_{args.num_shards:02d}"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return {
        "shard_tag": shard_tag,
        "output_csv": os.path.join(args.result_dir, f"submission_{shard_tag}.csv"),
        "raw_log": os.path.join(args.log_dir, f"raw_outputs_{shard_tag}.jsonl"),
        "run_log": os.path.join(args.log_dir, f"run_{shard_tag}_{timestamp}.log"),
        "temp_dir": os.path.join(args.log_dir, f"temp_{shard_tag}"),
    }


def setup_logging(run_log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(run_log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def write_predictions_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["video_id", "accident_time", "center_x", "center_y", "accident_type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: str, payload: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_existing_results(path: str) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "video_id" not in df.columns:
        return {}
    return {str(row["video_id"]): row.to_dict() for _, row in df.iterrows()}


def filter_shard_rows(df: pd.DataFrame, shard_id: int, num_shards: int, limit: int) -> pd.DataFrame:
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")

    shard_df = df.iloc[shard_id::num_shards].copy()
    shard_df.reset_index(drop=True, inplace=True)
    if limit > 0:
        shard_df = shard_df.iloc[:limit].copy()
        shard_df.reset_index(drop=True, inplace=True)
    return shard_df


def resolve_video_entry(row: pd.Series, dataset_root: str, videos_dir: str) -> Tuple[str, str]:
    if "video_id" in row.index and pd.notna(row["video_id"]):
        video_id = str(row["video_id"]).strip()
        return video_id, os.path.join(videos_dir, f"{video_id}.mp4")

    if "path" in row.index and pd.notna(row["path"]):
        rel_path = str(row["path"]).strip()
        video_id = os.path.splitext(os.path.basename(rel_path))[0]
        video_path = rel_path if os.path.isabs(rel_path) else os.path.join(dataset_root, rel_path)
        return video_id, video_path

    raise KeyError("Expected either 'video_id' or 'path' in metadata row")


def process_video(
    video_id: str,
    video_path: str,
    qwen: QwenVideoReasoner,
    temp_dir: str,
    raw_log_path: str,
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

    accident_time, scored_candidates = verify_accident_time(
        video_path=video_path,
        candidate_times=candidate_times,
        qwen=qwen,
        temp_dir=temp_dir,
    )
    append_jsonl(
        raw_log_path,
        {
            "video_id": video_id,
            "stage": "time_candidates",
            "candidate_times": candidate_times,
            "scored_candidates": scored_candidates,
        },
    )

    center_x, center_y = estimate_location(video_path=video_path, accident_time_sec=accident_time)
    accident_type = classify_accident_type(
        video_path=video_path,
        accident_time_sec=accident_time,
        qwen=qwen,
        temp_dir=temp_dir,
    )

    result = {
        "video_id": video_id,
        "accident_time": float(round(accident_time, 3)),
        "center_x": float(round(center_x, 6)),
        "center_y": float(round(center_y, 6)),
        "accident_type": accident_type,
    }
    append_jsonl(raw_log_path, {"video_id": video_id, "stage": "final_prediction", **result})
    return result


def main() -> None:
    args = parse_args()
    ensure_dirs(args.log_dir, args.result_dir)
    run_paths = get_run_paths(args)
    ensure_dirs(run_paths["temp_dir"])
    setup_logging(run_paths["run_log"])

    if os.path.exists(run_paths["raw_log"]):
        os.remove(run_paths["raw_log"])

    logging.info("Dataset root: %s", args.dataset_root)
    logging.info("Metadata CSV: %s", args.metadata_csv)
    logging.info("Videos dir: %s", args.videos_dir)
    logging.info("Model path: %s", args.model_path)
    logging.info("Shard: %d / %d", args.shard_id, args.num_shards)
    logging.info("Output CSV: %s", run_paths["output_csv"])
    logging.info("Raw log: %s", run_paths["raw_log"])
    logging.info("Temp dir: %s", run_paths["temp_dir"])

    metadata_df = pd.read_csv(args.metadata_csv)
    shard_df = filter_shard_rows(metadata_df, args.shard_id, args.num_shards, args.limit)
    logging.info("Rows assigned to this shard: %d", len(shard_df))

    existing_results = load_existing_results(run_paths["output_csv"]) if args.resume else {}
    predictions: List[Dict[str, object]] = list(existing_results.values())
    write_predictions_csv(run_paths["output_csv"], predictions)

    qwen = QwenVideoReasoner(model_path=args.model_path)

    for idx, row in shard_df.iterrows():
        video_id, video_path = resolve_video_entry(row, args.dataset_root, args.videos_dir)
        if video_id in existing_results:
            logging.info("[%d/%d] Skip existing: %s", idx + 1, len(shard_df), video_id)
            continue

        if not os.path.exists(video_path):
            logging.warning("[%d/%d] Missing video: %s", idx + 1, len(shard_df), video_path)
            result = {
                "video_id": video_id,
                "accident_time": 0.0,
                "center_x": 0.5,
                "center_y": 0.5,
                "accident_type": "other",
            }
        else:
            logging.info("[%d/%d] Processing %s", idx + 1, len(shard_df), video_id)
            result = process_video(
                video_id=video_id,
                video_path=video_path,
                qwen=qwen,
                temp_dir=run_paths["temp_dir"],
                raw_log_path=run_paths["raw_log"],
                sample_fps=args.sample_fps,
                top_k=args.top_k,
            )

        predictions.append(result)
        write_predictions_csv(run_paths["output_csv"], predictions)
        logging.info(
            "[%d/%d] Updated CSV | %s | t=%.3f xy=(%.6f, %.6f) type=%s",
            idx + 1,
            len(shard_df),
            result["video_id"],
            result["accident_time"],
            result["center_x"],
            result["center_y"],
            result["accident_type"],
        )

    logging.info("Finished shard run. Saved CSV: %s", run_paths["output_csv"])


if __name__ == "__main__":
    main()
