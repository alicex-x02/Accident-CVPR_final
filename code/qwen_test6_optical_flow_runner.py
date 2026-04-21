import argparse
import csv
import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, Iterator, Optional


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_SCRIPT = BASE_DIR / "qwen_test6 optical flow.py"
ACCIDENT_DIR = BASE_DIR / "accident"
RESULT_DIR = BASE_DIR.parent / "result"
METADATA_PATH = ACCIDENT_DIR / "test_metadata.csv"


def sanitize_part_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe or "part"


def count_metadata_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def load_original_module():
    spec = importlib.util.spec_from_file_location(
        "qwen_test6_optical_flow_original",
        ORIGINAL_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load original script: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_selected_read_metadata(module, start_row: int, end_row: Optional[int]):
    def selected_read_metadata(csv_path: str) -> Iterator[Dict[str, str]]:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                if idx < start_row:
                    continue
                if end_row is not None and idx > end_row:
                    break
                yield module.normalize_metadata(row)

    return selected_read_metadata


def run_worker(start_row: int, end_row: Optional[int], part_name: str, gpu_id: Optional[int]) -> int:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    module = load_original_module()
    safe_part = sanitize_part_name(part_name)
    module.PREDICTION_PATH = str(RESULT_DIR / f"qwen_test6_optical_flow_{safe_part}.csv")
    module.RAW_LOG_PATH = str(ACCIDENT_DIR / f"qwen_test6_optical_flow_{safe_part}.jsonl")
    module.read_metadata = make_selected_read_metadata(module, start_row, end_row)
    module.main()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a split qwen_test6 optical-flow worker")
    parser.add_argument("--start-row", type=int, required=True, help="1-based inclusive metadata row to start at")
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="1-based inclusive metadata row to end at; omit to run through the last row",
    )
    parser.add_argument("--part-name", required=True, help="Unique suffix for this shard's outputs")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id to expose via CUDA_VISIBLE_DEVICES")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not ORIGINAL_SCRIPT.exists():
        raise FileNotFoundError(f"Original script not found: {ORIGINAL_SCRIPT}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")
    if args.start_row < 1:
        raise ValueError("--start-row must be >= 1")
    if args.end_row is not None and args.end_row < args.start_row:
        raise ValueError("--end-row must be >= --start-row")
    return run_worker(args.start_row, args.end_row, args.part_name, args.gpu_id)


if __name__ == "__main__":
    raise SystemExit(main())
