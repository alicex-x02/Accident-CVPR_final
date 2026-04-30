#!/usr/bin/env python3
"""Resume qwen_test17 into the test17 result/log subdirectories.

This wrapper keeps qwen_test17.py unchanged while redirecting the prediction
CSV output into result/test17 so resumed rows can be merged with the earlier
partial CSVs there.
"""

from __future__ import annotations

import argparse
import os

import qwen_test17 as base


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST17_RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "test17")
TEST17_LOG_DIR = os.path.join(PROJECT_ROOT, "log", "test17")


def configure_output_paths() -> None:
    """Redirect qwen_test17 outputs to the test17 subdirectories."""
    base.RESULT_DIR = TEST17_RESULT_DIR
    base.PREDICTION_PATH = os.path.join(TEST17_RESULT_DIR, f"{base.SCRIPT_STEM}.csv")
    base.PART_PREDICTION_TEMPLATE = os.path.join(
        TEST17_RESULT_DIR, f"{base.SCRIPT_STEM}_{{part}}.csv"
    )
    base.PART_RUN_LOG_TEMPLATE = os.path.join(
        TEST17_LOG_DIR, f"{base.SCRIPT_STEM}_{{part}}_gpu{{gpu}}.out"
    )
    os.makedirs(TEST17_RESULT_DIR, exist_ok=True)
    os.makedirs(TEST17_LOG_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume qwen_test17 with test17-scoped output files"
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based inclusive start row from test_metadata.csv",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="1-based inclusive end row from test_metadata.csv",
    )
    parser.add_argument(
        "--part-name",
        default=None,
        help="Part name used in output filenames, e.g. part0_resume",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU id to expose to this process via CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--launch-two-gpus",
        action="store_true",
        help="Launch two worker subprocesses on GPU 0 and GPU 1 using a half-and-half row split",
    )
    return parser.parse_args()


def run_cli() -> int:
    args = parse_args()
    configure_output_paths()

    if args.launch_two_gpus:
        if not os.path.exists(base.METADATA_PATH):
            raise FileNotFoundError(f"Metadata CSV not found: {base.METADATA_PATH}")
        total_rows = base.count_metadata_rows(base.METADATA_PATH)
        return base.launch_two_gpu_shards(total_rows)

    base.main(
        start_row=args.start_row,
        end_row=args.end_row,
        part_name=args.part_name,
        gpu_id=args.gpu_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
