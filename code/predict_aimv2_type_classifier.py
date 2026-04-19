from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.aimv2_type_classifier import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    AIMv2TypePredictor,
    pick_best_cuda_device,
)


DEFAULT_VIDEO_ROOT = "/root/Desktop/workspace/woo/ACCIDENT@CVPR/data/raw/accident"
DEFAULT_CHECKPOINT_PATH = os.path.join(DEFAULT_CHECKPOINT_DIR, "best.pt")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict accident type using a trained AIMv2 classifier")
    parser.add_argument("--input-csv", required=True, help="Submission-style CSV with path and accident_time columns")
    parser.add_argument("--output-csv", default=None, help="Where to save the CSV with type predictions")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--device", default="auto")
    return parser


def resolve_device(name: str):
    normalized = (name or "auto").strip().lower()
    if normalized in {"auto", "cuda"}:
        return pick_best_cuda_device()
    return normalized


def predict_submission_csv(
    input_csv: str,
    output_csv: str,
    checkpoint_path: str,
    video_root: str | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    predictor = AIMv2TypePredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=resolve_device(device) if device is not None else None,
        video_root=video_root,
    )
    dataframe = pd.read_csv(input_csv)

    predictions: list[str] = []
    for _, row in dataframe.iterrows():
        try:
            prediction = predictor.predict_submission_type(row)
        except Exception as exc:  # pragma: no cover - fall back keeps submission generation robust
            logging.warning("Falling back to single for row %s because prediction failed: %s", row.get("path", "<unknown>"), exc)
            prediction = "single"
        predictions.append(prediction)

    dataframe["type"] = predictions
    dataframe.to_csv(output_csv, index=False)
    return dataframe


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv) if args.output_csv else input_path.with_name(f"{input_path.stem}_aimv2.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Input CSV: %s", input_path)
    logging.info("Output CSV: %s", output_path)
    logging.info("Checkpoint: %s", args.checkpoint)
    logging.info("Video root: %s", args.video_root)

    predict_submission_csv(
        input_csv=str(input_path),
        output_csv=str(output_path),
        checkpoint_path=args.checkpoint,
        video_root=args.video_root,
        device=args.device,
    )
    print(f"Saved type predictions to {output_path}")


if __name__ == "__main__":
    main()
