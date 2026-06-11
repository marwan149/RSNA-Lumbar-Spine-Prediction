import argparse
from pathlib import Path

from .core import RSNAConfig
from .pipeline import run_submission_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RSNA lumbar spine inference and export a submission CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the RSNA dataset directory containing test_series_descriptions.csv and sample_submission.csv.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="One or more model checkpoint paths to use for ensemble inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Path where the submission CSV will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers. Defaults to the system CPU count.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RSNAConfig.from_data_dir(
        data_dir=args.data_dir,
        output_path=args.output,
        checkpoint_paths=args.checkpoints,
        batch_size=args.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else None,
        use_amp=not args.no_amp,
    )

    run_submission_prediction(config)
