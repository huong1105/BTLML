"""Create cutoff-safe TV1 training/validation/test features from the base panel."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from store_sales.features import prepare_feature_datasets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data/processed/base_panel.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/features")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--train-step-days", type=int, default=16, help="16 gives non-overlapping training horizons")
    args = parser.parse_args()
    if not args.panel.is_file():
        parser.error("Base panel missing; run scripts/prepare_data.py first")
    report = prepare_feature_datasets(args.panel, args.output_dir, args.report_dir, step_days=args.train_step_days,
                                      progress=lambda message: print(message, flush=True))
    for name, info in report["stages"].items():
        print(f"{name}: {info['train_rows']:,} training rows; {info['forecast_rows']:,} forecast rows; origin={info['origin']}", flush=True)
    print(f"Feature manifest: {args.report_dir / 'features_manifest.json'}")


if __name__ == "__main__":
    main()
