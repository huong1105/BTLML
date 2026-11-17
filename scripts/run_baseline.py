"""Run the fixed-origin same-weekday mean baseline on a validation holdout."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from store_sales.baseline import run_baseline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "baseline")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--validation-start", default=None, help="YYYY-MM-DD; default: last 16 calendar days of train")
    parser.add_argument("--window-days", type=int, default=28)
    parser.add_argument("--horizon-days", type=int, default=16)
    args = parser.parse_args()
    result = run_baseline(args.raw_dir, args.output_dir, args.report_dir,
                          args.validation_start, args.window_days, args.horizon_days)
    print(f"Baseline RMSLE: {result['rmsle']:.6f}")
    print(f"Validation: {result['validation_start']} -> {result['validation_end']} ({result['n_rows']:,} rows)")
    print(f"Reports: {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
