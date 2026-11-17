"""Create measured TV1 EDA summaries, six figures and a Vietnamese report."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from store_sales.eda import run_eda  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    result = run_eda(args.raw_dir, args.figure_dir, args.report_dir)
    print(f"EDA complete: {result['train_rows']:,} train rows; 6 figures.")
    print(f"Read: {args.report_dir / 'eda_report.md'}")


if __name__ == "__main__":
    main()
