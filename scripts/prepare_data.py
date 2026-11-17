"""Build the TV1 base panel and data quality reports without training a model."""
from pathlib import Path
import argparse
import json
import sys
import platform
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from store_sales.data import (  # noqa: E402
    RAW_FILES, build_base_panel, load_raw_data, quality_report, sha256_file, write_quality_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    source_hashes = {name: sha256_file(args.raw_dir / name) for name in RAW_FILES}
    tables = load_raw_data(args.raw_dir)
    report = quality_report(tables)
    panel = build_base_panel(tables)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output_dir / "base_panel.parquet", index=False, engine="pyarrow")
    write_quality_report(report, args.report_dir)
    hashes_after = {name: sha256_file(args.raw_dir / name) for name in RAW_FILES}
    if hashes_after != source_hashes:
        raise RuntimeError("Raw source files changed while preparation was running")
    manifest = {
        "raw_files": {name: {"sha256": source_hashes[name], "bytes": (args.raw_dir / name).stat().st_size} for name in RAW_FILES},
        "base_panel_rows": len(panel), "base_panel_columns": list(panel.columns),
        "base_panel_dtypes": {name: str(dtype) for name, dtype in panel.dtypes.items()},
        "base_panel_sha256": sha256_file(args.output_dir / "base_panel.parquet"),
        "raw_files_unchanged": True,
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "pyarrow")}},
        "notes": "No sales imputation, no target lags, no model training; raw files unchanged.",
    }
    (args.report_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(panel):,} observed rows: {args.output_dir / 'base_panel.parquet'}")
    print(f"Quality reports: {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
