"""Cross-validate, fit and evaluate XGBoost / LightGBM on prepared feature stages."""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from store_sales.ml_models import (  # noqa: E402
    aligned_predictions,
    fit_model,
    predict_model,
    read_config,
    regression_metrics,
    rolling_origin_folds,
    score_by_id,
    subgroup_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=ROOT / "data/processed/features")
    parser.add_argument("--stage", choices=("tuning", "validation", "test"), default="tuning")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/models.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/models")
    parser.add_argument("--models", nargs="+", choices=("xgboost", "lightgbm"),
                        default=("xgboost", "lightgbm"))
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Number of final rolling-origin folds; use 0 to skip CV")
    parser.add_argument("--log-period", type=int, default=100,
                        help="Print training progress every N boosting rounds")
    parser.add_argument("--sample-submission", type=Path, default=ROOT / "data/raw/sample_submission.csv")
    return parser.parse_args()


def save_model_artifact(model, model_name: str, config: dict, category_levels: dict,
                        output_dir: Path, effective_rounds: int):
    folder = output_dir / model_name
    folder.mkdir(parents=True, exist_ok=True)
    if model_name == "xgboost":
        model.save_model(str(folder / "model.json"))
    else:
        # LightGBM's native writer can fail on Windows paths containing
        # non-ASCII characters. Serialize to text in memory, then let Python
        # open the Unicode path.
        model_text = model.booster_.model_to_string()
        (folder / "model.txt").write_text(model_text, encoding="utf-8")
    metadata = {
        "model": model_name,
        "target_transform": config["target_transform"],
        "categorical_features": config["categorical_features"],
        "category_levels": category_levels,
        "effective_n_estimators": effective_rounds,
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.cv_folds < 0:
        raise SystemExit("--cv-folds must be 0 or greater")
    if args.log_period < 1:
        raise SystemExit("--log-period must be at least 1")
    config = read_config(args.config)
    stage_dir = args.features_root / args.stage
    train_path = stage_dir / "train.parquet"
    forecast_path = stage_dir / "features.parquet"
    labels_path = stage_dir / "labels.parquet"
    for path in (train_path, forecast_path):
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")

    print(f"Reading {train_path}", flush=True)
    train = pd.read_parquet(train_path)
    forecast = pd.read_parquet(forecast_path)
    labels = pd.read_parquet(labels_path) if labels_path.is_file() else None
    if labels is not None and set(forecast["id"]) != set(labels["id"]):
        raise ValueError("Forecast feature IDs and label IDs do not match")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"stage": args.stage, "train_rows": len(train), "forecast_rows": len(forecast), "models": {}}

    for model_name in args.models:
        print(f"\n=== {model_name} ===", flush=True)
        model_config = copy.deepcopy(config)
        fold_scores = []
        best_rounds = []
        if args.cv_folds:
            folds = rolling_origin_folds(train, horizon_days=16, n_folds=args.cv_folds)
            for fold_number, (fold_train, fold_valid) in enumerate(folds, start=1):
                print(f"CV fold {fold_number}/{len(folds)}: {len(fold_train):,} train, "
                      f"{len(fold_valid):,} validation rows", flush=True)
                fit_started = time.perf_counter()
                model, columns, levels = fit_model(
                    model_name, model_config, fold_train, fold_valid, early_stopping=True,
                    log_period=args.log_period,
                )
                prediction = predict_model(model, model_name, fold_valid, model_config, columns, levels)
                score = regression_metrics(fold_valid["sales"], prediction)
                fold_scores.append(score)
                best_round = getattr(model, "best_iteration", None)
                if model_name == "xgboost" and best_round is not None:
                    # XGBoost reports a zero-based best_iteration; n_estimators is a count.
                    best_round = int(best_round) + 1
                elif best_round is None:
                    best_round = getattr(model, "best_iteration_", None)
                if best_round is None or best_round <= 0:
                    best_round = model_config["models"][model_name]["n_estimators"]
                best_rounds.append(int(best_round))
                elapsed = time.perf_counter() - fit_started
                print(f"  fold RMSLE={score['rmsle']:.6f}; best_iteration={best_round}; "
                      f"fit_time={elapsed:.1f}s", flush=True)
                del model, columns, levels, prediction, fold_train, fold_valid
                gc.collect()
            effective_rounds = max(1, int(np.median(best_rounds)))
            model_config["models"][model_name]["n_estimators"] = effective_rounds
        else:
            effective_rounds = int(model_config["models"][model_name]["n_estimators"])

        # The final stage labels are used only for scoring, never for early stopping.
        print(f"Fitting final model with {effective_rounds} boosting rounds", flush=True)
        fit_started = time.perf_counter()
        model, columns, levels = fit_model(
            model_name, model_config, train, validation_frame=None, early_stopping=False,
            log_period=args.log_period,
        )
        print(f"Final fit completed in {time.perf_counter() - fit_started:.1f}s", flush=True)
        prediction = predict_model(model, model_name, forecast, model_config, columns, levels)
        output = args.output_dir / f"{args.stage}_{model_name}_predictions.parquet"
        aligned_predictions(forecast, prediction).to_parquet(output, index=False)

        model_result = {
            "effective_n_estimators": effective_rounds,
            "cv_folds": fold_scores,
            "cv_mean_rmsle": float(np.mean([item["rmsle"] for item in fold_scores])) if fold_scores else None,
            "prediction_file": str(output),
        }
        if labels is not None:
            scores = score_by_id(labels, forecast, prediction)
            model_result["forecast_metrics"] = scores
            model_result["subgroup_metrics"] = subgroup_metrics(
                forecast.merge(labels[["id", "sales"]], on="id", how="left", validate="one_to_one"),
                prediction,
            )
            print("Forecast metrics: " + json.dumps(scores), flush=True)
            print(f"By-family/horizon metrics: {args.output_dir / f'{args.stage}_{model_name}_metrics.json'}", flush=True)
            (args.output_dir / f"{args.stage}_{model_name}_metrics.json").write_text(
                json.dumps(model_result["subgroup_metrics"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            if not args.sample_submission.is_file():
                raise SystemExit(f"Test prediction needs sample submission: {args.sample_submission}")
            sample = pd.read_csv(args.sample_submission)
            if list(sample.columns) != ["id", "sales"]:
                raise ValueError(f"Expected sample submission columns ['id', 'sales']; got {list(sample.columns)}")
            if sample["id"].isna().any() or sample["id"].duplicated().any():
                raise ValueError("Sample submission IDs must be present and unique")
            by_id = pd.DataFrame({"id": forecast["id"].to_numpy(), "sales": prediction})
            submission = sample[["id"]].merge(by_id, on="id", how="left", sort=False, validate="one_to_one")
            if submission["sales"].isna().any() or len(submission) != len(sample):
                raise ValueError("Predictions do not cover every sample submission ID")
            submission = submission[sample.columns]
            submission_path = args.output_dir / f"submission_{model_name}.csv"
            submission.to_csv(submission_path, index=False)
            model_result["submission_file"] = str(submission_path)
            print(f"Submission saved: {submission_path}", flush=True)

        save_model_artifact(model, model_name, model_config, levels, args.output_dir, effective_rounds)
        summary["models"][model_name] = model_result
        del model, columns, levels, prediction
        gc.collect()

    summary_path = args.output_dir / f"{args.stage}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Run summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
