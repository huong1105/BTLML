"""Training and evaluation helpers for global XGBoost and LightGBM models."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .metrics import evaluate_by_id, rmsle


def read_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("target_transform") not in {"log1p", "none"}:
        raise ValueError("target_transform must be 'log1p' or 'none'")
    return config


def feature_columns(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    excluded = set(config["metadata_columns"]) | {config["target_column"]}
    columns = [name for name in frame.columns if name not in excluded]
    if not columns:
        raise ValueError("No model features found")
    categorical = set(config["categorical_features"])
    missing = categorical.difference(columns)
    if missing:
        raise ValueError(f"Categorical columns missing from features: {sorted(missing)}")
    return columns


def prepare_features(
    frame: pd.DataFrame,
    columns: list[str],
    categorical_features: list[str],
    category_levels: dict[str, list],
) -> pd.DataFrame:
    """Use compact dtypes and stable train-derived category mappings."""
    x = frame[columns].copy()
    for name in categorical_features:
        levels = category_levels[name]
        x[name] = pd.Categorical(x[name], categories=levels, ordered=False)
    for name in x.columns:
        if name in categorical_features:
            continue
        if pd.api.types.is_bool_dtype(x[name].dtype):
            x[name] = x[name].astype("uint8")
        elif pd.api.types.is_integer_dtype(x[name].dtype):
            x[name] = pd.to_numeric(x[name], downcast="integer")
        elif pd.api.types.is_float_dtype(x[name].dtype):
            x[name] = x[name].astype("float32")
        else:
            raise ValueError(f"Unsupported non-numeric feature {name!r}: {x[name].dtype}")
    return x


def categories_from_train(frame: pd.DataFrame, categorical_features: list[str]) -> dict[str, list]:
    categories = {}
    for name in categorical_features:
        values = frame[name].dropna().unique().tolist()
        categories[name] = sorted(values, key=str)
        if not categories[name]:
            raise ValueError(f"Categorical feature {name!r} has no training values")
    return categories


def target_values(values, transform: str) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Training sales must be finite and nonnegative")
    return np.log1p(y) if transform == "log1p" else y


def inverse_target(values, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.expm1(values) if transform == "log1p" else values
    if not np.isfinite(result).all():
        raise ValueError("Model produced non-finite sales predictions")
    return np.maximum(result, 0.0)


def _xgboost_progress_callback(total_rounds: int, log_period: int):
    from xgboost.callback import TrainingCallback

    class ProgressCallback(TrainingCallback):
        def before_training(self, model):
            self.started = time.perf_counter()
            return model

        def after_iteration(self, model, epoch, evals_log):
            iteration = epoch + 1
            if iteration % log_period == 0 or iteration == total_rounds:
                elapsed = time.perf_counter() - self.started
                print(f"[xgboost] iteration {iteration:,}/{total_rounds:,}; elapsed={elapsed:.1f}s", flush=True)
            return False

    return ProgressCallback()


def _lightgbm_progress_callback(log_period: int):
    def callback(env):
        iteration = env.iteration + 1
        total_rounds = env.end_iteration
        if iteration % log_period == 0 or iteration == total_rounds:
            elapsed = time.perf_counter() - callback.started
            print(f"[lightgbm] iteration {iteration:,}/{total_rounds:,}; elapsed={elapsed:.1f}s", flush=True)

    callback.order = 30
    callback.before_iteration = False
    callback.started = time.perf_counter()
    return callback


def create_model(name: str, config: dict[str, Any], *, early_stopping: bool,
                 log_period: int, show_iteration_progress: bool):
    params = dict(config["models"][name])
    rounds = params.pop("early_stopping_rounds", 100)
    if name == "xgboost":
        from xgboost import XGBRegressor

        params.update({
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": config["random_state"],
            "verbosity": 0,
        })
        if early_stopping:
            params["early_stopping_rounds"] = rounds
        if show_iteration_progress:
            params["callbacks"] = [_xgboost_progress_callback(params["n_estimators"], log_period)]
        return XGBRegressor(**params)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        params.update({
            "objective": "regression",
            "metric": "rmse",
            "random_state": config["random_state"],
            "verbosity": -1,
        })
        return LGBMRegressor(**params)
    raise ValueError(f"Unknown model: {name}")


def fit_model(
    name: str,
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None = None,
    *,
    early_stopping: bool = True,
    log_period: int = 100,
):
    columns = feature_columns(train_frame, config)
    categoricals = list(config["categorical_features"])
    levels = categories_from_train(train_frame, categoricals)
    x_train = prepare_features(train_frame, columns, categoricals, levels)
    y_train = target_values(train_frame[config["target_column"]], config["target_transform"])
    model = create_model(
        name, config, early_stopping=early_stopping and validation_frame is not None,
        log_period=log_period, show_iteration_progress=validation_frame is None,
    )
    eval_set = None
    x_valid = None
    if validation_frame is not None:
        x_valid = prepare_features(validation_frame, columns, categoricals, levels)
        y_valid = target_values(validation_frame[config["target_column"]], config["target_transform"])
        eval_set = [(x_valid, y_valid)]
    if name == "lightgbm":
        from lightgbm import early_stopping as lgb_early_stopping, log_evaluation

        callbacks = [log_evaluation(period=log_period)] if eval_set is not None else [
            _lightgbm_progress_callback(log_period),
        ]
        if early_stopping and eval_set is not None:
            callbacks.append(lgb_early_stopping(
                int(config["models"][name]["early_stopping_rounds"]), verbose=False,
            ))
        model.fit(
            x_train, y_train, eval_set=eval_set,
            categorical_feature=categoricals, callbacks=callbacks,
        )
    else:
        model.fit(x_train, y_train, eval_set=eval_set,
                  verbose=log_period if eval_set is not None else False)
    return model, columns, levels


def predict_model(model, name: str, frame: pd.DataFrame, config: dict[str, Any],
                  columns: list[str], levels: dict[str, list]) -> np.ndarray:
    x = prepare_features(frame, columns, list(config["categorical_features"]), levels)
    return inverse_target(model.predict(x), config["target_transform"])


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return {
        "rmsle": rmsle(actual, predicted),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mae": float(np.mean(np.abs(actual - predicted))),
    }


def rolling_origin_folds(frame: pd.DataFrame, horizon_days: int, n_folds: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    origins = pd.to_datetime(frame["origin"]).drop_duplicates().sort_values().tolist()
    if n_folds <= 0:
        return []
    if len(origins) < n_folds + 1:
        raise ValueError(f"Need at least {n_folds + 1} origins; found {len(origins)}")
    folds = []
    for validation_origin in origins[-n_folds:]:
        validation_origin = pd.Timestamp(validation_origin)
        train_mask = pd.to_datetime(frame["origin"]) + pd.to_timedelta(horizon_days, unit="D") <= validation_origin
        valid_mask = pd.to_datetime(frame["origin"]).eq(validation_origin)
        fold_train = frame.loc[train_mask]
        fold_valid = frame.loc[valid_mask]
        if fold_train.empty or fold_valid.empty:
            raise ValueError(f"Empty rolling-origin fold at {validation_origin.date()}")
        folds.append((fold_train, fold_valid))
    return folds


def subgroup_metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for column in ("family", "horizon"):
        if column not in frame:
            continue
        groups = {}
        for value, indices in frame.groupby(column, observed=True, sort=True).groups.items():
            positions = np.asarray(list(indices), dtype=np.int64)
            groups[str(value)] = regression_metrics(frame.loc[positions, "sales"], predicted[positions])
        result[column] = groups
    return result


def aligned_predictions(features: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"id": features["id"].to_numpy(), "sales_pred": predicted})


def score_by_id(labels: pd.DataFrame, features: pd.DataFrame, predicted: np.ndarray) -> dict:
    scores = evaluate_by_id(labels[["id", "sales"]], aligned_predictions(features, predicted))
    scores.update(regression_metrics(
        labels.set_index("id").loc[features["id"], "sales"].to_numpy(), predicted,
    ))
    return scores
