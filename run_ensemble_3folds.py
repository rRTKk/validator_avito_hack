from pathlib import Path
import gc
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from folds import FOLDS, load_base_dataset, split_train_valid
from features import prepare_train_valid_for_catboost


OUTPUT_DIR = Path("outputs/ensemble")

TRAIN_ROWS_LIMIT = 500_000
VALID_ROWS_LIMIT = 300_000

RANDOM_SEED = 42

CATBOOST_1_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.06,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_SEED,
    "thread_count": 4,
    "verbose": 100,
    "allow_writing_files": False,
    "early_stopping_rounds": 50,
}

CATBOOST_2_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.045,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_SEED + 1,
    "thread_count": 4,
    "verbose": 100,
    "allow_writing_files": False,
    "early_stopping_rounds": 50,
}

LIGHTGBM_1_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 2.0,
    "lambda_l1": 0.1,
    "max_bin": 255,
    "verbose": -1,
    "num_threads": 4,
    "seed": RANDOM_SEED,
}

LIGHTGBM_2_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.035,
    "num_leaves": 255,
    "max_depth": -1,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.75,
    "bagging_freq": 5,
    "lambda_l2": 5.0,
    "lambda_l1": 0.2,
    "max_bin": 255,
    "verbose": -1,
    "num_threads": 4,
    "seed": RANDOM_SEED + 1,
}

XGBOOST_1_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "max_bin": 256,
    "nthread": 4,
    "seed": RANDOM_SEED,
    "verbosity": 0,
}

LGB_NUM_ROUNDS = 500
LGB_EARLY_STOPPING_ROUNDS = 50

XGB_NUM_ROUNDS = 500
XGB_EARLY_STOPPING_ROUNDS = 50

ENSEMBLE_WEIGHTS = {
    "catboost_1": 1.0,
    "catboost_2": 1.0,
    "lightgbm_1": 1.0,
    "lightgbm_2": 1.0,
    "xgboost_1": 1.0,
}


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        log(f"{self.name} started")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.start_time

        if exc_type is None:
            log(f"{self.name} finished in {elapsed:.1f} sec")
        else:
            log(f"{self.name} failed after {elapsed:.1f} sec")


def limit_lazy(lf: pl.LazyFrame, limit: int | None) -> pl.LazyFrame:
    if limit is None:
        return lf

    return lf.head(limit)


def find_best_threshold(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.03, 0.40, 0.005):
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    threshold: float,
    model_name: str,
) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def get_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y)
    positives = max(1, int((y_arr == 1).sum()))
    negatives = max(1, int((y_arr == 0).sum()))

    return negatives / positives


def prepare_for_catboost(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    cat_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = X_train.copy()
    valid = X_valid.copy()

    for column in train.columns:
        if column in cat_features:
            train[column] = train[column].astype("string").fillna("unknown")
            valid[column] = valid[column].astype("string").fillna("unknown")
        else:
            train[column] = pd.to_numeric(train[column], errors="coerce").fillna(-1.0).astype("float32")
            valid[column] = pd.to_numeric(valid[column], errors="coerce").fillna(-1.0).astype("float32")

    return train, valid


def prepare_for_lgb_xgb(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    cat_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = X_train.copy()
    valid = X_valid.copy()

    for column in train.columns:
        if column in cat_features:
            train[column] = train[column].astype("string").fillna("unknown")
            valid[column] = valid[column].astype("string").fillna("unknown")

            categories = pd.Index(train[column].unique()).union(pd.Index(valid[column].unique()))

            train[column] = pd.Categorical(train[column], categories=categories)
            valid[column] = pd.Categorical(valid[column], categories=categories)
        else:
            train[column] = pd.to_numeric(train[column], errors="coerce").fillna(-1.0).astype("float32")
            valid[column] = pd.to_numeric(valid[column], errors="coerce").fillna(-1.0).astype("float32")

    return train, valid


def train_catboost_model(
    name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
) -> tuple[np.ndarray, dict]:
    with Timer(f"{name} | preparing data"):
        train_cb, valid_cb = prepare_for_catboost(X_train, X_valid, cat_features)

    with Timer(f"{name} | training"):
        model = CatBoostClassifier(**params)

        model.fit(
            train_cb,
            y_train,
            cat_features=cat_features,
            eval_set=(valid_cb, y_valid),
            use_best_model=True,
        )

    with Timer(f"{name} | prediction and metrics"):
        proba = model.predict_proba(valid_cb)[:, 1]
        threshold, _ = find_best_threshold(y_valid, proba)
        metrics = calculate_metrics(y_valid, proba, threshold, name)
        metrics["best_iteration"] = int(model.get_best_iteration() or params["iterations"])

    del train_cb
    del valid_cb
    del model
    gc.collect()

    return proba, metrics


def train_lightgbm_model(
    name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
) -> tuple[np.ndarray, dict]:
    with Timer(f"{name} | preparing data"):
        train_lgb, valid_lgb = prepare_for_lgb_xgb(X_train, X_valid, cat_features)

        params = dict(params)
        params["scale_pos_weight"] = get_scale_pos_weight(y_train)

        train_dataset = lgb.Dataset(
            train_lgb,
            label=y_train,
            categorical_feature=cat_features,
            free_raw_data=True,
        )

        valid_dataset = lgb.Dataset(
            valid_lgb,
            label=y_valid,
            categorical_feature=cat_features,
            reference=train_dataset,
            free_raw_data=True,
        )

    with Timer(f"{name} | training"):
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=LGB_NUM_ROUNDS,
            valid_sets=[valid_dataset],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(100),
            ],
        )

    with Timer(f"{name} | prediction and metrics"):
        proba = model.predict(valid_lgb, num_iteration=model.best_iteration)
        threshold, _ = find_best_threshold(y_valid, proba)
        metrics = calculate_metrics(y_valid, proba, threshold, name)
        metrics["best_iteration"] = int(model.best_iteration)

    del train_lgb
    del valid_lgb
    del train_dataset
    del valid_dataset
    del model
    gc.collect()

    return proba, metrics


def train_xgboost_model(
    name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
) -> tuple[np.ndarray, dict]:
    with Timer(f"{name} | preparing data"):
        train_xgb, valid_xgb = prepare_for_lgb_xgb(X_train, X_valid, cat_features)

        params = dict(params)
        params["scale_pos_weight"] = get_scale_pos_weight(y_train)

        dtrain = xgb.DMatrix(train_xgb, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(valid_xgb, label=y_valid, enable_categorical=True)

    with Timer(f"{name} | training"):
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=XGB_NUM_ROUNDS,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            verbose_eval=100,
        )

    with Timer(f"{name} | prediction and metrics"):
        proba = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        threshold, _ = find_best_threshold(y_valid, proba)
        metrics = calculate_metrics(y_valid, proba, threshold, name)
        metrics["best_iteration"] = int(model.best_iteration)

    del train_xgb
    del valid_xgb
    del dtrain
    del dvalid
    del model
    gc.collect()

    return proba, metrics


def blend_predictions(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    usable_weights = {
        name: weight
        for name, weight in weights.items()
        if name in predictions
    }

    weight_sum = sum(usable_weights.values())

    if weight_sum <= 0:
        raise ValueError("No model predictions available for ensemble")

    result = None

    for name, weight in usable_weights.items():
        current = predictions[name] * (weight / weight_sum)

        if result is None:
            result = current.copy()
        else:
            result += current

    return result


def run_single_fold(data: pl.LazyFrame, fold) -> pd.DataFrame:
    print("\n================================")
    print(f"Fold: {fold.name}")
    print(f"Train: {fold.train_start} — {fold.train_end}")
    print(f"Valid: {fold.valid_start} — {fold.valid_end}")
    print("================================")

    train, valid = split_train_valid(data, fold)

    train = limit_lazy(train, TRAIN_ROWS_LIMIT)
    valid = limit_lazy(valid, VALID_ROWS_LIMIT)

    with Timer(f"{fold.name} | feature engineering from features.py"):
        X_train, y_train, X_valid, y_valid, used_features, cat_features = prepare_train_valid_for_catboost(
            train=train,
            valid=valid,
        )

    log(f"{fold.name} | Train shape: {X_train.shape}")
    log(f"{fold.name} | Valid shape: {X_valid.shape}")
    log(f"{fold.name} | Used features: {len(used_features)}")
    log(f"{fold.name} | Categorical features: {len(cat_features)}")

    predictions = {}
    metrics = []

    log(f"{fold.name} | starting model 1/5: catboost_1")
    proba, model_metrics = train_catboost_model(
        name="catboost_1",
        params=CATBOOST_1_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        cat_features=cat_features,
    )
    predictions["catboost_1"] = proba
    metrics.append(model_metrics)

    log(f"{fold.name} | starting model 2/5: catboost_2")
    proba, model_metrics = train_catboost_model(
        name="catboost_2",
        params=CATBOOST_2_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        cat_features=cat_features,
    )
    predictions["catboost_2"] = proba
    metrics.append(model_metrics)

    log(f"{fold.name} | starting model 3/5: lightgbm_1")
    proba, model_metrics = train_lightgbm_model(
        name="lightgbm_1",
        params=LIGHTGBM_1_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        cat_features=cat_features,
    )
    predictions["lightgbm_1"] = proba
    metrics.append(model_metrics)

    log(f"{fold.name} | starting model 4/5: lightgbm_2")
    proba, model_metrics = train_lightgbm_model(
        name="lightgbm_2",
        params=LIGHTGBM_2_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        cat_features=cat_features,
    )
    predictions["lightgbm_2"] = proba
    metrics.append(model_metrics)

    log(f"{fold.name} | starting model 5/5: xgboost_1")
    proba, model_metrics = train_xgboost_model(
        name="xgboost_1",
        params=XGBOOST_1_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        cat_features=cat_features,
    )
    predictions["xgboost_1"] = proba
    metrics.append(model_metrics)

    with Timer(f"{fold.name} | ensemble blending and metrics"):
        ensemble_proba = blend_predictions(predictions, ENSEMBLE_WEIGHTS)
        ensemble_threshold, _ = find_best_threshold(y_valid, ensemble_proba)
        ensemble_metrics = calculate_metrics(
            y_valid,
            ensemble_proba,
            ensemble_threshold,
            "ensemble_5_models",
        )
        ensemble_metrics["best_iteration"] = -1
        metrics.append(ensemble_metrics)

    result = pd.DataFrame(metrics)

    result.insert(0, "fold", fold.name)
    result.insert(1, "train_start", fold.train_start)
    result.insert(2, "train_end", fold.train_end)
    result.insert(3, "valid_start", fold.valid_start)
    result.insert(4, "valid_end", fold.valid_end)
    result.insert(5, "train_rows", len(X_train))
    result.insert(6, "valid_rows", len(X_valid))
    result.insert(7, "features_count", len(used_features))

    print("\nFold result:")
    print(
        result[
            [
                "fold",
                "model",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "threshold",
            ]
        ].sort_values("roc_auc", ascending=False)
    )

    del X_train
    del y_train
    del X_valid
    del y_valid
    del predictions
    gc.collect()

    return result


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results
        .groupby("model", as_index=False)
        .agg({
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "roc_auc": "mean",
            "pr_auc": "mean",
            "threshold": "mean",
            "features_count": "mean",
            "train_rows": "mean",
            "valid_rows": "mean",
        })
        .rename(columns={
            "precision": "mean_precision",
            "recall": "mean_recall",
            "f1": "mean_f1",
            "roc_auc": "mean_roc_auc",
            "pr_auc": "mean_pr_auc",
            "threshold": "mean_threshold",
            "features_count": "mean_features_count",
            "train_rows": "mean_train_rows",
            "valid_rows": "mean_valid_rows",
        })
        .sort_values("mean_roc_auc", ascending=False)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log(f"Run ID: {run_id}")
    log("Loading dataset")

    with Timer("loading dataset"):
        data = load_base_dataset()

    fold_results = []

    for fold in FOLDS:
        with Timer(f"{fold.name} | full fold pipeline"):
            fold_result = run_single_fold(data, fold)
            fold_results.append(fold_result)

    with Timer("summarizing and saving results"):
        results = pd.concat(fold_results, ignore_index=True)
        summary = summarize_results(results)

        metrics_path = OUTPUT_DIR / f"ensemble_3fold_metrics_{run_id}.csv"
        summary_path = OUTPUT_DIR / f"ensemble_3fold_summary_{run_id}.csv"
        latest_metrics_path = OUTPUT_DIR / "ensemble_3fold_metrics_latest.csv"
        latest_summary_path = OUTPUT_DIR / "ensemble_3fold_summary_latest.csv"
        config_path = OUTPUT_DIR / f"ensemble_config_{run_id}.json"

        results.to_csv(metrics_path, index=False)
        summary.to_csv(summary_path, index=False)
        results.to_csv(latest_metrics_path, index=False)
        summary.to_csv(latest_summary_path, index=False)

        config = {
            "run_id": run_id,
            "train_rows_limit": TRAIN_ROWS_LIMIT,
            "valid_rows_limit": VALID_ROWS_LIMIT,
            "random_seed": RANDOM_SEED,
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "catboost_1_params": CATBOOST_1_PARAMS,
            "catboost_2_params": CATBOOST_2_PARAMS,
            "lightgbm_1_params": LIGHTGBM_1_PARAMS,
            "lightgbm_2_params": LIGHTGBM_2_PARAMS,
            "xgboost_1_params": XGBOOST_1_PARAMS,
            "lgb_num_rounds": LGB_NUM_ROUNDS,
            "lgb_early_stopping_rounds": LGB_EARLY_STOPPING_ROUNDS,
            "xgb_num_rounds": XGB_NUM_ROUNDS,
            "xgb_early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
        }

        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

    print("\nAll metrics:")
    print(
        results[
            [
                "fold",
                "model",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "threshold",
            ]
        ]
    )

    print("\nSummary:")
    print(summary)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved latest metrics to: {latest_metrics_path}")
    print(f"Saved latest summary to: {latest_summary_path}")
    print(f"Saved config to: {config_path}")


if __name__ == "__main__":
    main()