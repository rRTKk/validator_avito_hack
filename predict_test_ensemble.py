from pathlib import Path
import gc
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from catboost import CatBoostClassifier

from features import prepare_train_test_for_catboost


TRAIN_PATH = Path("data/parquet/dataset_final.parquet")
TEST_PATH = Path("data/parquet/test_final.parquet")

OUTPUT_DIR = Path("outputs/submission")
ENSEMBLE_OUTPUT_DIR = Path("outputs/ensemble")

SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
PROBA_PATH = OUTPUT_DIR / "test_predictions_with_proba.csv"
CONFIG_PATH = OUTPUT_DIR / "prediction_config.json"

LATEST_ENSEMBLE_SUMMARY_PATH = ENSEMBLE_OUTPUT_DIR / "ensemble_3fold_summary_latest.csv"

TARGET_COL = "is_return"

TRAIN_ROWS_LIMIT = 1_500_000
TEST_ROWS_LIMIT = None

RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.83
FORCE_THRESHOLD = 0.83

CATBOOST_1_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.06,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_SEED,
    "thread_count": 6,
    "verbose": 100,
    "allow_writing_files": False,
}

CATBOOST_2_PARAMS = {
    "iterations": 350,
    "learning_rate": 0.045,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_SEED + 1,
    "thread_count": 6,
    "verbose": 100,
    "allow_writing_files": False,
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
    "nthread": 6,
    "seed": RANDOM_SEED,
    "verbosity": 0,
}

XGB_NUM_ROUNDS = 350

ENSEMBLE_WEIGHTS = {
    "catboost_1": 2.0,
    "catboost_2": 3.0,
    "xgboost_1": 1.0,
}

EXTRA_THRESHOLDS = [0.50, 0.5283333333333334, 0.55, 0.60, 0.65]


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


def load_threshold() -> float:
    if FORCE_THRESHOLD is not None:
        log(f"Using forced threshold: {FORCE_THRESHOLD}")
        return float(FORCE_THRESHOLD)

    if not LATEST_ENSEMBLE_SUMMARY_PATH.exists():
        log(f"Validation summary not found. Using default threshold: {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD

    summary = pd.read_csv(LATEST_ENSEMBLE_SUMMARY_PATH)

    if "model" not in summary.columns or "mean_threshold" not in summary.columns:
        log(f"mean_threshold not found. Using default threshold: {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD

    ensemble_row = summary[summary["model"] == "ensemble_5_models"]

    if ensemble_row.empty:
        log(f"ensemble_5_models threshold not found. Using default threshold: {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD

    threshold = float(ensemble_row.iloc[0]["mean_threshold"])
    log(f"Loaded threshold from validation: {threshold}")

    return threshold


def get_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y)
    positives = max(1, int((y_arr == 1).sum()))
    negatives = max(1, int((y_arr == 0).sum()))

    return negatives / positives


def prepare_for_catboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = X_train.copy()
    test = X_test.copy()

    for column in train.columns:
        if column in cat_features:
            train[column] = train[column].astype("string").fillna("unknown")
            test[column] = test[column].astype("string").fillna("unknown")
        else:
            train[column] = pd.to_numeric(train[column], errors="coerce").fillna(-1.0).astype("float32")
            test[column] = pd.to_numeric(test[column], errors="coerce").fillna(-1.0).astype("float32")

    return train, test


def prepare_for_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = X_train.copy()
    test = X_test.copy()

    for column in train.columns:
        if column in cat_features:
            train[column] = train[column].astype("string").fillna("unknown")
            test[column] = test[column].astype("string").fillna("unknown")

            categories = pd.Index(train[column].unique()).union(pd.Index(test[column].unique()))

            train[column] = pd.Categorical(train[column], categories=categories)
            test[column] = pd.Categorical(test[column], categories=categories)
        else:
            train[column] = pd.to_numeric(train[column], errors="coerce").fillna(-1.0).astype("float32")
            test[column] = pd.to_numeric(test[column], errors="coerce").fillna(-1.0).astype("float32")

    return train, test


def train_predict_catboost(
    name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list[str],
) -> np.ndarray:
    with Timer(f"{name} | preparing data"):
        train_cb, test_cb = prepare_for_catboost(X_train, X_test, cat_features)

    with Timer(f"{name} | training"):
        model = CatBoostClassifier(**params)
        model.fit(
            train_cb,
            y_train,
            cat_features=cat_features,
        )

    with Timer(f"{name} | prediction"):
        proba = model.predict_proba(test_cb)[:, 1]

    del train_cb
    del test_cb
    del model
    gc.collect()

    return proba


def train_predict_xgboost(
    name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list[str],
) -> np.ndarray:
    with Timer(f"{name} | preparing data"):
        train_xgb, test_xgb = prepare_for_xgboost(X_train, X_test, cat_features)

        params = dict(params)
        params["scale_pos_weight"] = get_scale_pos_weight(y_train)

        dtrain = xgb.DMatrix(train_xgb, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(test_xgb, enable_categorical=True)

    with Timer(f"{name} | training"):
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=XGB_NUM_ROUNDS,
            evals=[(dtrain, "train")],
            verbose_eval=100,
        )

    with Timer(f"{name} | prediction"):
        proba = model.predict(dtest)

    del train_xgb
    del test_xgb
    del dtrain
    del dtest
    del model
    gc.collect()

    return proba


def blend_predictions(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    usable_weights = {
        name: weight
        for name, weight in weights.items()
        if name in predictions and weight > 0
    }

    weight_sum = sum(usable_weights.values())

    if weight_sum <= 0:
        raise ValueError("No predictions available for ensemble")

    result = None

    for name, weight in usable_weights.items():
        current = predictions[name] * (weight / weight_sum)

        if result is None:
            result = current.copy()
        else:
            result += current

    return result


def load_data() -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train dataset not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    train = (
        pl.scan_parquet(TRAIN_PATH)
        .filter(pl.col(TARGET_COL).is_not_null())
    )

    test = pl.scan_parquet(TEST_PATH)

    test_rule_false = test.filter(pl.col(TARGET_COL) == False)
    test_ml = test.filter(pl.col(TARGET_COL).is_null())

    return train, test_rule_false, test_ml


def get_test_ids(test_ml: pl.LazyFrame) -> pd.DataFrame:
    return (
        test_ml
        .select([
            "deliveryorder_id",
            "item_id",
        ])
        .collect()
        .to_pandas()
    )


def build_submission(
    test_rule_false: pl.LazyFrame,
    test_ml_ids: pd.DataFrame,
    ensemble_proba: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ml_part = test_ml_ids.copy()
    ml_part["proba"] = ensemble_proba
    ml_part["is_return"] = ml_part["proba"] >= threshold

    rule_part = (
        test_rule_false
        .select([
            "deliveryorder_id",
            "item_id",
        ])
        .collect()
        .to_pandas()
    )

    rule_part["proba"] = 0.0
    rule_part["is_return"] = False

    full = pd.concat([rule_part, ml_part], ignore_index=True)

    submission = full[["item_id", "is_return"]].copy()
    submission["is_return"] = submission["is_return"].astype(bool)

    return submission, full


def save_extra_threshold_submissions(full_predictions: pd.DataFrame, run_id: str) -> list[dict]:
    saved = []

    for threshold in EXTRA_THRESHOLDS:
        submission = full_predictions[["item_id"]].copy()
        submission["is_return"] = full_predictions["proba"] >= threshold
        submission["is_return"] = submission["is_return"].astype(bool)

        true_count = int(submission["is_return"].sum())
        false_count = int(len(submission) - true_count)

        path = OUTPUT_DIR / f"submission_risk_thr_{threshold:.3f}_true_{true_count}_{run_id}.csv"
        submission.to_csv(path, index=False)

        saved.append({
            "threshold": float(threshold),
            "path": str(path),
            "true_count": true_count,
            "false_count": false_count,
            "true_rate": float(true_count / len(submission)),
        })

    return saved


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = load_threshold()

    log(f"Run ID: {run_id}")
    log(f"Using threshold: {threshold}")
    log("Models: catboost_1, catboost_2, xgboost_1")
    log(f"Ensemble weights: {ENSEMBLE_WEIGHTS}")

    with Timer("loading train and test"):
        train, test_rule_false, test_ml = load_data()

    with Timer("checking dataset sizes"):
        train_rows = train.select(pl.len()).collect().item()
        rule_false_rows = test_rule_false.select(pl.len()).collect().item()
        ml_rows = test_ml.select(pl.len()).collect().item()

        log(f"Train rows: {train_rows}")
        log(f"Test rule false rows: {rule_false_rows}")
        log(f"Test ML rows: {ml_rows}")
        log(f"Test total rows: {rule_false_rows + ml_rows}")

    train = limit_lazy(train, TRAIN_ROWS_LIMIT)
    test_ml = limit_lazy(test_ml, TEST_ROWS_LIMIT)

    with Timer("saving test ids"):
        test_ml_ids = get_test_ids(test_ml)

    with Timer("feature engineering from features.py"):
        X_train, y_train, X_test, used_features, cat_features, artifacts = prepare_train_test_for_catboost(
            train=train,
            test=test_ml,
        )

    log(f"X_train shape: {X_train.shape}")
    log(f"X_test shape: {X_test.shape}")
    log(f"Used features: {len(used_features)}")
    log(f"Categorical features: {len(cat_features)}")
    log(f"Positive rate in train: {float(y_train.mean()):.6f}")

    predictions = {}

    log("starting model 1/3: catboost_1")
    predictions["catboost_1"] = train_predict_catboost(
        name="catboost_1",
        params=CATBOOST_1_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        cat_features=cat_features,
    )

    log("starting model 2/3: catboost_2")
    predictions["catboost_2"] = train_predict_catboost(
        name="catboost_2",
        params=CATBOOST_2_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        cat_features=cat_features,
    )

    log("starting model 3/3: xgboost_1")
    predictions["xgboost_1"] = train_predict_xgboost(
        name="xgboost_1",
        params=XGBOOST_1_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        cat_features=cat_features,
    )

    with Timer("ensemble blending"):
        ensemble_proba = blend_predictions(predictions, ENSEMBLE_WEIGHTS)

    with Timer("building submission"):
        submission, full_predictions = build_submission(
            test_rule_false=test_rule_false,
            test_ml_ids=test_ml_ids,
            ensemble_proba=ensemble_proba,
            threshold=threshold,
        )

    timestamped_submission_path = OUTPUT_DIR / f"submission_{run_id}.csv"
    timestamped_proba_path = OUTPUT_DIR / f"test_predictions_with_proba_{run_id}.csv"
    stable_risk_path = OUTPUT_DIR / f"RISK_catboost_heavy_xgb_thr_{threshold:.3f}_{run_id}.csv"

    with Timer("saving files"):
        submission.to_csv(SUBMISSION_PATH, index=False)
        submission.to_csv(timestamped_submission_path, index=False)
        submission.to_csv(stable_risk_path, index=False)

        full_predictions.to_csv(PROBA_PATH, index=False)
        full_predictions.to_csv(timestamped_proba_path, index=False)

        extra_submissions = save_extra_threshold_submissions(full_predictions, run_id)

        config = {
            "run_id": run_id,
            "threshold": threshold,
            "force_threshold": FORCE_THRESHOLD,
            "train_rows": int(len(X_train)),
            "test_ml_rows": int(len(X_test)),
            "used_features_count": len(used_features),
            "cat_features_count": len(cat_features),
            "train_rows_limit": TRAIN_ROWS_LIMIT,
            "test_rows_limit": TEST_ROWS_LIMIT,
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "catboost_1_params": CATBOOST_1_PARAMS,
            "catboost_2_params": CATBOOST_2_PARAMS,
            "xgboost_1_params": XGBOOST_1_PARAMS,
            "xgb_num_rounds": XGB_NUM_ROUNDS,
            "extra_thresholds": EXTRA_THRESHOLDS,
            "extra_submissions": extra_submissions,
        }

        with open(CONFIG_PATH, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

    log(f"Saved submission: {SUBMISSION_PATH}")
    log(f"Saved timestamped submission: {timestamped_submission_path}")
    log(f"Saved risk submission: {stable_risk_path}")
    log(f"Saved predictions with proba: {PROBA_PATH}")

    log("Prediction distribution:")
    print(submission["is_return"].value_counts(dropna=False))

    log("Extra threshold submissions:")
    for item in extra_submissions:
        print(item)

    log("Done")

    del X_train
    del y_train
    del X_test
    del predictions
    del full_predictions
    gc.collect()


if __name__ == "__main__":
    main()