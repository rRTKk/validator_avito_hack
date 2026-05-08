import sys

import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from folds import FOLDS, load_base_dataset, split_train_valid
from hypotheses import HYPOTHESES
from features import prepare_train_valid_for_catboost


DEFAULT_HYPOTHESIS = "baseline"
DEFAULT_FOLD_INDEX = 2

TRAIN_ROWS_LIMIT = 500_000
VALID_ROWS_LIMIT = 500_000


def limit_lazy(lf: pl.LazyFrame, limit: int | None) -> pl.LazyFrame:
    if limit is None:
        return lf

    return lf.head(limit)


def make_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )


def find_best_threshold(y_true, proba) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.05, 0.95, 0.01):
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def calculate_metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def main() -> None:
    hypothesis_name = DEFAULT_HYPOTHESIS

    if len(sys.argv) > 1:
        hypothesis_name = sys.argv[1]

    if hypothesis_name not in HYPOTHESES:
        raise ValueError(f"Unknown hypothesis: {hypothesis_name}")

    fold = FOLDS[DEFAULT_FOLD_INDEX]

    print(f"Fold: {fold.name}")
    print(f"Hypothesis: {hypothesis_name}")
    print(HYPOTHESES[hypothesis_name]["description"])
    print(HYPOTHESES[hypothesis_name]["conclusion"])

    print("Loading base dataset...")
    data = load_base_dataset()

    print("Splitting train/valid...")
    train, valid = split_train_valid(data, fold)

    print("Limiting rows...")
    train = limit_lazy(train, TRAIN_ROWS_LIMIT)
    valid = limit_lazy(valid, VALID_ROWS_LIMIT)

    requested_features = HYPOTHESES[hypothesis_name]["features"]

    print("Preparing features...")
    X_train, y_train, X_valid, y_valid, used_features, cat_features = prepare_train_valid_for_catboost(
        train=train,
        valid=valid,
        requested_features=requested_features,
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Valid shape: {X_valid.shape}")
    print(f"Used features: {len(used_features)}")
    print(used_features)
    print(f"Categorical features: {cat_features}")

    print("Starting CatBoost training...")
    model = make_model()

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    print("Predicting validation probabilities...")
    proba = model.predict_proba(X_valid)[:, 1]

    print("Searching best threshold...")
    threshold, best_f1 = find_best_threshold(y_valid, proba)

    print("Calculating metrics...")
    metrics = calculate_metrics(y_valid, proba, threshold)

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()