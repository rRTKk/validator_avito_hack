from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from folds import FOLDS, load_base_dataset, split_train_valid
from hypotheses import HYPOTHESES
from features import prepare_train_valid_for_catboost


BASELINE_NAME = "baseline"

TRAIN_ROWS_LIMIT = 500_000
VALID_ROWS_LIMIT = 300_000

OUTPUT_DIR = Path("outputs/metrics")


def limit_lazy(lf: pl.LazyFrame, limit: int | None) -> pl.LazyFrame:
    if limit is None:
        return lf

    return lf.head(limit)


def make_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=300,
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


def run_single_fold(
    data: pl.LazyFrame,
    hypothesis_name: str,
    fold,
) -> dict:
    print("\n==============================")
    print(f"Hypothesis: {hypothesis_name}")
    print(f"Fold: {fold.name}")
    print(f"Train: {fold.train_start} — {fold.train_end}")
    print(f"Valid: {fold.valid_start} — {fold.valid_end}")
    print("==============================")

    train, valid = split_train_valid(data, fold)

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
    threshold, _ = find_best_threshold(y_valid, proba)

    print("Calculating metrics...")
    metrics = calculate_metrics(y_valid, proba, threshold)

    result = {
        "hypothesis": hypothesis_name,
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
        "train_rows": len(X_train),
        "valid_rows": len(X_valid),
        "features_count": len(used_features),
        **metrics,
    }

    print("\nFold metrics:")
    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def run_hypothesis_3folds(
    data: pl.LazyFrame,
    hypothesis_name: str,
) -> pd.DataFrame:
    if hypothesis_name not in HYPOTHESES:
        raise ValueError(f"Unknown hypothesis: {hypothesis_name}")

    print("\n################################")
    print(f"Running hypothesis: {hypothesis_name}")
    print(HYPOTHESES[hypothesis_name]["description"])
    print(HYPOTHESES[hypothesis_name]["conclusion"])
    print("################################")

    results = []

    for fold in FOLDS:
        result = run_single_fold(
            data=data,
            hypothesis_name=hypothesis_name,
            fold=fold,
        )
        results.append(result)

    return pd.DataFrame(results)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df
        .groupby("hypothesis", as_index=False)
        .agg({
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "roc_auc": "mean",
            "threshold": "mean",
        })
        .rename(columns={
            "precision": "mean_precision",
            "recall": "mean_recall",
            "f1": "mean_f1",
            "roc_auc": "mean_roc_auc",
            "threshold": "mean_threshold",
        })
    )

    return summary


def compare_with_baseline(summary_df: pd.DataFrame, hypothesis_name: str) -> pd.DataFrame:
    if hypothesis_name == BASELINE_NAME:
        return summary_df

    baseline_row = summary_df[summary_df["hypothesis"] == BASELINE_NAME]
    hypothesis_row = summary_df[summary_df["hypothesis"] == hypothesis_name]

    if baseline_row.empty or hypothesis_row.empty:
        return summary_df

    baseline = baseline_row.iloc[0]
    hypothesis = hypothesis_row.iloc[0]

    comparison = {
        "hypothesis": hypothesis_name,
        "delta_precision": hypothesis["mean_precision"] - baseline["mean_precision"],
        "delta_recall": hypothesis["mean_recall"] - baseline["mean_recall"],
        "delta_f1": hypothesis["mean_f1"] - baseline["mean_f1"],
        "delta_roc_auc": hypothesis["mean_roc_auc"] - baseline["mean_roc_auc"],
    }

    return pd.DataFrame([comparison])


def save_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    hypothesis_name: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / f"{hypothesis_name}_3fold_metrics.csv"
    summary_path = OUTPUT_DIR / f"{hypothesis_name}_3fold_summary.csv"
    comparison_path = OUTPUT_DIR / f"{hypothesis_name}_vs_baseline.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if not comparison_df.empty:
        comparison_df.to_csv(comparison_path, index=False)

    print(f"\nSaved fold metrics to: {results_path}")
    print(f"Saved summary to: {summary_path}")

    if not comparison_df.empty:
        print(f"Saved comparison to: {comparison_path}")


def main() -> None:
    hypothesis_name = BASELINE_NAME

    if len(sys.argv) > 1:
        hypothesis_name = sys.argv[1]

    if hypothesis_name not in HYPOTHESES:
        raise ValueError(f"Unknown hypothesis: {hypothesis_name}")

    print("Loading base dataset...")
    data = load_base_dataset()

    all_results = []

    if hypothesis_name != BASELINE_NAME:
        baseline_results = run_hypothesis_3folds(
            data=data,
            hypothesis_name=BASELINE_NAME,
        )
        all_results.append(baseline_results)

    hypothesis_results = run_hypothesis_3folds(
        data=data,
        hypothesis_name=hypothesis_name,
    )
    all_results.append(hypothesis_results)

    results_df = pd.concat(all_results, ignore_index=True)

    summary_df = summarize_results(results_df)
    comparison_df = compare_with_baseline(summary_df, hypothesis_name)

    print("\nAll fold metrics:")
    print(results_df)

    print("\nMean metrics:")
    print(summary_df)

    if not comparison_df.empty:
        print("\nComparison with baseline:")
        print(comparison_df)

    save_results(
        results_df=results_df,
        summary_df=summary_df,
        comparison_df=comparison_df,
        hypothesis_name=hypothesis_name,
    )


if __name__ == "__main__":
    main()