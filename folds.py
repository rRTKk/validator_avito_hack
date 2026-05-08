from dataclasses import dataclass
from pathlib import Path

import polars as pl


PROCESSED_DIR = Path("data/processed")
FOLDS_DIR = Path("data/folds")

BASE_DATASET_PATH = PROCESSED_DIR / "base_dataset.parquet"

DATE_COL = "order_create_date"
TARGET_COL = "is_return"


@dataclass(frozen=True)
class FoldConfig:
    name: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str


FOLDS = [
    FoldConfig(
        name="fold_1",
        train_start="2025-08-01",
        train_end="2025-08-31",
        valid_start="2025-09-01",
        valid_end="2025-09-10",
    ),
    FoldConfig(
        name="fold_2",
        train_start="2025-08-01",
        train_end="2025-09-10",
        valid_start="2025-09-11",
        valid_end="2025-09-20",
    ),
    FoldConfig(
        name="fold_3",
        train_start="2025-08-01",
        train_end="2025-09-20",
        valid_start="2025-09-21",
        valid_end="2025-09-30",
    ),
]


def load_base_dataset(path: Path = BASE_DATASET_PATH) -> pl.LazyFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base dataset not found: {path}")

    return pl.scan_parquet(path)


def validate_fold_config(fold: FoldConfig) -> None:
    train_start = pl.Series([fold.train_start]).str.to_date().item()
    train_end = pl.Series([fold.train_end]).str.to_date().item()
    valid_start = pl.Series([fold.valid_start]).str.to_date().item()
    valid_end = pl.Series([fold.valid_end]).str.to_date().item()

    if train_start > train_end:
        raise ValueError(f"{fold.name}: train_start must be <= train_end")

    if valid_start > valid_end:
        raise ValueError(f"{fold.name}: valid_start must be <= valid_end")

    if train_end >= valid_start:
        raise ValueError(f"{fold.name}: train period must be before validation period")


def validate_all_folds(folds: list[FoldConfig] = FOLDS) -> None:
    for fold in folds:
        validate_fold_config(fold)


def split_train_valid(
    data: pl.LazyFrame,
    fold: FoldConfig,
    date_col: str = DATE_COL,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    validate_fold_config(fold)

    train = data.filter(
        (pl.col(date_col).dt.date() >= pl.lit(fold.train_start).str.to_date())
        & (pl.col(date_col).dt.date() <= pl.lit(fold.train_end).str.to_date())
    )

    valid = data.filter(
        (pl.col(date_col).dt.date() >= pl.lit(fold.valid_start).str.to_date())
        & (pl.col(date_col).dt.date() <= pl.lit(fold.valid_end).str.to_date())
    )

    return train, valid


def get_fold_stats(
    data: pl.LazyFrame,
    fold: FoldConfig,
    date_col: str = DATE_COL,
    target_col: str = TARGET_COL,
) -> pl.DataFrame:
    train, valid = split_train_valid(data, fold, date_col=date_col)

    train_stats = train.select([
        pl.lit(fold.name).alias("fold"),
        pl.lit("train").alias("part"),
        pl.len().alias("rows"),
        pl.col(date_col).min().alias("min_date"),
        pl.col(date_col).max().alias("max_date"),
        pl.col(target_col).sum().alias("returns_count"),
        pl.col(target_col).mean().alias("return_rate"),
    ])

    valid_stats = valid.select([
        pl.lit(fold.name).alias("fold"),
        pl.lit("valid").alias("part"),
        pl.len().alias("rows"),
        pl.col(date_col).min().alias("min_date"),
        pl.col(date_col).max().alias("max_date"),
        pl.col(target_col).sum().alias("returns_count"),
        pl.col(target_col).mean().alias("return_rate"),
    ])

    return pl.concat([
        train_stats.collect(),
        valid_stats.collect(),
    ])


def get_all_fold_stats(
    data: pl.LazyFrame,
    folds: list[FoldConfig] = FOLDS,
    date_col: str = DATE_COL,
    target_col: str = TARGET_COL,
) -> pl.DataFrame:
    stats = [
        get_fold_stats(
            data=data,
            fold=fold,
            date_col=date_col,
            target_col=target_col,
        )
        for fold in folds
    ]

    return pl.concat(stats)


def save_folds(
    data: pl.LazyFrame,
    folds: list[FoldConfig] = FOLDS,
    output_dir: Path = FOLDS_DIR,
    date_col: str = DATE_COL,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        train, valid = split_train_valid(data, fold, date_col=date_col)

        train_path = output_dir / f"{fold.name}_train.parquet"
        valid_path = output_dir / f"{fold.name}_valid.parquet"

        train.sink_parquet(train_path)
        valid.sink_parquet(valid_path)

        print(f"Saved: {train_path}")
        print(f"Saved: {valid_path}")


def main() -> None:
    validate_all_folds()

    data = load_base_dataset()

    stats = get_all_fold_stats(data)
    print(stats)

    save_folds(data)


if __name__ == "__main__":
    main()