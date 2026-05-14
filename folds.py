from dataclasses import dataclass
from pathlib import Path

import polars as pl


PARQUET_DIR = Path("data/parquet")
FOLDS_DIR = Path("data/folds")

DATASET_PATH = PARQUET_DIR / "dataset_final.parquet"

DATE_COL = "order_create_date"
DATE_WORK_COL = "_order_create_date_parsed"
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
        train_start="2025-07-01",
        train_end="2025-08-31",
        valid_start="2025-09-01",
        valid_end="2025-09-10",
    ),
    FoldConfig(
        name="fold_2",
        train_start="2025-07-01",
        train_end="2025-09-10",
        valid_start="2025-09-11",
        valid_end="2025-09-20",
    ),
    FoldConfig(
        name="fold_3",
        train_start="2025-07-01",
        train_end="2025-09-20",
        valid_start="2025-09-21",
        valid_end="2025-09-30",
    ),
]


def load_base_dataset(path: Path = DATASET_PATH) -> pl.LazyFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

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


def add_parsed_date_column(
    data: pl.LazyFrame,
    date_col: str = DATE_COL,
    date_work_col: str = DATE_WORK_COL,
) -> pl.LazyFrame:
    schema = data.collect_schema()

    if date_col not in schema.names():
        raise ValueError(f"Date column not found: {date_col}")

    dtype = schema[date_col]

    if dtype == pl.Date:
        return data.with_columns([
            pl.col(date_col).alias(date_work_col)
        ])

    dtype_name = str(dtype)

    if dtype_name.startswith("Datetime"):
        return data.with_columns([
            pl.col(date_col).dt.date().alias(date_work_col)
        ])

    return data.with_columns([
        pl.col(date_col)
        .cast(pl.Utf8, strict=False)
        .str.slice(0, 10)
        .str.to_date(strict=False)
        .alias(date_work_col)
    ])


def prepare_dataset_for_folds(
    data: pl.LazyFrame,
    date_col: str = DATE_COL,
    date_work_col: str = DATE_WORK_COL,
    target_col: str = TARGET_COL,
) -> pl.LazyFrame:
    schema = data.collect_schema()
    columns = schema.names()

    if target_col not in columns:
        raise ValueError(f"Target column not found: {target_col}")

    data = add_parsed_date_column(
        data=data,
        date_col=date_col,
        date_work_col=date_work_col,
    )

    return (
        data
        .filter(pl.col(date_work_col).is_not_null())
        .filter(pl.col(target_col).is_not_null())
        .with_columns([
            pl.col(target_col).cast(pl.Boolean, strict=False).alias(target_col)
        ])
    )


def split_train_valid(
    data: pl.LazyFrame,
    fold: FoldConfig,
    date_col: str = DATE_COL,
    date_work_col: str = DATE_WORK_COL,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    validate_fold_config(fold)

    data = prepare_dataset_for_folds(
        data=data,
        date_col=date_col,
        date_work_col=date_work_col,
    )

    train_start = pl.lit(fold.train_start).str.to_date()
    train_end = pl.lit(fold.train_end).str.to_date()
    valid_start = pl.lit(fold.valid_start).str.to_date()
    valid_end = pl.lit(fold.valid_end).str.to_date()

    train = data.filter(
        (pl.col(date_work_col) >= train_start)
        & (pl.col(date_work_col) <= train_end)
    )

    valid = data.filter(
        (pl.col(date_work_col) >= valid_start)
        & (pl.col(date_work_col) <= valid_end)
    )

    return train, valid


def get_part_stats(
    data: pl.LazyFrame,
    fold: FoldConfig,
    part: str,
    date_work_col: str = DATE_WORK_COL,
    target_col: str = TARGET_COL,
) -> pl.LazyFrame:
    return data.select([
        pl.lit(fold.name).alias("fold"),
        pl.lit(part).alias("part"),
        pl.len().alias("rows"),
        pl.col(date_work_col).min().alias("min_date"),
        pl.col(date_work_col).max().alias("max_date"),
        pl.col(target_col).cast(pl.Int32).sum().alias("returns_count"),
        pl.col(target_col).cast(pl.Int32).mean().alias("return_rate"),
    ])


def get_fold_stats(
    data: pl.LazyFrame,
    fold: FoldConfig,
    date_col: str = DATE_COL,
    date_work_col: str = DATE_WORK_COL,
    target_col: str = TARGET_COL,
) -> pl.DataFrame:
    train, valid = split_train_valid(
        data=data,
        fold=fold,
        date_col=date_col,
        date_work_col=date_work_col,
    )

    train_stats = get_part_stats(
        data=train,
        fold=fold,
        part="train",
        date_work_col=date_work_col,
        target_col=target_col,
    )

    valid_stats = get_part_stats(
        data=valid,
        fold=fold,
        part="valid",
        date_work_col=date_work_col,
        target_col=target_col,
    )

    return pl.concat([
        train_stats.collect(),
        valid_stats.collect(),
    ])


def get_all_fold_stats(
    data: pl.LazyFrame,
    folds: list[FoldConfig] = FOLDS,
    date_col: str = DATE_COL,
    date_work_col: str = DATE_WORK_COL,
    target_col: str = TARGET_COL,
) -> pl.DataFrame:
    stats = [
        get_fold_stats(
            data=data,
            fold=fold,
            date_col=date_col,
            date_work_col=date_work_col,
            target_col=target_col,
        )
        for fold in folds
    ]

    return pl.concat(stats)


def get_dataset_month_stats(
    data: pl.LazyFrame,
    date_col: str = DATE_COL,
    date_work_col: str = DATE_WORK_COL,
    target_col: str = TARGET_COL,
) -> pl.DataFrame:
    data = prepare_dataset_for_folds(
        data=data,
        date_col=date_col,
        date_work_col=date_work_col,
        target_col=target_col,
    )

    return (
        data
        .with_columns([
            pl.col(date_work_col).dt.strftime("%Y-%m").alias("month")
        ])
        .group_by("month")
        .agg([
            pl.len().alias("rows"),
            pl.col(target_col).cast(pl.Int32).sum().alias("returns_count"),
            pl.col(target_col).cast(pl.Int32).mean().alias("return_rate"),
        ])
        .with_columns([
            (pl.col("return_rate") * 100).round(2).alias("return_rate_percent")
        ])
        .sort("month")
        .collect()
    )


def save_folds(
    data: pl.LazyFrame,
    folds: list[FoldConfig] = FOLDS,
    output_dir: Path = FOLDS_DIR,
    date_col: str = DATE_COL,
    date_work_col: str = DATE_WORK_COL,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        train, valid = split_train_valid(
            data=data,
            fold=fold,
            date_col=date_col,
            date_work_col=date_work_col,
        )

        train_path = output_dir / f"{fold.name}_train.parquet"
        valid_path = output_dir / f"{fold.name}_valid.parquet"

        train.drop(DATE_WORK_COL, strict=False).sink_parquet(train_path)
        valid.drop(DATE_WORK_COL, strict=False).sink_parquet(valid_path)

        print(f"Saved: {train_path}")
        print(f"Saved: {valid_path}")


def main() -> None:
    validate_all_folds()

    data = load_base_dataset()

    print("\nDataset month stats:")
    month_stats = get_dataset_month_stats(data)
    print(month_stats)

    print("\nFold stats:")
    fold_stats = get_all_fold_stats(data)
    print(fold_stats)

    save_folds(data)


if __name__ == "__main__":
    main()