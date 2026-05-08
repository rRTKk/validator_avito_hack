import polars as pl

from folds import (
    FOLDS,
    DATE_COL,
    TARGET_COL,
    load_base_dataset,
    validate_all_folds,
    split_train_valid,
    get_all_fold_stats,
)


def check_fold_not_empty(data: pl.LazyFrame) -> int:
    rows = data.select(pl.len().alias("rows")).collect().item()

    if rows == 0:
        raise ValueError("Fold part is empty")

    return rows


def check_required_columns(data: pl.LazyFrame) -> None:
    columns = set(data.collect_schema().names())

    required_columns = {
        DATE_COL,
        TARGET_COL,
        "deliveryorder_id",
        "item_id",
        "buyer_id",
        "seller_id",
    }

    missing_columns = required_columns - columns

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def check_train_before_valid(train: pl.LazyFrame, valid: pl.LazyFrame, fold_name: str) -> None:
    dates = pl.concat([
        train.select([
            pl.lit("train").alias("part"),
            pl.col(DATE_COL).min().alias("min_date"),
            pl.col(DATE_COL).max().alias("max_date"),
        ]).collect(),
        valid.select([
            pl.lit("valid").alias("part"),
            pl.col(DATE_COL).min().alias("min_date"),
            pl.col(DATE_COL).max().alias("max_date"),
        ]).collect(),
    ])

    train_max = dates.filter(pl.col("part") == "train").select("max_date").item()
    valid_min = dates.filter(pl.col("part") == "valid").select("min_date").item()

    if train_max >= valid_min:
        raise ValueError(
            f"{fold_name}: train max date must be before valid min date. "
            f"train_max={train_max}, valid_min={valid_min}"
        )


def check_target_distribution(valid: pl.LazyFrame, fold_name: str) -> None:
    stats = valid.select([
        pl.len().alias("rows"),
        pl.col(TARGET_COL).sum().alias("returns_count"),
        pl.col(TARGET_COL).mean().alias("return_rate"),
    ]).collect()

    row = stats.row(0, named=True)

    if row["returns_count"] == 0:
        raise ValueError(f"{fold_name}: validation has zero returns")

    print(f"\n{fold_name} validation target stats:")
    print(stats)


def main() -> None:
    validate_all_folds()

    data = load_base_dataset()

    check_required_columns(data)

    print("\nFold stats:")
    print(get_all_fold_stats(data))

    for fold in FOLDS:
        train, valid = split_train_valid(data, fold)

        train_rows = check_fold_not_empty(train)
        valid_rows = check_fold_not_empty(valid)

        print(f"\n{fold.name}: train rows = {train_rows}, valid rows = {valid_rows}")

        check_train_before_valid(train, valid, fold.name)
        check_target_distribution(valid, fold.name)

    print("\nFolds test completed successfully")


if __name__ == "__main__":
    main()