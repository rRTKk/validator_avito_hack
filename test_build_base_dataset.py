from pathlib import Path

import polars as pl

from build_base_dataset import build_base_dataset


PROCESSED_DIR = Path("data/processed")
BASE_DATASET_PATH = PROCESSED_DIR / "base_dataset.parquet"


def assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File was not created: {path}")


def main() -> None:
    build_base_dataset()

    assert_file_exists(BASE_DATASET_PATH)

    base = pl.scan_parquet(BASE_DATASET_PATH)

    print("\nbase_dataset schema:")
    print(base.collect_schema())

    print("\nbase_dataset head:")
    print(base.head(5).collect())

    stats = base.select([
        pl.len().alias("rows"),
        pl.col("deliveryorder_id").null_count().alias("deliveryorder_id_nulls"),
        pl.col("order_create_date").null_count().alias("order_create_date_nulls"),
        pl.col("item_id").null_count().alias("item_id_nulls"),
        pl.col("buyer_id").null_count().alias("buyer_id_nulls"),
        pl.col("seller_id").null_count().alias("seller_id_nulls"),
        pl.col("is_return").null_count().alias("is_return_nulls"),
    ]).collect()

    print("\nbase_dataset checks:")
    print(stats)

    row = stats.row(0, named=True)

    if row["rows"] == 0:
        raise ValueError("base_dataset.parquet is empty")

    if row["deliveryorder_id_nulls"] != 0:
        raise ValueError("base_dataset contains null deliveryorder_id")

    if row["order_create_date_nulls"] != 0:
        raise ValueError("base_dataset contains null order_create_date")

    if row["item_id_nulls"] != 0:
        raise ValueError("base_dataset contains null item_id")

    if row["buyer_id_nulls"] != 0:
        raise ValueError("base_dataset contains null buyer_id")

    if row["seller_id_nulls"] != 0:
        raise ValueError("base_dataset contains null seller_id")

    if row["is_return_nulls"] != 0:
        raise ValueError("base_dataset contains null is_return")

    print("\nBase dataset test completed successfully")


if __name__ == "__main__":
    main()