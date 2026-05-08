from pathlib import Path

import polars as pl

from preprocessing import run_preprocessing


PROCESSED_DIR = Path("data/processed")

ORDERS_CLEAN_PATH = PROCESSED_DIR / "orders_clean.parquet"
ITEMS_CLEAN_PATH = PROCESSED_DIR / "items_clean.parquet"
PAYMENTS_CLEAN_PATH = PROCESSED_DIR / "payments_clean.parquet"
USERS_CLEAN_PATH = PROCESSED_DIR / "users_clean.parquet"


def assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File was not created: {path}")


def print_schema_and_head(name: str, path: Path) -> None:
    lf = pl.scan_parquet(path)

    print(f"\n{name} schema:")
    print(lf.collect_schema())

    print(f"\n{name} head:")
    print(lf.head(5).collect())


def check_orders() -> None:
    orders = pl.scan_parquet(ORDERS_CLEAN_PATH)

    stats = orders.select([
        pl.len().alias("rows"),
        pl.col("deliveryorder_id").null_count().alias("deliveryorder_id_nulls"),
        pl.col("order_create_date").null_count().alias("order_create_date_nulls"),
        pl.col("item_id").null_count().alias("item_id_nulls"),
        pl.col("buyer_id").null_count().alias("buyer_id_nulls"),
        pl.col("seller_id").null_count().alias("seller_id_nulls"),
        pl.col("is_return").null_count().alias("is_return_nulls"),
        pl.col("cancel_date").is_not_null().sum().alias("not_null_cancel_date"),
    ]).collect()

    print("\norders checks:")
    print(stats)

    row = stats.row(0, named=True)

    if row["rows"] == 0:
        raise ValueError("orders_clean.parquet is empty")

    if row["order_create_date_nulls"] != 0:
        raise ValueError("orders_clean.parquet contains null order_create_date")

    if row["is_return_nulls"] != 0:
        raise ValueError("orders_clean.parquet contains null is_return")

    if row["not_null_cancel_date"] != 0:
        raise ValueError("orders_clean.parquet contains not-null cancel_date")


def check_items() -> None:
    items = pl.scan_parquet(ITEMS_CLEAN_PATH)

    stats = items.select([
        pl.len().alias("rows"),
        pl.col("item_id").null_count().alias("item_id_nulls"),
        pl.col("item_id").n_unique().alias("unique_item_id"),
    ]).collect()

    print("\nitems checks:")
    print(stats)

    row = stats.row(0, named=True)

    if row["rows"] == 0:
        raise ValueError("items_clean.parquet is empty")

    if row["item_id_nulls"] != 0:
        raise ValueError("items_clean.parquet contains null item_id")

    if row["rows"] != row["unique_item_id"]:
        raise ValueError("items_clean.parquet contains duplicated item_id")


def check_payments() -> None:
    payments = pl.scan_parquet(PAYMENTS_CLEAN_PATH)

    stats = payments.select([
        pl.len().alias("rows"),
        pl.col("deliveryorder_id").null_count().alias("deliveryorder_id_nulls"),
    ]).collect()

    print("\npayments checks:")
    print(stats)

    row = stats.row(0, named=True)

    if row["rows"] == 0:
        raise ValueError("payments_clean.parquet is empty")

    if row["deliveryorder_id_nulls"] != 0:
        raise ValueError("payments_clean.parquet contains null deliveryorder_id")


def check_users() -> None:
    users = pl.scan_parquet(USERS_CLEAN_PATH)

    stats = users.select([
        pl.len().alias("rows"),
        pl.col("user_id").null_count().alias("user_id_nulls"),
        pl.col("user_id").n_unique().alias("unique_user_id"),
    ]).collect()

    print("\nusers checks:")
    print(stats)

    row = stats.row(0, named=True)

    if row["rows"] == 0:
        raise ValueError("users_clean.parquet is empty")

    if row["user_id_nulls"] != 0:
        raise ValueError("users_clean.parquet contains null user_id")

    if row["rows"] != row["unique_user_id"]:
        raise ValueError("users_clean.parquet contains duplicated user_id")


def main() -> None:
    run_preprocessing()

    expected_files = [
        ORDERS_CLEAN_PATH,
        ITEMS_CLEAN_PATH,
        PAYMENTS_CLEAN_PATH,
        USERS_CLEAN_PATH,
    ]

    for path in expected_files:
        assert_file_exists(path)

    print_schema_and_head("orders_clean", ORDERS_CLEAN_PATH)
    print_schema_and_head("items_clean", ITEMS_CLEAN_PATH)
    print_schema_and_head("payments_clean", PAYMENTS_CLEAN_PATH)
    print_schema_and_head("users_clean", USERS_CLEAN_PATH)

    check_orders()
    check_items()
    check_payments()
    check_users()

    print("\nPreprocessing test completed successfully")


if __name__ == "__main__":
    main()