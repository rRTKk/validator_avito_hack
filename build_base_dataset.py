from pathlib import Path

import polars as pl


PROCESSED_DIR = Path("data/processed")

ORDERS_CLEAN_PATH = PROCESSED_DIR / "orders_clean.parquet"
ITEMS_CLEAN_PATH = PROCESSED_DIR / "items_clean.parquet"
PAYMENTS_CLEAN_PATH = PROCESSED_DIR / "payments_clean.parquet"
USERS_CLEAN_PATH = PROCESSED_DIR / "users_clean.parquet"

BASE_DATASET_PATH = PROCESSED_DIR / "base_dataset.parquet"


def scan_parquet(path: Path) -> pl.LazyFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pl.scan_parquet(path)


def require_columns(lf: pl.LazyFrame, required_columns: list[str], table_name: str) -> None:
    columns = set(lf.collect_schema().names())

    missing_columns = [
        column
        for column in required_columns
        if column not in columns
    ]

    if missing_columns:
        raise ValueError(f"{table_name}: missing columns: {missing_columns}")


def build_payments_agg(payments: pl.LazyFrame) -> pl.LazyFrame:
    require_columns(
        payments,
        [
            "deliveryorder_id",
            "payment_method",
            "amount",
        ],
        "payments_clean.parquet",
    )

    agg_exprs = [
        pl.len().alias("payment_count"),
        pl.col("amount").sum().alias("payment_total"),
        pl.col("amount").mean().alias("payment_mean"),
        pl.col("amount").max().alias("payment_max"),
        pl.col("amount").min().alias("payment_min"),
        pl.col("payment_method").n_unique().alias("payment_method_count"),
        pl.col("payment_method").drop_nulls().first().alias("payment_method_first"),
        pl.col("payment_method").drop_nulls().last().alias("payment_method_last"),
    ]

    payment_columns = set(payments.collect_schema().names())

    if "created_txtime" in payment_columns:
        agg_exprs.extend([
            pl.col("created_txtime").min().alias("first_created_txtime"),
            pl.col("created_txtime").max().alias("last_created_txtime"),
        ])

    return (
        payments
        .filter(pl.col("deliveryorder_id").is_not_null())
        .group_by("deliveryorder_id")
        .agg(agg_exprs)
    )


def build_buyer_users(users: pl.LazyFrame) -> pl.LazyFrame:
    columns = users.collect_schema().names()

    rename_mapping = {
        column: f"buyer_{column}"
        for column in columns
        if column != "user_id"
    }

    return (
        users
        .rename(rename_mapping)
        .rename({"user_id": "buyer_id"})
    )


def build_seller_users(users: pl.LazyFrame) -> pl.LazyFrame:
    columns = users.collect_schema().names()

    rename_mapping = {
        column: f"seller_{column}"
        for column in columns
        if column != "user_id"
    }

    return (
        users
        .rename(rename_mapping)
        .rename({"user_id": "seller_id"})
    )


def build_base_dataset() -> None:
    orders = scan_parquet(ORDERS_CLEAN_PATH)
    items = scan_parquet(ITEMS_CLEAN_PATH)
    payments = scan_parquet(PAYMENTS_CLEAN_PATH)
    users = scan_parquet(USERS_CLEAN_PATH)

    require_columns(
        orders,
        [
            "deliveryorder_id",
            "order_create_date",
            "item_id",
            "buyer_id",
            "seller_id",
            "is_return",
        ],
        "orders_clean.parquet",
    )

    require_columns(
        items,
        [
            "item_id",
        ],
        "items_clean.parquet",
    )

    require_columns(
        users,
        [
            "user_id",
        ],
        "users_clean.parquet",
    )

    payments_agg = build_payments_agg(payments)
    buyer_users = build_buyer_users(users)
    seller_users = build_seller_users(users)

    base_dataset = (
        orders
        .join(items, on="item_id", how="left")
        .join(payments_agg, on="deliveryorder_id", how="left")
        .join(buyer_users, on="buyer_id", how="left")
        .join(seller_users, on="seller_id", how="left")
        .with_columns([
            pl.col("payment_count").fill_null(0).cast(pl.Int32).alias(
                "payment_count"),
            pl.col("payment_total").fill_null(0).cast(pl.Float32).alias(
                "payment_total"),
            pl.col("payment_mean").fill_null(0).cast(pl.Float32).alias(
                "payment_mean"),
            pl.col("payment_max").fill_null(0).cast(pl.Float32).alias(
                "payment_max"),
            pl.col("payment_min").fill_null(0).cast(pl.Float32).alias(
                "payment_min"),
            pl.col("payment_method_count").fill_null(0).cast(pl.Int32).alias(
                "payment_method_count"),
            pl.col("payment_method_first").fill_null("unknown").alias(
                "payment_method_first"),
            pl.col("payment_method_last").fill_null("unknown").alias(
                "payment_method_last"),
        ])
        .filter(pl.col("deliveryorder_id").is_not_null())
        .filter(pl.col("order_create_date").is_not_null())
        .filter(pl.col("item_id").is_not_null())
        .filter(pl.col("buyer_id").is_not_null())
        .filter(pl.col("seller_id").is_not_null())
        .filter(pl.col("is_return").is_not_null())
    )

    base_dataset.sink_parquet(BASE_DATASET_PATH)


if __name__ == "__main__":
    build_base_dataset()