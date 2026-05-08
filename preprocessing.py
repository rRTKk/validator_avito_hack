from pathlib import Path

import polars as pl


RAW_DIR = Path("data")
PROCESSED_DIR = Path("data/processed")

ORDERS_PATH = RAW_DIR / "orders.csv"
ITEMS_PATH = RAW_DIR / "items.csv"
PAYMENTS_PATH = RAW_DIR / "payments.csv"
USERS_PATH = RAW_DIR / "users.csv"

ORDERS_CLEAN_PATH = PROCESSED_DIR / "orders_clean.parquet"
ITEMS_CLEAN_PATH = PROCESSED_DIR / "items_clean.parquet"
PAYMENTS_CLEAN_PATH = PROCESSED_DIR / "payments_clean.parquet"
USERS_CLEAN_PATH = PROCESSED_DIR / "users_clean.parquet"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        infer_schema_length=10000,
        ignore_errors=True,
        null_values=["", "null", "None", "NaN", "nan"],
    )


def drop_index_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    columns = lf.collect_schema().names()

    bad_columns = [
        column
        for column in columns
        if column == ""
        or column.startswith("Unnamed")
        or column in {"index", "__index_level_0__"}
    ]

    if bad_columns:
        return lf.drop(bad_columns)

    return lf


def require_columns(lf: pl.LazyFrame, required_columns: list[str], table_name: str) -> None:
    columns = set(lf.collect_schema().names())
    missing_columns = [
        column
        for column in required_columns
        if column not in columns
    ]

    if missing_columns:
        raise ValueError(f"{table_name}: missing columns: {missing_columns}")


def column_exists(lf: pl.LazyFrame, column: str) -> bool:
    return column in set(lf.collect_schema().names())


def parse_datetime_if_exists(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    if not column_exists(lf, column):
        return lf

    return lf.with_columns(
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.to_datetime(strict=False)
        .alias(column)
    )


def cast_if_exists(lf: pl.LazyFrame, column: str, dtype: pl.DataType) -> pl.LazyFrame:
    if not column_exists(lf, column):
        return lf

    return lf.with_columns(
        pl.col(column).cast(dtype, strict=False).alias(column)
    )


def fill_string_if_exists(lf: pl.LazyFrame, column: str, value: str) -> pl.LazyFrame:
    if not column_exists(lf, column):
        return lf

    return lf.with_columns(
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .fill_null(value)
        .alias(column)
    )


def clean_orders() -> None:
    orders = drop_index_columns(scan_csv(ORDERS_PATH))

    require_columns(
        orders,
        [
            "deliveryorder_id",
            "order_create_date",
            "item_id",
            "buyer_id",
            "seller_id",
            "is_return",
            "cancel_date",
        ],
        "orders.csv",
    )

    orders = parse_datetime_if_exists(orders, "order_create_date")
    orders = parse_datetime_if_exists(orders, "order_accept_date")
    orders = parse_datetime_if_exists(orders, "cancel_date")

    orders = cast_if_exists(orders, "order_price", pl.Float32)
    orders = cast_if_exists(orders, "platform_id", pl.Int16)
    orders = cast_if_exists(orders, "is_pod", pl.Boolean)
    orders = cast_if_exists(orders, "is_return", pl.Boolean)

    orders_clean = (
        orders
        .filter(pl.col("order_create_date").is_not_null())
        .filter(pl.col("is_return").is_not_null())
        .filter(pl.col("cancel_date").is_null())
        .unique(
            subset=[
                "deliveryorder_id",
                "item_id",
                "buyer_id",
                "seller_id",
                "order_create_date",
            ],
            keep="first",
        )
    )

    orders_clean.sink_parquet(ORDERS_CLEAN_PATH)


def clean_items() -> None:
    items = drop_index_columns(scan_csv(ITEMS_PATH))

    require_columns(
        items,
        [
            "item_id",
        ],
        "items.csv",
    )

    items = parse_datetime_if_exists(items, "starttime")
    items = parse_datetime_if_exists(items, "close_date")

    items_clean = (
        items
        .filter(pl.col("item_id").is_not_null())
        .unique(subset=["item_id"], keep="first")
    )

    items_clean.sink_parquet(ITEMS_CLEAN_PATH)


def clean_payments() -> None:
    payments = drop_index_columns(scan_csv(PAYMENTS_PATH))

    require_columns(
        payments,
        [
            "deliveryorder_id",
            "payment_method",
            "amount",
        ],
        "payments.csv",
    )

    payments = parse_datetime_if_exists(payments, "created_txtime")
    payments = cast_if_exists(payments, "amount", pl.Float32)

    payments_clean = (
        payments
        .filter(pl.col("deliveryorder_id").is_not_null())
    )

    payments_clean.sink_parquet(PAYMENTS_CLEAN_PATH)


def clean_users() -> None:
    users = drop_index_columns(scan_csv(USERS_PATH))

    require_columns(
        users,
        [
            "user_id",
        ],
        "users.csv",
    )

    users = parse_datetime_if_exists(users, "registrationtime")
    users = parse_datetime_if_exists(users, "firstlistingdate")

    users = cast_if_exists(users, "iscompany", pl.Boolean)
    users = cast_if_exists(users, "isblocked", pl.Boolean)
    users = fill_string_if_exists(users, "gender", "unknown")

    users_clean = (
        users
        .filter(pl.col("user_id").is_not_null())
        .unique(subset=["user_id"], keep="first")
    )

    users_clean.sink_parquet(USERS_CLEAN_PATH)


def run_preprocessing() -> None:
    ensure_dirs()

    if not ORDERS_CLEAN_PATH.exists():
        print(f"Start: {ORDERS_CLEAN_PATH}")
        clean_orders()
        print(f"Done: {ORDERS_CLEAN_PATH}")
    else:
        print(f"Skip existing: {ORDERS_CLEAN_PATH}")

    if not ITEMS_CLEAN_PATH.exists():
        print(f"Start: {ITEMS_CLEAN_PATH}")
        clean_items()
        print(f"Done: {ITEMS_CLEAN_PATH}")
    else:
        print(f"Skip existing: {ITEMS_CLEAN_PATH}")

    if not PAYMENTS_CLEAN_PATH.exists():
        print(f"Start: {PAYMENTS_CLEAN_PATH}")
        clean_payments()
        print(f"Done: {PAYMENTS_CLEAN_PATH}")
    else:
        print(f"Skip existing: {PAYMENTS_CLEAN_PATH}")

    if not USERS_CLEAN_PATH.exists():
        print(f"Start: {USERS_CLEAN_PATH}")
        clean_users()
        print(f"Done: {USERS_CLEAN_PATH}")
    else:
        print(f"Skip existing: {USERS_CLEAN_PATH}")


if __name__ == "__main__":
    run_preprocessing()