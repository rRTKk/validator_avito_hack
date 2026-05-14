from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl


BASE_DIR = Path(__file__).parent

CSV_DIR = BASE_DIR
PARQUET_DIR = BASE_DIR / "data" / "parquet"
RESULTS_DIR = BASE_DIR / "results"

FINAL_DATASET_PATH = PARQUET_DIR / "dataset_final.parquet"

OLD_TEST_START = "2025-10-01"
OLD_TEST_END = "2025-10-10"

CHUNK_SIZE = 100_000

CSV_DROP = {
    "orders": {"Unnamed: 0", "order_accept_date", "cancel_date"},
    "items": {"Unnamed: 0", "close_date"},
    "payments": {"Unnamed: 0"},
    "users": {"Unnamed: 0"},
}

CSV_DTYPE_OVERRIDES = {
    "orders": {"is_pod": str},
}

ORDER_COLS = [
    "deliveryorder_id",
    "order_create_date",
    "item_id",
    "buyer_terminal",
    "seller_terminal",
    "delivery_service",
    "is_pod",
    "buyer_id",
    "seller_id",
    "order_price",
    "city",
    "platform_id",
    "is_return",
]


def resolve_csv_path(name: str) -> Path:
    candidates = [
        CSV_DIR / f"{name}.csv",
        CSV_DIR / "data" / f"{name}.csv",
        CSV_DIR / "data" / "raw" / f"{name}.csv",
    ]

    if name == "payments":
        candidates.extend([
            CSV_DIR / "payment.csv",
            CSV_DIR / "data" / "payment.csv",
            CSV_DIR / "data" / "raw" / "payment.csv",
        ])

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"CSV file not found for table: {name}")


def align_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []

    for field in schema:
        column = table.column(field.name)

        if column.type != field.type:
            try:
                column = column.cast(field.type, safe=False)
            except Exception:
                column = column.cast(pa.string())

        arrays.append(column)

    return pa.table(
        {field.name: array for field, array in zip(schema, arrays)},
        schema=schema,
    )


def convert_csv_to_parquet(force: bool = False) -> None:
    print("\nStep 1: CSV to Parquet")
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    for name, drop_columns in CSV_DROP.items():
        src = resolve_csv_path(name)
        dst = PARQUET_DIR / f"{name}.parquet"

        if dst.exists() and not force:
            print(f"Skip existing: {dst}")
            continue

        print(f"Converting: {src} -> {dst}", flush=True)

        writer = None
        writer_schema = None
        total_rows = 0

        for chunk in pd.read_csv(
            src,
            chunksize=CHUNK_SIZE,
            on_bad_lines="skip",
            low_memory=False,
            dtype=CSV_DTYPE_OVERRIDES.get(name, {}),
        ):
            columns_to_drop = [
                column
                for column in drop_columns
                if column in chunk.columns
            ]

            if columns_to_drop:
                chunk = chunk.drop(columns=columns_to_drop)

            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer_schema = table.schema
                writer = pq.ParquetWriter(
                    str(dst),
                    writer_schema,
                    compression="snappy",
                )
            elif table.schema != writer_schema:
                table = align_to_schema(table, writer_schema)

            writer.write_table(table)
            total_rows += len(chunk)

        if writer is not None:
            writer.close()

        print(f"Done: {dst}, rows={total_rows:,}", flush=True)


def make_old_test_predictions(force: bool = False) -> None:
    print("\nStep 2: Old test pseudo-labels")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    orders_csv = resolve_csv_path("orders")

    out_full = RESULTS_DIR / "old_test_rule_predictions_full.csv"
    out_submission = RESULTS_DIR / "old_test_rule_predictions_item_id_is_return.csv"
    out_distribution = RESULTS_DIR / "old_test_rule_predictions_distribution.csv"

    if out_full.exists() and out_submission.exists() and out_distribution.exists() and not force:
        print(f"Skip existing: {out_full}")
        return

    lf = (
        pl.scan_csv(
            orders_csv,
            null_values=[""],
            infer_schema_length=0,
            truncate_ragged_lines=True,
        )
        .with_columns([
            pl.col("order_create_date")
            .str.slice(0, 10)
            .alias("_order_day")
        ])
        .filter(
            (pl.col("_order_day") >= OLD_TEST_START)
            & (pl.col("_order_day") <= OLD_TEST_END)
        )
        .drop("_order_day")
        .with_columns([
            pl.when(pl.col("cancel_date").is_not_null())
            .then(False)
            .when(pl.col("order_accept_date").is_not_null())
            .then(False)
            .when(pl.col("seller_terminal").is_not_null())
            .then(True)
            .otherwise(False)
            .alias("is_return")
        ])
    )

    lf.sink_csv(out_full)
    lf.select(["item_id", "is_return"]).sink_csv(out_submission)

    stats = (
        lf
        .group_by("is_return")
        .agg([
            pl.len().alias("rows")
        ])
        .with_columns([
            (pl.col("rows") / pl.sum("rows") * 100).alias("share_percent")
        ])
        .sort("is_return")
        .collect(engine="streaming")
    )

    stats.write_csv(out_distribution)

    print(f"Saved: {out_full}")
    print(f"Saved: {out_submission}")
    print(f"Saved: {out_distribution}")

    for row in stats.iter_rows(named=True):
        print(
            f"is_return={row['is_return']}: "
            f"{row['rows']:,} ({row['share_percent']:.2f}%)"
        )


def require_columns(lf: pl.LazyFrame, required_columns: list[str], table_name: str) -> None:
    existing_columns = set(lf.collect_schema().names())
    missing_columns = [
        column
        for column in required_columns
        if column not in existing_columns
    ]

    if missing_columns:
        raise ValueError(f"{table_name}: missing columns: {missing_columns}")


def build_final_dataset(force: bool = False) -> None:
    print("\nStep 3: Build final dataset")
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    if FINAL_DATASET_PATH.exists() and not force:
        print(f"Skip existing: {FINAL_DATASET_PATH}")
        return

    orders = pl.scan_parquet(PARQUET_DIR / "orders.parquet")
    items = pl.scan_parquet(PARQUET_DIR / "items.parquet")
    payments = pl.scan_parquet(PARQUET_DIR / "payments.parquet")
    users = pl.scan_parquet(PARQUET_DIR / "users.parquet")

    require_columns(orders, ORDER_COLS, "orders.parquet")

    print("Computing order_price quantiles...", flush=True)

    price_col = (
        orders
        .filter(
            pl.col("seller_terminal").is_not_null()
            & pl.col("is_return").is_not_null()
        )
        .select(
            pl.col("order_price")
            .cast(pl.Float64, strict=False)
            .drop_nulls()
        )
        .collect()["order_price"]
    )

    p01 = price_col.quantile(0.01)
    p99 = price_col.quantile(0.99)

    print(f"order_price clip: [{p01:.0f}, {p99:.0f}]", flush=True)

    payments = payments.unique(subset=["deliveryorder_id"], keep="last")

    train_lf = (
        orders
        .filter(
            pl.col("seller_terminal").is_not_null()
            & pl.col("is_return").is_not_null()
        )
        .select(ORDER_COLS)
        .with_columns([
            pl.col("is_return").cast(pl.Boolean, strict=False),
            pl.col("order_price").cast(pl.Float64, strict=False),
            pl.col("platform_id").cast(pl.Float64, strict=False),
        ])
    )

    old_test_path = RESULTS_DIR / "old_test_rule_predictions_full.csv"

    if not old_test_path.exists():
        raise FileNotFoundError(f"File not found: {old_test_path}")

    old_test_schema = (
        pl.scan_csv(
            old_test_path,
            null_values=[""],
            infer_schema_length=0,
            truncate_ragged_lines=True,
        )
        .collect_schema()
        .names()
    )

    columns_to_drop = [
        column
        for column in ["order_accept_date", "cancel_date", "Unnamed: 0"]
        if column in old_test_schema
    ]

    old_test_lf = (
        pl.scan_csv(
            old_test_path,
            null_values=[""],
            infer_schema_length=10000,
            schema_overrides={"is_pod": pl.Utf8},
            truncate_ragged_lines=True,
        )
        .drop(columns_to_drop)
        .filter(pl.col("seller_terminal").is_not_null())
        .select(ORDER_COLS)
        .with_columns([
            pl.col("is_return").cast(pl.Boolean, strict=False),
            pl.col("order_price").cast(pl.Float64, strict=False),
            pl.col("platform_id").cast(pl.Float64, strict=False),
        ])
    )

    print("Joining and writing dataset...", flush=True)

    final_lf = (
        pl.concat([train_lf, old_test_lf], how="vertical_relaxed")
        .join(items, on="item_id", how="left", suffix="_item")
        .join(users, left_on="buyer_id", right_on="user_id", how="left", suffix="_buyer")
        .join(users, left_on="seller_id", right_on="user_id", how="left", suffix="_seller")
        .join(payments, on="deliveryorder_id", how="left", suffix="_payment")
        .drop(["seller_terminal", "buyer_terminal"], strict=False)
        .with_columns([
            pl.col("delivery_service")
            .cast(pl.Utf8, strict=False)
            .fill_null("unknown")
            .alias("delivery_service"),

            pl.col("order_price")
            .cast(pl.Float64, strict=False)
            .clip(p01, p99)
            .alias("order_price"),
        ])
    )

    final_lf.sink_parquet(
        str(FINAL_DATASET_PATH),
        compression="snappy",
    )

    lf = pl.scan_parquet(FINAL_DATASET_PATH)

    median_price = (
        lf
        .select(pl.col("order_price").median())
        .collect()
        .item()
    )

    null_price_count = (
        lf
        .select(pl.col("order_price").is_null().sum())
        .collect()
        .item()
    )

    if null_price_count > 0:
        tmp_path = FINAL_DATASET_PATH.with_name("dataset_final_tmp.parquet")

        (
            lf
            .with_columns([
                pl.col("order_price")
                .fill_null(median_price)
                .alias("order_price")
            ])
            .sink_parquet(str(tmp_path), compression="snappy")
        )

        tmp_path.replace(FINAL_DATASET_PATH)

    final_lf_for_stats = pl.scan_parquet(FINAL_DATASET_PATH)

    rows_count = final_lf_for_stats.select(pl.len()).collect().item()
    cols_count = len(final_lf_for_stats.collect_schema().names())

    distribution = (
        final_lf_for_stats
        .group_by("is_return")
        .agg([
            pl.len().alias("rows")
        ])
        .sort("is_return")
        .collect()
    )

    print(
        f"Saved: {FINAL_DATASET_PATH}, "
        f"rows={rows_count:,}, cols={cols_count}"
    )

    for row in distribution.iter_rows(named=True):
        print(
            f"is_return={row['is_return']}: "
            f"{row['rows']:,} ({row['rows'] / rows_count * 100:.2f}%)"
        )


def run_pipeline(force: bool = False) -> None:
    convert_csv_to_parquet(force=force)
    make_old_test_predictions(force=force)
    build_final_dataset(force=force)
    print("\nPipeline complete")


if __name__ == "__main__":
    run_pipeline(force=False)