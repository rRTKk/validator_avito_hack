from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


TARGET_COL = "is_return"
ROW_ID_COL = "row_id"

FORBIDDEN_FEATURES = {
    "is_return",
    "order_accept_date",
    "cancel_date",
    "close_date",
    "seller_terminal",
    "buyer_terminal",
}


BASE_FEATURES = [
    "order_price",
    "log_order_price",
    "order_price_bin",
    "order_price_is_cheap",
    "order_price_is_plateau",
    "city",
    "platform_id",
    "platform_name",
    "platform_type",
    "platform_os_family",
    "delivery_service",
    "delivery_service_norm",
    "delivery_is_liberal",
    "delivery_is_strict",
    "delivery_is_avito",
    "delivery_is_yandex_market",
    "is_pod",
    "category_name",
    "microcat_name",
    "title_len_chars",
    "title_word_count",
    "title_digit_count",
    "has_digit",
    "avg_word_len",
    "kw_women",
    "kw_men",
    "kw_original",
    "kw_new",
    "kw_used",
    "has_payment",
    "payment_total",
    "log_payment_total",
    "payment_minus_order_price",
    "payment_to_order_price_ratio",
    "payment_total_bin",
    "payment_minus_order_price_bin",
    "delivery_fee",
    "delivery_fee_share",
    "n_items_in_order",
    "log_n_items_in_order",
    "n_items_bin",
    "order_is_multi_item",
    "city_buyer_type",
]


TARGET_ENCODING_FEATURES = [
    "category_order_count",
    "category_dataset_share",
    "log_category_order_count",
    "category_item_supply",
    "log_category_item_supply",
    "category_return_rate_feature",
    "category_smoothed_return_rate_feature",
    "microcat_order_count",
    "microcat_dataset_share",
    "log_microcat_order_count",
    "microcat_item_supply",
    "log_microcat_item_supply",
    "microcat_share_inside_category",
    "microcat_return_rate_feature",
    "microcat_smoothed_return_rate_feature",
    "microcat_lift_vs_category_feature",
    "delivery_service_order_count",
    "delivery_service_dataset_share",
    "log_delivery_service_order_count",
    "delivery_service_return_rate_feature",
    "delivery_service_smoothed_return_rate_feature",
    "delivery_service_lift_vs_global_feature",
]


FINAL_FEATURES = BASE_FEATURES + TARGET_ENCODING_FEATURES


PRICE_BREAKS = [0.0, 500.0, 1_500.0, 5_000.0, 15_000.0, 50_000.0, float("inf")]
PRICE_LABELS = ["0-500", "500-1.5k", "1.5k-5k", "5k-15k", "15k-50k", "50k+"]

PAYMENT_AMOUNT_BREAKS = [0.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 30_000.0, float("inf")]
PAYMENT_AMOUNT_LABELS = ["<500", "500-999", "1k-2k", "2k-5k", "5k-10k", "10k-30k", "30k+"]

PAYMENT_DIFF_BREAKS = [float("-inf"), 0.0, 1.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, float("inf")]
PAYMENT_DIFF_LABELS = ["<0", "0", "1-99", "100-199", "200-499", "500-999", "1k-2k", "2k+"]

BASKET_BREAKS = [0.0, 1.5, 2.5, 3.5, 5.5, float("inf")]
BASKET_LABELS = ["1", "2", "3", "4-5", "6+"]

LIBERAL_DELIVERY_RE = r"(?i)(avito|авито|yandex|яндекс|маркет)"
STRICT_DELIVERY_RE = r"(?i)(почта|post|cdek|сдэк|boxberry|боксберри|dpd|pickpoint|пикпойнт|5post)"

TITLE_KEYWORDS = {
    "kw_women": r"(?i)(женск|женщин)",
    "kw_men": r"(?i)(мужск|мужчин)",
    "kw_original": r"(?i)(оригинал|original)",
    "kw_new": r"(?i)(^|\s)(нов|new)",
    "kw_used": r"(?i)(б\s*/?\s*у|бу|used)",
}


def get_columns(lf: pl.LazyFrame) -> set[str]:
    return set(lf.collect_schema().names())


def to_lazy(df: pl.DataFrame | pl.LazyFrame) -> tuple[pl.LazyFrame, bool]:
    if isinstance(df, pl.LazyFrame):
        return df, True

    return df.lazy(), False


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return num / pl.when(den == 0.0).then(None).otherwise(den)


def log1p_expr(col: str, alias: str | None = None) -> pl.Expr:
    return (pl.col(col).cast(pl.Float64, strict=False) + 1.0).log(math.e).alias(alias or f"log_{col}")


def cut_expr(col: str, breaks: list[float], labels: list[str]) -> pl.Expr:
    expr = pl.lit(labels[-1], dtype=pl.Utf8)

    for label, bound in zip(reversed(labels[:-1]), reversed(breaks[1:-1])):
        expr = pl.when(pl.col(col) < bound).then(pl.lit(label, dtype=pl.Utf8)).otherwise(expr)

    return expr


def ensure_row_id(lf: pl.LazyFrame) -> pl.LazyFrame:
    if ROW_ID_COL in get_columns(lf):
        return lf

    return lf.with_row_index(ROW_ID_COL)


def add_group_folds(
    df: pl.DataFrame,
    group_col: str,
    n_folds: int,
    seed: int,
) -> pl.DataFrame:
    fold_map = (
        df
        .select(group_col)
        .unique()
        .with_columns([
            (pl.col(group_col).hash(seed=seed) % n_folds)
            .cast(pl.Int32)
            .alias("_fold")
        ])
    )

    return df.join(fold_map, on=group_col, how="left")


def apply_static_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    lf, lazy = to_lazy(df)
    lf = ensure_row_id(lf)

    delivery = (
        pl.col("delivery_service")
        .cast(pl.Utf8, strict=False)
        .fill_null("unknown")
        .str.to_lowercase()
    )

    title = pl.col("title").cast(pl.Utf8, strict=False).fill_null("")
    title_lower = title.str.to_lowercase()
    title_norm = title_lower.str.replace_all(r"[^0-9a-zа-яё]+", " ")

    price = pl.col("order_price").cast(pl.Float64, strict=False)
    amount = pl.col("amount").cast(pl.Float64, strict=False)

    buyer_type = (
        pl.when(pl.col("iscompany").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("iscompany"))
        .then(pl.lit("company"))
        .otherwise(pl.lit("private"))
    )

    lf = lf.with_columns([
        pl.col("city").cast(pl.Utf8, strict=False).fill_null("unknown").alias("city"),
        pl.col("category_name").cast(pl.Utf8, strict=False).fill_null("unknown").alias("category_name"),
        pl.col("microcat_name").cast(pl.Utf8, strict=False).fill_null("unknown").alias("microcat_name"),
        pl.col("delivery_service").cast(pl.Utf8, strict=False).fill_null("unknown").alias("delivery_service"),
        pl.col("is_pod").cast(pl.Utf8, strict=False).fill_null("unknown").alias("is_pod"),
        pl.col("payment_method").cast(pl.Utf8, strict=False).fill_null("unknown").alias("payment_method"),
        price.alias("order_price"),
        pl.col("platform_id").cast(pl.Float64, strict=False).alias("platform_id"),
        amount.alias("amount"),
    ])

    lf = lf.with_columns([
        delivery.alias("delivery_service_norm"),
        delivery.str.contains(LIBERAL_DELIVERY_RE).alias("delivery_is_liberal"),
        delivery.str.contains(STRICT_DELIVERY_RE).alias("delivery_is_strict"),
        delivery.str.contains(r"(?i)(avito|авито)").alias("delivery_is_avito"),
        delivery.str.contains(r"(?i)(yandex|яндекс|маркет)").alias("delivery_is_yandex_market"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("platform_id") == 1.0)
        .then(pl.lit("ios"))
        .when(pl.col("platform_id") == 2.0)
        .then(pl.lit("android"))
        .when(pl.col("platform_id") == 3.0)
        .then(pl.lit("web"))
        .when(pl.col("platform_id") == 4.0)
        .then(pl.lit("mav"))
        .otherwise(pl.lit("unknown"))
        .alias("platform_name"),

        pl.when(pl.col("platform_id").is_in([1.0, 2.0]))
        .then(pl.lit("mobile_app"))
        .when(pl.col("platform_id") == 3.0)
        .then(pl.lit("desktop_web"))
        .when(pl.col("platform_id") == 4.0)
        .then(pl.lit("mobile_web"))
        .otherwise(pl.lit("unknown"))
        .alias("platform_type"),

        pl.when(pl.col("platform_id") == 1.0)
        .then(pl.lit("ios"))
        .when(pl.col("platform_id") == 2.0)
        .then(pl.lit("android"))
        .when(pl.col("platform_id").is_in([3.0, 4.0]))
        .then(pl.lit("web"))
        .otherwise(pl.lit("unknown"))
        .alias("platform_os_family"),
    ])

    lf = lf.with_columns([
        title.str.len_chars().alias("title_len_chars"),
        title_norm.str.count_matches(r"\S+").alias("title_word_count"),
        title.str.count_matches(r"\d").alias("title_digit_count"),
    ])

    lf = lf.with_columns([
        (pl.col("title_digit_count") > 0).alias("has_digit"),
        safe_div(
            pl.col("title_len_chars").cast(pl.Float64, strict=False),
            pl.max_horizontal(pl.col("title_word_count").cast(pl.Float64, strict=False), pl.lit(1.0)),
        ).alias("avg_word_len"),
        *[
            title_lower.str.contains(pattern).fill_null(False).alias(name)
            for name, pattern in TITLE_KEYWORDS.items()
        ],
    ])

    lf = lf.with_columns([
        (pl.col("order_price") + 1.0).log(math.e).alias("log_order_price"),

        pl.when(pl.col("order_price").is_null())
        .then(pl.lit("unknown"))
        .otherwise(cut_expr("order_price", PRICE_BREAKS, PRICE_LABELS))
        .alias("order_price_bin"),

        (pl.col("order_price") < 500.0).alias("order_price_is_cheap"),
        (pl.col("order_price") >= 15_000.0).alias("order_price_is_plateau"),
    ])

    lf = lf.with_columns([
        pl.col("amount").is_not_null().alias("has_payment"),
        pl.col("amount").alias("payment_total"),
        (pl.col("amount") + 1.0).log(math.e).alias("log_payment_total"),
        (pl.col("amount") - pl.col("order_price")).alias("payment_minus_order_price"),
        safe_div(pl.col("amount"), pl.col("order_price")).alias("payment_to_order_price_ratio"),

        pl.when(pl.col("amount").is_null())
        .then(pl.lit("no_payment"))
        .otherwise(cut_expr("amount", PAYMENT_AMOUNT_BREAKS, PAYMENT_AMOUNT_LABELS))
        .alias("payment_total_bin"),

        pl.when(pl.col("amount").is_null())
        .then(None)
        .otherwise(pl.max_horizontal(pl.col("amount") - pl.col("order_price"), pl.lit(0.0)))
        .alias("delivery_fee"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("payment_minus_order_price").is_null())
        .then(pl.lit("no_payment"))
        .otherwise(cut_expr("payment_minus_order_price", PAYMENT_DIFF_BREAKS, PAYMENT_DIFF_LABELS))
        .alias("payment_minus_order_price_bin"),

        safe_div(pl.col("delivery_fee"), pl.col("amount")).alias("delivery_fee_share"),
    ])

    lf = lf.with_columns([
        pl.col("item_id").n_unique().over("deliveryorder_id").alias("n_items_in_order")
    ])

    lf = lf.with_columns([
        log1p_expr("n_items_in_order", "log_n_items_in_order"),
        cut_expr("n_items_in_order", BASKET_BREAKS, BASKET_LABELS).alias("n_items_bin"),
        (pl.col("n_items_in_order") > 1).alias("order_is_multi_item"),
    ])

    lf = lf.with_columns([
        pl.concat_str([
            pl.col("city").fill_null("unknown"),
            buyer_type,
        ], separator="_").alias("city_buyer_type")
    ])

    return lf if lazy else lf.collect()


def make_volume_stats(
    df: pl.DataFrame,
    group_col: str,
    total_n: int,
) -> pl.DataFrame:
    base = group_col.removesuffix("_name")

    return (
        df
        .group_by(group_col)
        .agg([
            pl.len().alias("_n")
        ])
        .with_columns([
            pl.col("_n").alias(f"{base}_order_count"),
            (pl.col("_n") / total_n).alias(f"{base}_dataset_share"),
            (pl.col("_n").cast(pl.Float64) + 1.0).log(math.e).alias(f"log_{base}_order_count"),
        ])
        .drop("_n")
    )


def make_return_rates(
    df: pl.DataFrame,
    group_col: str,
    global_rate: float,
    smoothing: float,
    suffix: str,
) -> pl.DataFrame:
    base = group_col.removesuffix("_name")

    return (
        df
        .group_by(group_col)
        .agg([
            pl.len().alias("_n"),
            pl.col(TARGET_COL).cast(pl.Float64).sum().alias("_s"),
        ])
        .with_columns([
            (pl.col("_s") / pl.col("_n")).alias(f"{base}_return_rate{suffix}"),
            ((pl.col("_s") + smoothing * global_rate) / (pl.col("_n") + smoothing)).alias(f"{base}_smoothed_return_rate{suffix}"),
        ])
        .drop(["_n", "_s"])
    )


def fit_category_stats(
    train_df: pl.DataFrame,
    n_folds: int = 5,
    smoothing: float = 100.0,
    seed: int = 42,
) -> dict[str, Any]:
    train_df = train_df.filter(pl.col(TARGET_COL).is_not_null())

    if ROW_ID_COL not in train_df.columns:
        train_df = train_df.with_row_index(ROW_ID_COL)

    total_n = len(train_df)
    global_rate = float(train_df[TARGET_COL].cast(pl.Float64).mean())

    category_stats = make_volume_stats(train_df, "category_name", total_n)
    microcat_stats = make_volume_stats(train_df, "microcat_name", total_n)

    category_supply = (
        train_df
        .group_by("category_name")
        .agg([
            pl.col("item_id").n_unique().alias("category_item_supply")
        ])
        .with_columns([
            log1p_expr("category_item_supply", "log_category_item_supply")
        ])
    )

    microcat_supply = (
        train_df
        .group_by("microcat_name")
        .agg([
            pl.col("item_id").n_unique().alias("microcat_item_supply")
        ])
        .with_columns([
            log1p_expr("microcat_item_supply", "log_microcat_item_supply")
        ])
    )

    microcat_category_map = (
        train_df
        .group_by(["microcat_name", "category_name"])
        .agg([
            pl.len().alias("_n")
        ])
        .sort("_n", descending=True)
        .unique(subset=["microcat_name"], keep="first")
        .select(["microcat_name", "category_name"])
    )

    category_stats = (
        category_stats
        .join(category_supply, on="category_name", how="left")
        .join(make_return_rates(train_df, "category_name", global_rate, smoothing, ""), on="category_name", how="left")
    )

    microcat_stats = (
        microcat_stats
        .join(microcat_supply, on="microcat_name", how="left")
        .join(microcat_category_map, on="microcat_name", how="left")
        .join(category_stats.select(["category_name", "category_order_count", "category_return_rate"]), on="category_name", how="left")
        .with_columns([
            safe_div(
                pl.col("microcat_order_count").cast(pl.Float64),
                pl.col("category_order_count").cast(pl.Float64),
            ).alias("microcat_share_inside_category")
        ])
        .drop("category_order_count")
        .join(make_return_rates(train_df, "microcat_name", global_rate, smoothing, ""), on="microcat_name", how="left")
        .with_columns([
            safe_div(
                pl.col("microcat_return_rate"),
                pl.col("category_return_rate"),
            ).alias("microcat_lift_vs_category")
        ])
        .drop(["category_name", "category_return_rate"])
    )

    train_df = add_group_folds(train_df, "deliveryorder_id", n_folds, seed)
    oof_parts = []

    for fold in range(n_folds):
        other = train_df.filter(pl.col("_fold") != fold)
        current = train_df.filter(pl.col("_fold") == fold)
        fold_rate = float(other[TARGET_COL].cast(pl.Float64).mean())

        category_oof = make_return_rates(other, "category_name", fold_rate, smoothing, "_oof")
        microcat_oof = make_return_rates(other, "microcat_name", fold_rate, smoothing, "_oof")

        part = (
            current
            .select([ROW_ID_COL, "category_name", "microcat_name"])
            .join(category_oof, on="category_name", how="left")
            .join(microcat_oof, on="microcat_name", how="left")
            .with_columns([
                pl.col("category_return_rate_oof").fill_null(fold_rate),
                pl.col("category_smoothed_return_rate_oof").fill_null(fold_rate),
                pl.col("microcat_return_rate_oof").fill_null(fold_rate),
                pl.col("microcat_smoothed_return_rate_oof").fill_null(fold_rate),
            ])
            .with_columns([
                safe_div(
                    pl.col("microcat_return_rate_oof"),
                    pl.col("category_return_rate_oof"),
                ).fill_null(1.0).alias("microcat_lift_vs_category_oof")
            ])
            .drop(["category_name", "microcat_name"])
        )

        oof_parts.append(part)

    return {
        "category_stats": category_stats,
        "microcat_stats": microcat_stats,
        "category_oof": pl.concat(oof_parts),
        "global_return_rate": global_rate,
    }


def fit_delivery_stats(
    train_df: pl.DataFrame,
    n_folds: int = 5,
    smoothing: float = 200.0,
    seed: int = 42,
) -> dict[str, Any]:
    train_df = train_df.filter(pl.col(TARGET_COL).is_not_null())

    if ROW_ID_COL not in train_df.columns:
        train_df = train_df.with_row_index(ROW_ID_COL)

    total_n = len(train_df)
    global_rate = float(train_df[TARGET_COL].cast(pl.Float64).mean())

    service_stats = (
        train_df
        .group_by("delivery_service_norm")
        .agg([
            pl.len().alias("_n"),
            pl.col(TARGET_COL).cast(pl.Float64).sum().alias("_s"),
        ])
        .with_columns([
            pl.col("_n").alias("delivery_service_order_count"),
            (pl.col("_n") / total_n).alias("delivery_service_dataset_share"),
            (pl.col("_n").cast(pl.Float64) + 1.0).log(math.e).alias("log_delivery_service_order_count"),
            (pl.col("_s") / pl.col("_n")).alias("delivery_service_return_rate"),
            ((pl.col("_s") + smoothing * global_rate) / (pl.col("_n") + smoothing)).alias("delivery_service_smoothed_return_rate"),
        ])
        .with_columns([
            safe_div(
                pl.col("delivery_service_return_rate"),
                pl.lit(global_rate),
            ).alias("delivery_service_lift_vs_global")
        ])
        .drop(["_n", "_s"])
    )

    train_df = add_group_folds(train_df, "deliveryorder_id", n_folds, seed)
    oof_parts = []

    for fold in range(n_folds):
        other = train_df.filter(pl.col("_fold") != fold)
        current = train_df.filter(pl.col("_fold") == fold)
        fold_rate = float(other[TARGET_COL].cast(pl.Float64).mean())

        rates = (
            other
            .group_by("delivery_service_norm")
            .agg([
                pl.len().alias("_n"),
                pl.col(TARGET_COL).cast(pl.Float64).sum().alias("_s"),
            ])
            .with_columns([
                (pl.col("_s") / pl.col("_n")).alias("delivery_service_return_rate_oof"),
                ((pl.col("_s") + smoothing * fold_rate) / (pl.col("_n") + smoothing)).alias("delivery_service_smoothed_return_rate_oof"),
            ])
            .drop(["_n", "_s"])
        )

        part = (
            current
            .select([ROW_ID_COL, "delivery_service_norm"])
            .join(rates, on="delivery_service_norm", how="left")
            .with_columns([
                pl.col("delivery_service_return_rate_oof").fill_null(fold_rate),
                pl.col("delivery_service_smoothed_return_rate_oof").fill_null(fold_rate),
            ])
            .with_columns([
                safe_div(
                    pl.col("delivery_service_return_rate_oof"),
                    pl.lit(fold_rate),
                ).fill_null(1.0).alias("delivery_service_lift_vs_global_oof")
            ])
            .drop("delivery_service_norm")
        )

        oof_parts.append(part)

    return {
        "service_stats": service_stats,
        "delivery_oof": pl.concat(oof_parts),
        "global_return_rate": global_rate,
    }


def fit_feature_artifacts(train: pl.LazyFrame) -> dict[str, Any]:
    train_static = apply_static_features(train)

    train_df = (
        train_static
        .select([
            ROW_ID_COL,
            "deliveryorder_id",
            "item_id",
            "category_name",
            "microcat_name",
            "delivery_service_norm",
            TARGET_COL,
        ])
        .filter(pl.col(TARGET_COL).is_not_null())
        .collect()
    )

    category_stats = fit_category_stats(train_df)
    delivery_stats = fit_delivery_stats(train_df)

    return {
        "category": category_stats,
        "delivery": delivery_stats,
    }


def apply_target_encoding_features(
    df: pl.LazyFrame,
    artifacts: dict[str, Any],
    is_train: bool,
) -> pl.LazyFrame:
    lf = df

    category = artifacts["category"]
    delivery = artifacts["delivery"]

    category_stats = category["category_stats"]
    microcat_stats = category["microcat_stats"]
    service_stats = delivery["service_stats"]

    lf = (
        lf
        .join(
            category_stats
            .select([
                "category_name",
                "category_order_count",
                "category_dataset_share",
                "log_category_order_count",
                "category_item_supply",
                "log_category_item_supply",
            ])
            .lazy(),
            on="category_name",
            how="left",
        )
        .join(
            microcat_stats
            .select([
                "microcat_name",
                "microcat_order_count",
                "microcat_dataset_share",
                "log_microcat_order_count",
                "microcat_item_supply",
                "log_microcat_item_supply",
                "microcat_share_inside_category",
            ])
            .lazy(),
            on="microcat_name",
            how="left",
        )
        .join(
            service_stats
            .select([
                "delivery_service_norm",
                "delivery_service_order_count",
                "delivery_service_dataset_share",
                "log_delivery_service_order_count",
            ])
            .lazy(),
            on="delivery_service_norm",
            how="left",
        )
    )

    global_category_rate = category["global_return_rate"]
    global_delivery_rate = delivery["global_return_rate"]

    if is_train:
        lf = (
            lf
            .join(category["category_oof"].lazy(), on=ROW_ID_COL, how="left")
            .join(delivery["delivery_oof"].lazy(), on=ROW_ID_COL, how="left")
            .with_columns([
                pl.col("category_return_rate_oof").fill_null(global_category_rate).alias("category_return_rate_feature"),
                pl.col("category_smoothed_return_rate_oof").fill_null(global_category_rate).alias("category_smoothed_return_rate_feature"),
                pl.col("microcat_return_rate_oof").fill_null(global_category_rate).alias("microcat_return_rate_feature"),
                pl.col("microcat_smoothed_return_rate_oof").fill_null(global_category_rate).alias("microcat_smoothed_return_rate_feature"),
                pl.col("microcat_lift_vs_category_oof").fill_null(1.0).alias("microcat_lift_vs_category_feature"),
                pl.col("delivery_service_return_rate_oof").fill_null(global_delivery_rate).alias("delivery_service_return_rate_feature"),
                pl.col("delivery_service_smoothed_return_rate_oof").fill_null(global_delivery_rate).alias("delivery_service_smoothed_return_rate_feature"),
                pl.col("delivery_service_lift_vs_global_oof").fill_null(1.0).alias("delivery_service_lift_vs_global_feature"),
            ])
        )
    else:
        lf = (
            lf
            .join(
                category_stats
                .select([
                    "category_name",
                    "category_return_rate",
                    "category_smoothed_return_rate",
                ])
                .lazy(),
                on="category_name",
                how="left",
            )
            .join(
                microcat_stats
                .select([
                    "microcat_name",
                    "microcat_return_rate",
                    "microcat_smoothed_return_rate",
                    "microcat_lift_vs_category",
                ])
                .lazy(),
                on="microcat_name",
                how="left",
            )
            .join(
                service_stats
                .select([
                    "delivery_service_norm",
                    "delivery_service_return_rate",
                    "delivery_service_smoothed_return_rate",
                    "delivery_service_lift_vs_global",
                ])
                .lazy(),
                on="delivery_service_norm",
                how="left",
            )
            .with_columns([
                pl.col("category_return_rate").fill_null(global_category_rate).alias("category_return_rate_feature"),
                pl.col("category_smoothed_return_rate").fill_null(global_category_rate).alias("category_smoothed_return_rate_feature"),
                pl.col("microcat_return_rate").fill_null(global_category_rate).alias("microcat_return_rate_feature"),
                pl.col("microcat_smoothed_return_rate").fill_null(global_category_rate).alias("microcat_smoothed_return_rate_feature"),
                pl.col("microcat_lift_vs_category").fill_null(1.0).alias("microcat_lift_vs_category_feature"),
                pl.col("delivery_service_return_rate").fill_null(global_delivery_rate).alias("delivery_service_return_rate_feature"),
                pl.col("delivery_service_smoothed_return_rate").fill_null(global_delivery_rate).alias("delivery_service_smoothed_return_rate_feature"),
                pl.col("delivery_service_lift_vs_global").fill_null(1.0).alias("delivery_service_lift_vs_global_feature"),
            ])
        )

    lf = lf.with_columns([
        pl.col("category_order_count").fill_null(0),
        pl.col("category_dataset_share").fill_null(0.0),
        pl.col("log_category_order_count").fill_null(0.0),
        pl.col("category_item_supply").fill_null(0),
        pl.col("log_category_item_supply").fill_null(0.0),
        pl.col("microcat_order_count").fill_null(0),
        pl.col("microcat_dataset_share").fill_null(0.0),
        pl.col("log_microcat_order_count").fill_null(0.0),
        pl.col("microcat_item_supply").fill_null(0),
        pl.col("log_microcat_item_supply").fill_null(0.0),
        pl.col("microcat_share_inside_category").fill_null(0.0),
        pl.col("delivery_service_order_count").fill_null(0),
        pl.col("delivery_service_dataset_share").fill_null(0.0),
        pl.col("log_delivery_service_order_count").fill_null(0.0),
    ])

    return lf


def apply_feature_engineering(
    df: pl.LazyFrame,
    artifacts: dict[str, Any] | None = None,
    is_train: bool = False,
) -> pl.LazyFrame:
    lf = apply_static_features(df)

    if artifacts is not None:
        lf = apply_target_encoding_features(lf, artifacts, is_train=is_train)

    return lf


def remove_forbidden_features(features: list[str]) -> list[str]:
    return [
        feature
        for feature in features
        if feature not in FORBIDDEN_FEATURES
    ]


def get_existing_features(
    lf: pl.LazyFrame,
    features: list[str],
) -> tuple[list[str], list[str]]:
    columns = get_columns(lf)

    existing = [
        feature
        for feature in features
        if feature in columns and feature not in FORBIDDEN_FEATURES
    ]

    missing = [
        feature
        for feature in features
        if feature not in columns and feature not in FORBIDDEN_FEATURES
    ]

    return existing, missing


def fill_nulls_for_catboost(
    lf: pl.LazyFrame,
    features: list[str],
) -> pl.LazyFrame:
    schema = lf.collect_schema()
    expressions = []

    for feature in features:
        dtype = schema[feature]
        dtype_name = str(dtype)

        if dtype == pl.String:
            expressions.append(pl.col(feature).fill_null("unknown").alias(feature))
        elif dtype == pl.Boolean:
            expressions.append(pl.col(feature).fill_null(False).alias(feature))
        elif dtype_name.startswith("Datetime") or dtype_name == "Date":
            expressions.append(pl.col(feature).cast(pl.Int64, strict=False).fill_null(-1).alias(feature))
        else:
            expressions.append(pl.col(feature).cast(pl.Float64, strict=False).fill_null(-1.0).alias(feature))

    if not expressions:
        return lf

    return lf.with_columns(expressions)


def get_categorical_features(
    lf: pl.LazyFrame,
    features: list[str],
) -> list[str]:
    schema = lf.collect_schema()
    categorical = []

    for feature in features:
        dtype = schema[feature]

        if dtype == pl.String or dtype == pl.Boolean:
            categorical.append(feature)

    return categorical


def prepare_train_valid_for_catboost(
    train: pl.LazyFrame,
    valid: pl.LazyFrame,
    requested_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str]]:
    features = requested_features or FINAL_FEATURES
    features = remove_forbidden_features(features)

    train = ensure_row_id(train)
    valid = ensure_row_id(valid)

    artifacts = fit_feature_artifacts(train)

    train_fe = apply_feature_engineering(train, artifacts=artifacts, is_train=True)
    valid_fe = apply_feature_engineering(valid, artifacts=artifacts, is_train=False)

    train_features, missing_train = get_existing_features(train_fe, features)
    valid_features, missing_valid = get_existing_features(valid_fe, features)

    final_features = [
        feature
        for feature in train_features
        if feature in valid_features
    ]

    missing_features = sorted(set(missing_train + missing_valid))

    if missing_features:
        print("Missing features will be skipped:")
        for feature in missing_features:
            print(feature)

    train_fe = fill_nulls_for_catboost(train_fe, final_features)
    valid_fe = fill_nulls_for_catboost(valid_fe, final_features)

    cat_features = get_categorical_features(train_fe, final_features)

    train_df = train_fe.select(final_features + [TARGET_COL]).collect().to_pandas()
    valid_df = valid_fe.select(final_features + [TARGET_COL]).collect().to_pandas()

    X_train = train_df[final_features]
    y_train = train_df[TARGET_COL].astype(int)

    X_valid = valid_df[final_features]
    y_valid = valid_df[TARGET_COL].astype(int)

    return X_train, y_train, X_valid, y_valid, final_features, cat_features


def prepare_train_test_for_catboost(
    train: pl.LazyFrame,
    test: pl.LazyFrame,
    requested_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], list[str], dict[str, Any]]:
    features = requested_features or FINAL_FEATURES
    features = remove_forbidden_features(features)

    train = ensure_row_id(train)
    test = ensure_row_id(test)

    artifacts = fit_feature_artifacts(train)

    train_fe = apply_feature_engineering(train, artifacts=artifacts, is_train=True)
    test_fe = apply_feature_engineering(test, artifacts=artifacts, is_train=False)

    train_features, missing_train = get_existing_features(train_fe, features)
    test_features, missing_test = get_existing_features(test_fe, features)

    final_features = [
        feature
        for feature in train_features
        if feature in test_features
    ]

    missing_features = sorted(set(missing_train + missing_test))

    if missing_features:
        print("Missing features will be skipped:")
        for feature in missing_features:
            print(feature)

    train_fe = fill_nulls_for_catboost(train_fe, final_features)
    test_fe = fill_nulls_for_catboost(test_fe, final_features)

    cat_features = get_categorical_features(train_fe, final_features)

    train_df = train_fe.select(final_features + [TARGET_COL]).collect().to_pandas()
    test_df = test_fe.select(final_features).collect().to_pandas()

    X_train = train_df[final_features]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[final_features]

    return X_train, y_train, X_test, final_features, cat_features, artifacts