import polars as pl


TARGET_COL = "is_return"

FORBIDDEN_FEATURES = {
    "is_return",
    "order_accept_date",
    "cancel_date",
}


def get_columns(lf: pl.LazyFrame) -> set[str]:
    return set(lf.collect_schema().names())


def has_columns(lf: pl.LazyFrame, columns: list[str]) -> bool:
    existing = get_columns(lf)
    return all(column in existing for column in columns)


def add_basic_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    columns = get_columns(lf)
    expressions = []

    if "order_create_date" in columns:
        expressions.extend([
            pl.col("order_create_date").dt.weekday().alias("order_weekday"),
            pl.col("order_create_date").dt.day().alias("order_day"),
            pl.col("order_create_date").dt.month().alias("order_month"),
        ])

    if "title" in columns:
        expressions.append(
            pl.col("title")
            .cast(pl.Utf8, strict=False)
            .str.len_chars()
            .alias("title_len")
        )

    if "starttime" in columns and "order_create_date" in columns:
        expressions.append(
            (pl.col("order_create_date") - pl.col("starttime"))
            .dt.total_days()
            .alias("item_age_days")
        )

    if "close_date" in columns:
        expressions.append(
            pl.col("close_date").is_not_null().alias("has_close_date")
        )

    if "close_date" in columns and "order_create_date" in columns:
        expressions.append(
            (pl.col("close_date") - pl.col("order_create_date"))
            .dt.total_days()
            .alias("days_to_close_after_order")
        )

    if "buyer_registrationtime" in columns and "order_create_date" in columns:
        expressions.append(
            (pl.col("order_create_date") - pl.col("buyer_registrationtime"))
            .dt.total_days()
            .alias("buyer_account_age_days")
        )

    if "seller_registrationtime" in columns and "order_create_date" in columns:
        expressions.append(
            (pl.col("order_create_date") - pl.col("seller_registrationtime"))
            .dt.total_days()
            .alias("seller_account_age_days")
        )

    if "buyer_firstlistingdate" in columns and "order_create_date" in columns:
        expressions.append(
            (pl.col("order_create_date") - pl.col("buyer_firstlistingdate"))
            .dt.total_days()
            .alias("buyer_firstlisting_age_days")
        )

    if "seller_firstlistingdate" in columns and "order_create_date" in columns:
        expressions.append(
            (pl.col("order_create_date") - pl.col("seller_firstlistingdate"))
            .dt.total_days()
            .alias("seller_firstlisting_age_days")
        )

    if "payment_total" in columns and "order_price" in columns:
        expressions.extend([
            (pl.col("payment_total") - pl.col("order_price")).alias("payment_minus_order_price"),
            pl.when(pl.col("order_price") > 0)
            .then(pl.col("payment_total") / pl.col("order_price"))
            .otherwise(None)
            .alias("payment_to_price_ratio"),
        ])

    if "first_created_txtime" in columns and "order_create_date" in columns:
        expressions.append(
            (pl.col("first_created_txtime") - pl.col("order_create_date"))
            .dt.total_days()
            .alias("payment_delay_days")
        )

    if "order_price" in columns:
        expressions.append(
            pl.when(pl.col("order_price") < 500)
            .then(pl.lit("very_low"))
            .when(pl.col("order_price") < 1500)
            .then(pl.lit("low"))
            .when(pl.col("order_price") < 5000)
            .then(pl.lit("middle"))
            .when(pl.col("order_price") < 15000)
            .then(pl.lit("high"))
            .otherwise(pl.lit("very_high"))
            .alias("price_bucket")
        )

    if not expressions:
        return lf

    return lf.with_columns(expressions)


def add_order_count_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    columns = get_columns(lf)

    if "deliveryorder_id" not in columns:
        return lf

    order_counts = (
        lf
        .group_by("deliveryorder_id")
        .agg(pl.len().alias("items_count_in_order"))
    )

    result = lf.join(order_counts, on="deliveryorder_id", how="left")

    if "category_name" in columns:
        category_counts = (
            lf
            .group_by(["deliveryorder_id", "category_name"])
            .agg(pl.len().alias("same_category_items_count_in_order"))
        )

        result = result.join(
            category_counts,
            on=["deliveryorder_id", "category_name"],
            how="left",
        )

    if "microcat_name" in columns:
        microcat_counts = (
            lf
            .group_by(["deliveryorder_id", "microcat_name"])
            .agg(pl.len().alias("same_microcat_items_count_in_order"))
        )

        result = result.join(
            microcat_counts,
            on=["deliveryorder_id", "microcat_name"],
            how="left",
        )

    if "order_price" in columns:
        order_price_sum = (
            lf
            .group_by("deliveryorder_id")
            .agg(pl.col("order_price").sum().alias("order_total_price"))
        )

        result = result.join(order_price_sum, on="deliveryorder_id", how="left")

        result = result.with_columns([
            pl.when(pl.col("order_total_price") > 0)
            .then(pl.col("order_price") / pl.col("order_total_price"))
            .otherwise(None)
            .alias("item_price_share_in_order")
        ])

    return result


def build_history_agg(train: pl.LazyFrame, keys: list[str], prefix: str) -> pl.LazyFrame:
    return (
        train
        .group_by(keys)
        .agg([
            pl.len().alias(f"{prefix}_orders_count"),
            pl.col(TARGET_COL).cast(pl.Int32).sum().alias(f"{prefix}_returns_count"),
        ])
        .with_columns([
            pl.when(pl.col(f"{prefix}_orders_count") > 0)
            .then(pl.col(f"{prefix}_returns_count") / pl.col(f"{prefix}_orders_count"))
            .otherwise(0.0)
            .alias(f"{prefix}_return_rate")
        ])
    )


def add_history_features(
    train: pl.LazyFrame,
    valid: pl.LazyFrame,
    keys: list[str],
    prefix: str,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    if not has_columns(train, keys + [TARGET_COL]):
        return train, valid

    agg = build_history_agg(train, keys, prefix)

    train_with_agg = train.join(agg, on=keys, how="left")
    valid_with_agg = valid.join(agg, on=keys, how="left")

    train_with_agg = train_with_agg.with_columns([
        (pl.col(f"{prefix}_orders_count") - 1)
        .clip(lower_bound=0)
        .alias(f"{prefix}_orders_count"),
        (pl.col(f"{prefix}_returns_count") - pl.col(TARGET_COL).cast(pl.Int32))
        .clip(lower_bound=0)
        .alias(f"{prefix}_returns_count"),
    ])

    train_with_agg = train_with_agg.with_columns([
        pl.when(pl.col(f"{prefix}_orders_count") > 0)
        .then(pl.col(f"{prefix}_returns_count") / pl.col(f"{prefix}_orders_count"))
        .otherwise(0.0)
        .alias(f"{prefix}_return_rate")
    ])

    valid_with_agg = valid_with_agg.with_columns([
        pl.col(f"{prefix}_orders_count").fill_null(0),
        pl.col(f"{prefix}_returns_count").fill_null(0),
        pl.col(f"{prefix}_return_rate").fill_null(0.0),
    ])

    return train_with_agg, valid_with_agg


def add_frequency_features(
    train: pl.LazyFrame,
    valid: pl.LazyFrame,
    key: str,
    prefix: str,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    if key not in get_columns(train):
        return train, valid

    total_rows = train.select(pl.len().alias("rows")).collect().item()

    freq = (
        train
        .group_by(key)
        .agg(pl.len().alias(f"{prefix}_orders_count"))
        .with_columns([
            (pl.col(f"{prefix}_orders_count") / total_rows).alias(f"{prefix}_frequency"),
            (pl.col(f"{prefix}_orders_count") < 1000).alias(f"is_rare_{prefix}"),
        ])
    )

    train = train.join(freq, on=key, how="left")
    valid = valid.join(freq, on=key, how="left")

    valid = valid.with_columns([
        pl.col(f"{prefix}_orders_count").fill_null(0),
        pl.col(f"{prefix}_frequency").fill_null(0.0),
        pl.col(f"is_rare_{prefix}").fill_null(True),
    ])

    return train, valid


def add_buyer_microcat_features(
    train: pl.LazyFrame,
    valid: pl.LazyFrame,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    keys = ["buyer_id", "microcat_name"]

    if not has_columns(train, keys + [TARGET_COL]):
        return train, valid

    agg = (
        train
        .group_by(keys)
        .agg([
            pl.len().alias("buyer_microcat_orders_count"),
            pl.col(TARGET_COL).cast(pl.Int32).sum().alias("buyer_microcat_returns_count"),
        ])
        .with_columns([
            pl.when(pl.col("buyer_microcat_orders_count") > 0)
            .then(pl.col("buyer_microcat_returns_count") / pl.col("buyer_microcat_orders_count"))
            .otherwise(0.0)
            .alias("buyer_microcat_return_rate")
        ])
    )

    train = train.join(agg, on=keys, how="left")
    valid = valid.join(agg, on=keys, how="left")

    train = train.with_columns([
        pl.when(pl.col("buyer_microcat_orders_count") <= 1)
        .then(True)
        .otherwise(False)
        .alias("buyer_is_first_microcat_order")
    ])

    valid = valid.with_columns([
        pl.col("buyer_microcat_orders_count").fill_null(0),
        pl.col("buyer_microcat_returns_count").fill_null(0),
        pl.col("buyer_microcat_return_rate").fill_null(0.0),
        pl.when(pl.col("buyer_microcat_orders_count").fill_null(0) == 0)
        .then(True)
        .otherwise(False)
        .alias("buyer_is_first_microcat_order")
    ])

    return train, valid


def add_requested_features(
    train: pl.LazyFrame,
    valid: pl.LazyFrame,
    requested_features: list[str],
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    train = add_basic_features(train)
    valid = add_basic_features(valid)

    order_count_features = {
        "items_count_in_order",
        "same_category_items_count_in_order",
        "same_microcat_items_count_in_order",
        "item_price_share_in_order",
    }

    if any(feature in requested_features for feature in order_count_features):
        train = add_order_count_features(train)
        valid = add_order_count_features(valid)

    seller_history_features = {
        "seller_orders_count",
        "seller_returns_count",
        "seller_return_rate",
        "seller_return_rate_7d",
        "seller_return_rate_14d",
        "seller_return_rate_30d",
        "seller_is_bad_actor",
    }

    if any(feature in requested_features for feature in seller_history_features):
        train, valid = add_history_features(train, valid, ["seller_id"], "seller")

    buyer_history_features = {
        "buyer_orders_count",
        "buyer_returns_count",
        "buyer_return_rate",
        "buyer_return_rate_7d",
        "buyer_return_rate_14d",
        "buyer_return_rate_30d",
        "buyer_is_bad_actor",
    }

    if any(feature in requested_features for feature in buyer_history_features):
        train, valid = add_history_features(train, valid, ["buyer_id"], "buyer")

    category_history_features = {
        "category_orders_count",
        "category_returns_count",
        "category_return_rate",
    }

    if any(feature in requested_features for feature in category_history_features):
        train, valid = add_history_features(train, valid, ["category_name"], "category")

    microcat_history_features = {
        "microcat_orders_count",
        "microcat_returns_count",
        "microcat_return_rate",
    }

    if any(feature in requested_features for feature in microcat_history_features):
        train, valid = add_history_features(train, valid, ["microcat_name"], "microcat")

    delivery_history_features = {
        "delivery_service_orders_count",
        "delivery_service_returns_count",
        "delivery_service_return_rate",
    }

    if any(feature in requested_features for feature in delivery_history_features):
        train, valid = add_history_features(train, valid, ["delivery_service"], "delivery_service")

    city_history_features = {
        "city_orders_count",
        "city_returns_count",
        "city_return_rate",
    }

    if any(feature in requested_features for feature in city_history_features):
        train, valid = add_history_features(train, valid, ["city"], "city")

    category_frequency_features = {
        "category_frequency",
        "is_rare_category",
    }

    if any(feature in requested_features for feature in category_frequency_features):
        train, valid = add_frequency_features(train, valid, "category_name", "category")

    microcat_frequency_features = {
        "microcat_frequency",
        "is_rare_microcat",
    }

    if any(feature in requested_features for feature in microcat_frequency_features):
        train, valid = add_frequency_features(train, valid, "microcat_name", "microcat")

    buyer_microcat_features = {
        "buyer_microcat_orders_count",
        "buyer_microcat_returns_count",
        "buyer_microcat_return_rate",
        "buyer_is_first_microcat_order",
    }

    if any(feature in requested_features for feature in buyer_microcat_features):
        train, valid = add_buyer_microcat_features(train, valid)

    return train, valid


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

        if dtype == pl.String:
            expressions.append(
                pl.col(feature).fill_null("unknown").alias(feature)
            )
        elif dtype == pl.Boolean:
            expressions.append(
                pl.col(feature).fill_null(False).alias(feature)
            )
        else:
            expressions.append(
                pl.col(feature).fill_null(-1).alias(feature)
            )

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
    requested_features: list[str],
) -> tuple:
    train, valid = add_requested_features(train, valid, requested_features)

    requested_features = remove_forbidden_features(requested_features)

    train_features, missing_train = get_existing_features(train, requested_features)
    valid_features, missing_valid = get_existing_features(valid, requested_features)

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

    train = fill_nulls_for_catboost(train, final_features)
    valid = fill_nulls_for_catboost(valid, final_features)

    cat_features = get_categorical_features(train, final_features)

    train_df = train.select(final_features + [TARGET_COL]).collect().to_pandas()
    valid_df = valid.select(final_features + [TARGET_COL]).collect().to_pandas()

    X_train = train_df[final_features]
    y_train = train_df[TARGET_COL].astype(int)

    X_valid = valid_df[final_features]
    y_valid = valid_df[TARGET_COL].astype(int)

    return X_train, y_train, X_valid, y_valid, final_features, cat_features