"""
Подготовка тестового датасета.

Логика:
  seller_terminal IS NULL     → is_return = False  (правило, записываем сразу)
  seller_terminal IS NOT NULL → is_return = null   (нужна ML-модель)

Вывод: data/parquet/test_final.parquet
  - все строки теста с фичами
  - is_return: False для нетерминальных, null для терминальных
"""
import polars as pl
from pathlib import Path

BASE     = Path(__file__).parent
TEST_DIR = BASE / "test_data"
OUT      = BASE / "data" / "parquet" / "test_final.parquet"

# Цена: те же перцентили, что использовались при сборке трейна
PRICE_P01 = 10.0
PRICE_P99 = 30_000.0

# Колонки из orders, которые берём дальше
ORDER_COLS = [
    "deliveryorder_id", "order_create_date", "item_id",
    "seller_terminal", "delivery_service",
    "is_pod", "buyer_id", "seller_id", "order_price", "city",
    "platform_id", "is_return",
]

print("Loading test tables...", flush=True)
items    = pl.scan_parquet(TEST_DIR / "items.parquet")
payments = pl.scan_parquet(TEST_DIR / "payments.parquet").unique(subset=["deliveryorder_id"], keep="last")
users    = pl.scan_parquet(TEST_DIR / "users.parquet")

# Проверяем схему items заранее — имена колонок могут отличаться от трейна
_items_cols = set(items.collect_schema().names())
_items_rename = {"StartTime": "starttime"} if "StartTime" in _items_cols else {}
_items_drop   = [c for c in ("Close_Date", "close_date") if c in _items_cols]
if _items_rename or _items_drop:
    items = items.rename(_items_rename).drop(_items_drop)
    print(f"  items: renamed={_items_rename}, dropped={_items_drop}", flush=True)

print("Preparing test orders...", flush=True)
orders_lf = (
    pl.scan_parquet(TEST_DIR / "orders.parquet")
    # Лик-колонки всё равно пустые, но убираем для чистоты
    .drop(["cancel_date", "order_accept_date", "buyer_terminal"])
    # Приводим типы к тем же, что в трейне
    .with_columns([
        pl.col("order_price").cast(pl.Float64),
        pl.col("platform_id").cast(pl.Float64),
        pl.col("is_pod").cast(pl.Utf8),
        pl.col("order_create_date").cast(pl.Utf8),
    ])
    # Правило: нет терминала → is_return = False
    .with_columns(
        pl.when(pl.col("seller_terminal").is_null())
        .then(pl.lit(False))
        .otherwise(pl.lit(None).cast(pl.Boolean))
        .alias("is_return")
    )
    .select(ORDER_COLS)
)

print("Joining with lookup tables...", flush=True)
dataset_lf = (
    orders_lf
    .join(items,    on="item_id",          how="left", suffix="_item")
    .join(users,    left_on="buyer_id",    right_on="user_id", how="left", suffix="_buyer")
    .join(users,    left_on="seller_id",   right_on="user_id", how="left", suffix="_seller")
    .join(payments, on="deliveryorder_id", how="left", suffix="_payment")
    .drop(["seller_terminal"])
    .with_columns([
        pl.col("delivery_service").fill_null("unknown"),
        pl.col("order_price").clip(PRICE_P01, PRICE_P99),
    ])
)

print(f"Writing to {OUT} (streaming)...", flush=True)
dataset_lf.sink_parquet(str(OUT), compression="snappy")

# Статистика
lf = pl.scan_parquet(OUT)
n = lf.select(pl.len()).collect().item()
n_false  = lf.filter(pl.col("is_return") == False).select(pl.len()).collect().item()
n_null   = lf.filter(pl.col("is_return").is_null()).select(pl.len()).collect().item()
cols = len(lf.collect_schema())

print(f"\nOutput: {OUT.name}  rows={n:,}  cols={cols}")
print(f"  is_return=False (rule): {n_false:,} ({n_false/n*100:.1f}%)")
print(f"  is_return=null  (ML):   {n_null:,}  ({n_null/n*100:.1f}%)")
print("Done.")
