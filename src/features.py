"""Variables del baseline: solo información conocida al momento de la compra.

Se usan:
  - fecha de compra (mes, día de semana, hora)
  - plazo prometido: días entre la compra y la fecha estimada (Olist lo muestra al comprar)
  - carrito: cantidad de ítems, de vendedores, precio y flete totales
  - producto: peso y volumen totales
  - geografía: estado del cliente y si algún vendedor está en otro estado
  - pago: tipo de pago principal y cantidad máxima de cuotas

Se excluyen por fuga (ocurren después de la compra o dependen del resultado):
  order_approved_at, order_delivered_carrier_date, order_delivered_customer_date,
  order_status, reviews y cualquier duración calculada con esas fechas.
  shipping_limit_date también queda afuera por precaución: no está claro si
  se actualiza después de la compra.
"""

import pandas as pd

from src.target import TARGET_COL

PURCHASE_COL = "order_purchase_timestamp"

NUMERIC_FEATURES = [
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "promised_days",
    "n_items",
    "n_sellers",
    "total_price",
    "total_freight",
    "freight_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "max_installments",
]
CATEGORICAL_FEATURES = ["customer_state", "main_payment_type", "seller_other_state"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LEAKY_COLUMNS = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_status",
    "shipping_limit_date",
    "review_score",
]


def _items_features(items: pd.DataFrame, products: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    enriched = (
        items.merge(products, on="product_id", how="left")
        .merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
        .assign(
            volume_cm3=lambda d: d["product_length_cm"] * d["product_height_cm"] * d["product_width_cm"]
        )
    )
    return enriched.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        seller_states=("seller_state", lambda s: frozenset(s.dropna())),
    )


def _payments_features(payments: pd.DataFrame) -> pd.DataFrame:
    main_type = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")
        .set_index("order_id")["payment_type"]
        .rename("main_payment_type")
    )
    installments = payments.groupby("order_id")["payment_installments"].max().rename("max_installments")
    return pd.concat([main_type, installments], axis=1)


def build_features(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    customers: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """Una fila por pedido con order_id, fecha de compra, FEATURES y el target (si está en orders)."""
    purchase = pd.to_datetime(orders[PURCHASE_COL])
    estimated = pd.to_datetime(orders["order_estimated_delivery_date"])

    base = orders[["order_id", "customer_id"]].assign(
        **{
            PURCHASE_COL: purchase,
            "purchase_month": purchase.dt.month,
            "purchase_dayofweek": purchase.dt.dayofweek,
            "purchase_hour": purchase.dt.hour,
            "promised_days": (estimated.dt.normalize() - purchase.dt.normalize()).dt.days,
        }
    )
    if TARGET_COL in orders:
        base = base.assign(**{TARGET_COL: orders[TARGET_COL].to_numpy()})

    table = (
        base.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
        .merge(_items_features(items, products, sellers), left_on="order_id", right_index=True, how="left")
        .merge(_payments_features(payments), left_on="order_id", right_index=True, how="left")
    )

    seller_other_state = [
        "unknown" if not isinstance(states, frozenset) or not states
        else str(any(s != cust for s in states))
        for states, cust in zip(table["seller_states"], table["customer_state"])
    ]

    return table.assign(
        freight_ratio=table["total_freight"] / table["total_price"].where(table["total_price"] > 0),
        seller_other_state=seller_other_state,
    ).drop(columns=["seller_states", "customer_id"])
