"""Variables del baseline: solo información conocida en el momento de predicción.

Momento de predicción (anchor):
  - "aprobacion" (por defecto, alineado al README): cuando Olist aprueba el pago.
    Se descartan los pedidos sin order_approved_at y se agrega como variable las
    horas entre la compra y la aprobación.
  - "compra": cuando el cliente compra. Más estricto: no usa order_approved_at.

Se usan:
  - fecha de compra (mes, día de semana, hora)
  - plazo prometido: días entre la compra y la fecha estimada (Olist lo muestra al comprar)
  - carrito: cantidad de ítems, de vendedores, precio y flete totales
  - producto: peso y volumen totales
  - geografía: estado del cliente y si algún vendedor está en otro estado
  - pago: tipo de pago principal y cantidad máxima de cuotas

Se excluyen por fuga (ocurren después del momento de predicción o dependen del resultado):
  order_delivered_carrier_date, order_delivered_customer_date, order_status,
  reviews y cualquier duración calculada con esas fechas. Con anchor "compra"
  también order_approved_at. shipping_limit_date queda afuera por precaución:
  no está claro si se actualiza después de la compra.
"""

import pandas as pd

from src.target import TARGET_COL

PURCHASE_COL = "order_purchase_timestamp"
APPROVED_COL = "order_approved_at"
PREDICTION_TIME_COL = "prediction_time"

ANCHORS = {"aprobacion": APPROVED_COL, "compra": PURCHASE_COL}
DEFAULT_ANCHOR = "aprobacion"

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
APPROVAL_FEATURES = ["approval_hours"]

LEAKY_COLUMNS = [
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_status",
    "shipping_limit_date",
    "review_score",
]


def _check_anchor(anchor: str) -> None:
    if anchor not in ANCHORS:
        raise ValueError(f"anchor debe ser uno de {list(ANCHORS)}, no {anchor!r}")


def numeric_features(anchor: str = DEFAULT_ANCHOR) -> list[str]:
    _check_anchor(anchor)
    return NUMERIC_FEATURES + (APPROVAL_FEATURES if anchor == "aprobacion" else [])


def feature_columns(anchor: str = DEFAULT_ANCHOR) -> list[str]:
    return numeric_features(anchor) + CATEGORICAL_FEATURES


def leaky_columns(anchor: str = DEFAULT_ANCHOR) -> list[str]:
    """Columnas que no pueden ser variables porque se conocen después del momento de predicción."""
    _check_anchor(anchor)
    return LEAKY_COLUMNS + ([APPROVED_COL] if anchor == "compra" else [])


def _items_features(
    items: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
) -> pd.DataFrame:
    enriched = (
        items.merge(products, on="product_id", how="left")
        .merge(
            sellers[["seller_id", "seller_state"]],
            on="seller_id",
            how="left",
        )
    )

    physical_columns = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    # Los valores físicos iguales o menores que cero no son válidos.
    # Se conservan como faltantes para que el pipeline los impute.
    enriched[physical_columns] = enriched[physical_columns].mask(
        enriched[physical_columns] <= 0
    )

    enriched = enriched.assign(
        volume_cm3=lambda d: (
            d["product_length_cm"]
            * d["product_height_cm"]
            * d["product_width_cm"]
        )
    )

    return enriched.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=(
            "product_weight_g",
            lambda values: values.sum(min_count=len(values)),
        ),
        total_volume_cm3=(
            "volume_cm3",
            lambda values: values.sum(min_count=len(values)),
        ),
        seller_states=("seller_state", lambda values: frozenset(values.dropna())),
    )


def _payments_features(payments: pd.DataFrame) -> pd.DataFrame:
    main_type = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")
        .set_index("order_id")["payment_type"]
        .rename("main_payment_type")
    )

    valid_installments = payments.assign(
        payment_installments=payments["payment_installments"].mask(
            payments["payment_installments"] <= 0
        )
    )

    installments = (
        valid_installments.groupby("order_id")["payment_installments"]
        .max()
        .rename("max_installments")
    )

    return pd.concat([main_type, installments], axis=1)


def build_features(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    customers: pd.DataFrame,
    payments: pd.DataFrame,
    anchor: str = DEFAULT_ANCHOR,
) -> pd.DataFrame:
    """Una fila por pedido con order_id, prediction_time, las variables del anchor y el target.

    Con anchor "aprobacion" se descartan los pedidos sin fecha de aprobación.
    """
    _check_anchor(anchor)
    purchase = pd.to_datetime(orders[PURCHASE_COL])
    approved = pd.to_datetime(orders[APPROVED_COL])
    estimated = pd.to_datetime(orders["order_estimated_delivery_date"])
    prediction_time = approved if anchor == "aprobacion" else purchase

    base = orders[["order_id", "customer_id"]].assign(
        **{
            PURCHASE_COL: purchase,
            PREDICTION_TIME_COL: prediction_time,
            "purchase_month": purchase.dt.month,
            "purchase_dayofweek": purchase.dt.dayofweek,
            "purchase_hour": purchase.dt.hour,
            "promised_days": (estimated.dt.normalize() - purchase.dt.normalize()).dt.days,
        }
    )
    if anchor == "aprobacion":
        base = base.assign(approval_hours=(approved - purchase).dt.total_seconds() / 3600)
    if TARGET_COL in orders:
        base = base.assign(**{TARGET_COL: orders[TARGET_COL].to_numpy()})
    base = base.loc[prediction_time.notna().to_numpy()]

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
    ).drop(columns=["seller_states", "customer_id"]).reset_index(drop=True)
