import pandas as pd
import pytest

from src.features import (
    APPROVED_COL,
    PREDICTION_TIME_COL,
    PURCHASE_COL,
    build_features,
    feature_columns,
    leaky_columns,
)
from src.target import TARGET_COL, build_target

ANCHORS = ["aprobacion", "compra"]


def _features(raw_tables, anchor="aprobacion"):
    tables = {**raw_tables, "orders": build_target(raw_tables["orders"])}
    return build_features(**tables, anchor=anchor).set_index("order_id")


@pytest.mark.parametrize("anchor", ANCHORS)
def test_one_row_per_order_with_all_features(raw_tables, anchor):
    result = _features(raw_tables, anchor)
    assert len(result) == 3
    assert set(feature_columns(anchor)) <= set(result.columns)
    assert TARGET_COL in result.columns


@pytest.mark.parametrize("anchor", ANCHORS)
def test_no_post_prediction_columns_are_used_as_features(raw_tables, anchor):
    leaky = set(leaky_columns(anchor))
    assert not leaky & set(feature_columns(anchor))
    assert not leaky & set(_features(raw_tables, anchor).columns)


def test_purchase_anchor_never_uses_approval_information(raw_tables):
    result = _features(raw_tables, "compra")
    assert APPROVED_COL in leaky_columns("compra")
    assert "approval_hours" not in result.columns
    assert result[PREDICTION_TIME_COL].equals(result[PURCHASE_COL])


def test_approval_anchor_uses_approval_time_and_hours(raw_tables):
    result = _features(raw_tables, "aprobacion")
    assert result.loc["o1", PREDICTION_TIME_COL] == pd.Timestamp("2017-01-05 11:00")
    assert result.loc["o1", "approval_hours"] == pytest.approx(0.5)
    assert result.loc["o2", "approval_hours"] == pytest.approx(3.0)


def test_approval_anchor_drops_orders_without_approval_date(raw_tables):
    orders = raw_tables["orders"].assign(
        order_approved_at=raw_tables["orders"]["order_approved_at"].where(lambda s: s.index != 1)
    )
    result = _features({**raw_tables, "orders": orders}, "aprobacion")
    assert list(result.index) == ["o1", "o3"]
    assert len(_features({**raw_tables, "orders": orders}, "compra")) == 3


def test_unknown_anchor_is_rejected(raw_tables):
    with pytest.raises(ValueError):
        _features(raw_tables, "despacho")


def test_cart_and_payment_aggregations(raw_tables):
    o1 = _features(raw_tables).loc["o1"]
    assert o1["n_items"] == 2
    assert o1["n_sellers"] == 2
    assert o1["total_price"] == 150.0
    assert o1["total_freight"] == 15.0
    assert o1["total_weight_g"] == 1500.0
    assert o1["main_payment_type"] == "credit_card"
    assert o1["max_installments"] == 3
    assert o1["seller_other_state"] == "True"


def test_promised_days_counts_calendar_days_from_purchase(raw_tables):
    result = _features(raw_tables)
    assert result.loc["o1", "promised_days"] == 20
    assert result.loc["o3", "promised_days"] == 9


def test_seller_in_same_state_is_flagged_false(raw_tables):
    assert _features(raw_tables).loc["o2", "seller_other_state"] == "False"


def test_invalid_product_measurements_remain_missing(raw_tables):
    """Un cero físico no debe convertirse en un peso o volumen válido."""
    products = raw_tables["products"].copy()
    products.loc[products["product_id"] == "p1", "product_weight_g"] = 0
    products.loc[products["product_id"] == "p1", "product_length_cm"] = 0

    result = _features({**raw_tables, "products": products})

    # o1 tiene p1 y p2. Si un ítem no tiene una medida válida, no inventamos
    # un total parcial para todo el pedido: el pipeline lo imputará después.
    assert pd.isna(result.loc["o1", "total_weight_g"])
    assert pd.isna(result.loc["o1", "total_volume_cm3"])


def test_nonpositive_installments_remain_missing(raw_tables):
    """Cero cuotas es una anomalía del origen, no una financiación válida."""
    payments = raw_tables["payments"].copy()
    payments.loc[payments["order_id"] == "o2", "payment_installments"] = 0

    result = _features({**raw_tables, "payments": payments})

    assert pd.isna(result.loc["o2", "max_installments"])
