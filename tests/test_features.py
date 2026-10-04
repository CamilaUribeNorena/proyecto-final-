from src.features import FEATURES, LEAKY_COLUMNS, build_features
from src.target import TARGET_COL, build_target


def _features(raw_tables):
    tables = {**raw_tables, "orders": build_target(raw_tables["orders"])}
    return build_features(**tables).set_index("order_id")


def test_one_row_per_order_with_all_features(raw_tables):
    result = _features(raw_tables)
    assert len(result) == 3
    assert set(FEATURES) <= set(result.columns)
    assert TARGET_COL in result.columns


def test_no_post_purchase_columns_are_used_as_features(raw_tables):
    assert not set(LEAKY_COLUMNS) & set(FEATURES)
    assert not set(LEAKY_COLUMNS) & set(_features(raw_tables).columns)


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
