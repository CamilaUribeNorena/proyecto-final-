import pandas as pd

from src.target import TARGET_COL, build_target


def _orders():
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d", "e"],
            "order_status": ["delivered", "delivered", "delivered", "canceled", "delivered"],
            "order_delivered_customer_date": [
                "2018-01-10 15:00:00",  # mismo día prometido, con hora
                "2018-01-11 09:00:00",  # un día después
                "2018-01-08 10:00:00",  # antes
                "2018-01-20 10:00:00",  # cancelado: se descarta
                None,  # sin fecha de entrega: se descarta
            ],
            "order_estimated_delivery_date": ["2018-01-10"] * 5,
            "order_purchase_timestamp": ["2017-12-28 10:00:00"] * 5,
        }
    )


def test_only_delivered_orders_with_dates_are_kept():
    result = build_target(_orders())
    assert list(result["order_id"]) == ["a", "b", "c"]


def test_same_day_delivery_is_not_late_by_calendar_day():
    result = build_target(_orders()).set_index("order_id")[TARGET_COL]
    assert result.to_dict() == {"a": 0, "b": 1, "c": 0}


def test_timestamp_comparison_marks_same_day_as_late():
    result = build_target(_orders(), by_calendar_day=False).set_index("order_id")[TARGET_COL]
    assert result.to_dict() == {"a": 1, "b": 1, "c": 0}


def test_input_is_not_mutated():
    orders = _orders()
    before = orders.copy()
    build_target(orders)
    pd.testing.assert_frame_equal(orders, before)


def test_only_purchases_inside_study_period_are_kept():
    orders = pd.concat([_orders().iloc[[0]]] * 4, ignore_index=True)
    orders["order_id"] = ["2016", "inicio", "fin", "sep2018"]
    orders["order_purchase_timestamp"] = [
        "2016-12-31 23:59:00",  # antes del período: se descarta
        "2017-01-01 00:00:00",  # primer instante del período
        "2018-08-31 23:59:00",  # último día del período
        "2018-09-01 00:00:00",  # después del período: se descarta
    ]
    assert list(build_target(orders)["order_id"]) == ["inicio", "fin"]
    assert len(build_target(orders, only_study_period=False)) == 4
