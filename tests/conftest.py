import pandas as pd
import pytest


@pytest.fixture
def raw_tables():
    """Mini dataset sintético con la misma forma que los CSV de Olist."""
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c2", "c3"],
            "order_status": ["delivered"] * 3,
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-01-05 10:30", "2017-06-10 22:00", "2018-02-01 08:00"]
            ),
            "order_approved_at": pd.to_datetime(["2017-01-05 11:00", "2017-06-11 01:00", "2018-02-01 09:00"]),
            "order_delivered_carrier_date": pd.to_datetime(["2017-01-07", "2017-06-13", "2018-02-03"]),
            "order_delivered_customer_date": pd.to_datetime(
                ["2017-01-20 14:00", "2017-07-15 10:00", "2018-02-10 12:00"]
            ),
            "order_estimated_delivery_date": pd.to_datetime(["2017-01-25", "2017-07-01", "2018-02-10"]),
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3"],
            "order_item_id": [1, 2, 1, 1],
            "product_id": ["p1", "p2", "p1", "p2"],
            "seller_id": ["s1", "s2", "s1", "s2"],
            "shipping_limit_date": pd.to_datetime(["2017-01-08"] * 2 + ["2017-06-14", "2018-02-04"]),
            "price": [100.0, 50.0, 100.0, 50.0],
            "freight_value": [10.0, 5.0, 20.0, 8.0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_weight_g": [500.0, 1000.0],
            "product_length_cm": [10.0, 20.0],
            "product_height_cm": [10.0, 10.0],
            "product_width_cm": [10.0, 10.0],
        }
    )
    sellers = pd.DataFrame({"seller_id": ["s1", "s2"], "seller_state": ["SP", "RJ"]})
    customers = pd.DataFrame({"customer_id": ["c1", "c2", "c3"], "customer_state": ["SP", "SP", "RJ"]})
    payments = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3"],
            "payment_type": ["voucher", "credit_card", "boleto", "credit_card"],
            "payment_installments": [1, 3, 1, 6],
            "payment_value": [20.0, 145.0, 120.0, 58.0],
        }
    )
    return {
        "orders": orders,
        "items": items,
        "products": products,
        "sellers": sellers,
        "customers": customers,
        "payments": payments,
    }
