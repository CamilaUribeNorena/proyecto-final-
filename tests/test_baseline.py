import numpy as np
import pandas as pd
import pytest

from src.baseline import run_baseline

N_ORDERS = 400


def _synthetic_tables(seed: int = 0) -> dict[str, pd.DataFrame]:
    """Pedidos sintéticos donde un plazo prometido corto aumenta el retraso."""
    rng = np.random.default_rng(seed)
    ids = [f"o{i}" for i in range(N_ORDERS)]
    purchase = pd.Timestamp("2017-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 600, N_ORDERS)), unit="D")
    promised = rng.integers(5, 40, N_ORDERS)
    actual = promised + rng.integers(-10, 6, N_ORDERS) - (promised > 20) * 5
    orders = pd.DataFrame(
        {
            "order_id": ids,
            "customer_id": [f"c{i}" for i in range(N_ORDERS)],
            "order_status": "delivered",
            "order_purchase_timestamp": purchase,
            "order_approved_at": purchase + pd.to_timedelta(rng.integers(0, 48, N_ORDERS), unit="h"),
            "order_estimated_delivery_date": purchase + pd.to_timedelta(promised, unit="D"),
            "order_delivered_customer_date": purchase + pd.to_timedelta(actual, unit="D") + pd.Timedelta(hours=12),
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ids,
            "order_item_id": 1,
            "product_id": "p1",
            "seller_id": rng.choice(["s1", "s2"], N_ORDERS),
            "price": rng.uniform(10, 200, N_ORDERS),
            "freight_value": rng.uniform(5, 40, N_ORDERS),
        }
    )
    products = pd.DataFrame(
        {"product_id": ["p1"], "product_weight_g": [500.0], "product_length_cm": [10.0],
         "product_height_cm": [10.0], "product_width_cm": [10.0]}
    )
    sellers = pd.DataFrame({"seller_id": ["s1", "s2"], "seller_state": ["SP", "RJ"]})
    customers = pd.DataFrame(
        {"customer_id": orders["customer_id"], "customer_state": rng.choice(["SP", "RJ", "BA"], N_ORDERS)}
    )
    payments = pd.DataFrame(
        {"order_id": ids, "payment_type": rng.choice(["credit_card", "boleto"], N_ORDERS),
         "payment_installments": rng.integers(1, 10, N_ORDERS), "payment_value": items["price"]}
    )
    return {"orders": orders, "items": items, "products": products,
            "sellers": sellers, "customers": customers, "payments": payments}


@pytest.mark.parametrize("anchor", ["aprobacion", "compra"])
def test_baseline_runs_end_to_end_and_reports_all_metrics(anchor):
    results = run_baseline(_synthetic_tables(), test_fraction=0.25, anchor=anchor)

    assert results["anchor"] == anchor
    assert ("approval_hours" in results["features"]) == (anchor == "aprobacion")
    assert results["train_period"][1] < results["test_period"][0]
    metrics = results["test_metrics"]
    for key in ["confusion_matrix", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert key in metrics
    assert metrics["roc_auc"] > 0.6  # el plazo prometido es informativo en los datos sintéticos
