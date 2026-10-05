"""Carga de los CSV de Olist desde data/raw (descargados de Kaggle, fuera de Git)."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
}

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
}


def load_raw_tables(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Lee las tablas necesarias para el baseline. Falla con un mensaje claro si falta alguna."""
    raw_dir = Path(raw_dir)
    missing = [name for name in FILES.values() if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan CSV en {raw_dir}: {', '.join(missing)}. "
            "Descargalos de Kaggle (olistbr/brazilian-ecommerce) y descomprimilos ahí."
        )

    return {
        key: pd.read_csv(
            raw_dir / filename,
            parse_dates=DATE_COLUMNS.get(key, []),
            dtype={"customer_zip_code_prefix": str, "seller_zip_code_prefix": str},
        )
        for key, filename in FILES.items()
    }
