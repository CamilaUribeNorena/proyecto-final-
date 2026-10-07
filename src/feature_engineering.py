"""Features del modelo principal (parte 2): ruta, producto e historial del vendedor.

Se agregan sobre la tabla del baseline (`src/features.py::build_features`), que ya
trae fecha de compra, plazo prometido, carrito, pago y estado del cliente.

Cómo sumar una feature nueva (por ejemplo, las del EDA de Mathias):
  1. Escribir una función `(context: FeatureContext) -> pd.DataFrame` que devuelva
     una fila por order_id (columna order_id + las columnas nuevas).
  2. Registrarla en FEATURE_BUILDERS con sus columnas numéricas y categóricas.
  3. Agregar un test en tests/test_feature_engineering.py.
Toda feature tiene que conocerse en el momento de predicción (`prediction_time`).
El test anti-fuga recorre FEATURE_BUILDERS y falla si alguna usa columnas prohibidas.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data_loading import RAW_DIR
from src.features import (
    CATEGORICAL_FEATURES,
    DEFAULT_ANCHOR,
    PREDICTION_TIME_COL,
    build_features,
    feature_columns,
)
from src.target import TARGET_COL, build_target

GEOLOCATION_FILE = "olist_geolocation_dataset.csv"
EARTH_RADIUS_KM = 6371.0
MIN_SELLER_HISTORY = 5  # con menos pedidos previos la tasa del vendedor es ruido

REGIONS = {
    "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MS", "MT"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}
STATE_TO_REGION = {state: region for region, states in REGIONS.items() for state in states}


@dataclass(frozen=True)
class FeatureContext:
    """Lo que recibe cada builder: la tabla base y las tablas crudas."""

    base: pd.DataFrame  # una fila por pedido, con order_id y prediction_time
    tables: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class FeatureBuilder:
    name: str
    build: Callable[[FeatureContext], pd.DataFrame]
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()


def load_geolocation(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = Path(raw_dir) / GEOLOCATION_FILE
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Viene en el mismo zip de Kaggle que el resto de los CSV.")
    return pd.read_csv(path, dtype={"geolocation_zip_code_prefix": str})


def _zip_centroids(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Lat/lng promedio por prefijo de código postal (el geolocation trae varias filas por prefijo)."""
    return (
        geolocation.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    lat1, lng1, lat2, lng2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _main_item(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """El ítem más caro de cada pedido define vendedor y categoría principales."""
    return (
        tables["items"].sort_values(["order_id", "price"], ascending=[True, False])
        .drop_duplicates("order_id")[["order_id", "seller_id", "product_id"]]
    )


def route_features(context: FeatureContext) -> pd.DataFrame:
    """Distancia cliente-vendedor, región de cada punta y si la ruta cruza regiones."""
    tables = context.tables
    centroids = _zip_centroids(tables["geolocation"])
    orders = tables["orders"][["order_id", "customer_id"]]
    customers = tables["customers"][["customer_id", "customer_zip_code_prefix", "customer_state"]]
    sellers = tables["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]]

    pairs = (
        tables["items"][["order_id", "seller_id"]].drop_duplicates()
        .merge(orders, on="order_id")
        .merge(customers, on="customer_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .merge(centroids.add_prefix("c_"), left_on="customer_zip_code_prefix", right_index=True, how="left")
        .merge(centroids.add_prefix("s_"), left_on="seller_zip_code_prefix", right_index=True, how="left")
    )
    pairs = pairs.assign(distance_km=haversine_km(pairs["c_lat"], pairs["c_lng"], pairs["s_lat"], pairs["s_lng"]))
    distance = pairs.groupby("order_id")["distance_km"].max().rename("max_distance_km")

    main_seller_state = (
        _main_item(tables).merge(sellers, on="seller_id", how="left").set_index("order_id")["seller_state"]
    )
    customer_state = orders.merge(customers, on="customer_id", how="left").set_index("order_id")["customer_state"]
    customer_region = customer_state.map(STATE_TO_REGION).fillna("unknown")
    seller_region = main_seller_state.reindex(customer_state.index).map(STATE_TO_REGION).fillna("unknown")

    return pd.DataFrame(
        {
            "max_distance_km": distance.reindex(customer_state.index),
            "customer_region": customer_region,
            "seller_region": seller_region,
            "cross_region": np.where(
                (customer_region == "unknown") | (seller_region == "unknown"),
                "unknown",
                (customer_region != seller_region).astype(str),
            ),
        }
    ).rename_axis("order_id").reset_index()


def product_features(context: FeatureContext) -> pd.DataFrame:
    """Categoría del producto principal del pedido (el ítem más caro)."""
    tables = context.tables
    category = _main_item(tables).merge(
        tables["products"][["product_id", "product_category_name"]], on="product_id", how="left"
    )
    return pd.DataFrame(
        {
            "order_id": category["order_id"],
            "main_category": category["product_category_name"].fillna("unknown"),
        }
    )


def seller_history_features(context: FeatureContext) -> pd.DataFrame:
    """Pedidos previos y tasa de retraso histórica del vendedor principal.

    Sin fuga: para un pedido con prediction_time t solo cuentan los pedidos del
    mismo vendedor que se ENTREGARON antes de t (estrictamente). Un pedido que a
    esa hora todavía no llegó no aporta información, aunque luego se retrase.
    """
    tables = context.tables
    delivered = build_target(tables["orders"], only_study_period=False)[
        ["order_id", "order_delivered_customer_date", TARGET_COL]
    ]
    events = (
        delivered.merge(_main_item(tables)[["order_id", "seller_id"]], on="order_id")
        .sort_values("order_delivered_customer_date")
        .assign(
            prev_orders=lambda d: d.groupby("seller_id").cumcount() + 1,
            prev_late=lambda d: d.groupby("seller_id")[TARGET_COL].cumsum(),
        )[["seller_id", "order_delivered_customer_date", "prev_orders", "prev_late"]]
    )

    queries = (
        context.base[["order_id", PREDICTION_TIME_COL]]
        .merge(_main_item(tables)[["order_id", "seller_id"]], on="order_id", how="left")
        .dropna(subset=[PREDICTION_TIME_COL])
        .sort_values(PREDICTION_TIME_COL)
    )
    history = pd.merge_asof(
        queries,
        events,
        left_on=PREDICTION_TIME_COL,
        right_on="order_delivered_customer_date",
        by="seller_id",
        direction="backward",
        allow_exact_matches=False,
    )
    prev_orders = history["prev_orders"].fillna(0)
    late_rate = (history["prev_late"] / prev_orders).where(prev_orders >= MIN_SELLER_HISTORY)
    return pd.DataFrame(
        {
            "order_id": history["order_id"],
            "seller_prev_orders": prev_orders.to_numpy(),
            "seller_prev_late_rate": late_rate.to_numpy(),
        }
    )


FEATURE_BUILDERS: list[FeatureBuilder] = [
    FeatureBuilder(
        "ruta",
        route_features,
        numeric=("max_distance_km",),
        categorical=("customer_region", "seller_region", "cross_region"),
    ),
    FeatureBuilder("producto", product_features, categorical=("main_category",)),
    FeatureBuilder(
        "historial_vendedor",
        seller_history_features,
        numeric=("seller_prev_orders", "seller_prev_late_rate"),
    ),
]


def model_feature_lists(anchor: str = DEFAULT_ANCHOR) -> tuple[list[str], list[str]]:
    """(numéricas, categóricas) del modelo principal: las del baseline más las de FEATURE_BUILDERS."""
    base = feature_columns(anchor)
    numeric = [c for c in base if c not in CATEGORICAL_FEATURES]
    categorical = list(CATEGORICAL_FEATURES)
    for builder in FEATURE_BUILDERS:
        numeric += list(builder.numeric)
        categorical += list(builder.categorical)
    return numeric, categorical


def build_model_dataset(tables: dict[str, pd.DataFrame], anchor: str = DEFAULT_ANCHOR) -> pd.DataFrame:
    """Dataset de modelado: una fila por pedido entregado del período, target y todas las features."""
    base_tables = {k: v for k, v in tables.items() if k != "geolocation"}
    base = build_features(**{**base_tables, "orders": build_target(tables["orders"])}, anchor=anchor)
    context = FeatureContext(base=base, tables=tables)
    dataset = base
    for builder in FEATURE_BUILDERS:
        extra = builder.build(context)
        expected = {"order_id", *builder.numeric, *builder.categorical}
        missing = expected - set(extra.columns)
        if missing:
            raise ValueError(f"El builder {builder.name} no devolvió {sorted(missing)}")
        dataset = dataset.merge(extra[list(expected)], on="order_id", how="left", validate="one_to_one")
    return dataset
