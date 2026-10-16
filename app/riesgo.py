"""Lógica de la app de riesgo por pedido, separada de Streamlit para poder testearla.

Usa el artefacto de src/umbral.py (models/modelo_principal_calibrado.joblib):
modelo reentrenado con train + validación, calibrador de Platt y cortes de riesgo
fijados en validación. El nivel (alto, medio, bajo) es el dato principal; la
probabilidad calibrada está inflada en los meses nuevos (ver docs/metricas_evaluacion.md).
"""

from pathlib import Path

import joblib
import pandas as pd

from src.calibration import LEVELS
from src.features import PREDICTION_TIME_COL
from src.target import TARGET_COL
from src.umbral import CALIBRATED_MODEL_PATH, predict_risk

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / CALIBRATED_MODEL_PATH
SEMAFORO = {"alto": "🔴", "medio": "🟡", "bajo": "🟢"}
# Columnas que se muestran en la tabla de pedidos, además del riesgo.
DETAIL_COLUMNS = [
    "order_id",
    PREDICTION_TIME_COL,
    "customer_state",
    "promised_days",
    "max_distance_km",
    "total_price",
    "total_freight",
    "n_items",
]


def load_artifact(path: Path = ARTIFACT_PATH) -> dict:
    """Carga el modelo calibrado. Falla con un mensaje claro si no se generó."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No está {path}. Generalo con `python -m src.umbral`.")
    return joblib.load(path)


def orders_with_risk(artifact: dict, orders: pd.DataFrame) -> pd.DataFrame:
    """Pedidos con probabilidad calibrada, nivel y semáforo, del más riesgoso al menos riesgoso.

    Si los pedidos traen el target (datos históricos), se conserva para comparar.
    """
    risk = predict_risk(artifact, orders)
    columns = [c for c in [*DETAIL_COLUMNS, TARGET_COL] if c in orders.columns]
    result = orders[columns].join(risk)
    result.insert(0, "semaforo", result["nivel_riesgo"].map(SEMAFORO))
    return result.sort_values("prob_retraso", ascending=False).reset_index(drop=True)


def summary_by_level(scored: pd.DataFrame) -> pd.DataFrame:
    """Por nivel: cantidad y % de pedidos y, si hay target, tasa real de retraso."""
    grouped = scored.groupby("nivel_riesgo")
    summary = pd.DataFrame({"pedidos": grouped.size()}).reindex(list(LEVELS), fill_value=0)
    total = summary["pedidos"].sum()
    summary["pct_pedidos"] = summary["pedidos"] / total if total else 0.0
    if TARGET_COL in scored.columns:
        summary["tasa_retraso_real"] = grouped[TARGET_COL].mean().reindex(list(LEVELS))
    summary.insert(0, "semaforo", [SEMAFORO[level] for level in LEVELS])
    return summary
