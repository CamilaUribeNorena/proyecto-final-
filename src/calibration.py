"""Calibración, niveles de riesgo y umbral por costos (decisión 9 de docs/decisiones.md).

El baseline entrena con class_weight="balanced": ordena bien los pedidos, pero sus
probabilidades están infladas (predice ~49 % de riesgo medio cuando la tasa real
es ~7 %). Por eso:

1. Las probabilidades se recalibran con validación (recalibración de Platt: una
   regresión logística sobre el logit del score del modelo).
2. Los niveles de riesgo se definen por volumen de pedidos (alto = 5 % más
   riesgoso, medio = 15 % siguiente), con cortes fijados en validación. Así el
   volumen de alertas no depende de la prevalencia, que cambia mucho entre meses.
3. El umbral por costos (1 / (1 + r), con r = costo FN / costo FP) se reporta como
   análisis de sensibilidad, no como regla única.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HIGH_SHARE = 0.05
MEDIUM_SHARE = 0.15
LEVELS = ("alto", "medio", "bajo")
COST_RATIOS = (5, 10, 20)
_EPS = 1e-6


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Recalibra scores de un modelo con una regresión logística ajustada en validación."""

    def fit(self, scores, y) -> "PlattCalibrator":
        self._model = LogisticRegression().fit(_logit(scores), np.asarray(y))
        return self

    def transform(self, scores) -> np.ndarray:
        return self._model.predict_proba(_logit(scores))[:, 1]


def risk_level_cuts(
    validation_probs, high_share: float = HIGH_SHARE, medium_share: float = MEDIUM_SHARE
) -> dict:
    """Cortes de probabilidad que dejan high_share de pedidos en alto y medium_share en medio."""
    probs = np.asarray(validation_probs, dtype=float)
    return {
        "alto": float(np.quantile(probs, 1 - high_share)),
        "medio": float(np.quantile(probs, 1 - high_share - medium_share)),
    }


def assign_risk_level(probs, cuts: dict) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    return np.where(probs >= cuts["alto"], "alto", np.where(probs >= cuts["medio"], "medio", "bajo"))


def risk_level_report(y_true, levels) -> list[dict]:
    """Por nivel: % de pedidos, tasa de retraso, % de los retrasos que captura y lift."""
    df = pd.DataFrame({"y": np.asarray(y_true), "nivel": np.asarray(levels)})
    base = df["y"].mean()
    total_late = df["y"].sum()
    rows = []
    for level in LEVELS:
        part = df[df["nivel"] == level]
        rate = float(part["y"].mean()) if len(part) else 0.0
        rows.append(
            {
                "nivel": level,
                "pct_pedidos": float(len(part) / len(df)),
                "tasa_retraso": rate,
                "pct_retrasos": float(part["y"].sum() / total_late) if total_late else 0.0,
                "lift": float(rate / base) if base else 0.0,
            }
        )
    return rows


def cost_threshold(cost_ratio: float) -> float:
    """Umbral óptimo con probabilidades calibradas si un FN cuesta cost_ratio veces un FP."""
    return 1.0 / (1.0 + cost_ratio)


def calibration_by_decile(y_true, probs) -> list[dict]:
    """Riesgo medio predicho frente a tasa observada, por decil de probabilidad."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(probs, dtype=float)})
    df["decil"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    grouped = df.groupby("decil").agg(predicho=("p", "mean"), observado=("y", "mean"))
    return [{"predicho": float(r.predicho), "observado": float(r.observado)} for r in grouped.itertuples()]
