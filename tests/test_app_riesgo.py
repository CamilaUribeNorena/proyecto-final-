import numpy as np
import pandas as pd
import pytest

from app.riesgo import SEMAFORO, load_artifact, orders_with_risk, summary_by_level
from src.calibration import PlattCalibrator
from src.features import PREDICTION_TIME_COL
from src.target import TARGET_COL


class _ScoreIsColumn:
    """Modelo de mentira: el puntaje es la columna 'score'."""

    def predict_proba(self, X):
        score = X["score"].to_numpy()
        return np.column_stack([1 - score, score])


@pytest.fixture
def artifact():
    scores = np.linspace(0.05, 0.95, 200)
    y = (scores > 0.7).astype(int)
    return {
        "model": _ScoreIsColumn(),
        "features": ["score"],
        "calibrator": PlattCalibrator().fit(scores, y),
        "cuts": {"alto": 0.6, "medio": 0.3},
        "anchor": "aprobacion",
    }


@pytest.fixture
def orders():
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            PREDICTION_TIME_COL: pd.to_datetime(["2018-06-01"] * 4),
            "customer_state": ["SP", "RJ", "MG", "SP"],
            "score": [0.1, 0.9, 0.5, 0.75],
            TARGET_COL: [0, 1, 0, 0],
        }
    )


def test_orders_are_sorted_by_risk_with_traffic_light(artifact, orders):
    scored = orders_with_risk(artifact, orders)
    assert scored["prob_retraso"].is_monotonic_decreasing
    assert scored["order_id"].iloc[0] == "b"
    assert (scored["semaforo"] == scored["nivel_riesgo"].map(SEMAFORO)).all()
    assert "score" not in scored.columns  # solo columnas para mostrar
    assert TARGET_COL in scored.columns


def test_summary_has_every_level_even_if_empty(artifact, orders):
    scored = orders_with_risk(artifact, orders.iloc[[0]])
    summary = summary_by_level(scored)
    assert list(summary.index) == ["alto", "medio", "bajo"]
    assert summary["pedidos"].sum() == 1
    assert summary["pct_pedidos"].sum() == pytest.approx(1.0)


def test_summary_reports_real_late_rate_when_target_is_present(artifact, orders):
    summary = summary_by_level(orders_with_risk(artifact, orders))
    assert summary.loc["alto", "tasa_retraso_real"] == pytest.approx(0.5)


def test_missing_artifact_explains_how_to_generate_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.umbral"):
        load_artifact(tmp_path / "no_existe.joblib")
