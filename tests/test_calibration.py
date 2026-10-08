import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    PlattCalibrator,
    assign_risk_level,
    cost_threshold,
    risk_level_cuts,
    risk_level_report,
)
from src.features import PREDICTION_TIME_COL
from src.validation import temporal_three_way_split


def test_three_way_split_is_ordered_in_time_and_complete():
    table = pd.DataFrame({PREDICTION_TIME_COL: pd.date_range("2017-01-01", periods=100, freq="5D")})
    train, validation, test = temporal_three_way_split(table)
    assert len(train) + len(validation) + len(test) == len(table)
    assert train[PREDICTION_TIME_COL].max() < validation[PREDICTION_TIME_COL].min()
    assert validation[PREDICTION_TIME_COL].max() < test[PREDICTION_TIME_COL].min()


def test_platt_fixes_inflated_probabilities():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.07).astype(int)
    inflated = np.clip(0.45 + 0.2 * y + rng.normal(0, 0.1, 5000), 0.01, 0.99)
    calibrated = PlattCalibrator().fit(inflated, y).transform(inflated)
    assert inflated.mean() > 0.4
    assert calibrated.mean() == pytest.approx(y.mean(), abs=0.01)


def test_risk_levels_follow_the_validation_shares():
    probs = np.linspace(0, 1, 1000)
    levels = assign_risk_level(probs, risk_level_cuts(probs))
    shares = pd.Series(levels).value_counts(normalize=True)
    assert shares["alto"] == pytest.approx(0.05, abs=0.01)
    assert shares["medio"] == pytest.approx(0.15, abs=0.01)
    assert shares["bajo"] == pytest.approx(0.80, abs=0.01)


def test_risk_level_report_computes_lift():
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    levels = ["alto", "alto", "medio", "medio", "bajo", "bajo", "bajo", "bajo", "bajo", "bajo"]
    report = {r["nivel"]: r for r in risk_level_report(y, levels)}
    assert report["alto"]["tasa_retraso"] == 1.0
    assert report["alto"]["lift"] == pytest.approx(5.0)
    assert report["alto"]["pct_retrasos"] == 1.0
    assert report["bajo"]["pct_pedidos"] == pytest.approx(0.6)


def test_cost_threshold():
    assert cost_threshold(1) == pytest.approx(0.5)
    assert cost_threshold(10) == pytest.approx(1 / 11)


def test_predict_risk_returns_calibrated_probability_and_level():
    from sklearn.linear_model import LogisticRegression

    from src.umbral import predict_risk

    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 1))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(x[:, 0] * 2 - 3)))).astype(int)
    model = LogisticRegression(class_weight="balanced").fit(x, y)
    orders = pd.DataFrame({"x": x[:, 0]})
    scores = model.predict_proba(orders[["x"]].to_numpy())[:, 1]
    calibrator = PlattCalibrator().fit(scores, y)

    class _Wrapper:
        def predict_proba(self, frame):
            return model.predict_proba(frame.to_numpy())

    artifact = {
        "model": _Wrapper(),
        "features": ["x"],
        "calibrator": calibrator,
        "cuts": risk_level_cuts(calibrator.transform(scores)),
    }
    result = predict_risk(artifact, orders)
    assert list(result.columns) == ["prob_retraso", "nivel_riesgo"]
    assert result["prob_retraso"].mean() == pytest.approx(y.mean(), abs=0.02)
    assert set(result["nivel_riesgo"]) == {"alto", "medio", "bajo"}


def test_retraining_with_validation_keeps_calibration_and_cuts(monkeypatch):
    import tests.test_baseline as synthetic
    from src.umbral import run

    monkeypatch.setattr(synthetic, "N_ORDERS", 1500)
    tables = synthetic._synthetic_tables(seed=2)

    report_train, artifact_train = run(tables, "baseline", reentrenar=False)
    report_final, artifact_final = run(tables, "baseline", reentrenar=True)

    # calibrador y cortes salen de validación con el modelo de train: no cambian
    assert artifact_final["cuts"] == artifact_train["cuts"]
    assert artifact_final["calibrator"].transform(np.array([0.5])) == pytest.approx(
        artifact_train["calibrator"].transform(np.array([0.5]))
    )
    # el modelo guardado sí cambia: se reentrenó con más pedidos
    assert artifact_final["entrenado_con"] == "train + validacion"
    assert artifact_train["entrenado_con"] == "train"
    coef_train = artifact_train["model"].named_steps["model"].coef_
    coef_final = artifact_final["model"].named_steps["model"].coef_
    assert not np.allclose(coef_train, coef_final)
    # el test del modelo final se reporta aparte, sin pisar el del modelo de train
    assert "test_reentrenado" in report_final["cohortes"]
    assert "test_reentrenado" not in report_train["cohortes"]
    assert report_final["cohortes"]["test"] == report_train["cohortes"]["test"]
