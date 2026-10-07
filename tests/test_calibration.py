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
