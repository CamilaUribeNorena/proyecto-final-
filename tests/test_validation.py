import pandas as pd
import pytest

from src.features import PURCHASE_COL
from src.validation import classification_metrics, temporal_cutoff, temporal_split


def _table():
    dates = pd.date_range("2017-01-01", periods=10, freq="30D")
    return pd.DataFrame({PURCHASE_COL: dates, "x": range(10)})


def test_split_keeps_all_train_before_test():
    table = _table()
    train, test = temporal_split(table, temporal_cutoff(table, test_fraction=0.2))
    assert len(train) + len(test) == len(table)
    assert train[PURCHASE_COL].max() < test[PURCHASE_COL].min()


def test_cutoff_leaves_roughly_the_requested_fraction_in_test():
    table = _table()
    _, test = temporal_split(table, temporal_cutoff(table, test_fraction=0.3))
    assert len(test) == 3


def test_metrics_on_known_predictions():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.6, 0.4, 0.9]
    m = classification_metrics(y_true, y_score)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["positive_rate"] == pytest.approx(0.5)
    assert 0 < m["pr_auc"] <= 1
