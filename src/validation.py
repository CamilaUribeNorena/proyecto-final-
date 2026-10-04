"""Validación temporal: se entrena con el pasado y se evalúa con el futuro.

Nunca se mezclan pedidos al azar: todo pedido de test fue comprado en o después
de la fecha de corte, y todo pedido de train antes de esa fecha.
"""

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import PURCHASE_COL

DEFAULT_TEST_FRACTION = 0.2
DEFAULT_THRESHOLD = 0.5


def temporal_cutoff(table: pd.DataFrame, test_fraction: float = DEFAULT_TEST_FRACTION) -> pd.Timestamp:
    """Fecha (día) a partir de la cual queda aproximadamente el último test_fraction de pedidos."""
    return table[PURCHASE_COL].quantile(1 - test_fraction).normalize()


def temporal_split(table: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train: compras antes del corte. Test: compras desde el corte en adelante."""
    is_test = table[PURCHASE_COL] >= cutoff
    return table.loc[~is_test].copy(), table.loc[is_test].copy()


def classification_metrics(y_true, y_score, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Matriz de confusión, precision, recall, F1, ROC-AUC y PR-AUC."""
    y_pred = (pd.Series(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n": int(len(y_pred)),
        "positive_rate": float(pd.Series(y_true).mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }
