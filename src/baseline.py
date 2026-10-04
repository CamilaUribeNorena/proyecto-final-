"""Baseline: regresión logística con validación temporal.

Uso (desde la raíz del repo, con los CSV en data/raw):
    python -m src.baseline
    python -m src.baseline --test-fraction 0.2 --output reports/baseline_metrics.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loading import RAW_DIR, load_raw_tables
from src.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, PURCHASE_COL, build_features
from src.target import TARGET_COL, build_target
from src.validation import DEFAULT_TEST_FRACTION, classification_metrics, temporal_cutoff, temporal_split

RANDOM_STATE = 42
MAX_ITER = 2000


def make_model() -> Pipeline:
    """Imputación + escalado + one-hot + regresión logística balanceada (el retraso es minoritario)."""
    preprocess = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = LogisticRegression(class_weight="balanced", max_iter=MAX_ITER, random_state=RANDOM_STATE)
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def build_model_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_features(**{**tables, "orders": build_target(tables["orders"])})


def run_baseline(tables: dict[str, pd.DataFrame], test_fraction: float = DEFAULT_TEST_FRACTION) -> dict:
    table = build_model_table(tables)
    cutoff = temporal_cutoff(table, test_fraction)
    train, test = temporal_split(table, cutoff)

    model = make_model().fit(train[FEATURES], train[TARGET_COL])
    scores = model.predict_proba(test[FEATURES])[:, 1]

    return {
        "cutoff": str(cutoff.date()),
        "train_period": [str(train[PURCHASE_COL].min().date()), str(train[PURCHASE_COL].max().date())],
        "test_period": [str(test[PURCHASE_COL].min().date()), str(test[PURCHASE_COL].max().date())],
        "train_size": int(len(train)),
        "train_positive_rate": float(train[TARGET_COL].mean()),
        "late_rate_all_delivered": float(table[TARGET_COL].mean()),
        "late_rate_timestamp_comparison": float(
            build_target(tables["orders"], by_calendar_day=False)[TARGET_COL].mean()
        ),
        "test_metrics": classification_metrics(test[TARGET_COL], scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--output", type=Path, default=Path("reports/baseline_metrics.json"))
    args = parser.parse_args()

    results = run_baseline(load_raw_tables(args.raw_dir), args.test_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
