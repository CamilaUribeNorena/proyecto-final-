"""Baseline: regresión logística con validación temporal.

Uso (desde la raíz del repo, con los CSV en data/raw):
    python -m src.baseline                      # predice al aprobar la compra (por defecto)
    python -m src.baseline --anchor compra      # predice al momento de la compra
    python -m src.baseline --anchor ambas       # corre las dos variantes para compararlas
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
from src.features import (
    ANCHORS,
    CATEGORICAL_FEATURES,
    DEFAULT_ANCHOR,
    PREDICTION_TIME_COL,
    build_features,
    feature_columns,
    numeric_features,
)
from src.target import TARGET_COL, build_target
from src.validation import DEFAULT_TEST_FRACTION, classification_metrics, temporal_cutoff, temporal_split

RANDOM_STATE = 42
MAX_ITER = 2000


def make_model(anchor: str = DEFAULT_ANCHOR) -> Pipeline:
    """Imputación + escalado + one-hot + regresión logística balanceada (el retraso es minoritario)."""
    preprocess = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_features(anchor)),
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


def build_model_table(tables: dict[str, pd.DataFrame], anchor: str = DEFAULT_ANCHOR) -> pd.DataFrame:
    return build_features(**{**tables, "orders": build_target(tables["orders"])}, anchor=anchor)


def run_baseline(
    tables: dict[str, pd.DataFrame],
    test_fraction: float = DEFAULT_TEST_FRACTION,
    anchor: str = DEFAULT_ANCHOR,
) -> dict:
    table = build_model_table(tables, anchor)
    cutoff = temporal_cutoff(table, test_fraction)
    train, test = temporal_split(table, cutoff)
    features = feature_columns(anchor)

    model = make_model(anchor).fit(train[features], train[TARGET_COL])
    scores = model.predict_proba(test[features])[:, 1]

    def period(part: pd.DataFrame) -> list[str]:
        return [str(part[PREDICTION_TIME_COL].min().date()), str(part[PREDICTION_TIME_COL].max().date())]

    return {
        "anchor": anchor,
        "features": features,
        "cutoff": str(cutoff.date()),
        "train_period": period(train),
        "test_period": period(test),
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
    parser.add_argument("--anchor", choices=[*ANCHORS, "ambas"], default=DEFAULT_ANCHOR)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    tables = load_raw_tables(args.raw_dir)
    anchors = list(ANCHORS) if args.anchor == "ambas" else [args.anchor]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for anchor in anchors:
        results = run_baseline(tables, args.test_fraction, anchor)
        output = args.output_dir / f"baseline_metrics_{anchor}.json"
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
