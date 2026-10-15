"""Umbral, calibración y niveles de riesgo (decisión 9), aplicados al modelo principal.

Uso (desde la raíz del repo, con los 9 CSV en data/raw):
    python -m src.umbral                     # modelo principal (decisión 10), por defecto
    python -m src.umbral --modelo baseline   # el baseline, como referencia

Entrena el modelo con train, recalibra y fija los cortes de riesgo con validación,
y reporta en validación y en test (una sola vez). Después reentrena el mismo
pipeline con train + validación (mismas features e hiperparámetros) y lo mide en
test con el calibrador y los cortes ya fijados: ese es el modelo final que se
guarda. Con --sin-reentrenar se guarda el entrenado solo con train. Escribe
reports/umbral_calibracion_<modelo>.json y, para el modelo principal, guarda en
models/modelo_principal_calibrado.joblib todo lo que necesita la API para dar un
riesgo por pedido: el modelo, sus columnas, el calibrador y los cortes.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from src.baseline import build_model_table, make_model
from src.calibration import (
    COST_RATIOS,
    PlattCalibrator,
    assign_risk_level,
    calibration_by_decile,
    cost_threshold,
    risk_level_cuts,
    risk_level_report,
)
from src.data_loading import RAW_DIR, load_raw_tables
from src.feature_engineering import build_model_dataset, load_geolocation
from src.features import DEFAULT_ANCHOR, PREDICTION_TIME_COL, feature_columns
from src.target import TARGET_COL
from src.training import make_logistic, selected_feature_lists
from src.validation import temporal_three_way_split

MODELS = ("principal", "baseline")
CALIBRATED_MODEL_PATH = Path("models/modelo_principal_calibrado.joblib")


def _period(part) -> list[str]:
    return [str(part[PREDICTION_TIME_COL].min().date()), str(part[PREDICTION_TIME_COL].max().date())]


def _cost_rows(y, prob) -> list[dict]:
    rows = []
    for ratio in COST_RATIOS:
        alert = prob >= cost_threshold(ratio)
        rows.append(
            {
                "fn_cuesta_x_fp": ratio,
                "umbral": cost_threshold(ratio),
                "pct_alertas": float(alert.mean()),
                "recall": float(y[alert].sum() / y.sum()) if y.sum() else 0.0,
                "precision": float(y[alert].mean()) if alert.any() else 0.0,
            }
        )
    return rows


def _cohort_report(y, raw, calibrator, cuts) -> dict:
    """Métricas de una cohorte: discriminación, calibración, niveles de riesgo y umbral por costos."""
    prob = calibrator.transform(raw)
    return {
        "prevalencia": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, raw)),
        "pr_auc": float(average_precision_score(y, raw)),
        "riesgo_medio_sin_calibrar": float(raw.mean()),
        "riesgo_medio_calibrado": float(prob.mean()),
        "calibracion_sin_calibrar": calibration_by_decile(y, raw),
        "calibracion_calibrada": calibration_by_decile(y, prob),
        "niveles": risk_level_report(y, assign_risk_level(prob, cuts)),
        "umbral_por_costos": _cost_rows(y, prob),
    }


def prepare(tables: dict, modelo: str = "principal", anchor: str = DEFAULT_ANCHOR):
    """(tabla de modelado, pipeline sin entrenar, columnas) del modelo pedido."""
    if modelo == "principal":
        dataset = build_model_dataset(tables, anchor)
        numeric, categorical = selected_feature_lists(anchor)
        return dataset, make_logistic(numeric, categorical), numeric + categorical
    if modelo == "baseline":
        base_tables = {k: v for k, v in tables.items() if k != "geolocation"}
        return build_model_table(base_tables, anchor), make_model(anchor), feature_columns(anchor)
    raise ValueError(f"modelo desconocido: {modelo}")


def run(
    tables: dict, modelo: str = "principal", anchor: str = DEFAULT_ANCHOR, reentrenar: bool = True
) -> tuple[dict, dict]:
    """Devuelve (reporte, artefacto calibrado listo para predecir).

    Con reentrenar=True el artefacto usa el modelo reentrenado con train + validación;
    el calibrador y los cortes de riesgo siguen siendo los fijados en validación.
    """
    table, pipeline, features = prepare(tables, modelo, anchor)
    train, validation, test = temporal_three_way_split(table)
    model = clone(pipeline).fit(train[features], train[TARGET_COL])

    parts = {"validacion": validation, "test": test}
    raw = {name: model.predict_proba(part[features])[:, 1] for name, part in parts.items()}
    y = {name: part[TARGET_COL].to_numpy() for name, part in parts.items()}

    calibrator = PlattCalibrator().fit(raw["validacion"], y["validacion"])
    cuts = risk_level_cuts(calibrator.transform(raw["validacion"]))

    cohorts = {name: _cohort_report(y[name], raw[name], calibrator, cuts) for name in parts}
    final_model = model
    if reentrenar:
        train_val = pd.concat([train, validation])
        final_model = clone(pipeline).fit(train_val[features], train_val[TARGET_COL])
        cohorts["test_reentrenado"] = _cohort_report(
            y["test"], final_model.predict_proba(test[features])[:, 1], calibrator, cuts
        )
    report = {
        "modelo": modelo,
        "anchor": anchor,
        "features": features,
        "periodos": {"train": _period(train), "validacion": _period(validation), "test": _period(test)},
        "tamanos": {"train": len(train), "validacion": len(validation), "test": len(test)},
        "cortes_riesgo_validacion": cuts,
        "reentrenado_con_validacion": reentrenar,
        "cohortes": cohorts,
    }
    artifact = {
        "model": final_model,
        "features": features,
        "calibrator": calibrator,
        "cuts": cuts,
        "anchor": anchor,
        "entrenado_con": "train + validacion" if reentrenar else "train",
    }
    return report, artifact


def predict_risk(artifact: dict, orders: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad calibrada de retraso y nivel de riesgo (alto, medio, bajo) por pedido."""
    scores = artifact["model"].predict_proba(orders[artifact["features"]])[:, 1]
    prob = artifact["calibrator"].transform(scores)
    return pd.DataFrame(
        {"prob_retraso": prob, "nivel_riesgo": assign_risk_level(prob, artifact["cuts"])},
        index=orders.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--modelo", choices=MODELS, default="principal")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--sin-reentrenar", action="store_true", help="guarda el modelo entrenado solo con train"
    )
    args = parser.parse_args()

    tables = load_raw_tables(args.raw_dir)
    if args.modelo == "principal":
        tables = {**tables, "geolocation": load_geolocation(args.raw_dir)}
    report, artifact = run(tables, args.modelo, reentrenar=not args.sin_reentrenar)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"umbral_calibracion_{args.modelo}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    if args.modelo == "principal":
        CALIBRATED_MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(artifact, CALIBRATED_MODEL_PATH)

    for name, c in report["cohortes"].items():
        print(
            f"\n{name}: prevalencia {c['prevalencia']:.1%} | PR-AUC {c['pr_auc']:.3f} | "
            f"riesgo medio {c['riesgo_medio_sin_calibrar']:.1%} -> {c['riesgo_medio_calibrado']:.1%} calibrado"
        )
        for row in c["niveles"]:
            print(
                f"  {row['nivel']:<6} {row['pct_pedidos']:6.1%} pedidos  "
                f"tasa {row['tasa_retraso']:5.1%}  capta {row['pct_retrasos']:5.1%}  lift {row['lift']:.2f}"
            )
    print(f"\nReporte: {out}")
    if args.modelo == "principal":
        print(f"Modelo calibrado: {CALIBRATED_MODEL_PATH}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
