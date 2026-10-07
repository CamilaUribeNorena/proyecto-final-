"""Umbral, calibración y niveles de riesgo del baseline (tarea 3, decisión 9).

Uso (desde la raíz del repo, con los CSV en data/raw):
    python -m src.umbral

Entrena el baseline con train, recalibra y fija los cortes de riesgo con
validación, y reporta en validación y en test. Escribe reports/umbral_calibracion.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
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
from src.features import DEFAULT_ANCHOR, PREDICTION_TIME_COL, feature_columns
from src.target import TARGET_COL
from src.validation import temporal_three_way_split


def _period(part) -> list[str]:
    return [str(part[PREDICTION_TIME_COL].min().date()), str(part[PREDICTION_TIME_COL].max().date())]


def run(tables: dict, anchor: str = DEFAULT_ANCHOR) -> dict:
    table = build_model_table(tables, anchor)
    train, validation, test = temporal_three_way_split(table)
    features = feature_columns(anchor)
    model = make_model(anchor).fit(train[features], train[TARGET_COL])

    raw = {name: model.predict_proba(part[features])[:, 1] for name, part in (("validacion", validation), ("test", test))}
    y = {"validacion": validation[TARGET_COL].to_numpy(), "test": test[TARGET_COL].to_numpy()}

    calibrator = PlattCalibrator().fit(raw["validacion"], y["validacion"])
    prob = {name: calibrator.transform(s) for name, s in raw.items()}
    cuts = risk_level_cuts(prob["validacion"])

    cohorts = {}
    for name in ("validacion", "test"):
        cohorts[name] = {
            "prevalencia": float(y[name].mean()),
            "roc_auc": float(roc_auc_score(y[name], raw[name])),
            "pr_auc": float(average_precision_score(y[name], raw[name])),
            "riesgo_medio_sin_calibrar": float(raw[name].mean()),
            "riesgo_medio_calibrado": float(prob[name].mean()),
            "calibracion_sin_calibrar": calibration_by_decile(y[name], raw[name]),
            "calibracion_calibrada": calibration_by_decile(y[name], prob[name]),
            "niveles": risk_level_report(y[name], assign_risk_level(prob[name], cuts)),
            "umbral_por_costos": [
                {
                    "fn_cuesta_x_fp": r,
                    "umbral": cost_threshold(r),
                    "pct_alertas": float((prob[name] >= cost_threshold(r)).mean()),
                    "recall": float(y[name][prob[name] >= cost_threshold(r)].sum() / y[name].sum()),
                    "precision": float(y[name][prob[name] >= cost_threshold(r)].mean())
                    if (prob[name] >= cost_threshold(r)).any()
                    else 0.0,
                }
                for r in COST_RATIOS
            ],
        }

    return {
        "anchor": anchor,
        "periodos": {"train": _period(train), "validacion": _period(validation), "test": _period(test)},
        "tamanos": {"train": len(train), "validacion": len(validation), "test": len(test)},
        "cortes_riesgo_validacion": cuts,
        "cohortes": cohorts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    results = run(load_raw_tables(args.raw_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "umbral_calibracion.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for name, c in results["cohortes"].items():
        print(f"\n{name}: prevalencia {c['prevalencia']:.1%} | riesgo medio {c['riesgo_medio_sin_calibrar']:.1%} -> {c['riesgo_medio_calibrado']:.1%} calibrado")
        for row in c["niveles"]:
            print(f"  {row['nivel']:<6} {row['pct_pedidos']:6.1%} pedidos  tasa {row['tasa_retraso']:5.1%}  lift {row['lift']:.2f}")
    print(f"\nReporte: {out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
