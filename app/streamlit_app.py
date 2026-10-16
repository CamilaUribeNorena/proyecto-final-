"""App de riesgo de retraso por pedido (esqueleto, Sprint 2).

Uso, desde la raíz del repo, con los CSV en data/raw y el modelo generado
(`python -m src.umbral`):
    streamlit run app/streamlit_app.py

Por ahora arma los pedidos desde los CSV de Kaggle. Cuando esté la API, la app
le va a pedir el riesgo a la API en lugar de cargar el modelo.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # para importar src y app

from app.riesgo import load_artifact, orders_with_risk, summary_by_level  # noqa: E402
from src.calibration import LEVELS  # noqa: E402
from src.data_loading import load_raw_tables  # noqa: E402
from src.feature_engineering import build_model_dataset, load_geolocation  # noqa: E402
from src.features import PREDICTION_TIME_COL  # noqa: E402
from src.target import TARGET_COL  # noqa: E402
from src.validation import temporal_three_way_split  # noqa: E402

LEVEL_LABELS = {"alto": "Riesgo alto", "medio": "Riesgo medio", "bajo": "Riesgo bajo"}


@st.cache_resource
def get_artifact() -> dict:
    return load_artifact()


@st.cache_data(show_spinner="Armando los pedidos desde los CSV...")
def get_scored_orders():
    """Pedidos del período de test (los que el modelo no vio), con su riesgo."""
    tables = {**load_raw_tables(), "geolocation": load_geolocation()}
    dataset = build_model_dataset(tables, get_artifact()["anchor"])
    _, _, test = temporal_three_way_split(dataset)
    return orders_with_risk(get_artifact(), test)


def show_summary(scored) -> None:
    summary = summary_by_level(scored)
    for column, level in zip(st.columns(len(LEVELS)), LEVELS):
        row = summary.loc[level]
        column.metric(
            f"{row['semaforo']} {LEVEL_LABELS[level]}",
            f"{int(row['pedidos']):,}".replace(",", "."),
            f"{row['pct_pedidos']:.1%} de los pedidos",
            delta_color="off",
        )


def main() -> None:
    st.set_page_config(page_title="Riesgo de retraso", page_icon="🚚", layout="wide")
    st.title("Riesgo de retraso por pedido")
    st.caption(
        "Olist · modelo principal calibrado. El nivel ordena los pedidos para priorizar el "
        "seguimiento; la probabilidad está inflada en los meses nuevos, así que se lee como orden, "
        "no como valor exacto."
    )

    try:
        scored = get_scored_orders()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.header("Filtros")
        dates = scored[PREDICTION_TIME_COL].dt.date
        start, end = st.date_input(
            "Fecha de aprobación", (dates.min(), dates.max()), min_value=dates.min(), max_value=dates.max()
        )
        levels = st.multiselect("Nivel de riesgo", list(LEVELS), default=["alto"], format_func=LEVEL_LABELS.get)
        show_real = st.checkbox("Mostrar si se retrasó (dato histórico)", value=False)

    in_dates = scored[(dates >= start) & (dates <= end)]
    show_summary(in_dates)

    st.subheader("Pedidos a revisar")
    table = in_dates[in_dates["nivel_riesgo"].isin(levels)]
    if not show_real:
        table = table.drop(columns=[TARGET_COL], errors="ignore")
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "prob_retraso": st.column_config.ProgressColumn("Riesgo", format="%.2f", min_value=0, max_value=1),
            PREDICTION_TIME_COL: st.column_config.DatetimeColumn("Aprobación", format="DD/MM/YYYY HH:mm"),
        },
    )
    st.caption(f"{len(table):,} pedidos".replace(",", "."))


main()
