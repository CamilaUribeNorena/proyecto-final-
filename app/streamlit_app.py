"""App de riesgo de retraso por pedido, para el equipo de operaciones.

Uso, desde la raíz del repo, con los CSV en data/raw y el modelo generado
(`python -m src.umbral`):
    streamlit run app/streamlit_app.py

Por ahora arma los pedidos desde los CSV de Kaggle (período de test, los meses
que el modelo no vio). Cuando esté la API, la app le va a pedir el riesgo a la
API en lugar de cargar el modelo.
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # para importar src y app

from app.riesgo import (  # noqa: E402
    SEMAFORO,
    format_feature_value,
    load_artifact,
    orders_with_risk,
    risk_factors,
    summary_by_level,
)
from src.calibration import LEVELS  # noqa: E402
from src.data_loading import load_raw_tables  # noqa: E402
from src.feature_engineering import build_model_dataset, load_geolocation  # noqa: E402
from src.features import PREDICTION_TIME_COL  # noqa: E402
from src.target import TARGET_COL  # noqa: E402
from src.validation import temporal_three_way_split  # noqa: E402

# Marca DataCraft y semáforo. El color nunca va solo: siempre con ícono y texto.
NAVY, CREAM, ORANGE, INK_SOFT = "#13294B", "#FBF8F2", "#E8833A", "#4A5B78"
LEVEL_COLORS = {"alto": "#C0392B", "medio": "#D69E2E", "bajo": "#2F855A"}
LEVEL_INFO = {
    "alto": ("Riesgo alto", "Revisar hoy: contactar al vendedor y avisar al cliente."),
    "medio": ("Riesgo medio", "Seguir de cerca durante el envío."),
    "bajo": ("Riesgo bajo", "Seguimiento normal."),
}
FACTORS_UP, FACTORS_DOWN = 3, 2

CSS = f"""
<style>
h1, h2, h3, .dc-serif {{ font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
.dc-header {{ background: {NAVY}; color: {CREAM}; border-radius: 14px; padding: 18px 24px;
  display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap; margin-bottom: 18px; }}
.dc-logo {{ font-family: Georgia, serif; font-size: 1.7rem; font-weight: 600; color: {CREAM}; }}
.dc-logo span {{ color: {ORANGE}; }}
.dc-sub {{ color: #D9DEE8; font-size: 1rem; }}
.dc-hero {{ font-family: Georgia, serif; font-size: 1.45rem; line-height: 1.35; color: {NAVY}; margin: 4px 0 2px; }}
.dc-hero b {{ color: #B4541A; }}
.dc-muted {{ color: {INK_SOFT}; font-size: 0.9rem; }}
.dc-card {{ background: #FFFFFF; border: 1px solid #E6DECF; border-left: 6px solid var(--lvl);
  border-radius: 12px; padding: 14px 16px; height: 100%; }}
.dc-card .t {{ font-weight: 600; color: {NAVY}; font-size: 0.95rem; }}
.dc-card .n {{ font-family: Georgia, serif; font-size: 2rem; color: {NAVY}; line-height: 1.2; }}
.dc-card .p {{ color: {INK_SOFT}; font-size: 0.9rem; }}
.dc-card .a {{ color: {NAVY}; font-size: 0.88rem; margin-top: 8px; }}
.dc-badge {{ display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 0.85rem;
  font-weight: 600; color: {NAVY}; background: #FFFFFF; border: 2px solid var(--lvl); }}
.dc-factor {{ padding: 6px 0; border-bottom: 1px solid #EFE8DA; color: {NAVY}; }}
.dc-factor small {{ color: {INK_SOFT}; }}
.dc-facts {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 6px 0 10px; }}
.dc-facts small {{ display: block; color: {INK_SOFT}; font-size: 0.8rem; }}
.dc-facts b {{ font-family: Georgia, serif; font-size: 1.15rem; color: {NAVY}; }}
</style>
"""


def _int(value) -> str:
    return f"{int(value):,}".replace(",", ".")


def _pct(value: float) -> str:
    return f"{100 * value:.1f} %".replace(".", ",")


@st.cache_resource
def get_artifact() -> dict:
    return load_artifact()


@st.cache_data(show_spinner="Preparando los pedidos...")
def get_orders() -> pd.DataFrame:
    """Pedidos del período de test (los que el modelo no vio), con todas sus variables."""
    tables = {**load_raw_tables(), "geolocation": load_geolocation()}
    dataset = build_model_dataset(tables, get_artifact()["anchor"])
    _, _, test = temporal_three_way_split(dataset)
    return test.reset_index(drop=True)


@st.cache_data
def get_scored_orders() -> pd.DataFrame:
    return orders_with_risk(get_artifact(), get_orders())


def show_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="dc-header"><div class="dc-logo">DataCraft<span>.</span></div>'
        '<div class="dc-sub">Riesgo de retraso por pedido · Olist</div></div>',
        unsafe_allow_html=True,
    )


def show_filters(scored: pd.DataFrame) -> tuple:
    dates = scored[PREDICTION_TIME_COL].dt.date
    left, middle, right = st.columns([2, 2, 1.4], vertical_alignment="bottom")
    period = left.date_input(
        "Pedidos aprobados entre",
        (dates.min(), dates.max()),
        min_value=dates.min(),
        max_value=dates.max(),
        format="DD/MM/YYYY",
    )
    search = middle.text_input("Buscar un pedido", placeholder="Pegá el código del pedido")
    show_real = right.toggle("Mostrar si se retrasó", help="Dato histórico: en un pedido nuevo todavía no se sabe.")
    start, end = period if len(period) == 2 else (period[0], period[0])
    return scored[(dates >= start) & (dates <= end)], search.strip(), show_real


def show_hero(summary: pd.DataFrame, n_orders: int) -> None:
    high = summary.loc["alto"]
    if "pct_retrasos" in summary.columns and n_orders:
        text = (
            f"Revisando el <b>{_pct(high['pct_pedidos'])}</b> de los pedidos (nivel alto) "
            f"se encuentra el <b>{_pct(high['pct_retrasos'])}</b> de los retrasos."
        )
    else:
        text = f"Hay <b>{_int(high['pedidos'])}</b> pedidos en riesgo alto para revisar."
    st.markdown(f'<p class="dc-hero">{text}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="dc-muted">{_int(n_orders)} pedidos en el período elegido. '
        "El nivel ordena los pedidos por riesgo para decidir por dónde empezar.</p>",
        unsafe_allow_html=True,
    )


def show_level_cards(summary: pd.DataFrame) -> None:
    for column, level in zip(st.columns(len(LEVELS)), LEVELS):
        row = summary.loc[level]
        title, action = LEVEL_INFO[level]
        late = (
            f" · se retrasó el {_pct(row['tasa_retraso_real'])}"
            if "tasa_retraso_real" in summary.columns and row["pedidos"]
            else ""
        )
        column.markdown(
            f'<div class="dc-card" style="--lvl:{LEVEL_COLORS[level]}">'
            f'<div class="t">{SEMAFORO[level]} {title}</div>'
            f'<div class="n">{_int(row["pedidos"])}</div>'
            f'<div class="p">{_pct(row["pct_pedidos"])} de los pedidos{late}</div>'
            f'<div class="a">{action}</div></div>',
            unsafe_allow_html=True,
        )


def show_share_chart(summary: pd.DataFrame) -> None:
    """Dos barras al 100 %: cómo se reparten los pedidos y cómo se reparten los retrasos."""
    if "pct_retrasos" not in summary.columns:
        return
    rows = [
        {"grupo": group, "nivel": LEVEL_INFO[level][0], "orden": i, "parte": summary.loc[level, column]}
        for group, column in [("Pedidos", "pct_pedidos"), ("Retrasos", "pct_retrasos")]
        for i, level in enumerate(LEVELS)
    ]
    data = pd.DataFrame(rows)
    data["fin"] = data.groupby("grupo")["parte"].cumsum()
    data["inicio"] = data["fin"] - data["parte"]
    data["etiqueta"] = data["parte"].map(_pct)
    color = alt.Color(
        "nivel:N",
        scale=alt.Scale(domain=[LEVEL_INFO[l][0] for l in LEVELS], range=[LEVEL_COLORS[l] for l in LEVELS]),
        legend=alt.Legend(title=None, orient="top", labelColor=NAVY, labelFontSize=13),
    )
    y = alt.Y(
        "grupo:N",
        title=None,
        sort=["Pedidos", "Retrasos"],
        axis=alt.Axis(labelColor=NAVY, labelFontSize=13, domain=False, ticks=False, labelPadding=8),
    )
    x = alt.X("inicio:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 1]))
    base = alt.Chart(data).encode(y=y)
    bars = base.mark_bar(size=28, stroke=CREAM, strokeWidth=2, cornerRadius=4).encode(
        x=x,
        x2="fin:Q",
        color=color,
        tooltip=[alt.Tooltip("grupo:N", title=""), alt.Tooltip("nivel:N", title="Nivel"), alt.Tooltip("etiqueta:N", title="Parte")],
    )
    labels = (
        base.transform_filter(alt.datum.orden == 0)
        .mark_text(align="left", dx=8, color="#FFFFFF", fontWeight="bold", fontSize=13)
        .encode(x=x, text="etiqueta:N")
    )
    chart = (bars + labels).properties(height=alt.Step(40)).configure_view(stroke=None)
    st.markdown("##### Dónde están los retrasos")
    st.altair_chart(chart, width="stretch")
    st.caption("La parte roja de los retrasos es más grande que la de los pedidos: el nivel alto concentra retrasos.")


def level_picker(summary: pd.DataFrame) -> list[str]:
    picked = st.segmented_control(
        "Mostrar",
        list(LEVELS),
        default=["alto"],
        selection_mode="multi",
        format_func=lambda l: f"{SEMAFORO[l]} {LEVEL_INFO[l][0]} ({_int(summary.loc[l, 'pedidos'])})",
    )
    return picked or list(LEVELS)


def table_view(orders: pd.DataFrame, show_real: bool) -> pd.DataFrame:
    """Tabla con nombres de columna para operaciones."""
    view = pd.DataFrame(
        {
            "Nivel": orders["semaforo"] + " " + orders["nivel_riesgo"].str.capitalize(),
            "Pedido": orders["order_id"],
            "Aprobado": orders[PREDICTION_TIME_COL],
            "Estado": orders["customer_state"],
            "Plazo (días)": orders["promised_days"],
            "Distancia (km)": orders["max_distance_km"].round(0),
            "Precio (R$)": orders["total_price"],
            "Flete (R$)": orders["total_freight"],
            "Puntaje": orders["prob_retraso"],
        }
    )
    if show_real and TARGET_COL in orders.columns:
        # Al lado del nivel, para que se vea sin desplazar la tabla.
        view.insert(1, "¿Se retrasó?", orders[TARGET_COL].map({1: "Sí", 0: "No"}).to_numpy())
    return view


def show_table(orders: pd.DataFrame, show_real: bool):
    """Tabla seleccionable. Devuelve el order_id elegido (o el primero de la lista)."""
    view = table_view(orders, show_real)
    event = st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Pedido": st.column_config.TextColumn(width="small", help="Código completo del pedido"),
            "Aprobado": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
            "Distancia (km)": st.column_config.NumberColumn(format="%d"),
            "Precio (R$)": st.column_config.NumberColumn(format="%.0f"),
            "Flete (R$)": st.column_config.NumberColumn(format="%.0f"),
            "Puntaje": st.column_config.ProgressColumn(
                format="%.2f", min_value=0, max_value=1, help="Sirve para ordenar: más alto, más riesgo."
            ),
        },
    )
    left, right = st.columns([3, 1], vertical_alignment="center")
    note = f"{_int(len(view))} pedidos, del más riesgoso al menos riesgoso. "
    if "¿Se retrasó?" in view.columns:
        late = view["¿Se retrasó?"] == "Sí"
        note += f"Se retrasaron {_int(late.sum())} ({_pct(late.mean())}). "
    left.caption(note + "Marcá la casilla de una fila para ver su detalle a la derecha.")
    right.download_button(
        "Descargar lista (CSV)",
        view.to_csv(index=False).encode("utf-8-sig"),
        file_name="pedidos_a_revisar.csv",
        mime="text/csv",
        width="stretch",
    )
    rows = event.selection.rows
    return orders["order_id"].iloc[rows[0] if rows else 0]


def show_detail(order_id: str, scored: pd.DataFrame, orders: pd.DataFrame) -> None:
    order = scored.loc[scored["order_id"] == order_id].iloc[0]
    level = order["nivel_riesgo"]
    title, action = LEVEL_INFO[level]
    with st.container(border=True):
        st.markdown(f"**Pedido** `{order_id[:12]}…`", help=order_id)
        st.markdown(
            f'<span class="dc-badge" style="--lvl:{LEVEL_COLORS[level]}">{SEMAFORO[level]} {title}</span>'
            f'<p class="dc-muted" style="margin-top:8px">{action}</p>',
            unsafe_allow_html=True,
        )
        facts = [
            ("Plazo prometido", format_feature_value("promised_days", order["promised_days"])),
            ("Distancia", format_feature_value("max_distance_km", order["max_distance_km"])),
            ("Estado del cliente", order["customer_state"]),
        ]
        st.markdown(
            '<div class="dc-facts">'
            + "".join(f"<div><small>{label}</small><b>{value}</b></div>" for label, value in facts)
            + "</div>",
            unsafe_allow_html=True,
        )

        factors = risk_factors(get_artifact(), orders, order_id)
        up = factors[factors["aporte"] > 0].head(FACTORS_UP)
        down = factors[factors["aporte"] < 0].sort_values("aporte").head(FACTORS_DOWN)
        st.markdown("###### Por qué tiene este nivel")
        _factor_list(up, "▲", "Sube el riesgo")
        if not down.empty:
            st.markdown("###### Lo que juega a favor")
            _factor_list(down, "▼", "Baja el riesgo")
        st.caption("Comparado con el pedido promedio del período.")


def _factor_list(factors: pd.DataFrame, arrow: str, meaning: str) -> None:
    for row in factors.itertuples():
        st.markdown(
            f'<div class="dc-factor">{arrow} <b>{row.etiqueta}</b>: {row.valor} <small>· {meaning}</small></div>',
            unsafe_allow_html=True,
        )


def show_help() -> None:
    with st.expander("Cómo leer esta pantalla"):
        st.markdown(
            "- **El nivel es lo importante.** Alto, medio y bajo salen de ordenar los pedidos por riesgo: "
            "el alto se pensó para el 5 % más riesgoso, y en estos meses marca cerca del 10 %.\n"
            "- **El puntaje sirve para ordenar, no es una probabilidad exacta.** En los meses nuevos sale "
            "más alto que la tasa real de retrasos.\n"
            "- **El detalle muestra qué variables pesan** en ese pedido frente al promedio, según el modelo. "
            "No quiere decir que esa sea la causa del retraso.\n"
            "- El modelo usa solo datos conocidos al aprobar el pago. Detalle en `docs/metricas_evaluacion.md`."
        )


def main() -> None:
    st.set_page_config(page_title="Riesgo de retraso · DataCraft", page_icon="🚚", layout="wide")
    show_header()
    try:
        orders, scored = get_orders(), get_scored_orders()
    except FileNotFoundError as error:
        st.error(f"No se pudo cargar la app. {error}")
        st.stop()

    in_period, search, show_real = show_filters(scored)
    summary = summary_by_level(in_period)
    show_hero(summary, len(in_period))
    show_level_cards(summary)
    st.write("")
    show_share_chart(summary)

    st.markdown("### Pedidos a revisar")
    levels = level_picker(summary)
    listed = in_period[in_period["nivel_riesgo"].isin(levels)]
    if search:
        listed = in_period[in_period["order_id"].str.contains(search, case=False, regex=False)]
    if listed.empty:
        st.info("No hay pedidos con esos filtros. Probá con otro período, otro nivel u otro código.")
    else:
        table_col, detail_col = st.columns([2.5, 1], gap="medium")
        with table_col:
            order_id = show_table(listed, show_real)
        with detail_col:
            show_detail(order_id, scored, orders)
    show_help()
    trained_with = get_artifact()["entrenado_con"].replace("validacion", "validación")
    st.caption(f"Modelo: regresión logística calibrada, entrenada con {trained_with}.")


main()
