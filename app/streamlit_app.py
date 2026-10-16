"""App de riesgo de retraso por pedido, para el equipo de operaciones.

Uso, desde la raíz del repo, con los CSV en data/raw y el modelo generado
(`python -m src.umbral`):
    streamlit run app/streamlit_app.py

Por ahora arma los pedidos desde los CSV de Kaggle (período de test, los meses
que el modelo no vio). Cuando esté la API, la app le va a pedir el riesgo a la
API en lugar de cargar el modelo.

La lógica de negocio vive en app/riesgo.py, los helpers de pantalla en
app/vista.py y los estilos en app/estilos.py: este archivo compone la pantalla.
"""

import json
import sys
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # para importar src y app

from app.estilos import CREAM, CSS, LEVEL_COLORS, LEVEL_SOFT, LEVEL_TEXT, NAVY, WHITE  # noqa: E402
from app.riesgo import (  # noqa: E402
    SEMAFORO,
    format_feature_value,
    load_artifact,
    orders_with_risk,
    risk_factors,
    summary_by_level,
)
from app.vista import (  # noqa: E402
    LEVEL_ALL,
    SCORE_SCALE,
    filter_orders,
    format_factor,
    format_int,
    format_pct,
    impact_weights,
    lift_high,
    order_exists,
    plural,
    table_view,
)
from src.calibration import LEVELS  # noqa: E402
from src.data_loading import load_raw_tables  # noqa: E402
from src.feature_engineering import build_model_dataset, load_geolocation  # noqa: E402
from src.features import PREDICTION_TIME_COL  # noqa: E402
from src.validation import temporal_three_way_split  # noqa: E402

LEVEL_INFO = {
    "alto": ("Riesgo alto", "Revisar hoy: contactar al vendedor y avisar al cliente."),
    "medio": ("Riesgo medio", "Seguir de cerca durante el envío."),
    "bajo": ("Riesgo bajo", "Seguimiento normal."),
}
FACTORS_UP, FACTORS_DOWN = 3, 2
ARROWS = {"up": "↑", "down": "↓"}
TABLE_HEIGHT = 420
CHART_STEP, CHART_BAR = 40, 28  # alto de cada fila del gráfico y de la barra dentro de la fila
HINT_DEFAULT = "Viendo el pedido más riesgoso de la lista (seleccioná otra fila para inspeccionar)"
COPY_LABEL = "⧉ Copiar ID del pedido"


# ----------------------------------------------------------------------------- datos


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


# ----------------------------------------------------------------------------- filtros


def _reset_filters(seed: dict) -> None:
    """Vuelve a crear los filtros con estos valores: los widgets cambian de clave y arrancan de cero."""
    st.session_state["seed"] = seed
    st.session_state["gen"] = st.session_state.get("gen", 0) + 1


def _widget_key(name: str) -> str:
    return f"{name}-{st.session_state.get('gen', 0)}"


def _seed() -> dict:
    return st.session_state.get("seed", {})


def show_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="dc-header"><div class="dc-logo">DataCraft<span>.</span></div>'
        '<div class="dc-sub">Riesgo de retraso por pedido · Olist</div></div>',
        unsafe_allow_html=True,
    )


def show_toolbar(scored: pd.DataFrame) -> tuple[pd.DataFrame, str, bool, tuple]:
    """Filtros globales en una barra: período, búsqueda y dato histórico."""
    dates = scored[PREDICTION_TIME_COL].dt.date
    full_period = (dates.min(), dates.max())
    with st.container(key="dc-toolbar"):
        st.markdown('<div class="dc-toolbar-title">Filtros</div>', unsafe_allow_html=True)
        left, middle, right = st.columns([2, 2, 1.4], vertical_alignment="bottom")
        period = left.date_input(
            "Pedidos aprobados entre",
            _seed().get("period", full_period),
            min_value=full_period[0],
            max_value=full_period[1],
            format="DD/MM/YYYY",
            key=_widget_key("period"),
        )
        search = middle.text_input(
            "Buscar un pedido",
            _seed().get("search", ""),
            placeholder="Pegá el código del pedido",
            key=_widget_key("search"),
        )
        show_real = right.toggle(
            "Mostrar si se retrasó", key="show_real", help="Dato histórico: en un pedido nuevo todavía no se sabe."
        )
    start, end = period if len(period) == 2 else (period[0], period[0])
    return scored[(dates >= start) & (dates <= end)], search.strip(), show_real, full_period


# ----------------------------------------------------------------------------- resumen


def show_hero(summary: pd.DataFrame, n_orders: int) -> None:
    high = summary.loc["alto"]
    if "pct_retrasos" in summary.columns and n_orders:
        text = (
            f"Revisando el <b>{format_pct(high['pct_pedidos'])}</b> de los pedidos (nivel alto) "
            f"se encuentra el <b>{format_pct(high['pct_retrasos'])}</b> de los retrasos."
        )
    else:
        text = f"Hay <b>{format_int(high['pedidos'])}</b> pedidos en riesgo alto para revisar."
    st.markdown(f'<p class="dc-hero">{text}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="dc-muted">{format_int(n_orders)} pedidos en el período elegido. '
        "El nivel ordena los pedidos por riesgo para decidir por dónde empezar.</p>",
        unsafe_allow_html=True,
    )


def show_level_cards(summary: pd.DataFrame) -> None:
    for column, level in zip(st.columns(len(LEVELS)), LEVELS):
        row = summary.loc[level]
        title, action = LEVEL_INFO[level]
        late = (
            f" · se retrasó el {format_pct(row['tasa_retraso_real'])}"
            if "tasa_retraso_real" in summary.columns and row["pedidos"]
            else ""
        )
        column.markdown(
            f'<div class="dc-card" style="--lvl:{LEVEL_COLORS[level]}">'
            f'<div class="t">{SEMAFORO[level]} {title}</div>'
            f'<div class="n">{format_int(row["pedidos"])}</div>'
            f'<div class="p">{format_pct(row["pct_pedidos"])} de los pedidos{late}</div>'
            f'<div class="a">{action}</div></div>',
            unsafe_allow_html=True,
        )


def show_share_chart(summary: pd.DataFrame) -> None:
    """Dos barras al 100 % (pedidos y retrasos) y, al lado, cuánto concentra el nivel alto."""
    if "pct_retrasos" not in summary.columns:
        return
    st.markdown("##### Dónde están los retrasos")
    chart_col, lift_col = st.columns([3, 1.15], gap="medium", vertical_alignment="center")
    with chart_col:
        st.altair_chart(_share_chart(summary), width="stretch")
        st.caption("La parte roja de los retrasos es más grande que la de los pedidos: el nivel alto concentra retrasos.")
    with lift_col:
        show_lift(summary)


def show_lift(summary: pd.DataFrame) -> None:
    lift = lift_high(summary)
    if lift is None:
        return
    high = summary.loc["alto"]
    st.markdown(
        f'<div class="dc-lift"><div class="n">{format_factor(lift)}</div>'
        '<div class="t">más concentración de retrasos en el nivel alto</div>'
        f'<div class="s">{SEMAFORO["alto"]} El {format_pct(high["pct_pedidos"])} de los pedidos junta '
        f'el {format_pct(high["pct_retrasos"])} de los retrasos.</div></div>',
        unsafe_allow_html=True,
    )


def _share_chart(summary: pd.DataFrame) -> alt.LayerChart:
    rows = [
        {"grupo": group, "nivel": LEVEL_INFO[level][0], "orden": i, "parte": summary.loc[level, column]}
        for group, column in [("Pedidos", "pct_pedidos"), ("Retrasos", "pct_retrasos")]
        for i, level in enumerate(LEVELS)
    ]
    data = pd.DataFrame(rows)
    data["fin"] = data.groupby("grupo")["parte"].cumsum()
    data["inicio"] = data["fin"] - data["parte"]
    data["etiqueta"] = data["parte"].map(format_pct)
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
    bars = base.mark_bar(size=CHART_BAR, stroke=CREAM, strokeWidth=2, cornerRadius=4).encode(
        x=x,
        x2="fin:Q",
        color=color,
        tooltip=[
            alt.Tooltip("grupo:N", title=""),
            alt.Tooltip("nivel:N", title="Nivel"),
            alt.Tooltip("etiqueta:N", title="Parte"),
        ],
    )
    labels = (
        base.transform_filter(alt.datum.orden == 0)
        .mark_text(align="left", dx=8, color=WHITE, fontWeight="bold", fontSize=13)
        .encode(x=x, text="etiqueta:N")
    )
    # Línea punteada que une el final de la parte roja de una barra con el de la otra, cruzando
    # solo el hueco entre ambas: con filas de CHART_STEP píxeles, el borde inferior de la barra
    # de arriba y el superior de la de abajo quedan en posiciones fijas.
    high = summary.loc["alto"]
    gap = (CHART_STEP - CHART_BAR) / 2
    link = pd.DataFrame({"x": [high["pct_pedidos"]], "x2": [high["pct_retrasos"]]})
    connector = (
        alt.Chart(link)
        .mark_rule(color=LEVEL_COLORS["alto"], strokeDash=[4, 3], strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
            x2="x2:Q",
            y=alt.value(CHART_STEP - gap),
            y2=alt.value(CHART_STEP + gap),
        )
    )
    return (bars + labels + connector).properties(height=alt.Step(CHART_STEP)).configure_view(stroke=None)


# ----------------------------------------------------------------------------- tabla


def level_picker(summary: pd.DataFrame) -> str:
    """Un solo nivel activo (o todos), con la cantidad de pedidos de cada uno."""
    counts = {LEVEL_ALL: int(summary["pedidos"].sum()), **{l: int(summary.loc[l, "pedidos"]) for l in LEVELS}}
    labels = {LEVEL_ALL: "Todos", **{l: f"{SEMAFORO[l]} {LEVEL_INFO[l][0]}" for l in LEVELS}}
    picked = st.segmented_control(
        "Nivel de riesgo",
        [LEVEL_ALL, *LEVELS],
        default=_seed().get("level", "alto"),
        selection_mode="single",
        required=True,
        format_func=lambda l: f"{labels[l]} ({format_int(counts[l])})",
        key=_widget_key("level"),
        label_visibility="collapsed",
    )
    return picked or LEVEL_ALL


def show_table_bar(
    in_period: pd.DataFrame, summary: pd.DataFrame, search: str, show_real: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selector de nivel y descarga en la misma línea, sobre la tabla."""
    with st.container(key="dc-tablebar"):
        left, right = st.columns([3.2, 1], vertical_alignment="center")
        with left:
            level = level_picker(summary)
        listed = filter_orders(in_period, level, search)
        view = table_view(listed, show_real)
        right.download_button(
            "Descargar lista (CSV)",
            view.to_csv(index=False).encode("utf-8-sig"),
            file_name="pedidos_a_revisar.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            disabled=listed.empty,
        )
    return listed, view


def show_table(listed: pd.DataFrame, view: pd.DataFrame) -> tuple[str, int | None]:
    """Tabla seleccionable. Devuelve el pedido a inspeccionar y la fila marcada (None si no hay)."""
    event = st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=TABLE_HEIGHT,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Nivel": st.column_config.TextColumn(width="small"),
            "Puntaje": st.column_config.ProgressColumn(
                format="%d",
                min_value=0,
                max_value=SCORE_SCALE,
                width="small",
                help="De 0 a 100. Sirve para ordenar: más alto, más riesgo.",
            ),
            "Pedido": st.column_config.TextColumn(width="medium", help="Código completo del pedido"),
            "Aprobado": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
            "Estado": st.column_config.TextColumn(width="small"),
            "Plazo": st.column_config.NumberColumn(format="%d días"),
            "Distancia": st.column_config.NumberColumn(format="%d km"),
            "Precio": st.column_config.NumberColumn(format="R$ %.2f"),
            "Flete": st.column_config.NumberColumn(format="R$ %.2f"),
        },
    )
    rows = event.selection.rows
    picked = rows[0] if rows and rows[0] < len(view) else None
    st.caption(
        f"{plural(len(view), 'pedido')}, del más riesgoso al menos riesgoso. "
        "Marcá la casilla al inicio de una fila para inspeccionarla a la derecha."
    )
    return listed["order_id"].iloc[picked if picked is not None else 0], picked


def show_empty_state(scored: pd.DataFrame, search: str, full_period: tuple) -> None:
    exists = order_exists(scored, search)
    if exists:
        title = "El pedido existe, pero está fuera de los filtros actuales"
        sub = "Está en otro período o en otro nivel de riesgo. Podés verlo quitando los filtros."
    elif search:
        title = "No encontramos ningún pedido con ese código"
        sub = "Revisá que el código esté bien copiado, o limpiá la búsqueda para volver a la lista."
    else:
        title = "No hay pedidos con esos filtros"
        sub = "Probá con otro período u otro nivel de riesgo."
    st.markdown(
        f'<div class="dc-empty"><div class="i">🔎</div><div class="t">{title}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )
    with st.container(horizontal=True, horizontal_alignment="center"):
        if exists:
            st.button(
                "Ver el pedido sin filtros",
                type="primary",
                icon=":material/filter_alt_off:",
                on_click=_reset_filters,
                args=({"search": search, "level": LEVEL_ALL, "period": full_period},),
            )
        st.button(
            "Limpiar filtros",
            type="secondary" if exists else "primary",
            icon=":material/restart_alt:",
            on_click=_reset_filters,
            args=({"period": full_period},),
        )


# ----------------------------------------------------------------------------- inspector


def show_detail(order_id: str, scored: pd.DataFrame, orders: pd.DataFrame, picked: int | None) -> None:
    order = scored.loc[scored["order_id"] == order_id].iloc[0]
    level = order["nivel_riesgo"]
    title, action = LEVEL_INFO[level]
    palette = f"--lvl:{LEVEL_COLORS[level]};--soft:{LEVEL_SOFT[level]};--ink:{LEVEL_TEXT[level]}"
    hint = HINT_DEFAULT if picked is None else f"Fila {picked + 1} seleccionada en la tabla"
    with st.container(key="dc-inspector"):
        st.markdown(
            '<div class="dc-eyebrow">Inspector de pedido</div>'
            f'<div class="dc-hint{"" if picked is None else " picked"}">{hint}</div>'
            f'<div class="dc-oid">{escape(order_id)}</div>'
            f'<div class="dc-level" style="{palette}"><span class="dc-badge">{SEMAFORO[level]} {title}</span>'
            f'<span class="dc-score">Puntaje <b>{round(order["prob_retraso"] * SCORE_SCALE)}</b>/{SCORE_SCALE}</span></div>',
            unsafe_allow_html=True,
        )
        facts = [
            ("Plazo prometido", format_feature_value("promised_days", order["promised_days"])),
            ("Distancia", format_feature_value("max_distance_km", order["max_distance_km"])),
            ("Estado del cliente", order["customer_state"]),
        ]
        st.markdown(
            '<div class="dc-facts">'
            + "".join(f"<div><small>{label}</small><b>{escape(str(value))}</b></div>" for label, value in facts)
            + "</div>",
            unsafe_allow_html=True,
        )
        up, down = impact_weights(risk_factors(get_artifact(), orders, order_id), FACTORS_UP, FACTORS_DOWN)
        _factor_block("Por qué tiene este nivel", "Sube el riesgo", up, "up")
        if not down.empty:
            _factor_block("Lo que juega a favor", "Baja el riesgo", down, "down")
        st.markdown(
            '<div class="dc-fx-note">Comparado con el pedido promedio del período.</div>'
            f'<div class="dc-action" style="{palette}"><div class="k">Acción sugerida</div>'
            f'<div class="t">{SEMAFORO[level]} {action}</div></div>',
            unsafe_allow_html=True,
        )
        _copy_button(order_id)


def _factor_block(title: str, meaning: str, factors: pd.DataFrame, kind: str) -> None:
    """Chips con flecha y mini barra de impacto: se lee de un vistazo qué pesa más."""
    arrow = ARROWS[kind]
    rows = "".join(
        f'<div class="dc-fx dc-fx-{kind}" title="{meaning}">'
        f'<div class="dc-fx-row"><span class="dc-chip"><i>{arrow}</i>{escape(row.etiqueta)}</span>'
        f'<span class="dc-fx-val">{escape(row.valor)}</span></div>'
        f'<div class="dc-fx-bar"><i style="width:{row.peso:.0%}"></i></div></div>'
        for row in factors.itertuples()
    )
    st.markdown(
        f'<div class="dc-sec"><span class="dc-sec-t">{title}</span>'
        f'<span class="dc-sec-m dc-sec-{kind}">{arrow} {meaning}</span></div>{rows}',
        unsafe_allow_html=True,
    )


def _copy_button(order_id: str) -> None:
    """Botón secundario que copia el código al portapapeles (JS, en un iframe propio)."""
    st.iframe(
        f"""
<style>
  body {{ margin: 0; padding-top: 6px; font-family: "Source Sans Pro", "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif; }}
  button {{ width: 100%; height: 38px; border-radius: 8px; border: 1px solid {NAVY}; background: {WHITE}; color: {NAVY};
    font-size: 14px; font-weight: 600; cursor: pointer; transition: background .15s ease, transform .1s ease; }}
  button:hover {{ background: {CREAM}; }}
  button:active {{ transform: scale(.985); }}
  button.ok {{ background: {LEVEL_SOFT["bajo"]}; border-color: {LEVEL_COLORS["bajo"]}; color: {LEVEL_TEXT["bajo"]}; }}
</style>
<button id="copy" type="button">{COPY_LABEL}</button>
<script>
  const orderId = {json.dumps(order_id)};
  const button = document.getElementById("copy");
  const done = () => {{
    button.textContent = "✓ ID copiado";
    button.classList.add("ok");
    setTimeout(() => {{ button.textContent = {json.dumps(COPY_LABEL)}; button.classList.remove("ok"); }}, 1600);
  }};
  const fallback = () => {{
    const area = document.createElement("textarea");
    area.value = orderId;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    done();
  }};
  button.addEventListener("click", () => {{
    if (navigator.clipboard && window.isSecureContext) {{
      navigator.clipboard.writeText(orderId).then(done).catch(fallback);
    }} else {{
      fallback();
    }}
  }});
</script>
""",
        height=48,
    )


# ----------------------------------------------------------------------------- página


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

    in_period, search, show_real, full_period = show_toolbar(scored)
    summary = summary_by_level(in_period)
    show_hero(summary, len(in_period))
    show_level_cards(summary)
    st.write("")
    show_share_chart(summary)

    st.markdown("### Pedidos a revisar")
    listed, view = show_table_bar(in_period, summary, search, show_real)
    if listed.empty:
        show_empty_state(scored, search, full_period)
    else:
        table_col, detail_col = st.columns([2.5, 1], gap="medium")
        with table_col:
            order_id, picked = show_table(listed, view)
        with detail_col:
            show_detail(order_id, scored, orders, picked)
    show_help()
    trained_with = get_artifact()["entrenado_con"].replace("validacion", "validación")
    st.caption(f"Modelo: regresión logística calibrada, entrenada con {trained_with}.")


main()
