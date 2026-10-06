"""Pruebas sobre los archivos SQL de la base de datos (no necesitan BigQuery)."""

import re
from pathlib import Path

import pytest

from src.features import LEAKY_COLUMNS

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def test_archivos_sql_existen():
    for nombre in ["01_esquema.sql", "02_tabla_analitica.sql", "03_validaciones.sql", "cargar_bigquery.py"]:
        assert (SQL_DIR / nombre).exists(), f"falta sql/{nombre}"


def test_esquema_crea_las_nueve_tablas():
    sql = (SQL_DIR / "01_esquema.sql").read_text(encoding="utf-8")
    tablas = re.findall(r"CREATE TABLE IF NOT EXISTS `PROYECTO\.olist\.(\w+)`", sql)
    esperadas = {
        "customers", "sellers", "products", "product_category_name_translation",
        "geolocation", "orders", "order_items", "order_payments", "order_reviews",
    }
    assert set(tablas) == esperadas


def test_vista_analitica_no_usa_columnas_con_fuga_como_variable():
    """La vista puede usar order_delivered_customer_date SOLO para construir pedido_retrasado
    (dentro del CTE `entregados`), nunca en el SELECT final de variables."""
    sql = (SQL_DIR / "02_tabla_analitica.sql").read_text(encoding="utf-8")
    select_final = sql.split("SELECT\n  e.order_id")[1]
    for col in LEAKY_COLUMNS:
        assert col not in select_final, f"{col} aparece como variable en la vista"


def test_target_en_sql_compara_por_dia_calendario():
    sql = (SQL_DIR / "02_tabla_analitica.sql").read_text(encoding="utf-8")
    assert "DATE(order_delivered_customer_date) > DATE(order_estimated_delivery_date) AS INT64) AS pedido_retrasado" in sql


def test_vista_usa_el_mismo_periodo_de_estudio_que_src():
    sql = (SQL_DIR / "02_tabla_analitica.sql").read_text(encoding="utf-8")
    assert "TIMESTAMP('2017-01-01')" in sql and "TIMESTAMP('2018-09-01')" in sql


def test_las_sentencias_sql_se_parten_sin_comentarios():
    """Un ';' dentro de un comentario no debe generar una sentencia basura."""
    pytest.importorskip("google.cloud.bigquery")  # solo quien tenga la librería corre esta prueba
    from sql.cargar_bigquery import sentencias

    for nombre in ["01_esquema.sql", "02_tabla_analitica.sql", "03_validaciones.sql"]:
        for s in sentencias(nombre, "p"):
            assert s.upper().startswith(("CREATE", "SELECT", "WITH")), f"{nombre}: sentencia rara: {s[:40]!r}"
