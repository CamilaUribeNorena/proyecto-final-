"""Carga los 9 CSV de Olist (data/raw/) a BigQuery y crea la tabla analítica.

Uso (desde la raíz del repo, con el entorno activado):

    pip install google-cloud-bigquery pandas-gbq db-dtypes
    gcloud auth application-default login      # una sola vez
    python -m sql.cargar_bigquery --proyecto MI-PROYECTO-GCP
    python -m sql.cargar_bigquery --proyecto MI-PROYECTO-GCP --validar

Qué hace:
  1. Crea el dataset `olist` si no existe.
  2. Ejecuta sql/01_esquema.sql (tablas con tipos explícitos).
  3. Carga cada CSV con el esquema de la tabla (WRITE_TRUNCATE: se puede
     repetir sin duplicar filas).
  4. Ejecuta sql/02_tabla_analitica.sql (vista pedidos_analitica).
  5. Con --validar, corre sql/03_validaciones.sql e imprime los resultados.

Es idempotente: correrlo dos veces deja la base igual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
RAW_DIR = ROOT / "data" / "raw"
DATASET = "olist"

# tabla en BigQuery -> archivo CSV de Kaggle
CSV_POR_TABLA = {
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "product_category_name_translation": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
}


def leer_sql(nombre: str, proyecto: str) -> str:
    return (SQL_DIR / nombre).read_text(encoding="utf-8").replace("PROYECTO", proyecto)


def sentencias(nombre: str, proyecto: str) -> list[str]:
    """Quita los comentarios '--' y parte el .sql en sentencias separadas por ';'.

    Los comentarios se quitan ANTES de partir, porque un ';' dentro de un
    comentario rompería la sentencia.
    """
    sin_comentarios = "\n".join(
        linea.split("--", 1)[0] for linea in leer_sql(nombre, proyecto).splitlines()
    )
    return [s.strip() for s in sin_comentarios.split(";") if s.strip()]


def ejecutar_script(client: bigquery.Client, nombre: str, proyecto: str) -> None:
    """Ejecuta un .sql con varias sentencias separadas por ';'."""
    for sentencia in sentencias(nombre, proyecto):
        client.query(sentencia).result()


def cargar_csv(client: bigquery.Client, proyecto: str, tabla: str, archivo: str) -> int:
    ruta = RAW_DIR / archivo
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Descarga el dataset de Kaggle (olistbr/brazilian-ecommerce) en data/raw/.")

    table_id = f"{proyecto}.{DATASET}.{tabla}"
    esquema = client.get_table(table_id).schema  # el esquema ya lo creó 01_esquema.sql
    config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        schema=esquema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        allow_quoted_newlines=True,  # las reseñas tienen saltos de línea dentro de comillas
        null_marker="",
    )
    with ruta.open("rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=config)
    job.result()
    return client.get_table(table_id).num_rows


def validar(client: bigquery.Client, proyecto: str) -> bool:
    """Corre 03_validaciones.sql y devuelve True si todas las columnas `ok` son TRUE."""
    todo_ok = True
    for sentencia in sentencias("03_validaciones.sql", proyecto):
        df = client.query(sentencia).result().to_dataframe()
        print(df.to_string(index=False))
        print("-" * 60)
        for col in df.columns:
            if col.startswith("ok") and not df[col].all():
                todo_ok = False
    return todo_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proyecto", required=True, help="id del proyecto de Google Cloud")
    parser.add_argument("--validar", action="store_true", help="correr sql/03_validaciones.sql al final")
    parser.add_argument("--solo-validar", action="store_true", help="no cargar, solo validar")
    args = parser.parse_args(argv)

    client = bigquery.Client(project=args.proyecto)

    if not args.solo_validar:
        client.create_dataset(bigquery.Dataset(f"{args.proyecto}.{DATASET}"), exists_ok=True)
        print(f"Dataset {args.proyecto}.{DATASET} listo")

        ejecutar_script(client, "01_esquema.sql", args.proyecto)
        print("Esquema creado")

        for tabla, archivo in CSV_POR_TABLA.items():
            filas = cargar_csv(client, args.proyecto, tabla, archivo)
            print(f"  {tabla:<36} {filas:>9,} filas")

        ejecutar_script(client, "02_tabla_analitica.sql", args.proyecto)
        print("Vista pedidos_analitica creada")

    if args.validar or args.solo_validar:
        ok = validar(client, args.proyecto)
        print("VALIDACIÓN:", "OK" if ok else "FALLÓ (revisar filas con ok = False)")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
