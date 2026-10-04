"""Definición única del target "pedido retrasado".

Regla del equipo:
    retrasado = 1 si el pedido se entregó DESPUÉS de la fecha estimada.

Decisión sobre la hora (ver docs/decisiones_baseline.md):
    order_estimated_delivery_date siempre viene a las 00:00:00, mientras que
    order_delivered_customer_date trae hora real. Si comparamos timestamps
    completos, un pedido entregado el mismo día prometido (por ejemplo a las
    15:00) cuenta como retrasado, y eso infla la tasa. Por eso comparamos por
    día calendario: solo es retraso si se entregó un día posterior al prometido.
"""

import pandas as pd

DELIVERED_COL = "order_delivered_customer_date"
ESTIMATED_COL = "order_estimated_delivery_date"
TARGET_COL = "is_late"


def build_target(orders: pd.DataFrame, by_calendar_day: bool = True) -> pd.DataFrame:
    """Devuelve una copia con solo los pedidos entregados y la columna is_late.

    Se descartan los pedidos sin estado "delivered" o sin fecha de entrega o
    estimada, porque en ellos el retraso no se puede observar.

    by_calendar_day=False reproduce la comparación por timestamp completo,
    útil solo para comparar contra la cifra alternativa.
    """
    delivered = pd.to_datetime(orders[DELIVERED_COL], errors="coerce")
    estimated = pd.to_datetime(orders[ESTIMATED_COL], errors="coerce")

    mask = (
        orders["order_status"].eq("delivered")
        & delivered.notna()
        & estimated.notna()
    )

    if by_calendar_day:
        late = delivered.dt.normalize() > estimated.dt.normalize()
    else:
        late = delivered > estimated

    return (
        orders.assign(
            **{DELIVERED_COL: delivered, ESTIMATED_COL: estimated, TARGET_COL: late.astype(int)}
        )
        .loc[mask]
        .reset_index(drop=True)
    )
