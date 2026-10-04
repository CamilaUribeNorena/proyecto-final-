# Decisiones del baseline (E3)

## Target: pedido retrasado

- Función única: `src/target.py::build_target`. Todo el equipo debería usar esta función y no recalcular el target en notebooks.
- `is_late = 1` si el pedido se entregó después de `order_estimated_delivery_date`.
- Solo pedidos con `order_status == "delivered"` y con ambas fechas presentes. En los demás el retraso no se observa.

### Por qué comparamos por día calendario

`order_estimated_delivery_date` siempre viene a las 00:00, pero `order_delivered_customer_date` trae la hora real. Si se comparan los timestamps completos, un pedido entregado **el mismo día prometido** (por ejemplo a las 15:00) cuenta como retrasado. Por eso aparecían dos tasas distintas en los chats del equipo (8,1 % y 6,8 %).

Confirmado con los datos (96.470 pedidos entregados): por timestamp da 8,11 % y por día calendario 6,77 %.

Decisión: comparamos solo la fecha (`.dt.normalize()`). Es retraso únicamente si llegó un día posterior al prometido, que es como lo vive el cliente. La comparación por timestamp queda disponible con `build_target(..., by_calendar_day=False)` y el script reporta las dos tasas (`late_rate_all_delivered` y `late_rate_timestamp_comparison`) para que quede trazable.

## Variables: solo lo que se conoce al comprar

Ver `src/features.py`. Se usan fecha de compra (mes, día de semana, hora), plazo prometido en días, carrito (ítems, vendedores, precio, flete, peso, volumen), estado del cliente, si algún vendedor está en otro estado y forma de pago.

Quedan afuera por fuga de datos: `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_status`, reseñas y cualquier duración calculada con esas fechas. `shipping_limit_date` también queda afuera por precaución, porque no está claro si se actualiza después de la compra. Hay un test (`tests/test_features.py`) que falla si alguna de esas columnas entra como variable.

## Validación temporal

- Se ordena por `order_purchase_timestamp`. Train = compras antes de la fecha de corte; test = compras desde esa fecha.
- Por defecto el corte deja aproximadamente el último 20 % de pedidos en test (`--test-fraction`).
- Nunca se usa un split aleatorio: el modelo se evalúa sobre el "futuro", como pasaría en producción.
- Ojo: los últimos meses del dataset tienen pocos pedidos entregados (los no entregados se excluyen), así que el test puede tener una tasa de retraso distinta a la de train. El script reporta ambas tasas.

## Modelo y métricas

- Regresión logística con `class_weight="balanced"` (el retraso es minoritario), imputación por mediana, escalado y one-hot.
- Métricas en test con umbral 0,5: matriz de confusión, precision, recall, F1, ROC-AUC y PR-AUC. PR-AUC es la más informativa con clases desbalanceadas; su piso es la tasa de retraso del test.

## Cómo correrlo

```bash
pip install -r requirements.txt
# CSV de Kaggle (olistbr/brazilian-ecommerce) descomprimidos en data/raw/
python -m src.baseline
python -m pytest
```

El resultado queda en `reports/baseline_metrics.json` (ignorado por Git).
