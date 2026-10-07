# Modelo principal (parte 2: features y modelos)

Cómo correrlo, con los 9 CSV de Kaggle en `data/raw/`:

```bash
python -m src.training            # predice al aprobar la compra (decisión 3)
python -m pytest
```

Escribe `reports/comparacion_modelos.json` y guarda el modelo elegido en `models/modelo_principal.joblib` (los dos ignorados por Git). Tarda menos de un minuto.

## Dataset de modelado

`src/feature_engineering.py::build_model_dataset` arma una fila por pedido: el target de `build_target` (período de la decisión 8), las variables del baseline (`src/features.py`) y las features nuevas. Con el momento de predicción en la aprobación quedan **96.189 pedidos**: los 96.203 de `build_target` menos 14 entregados sin fecha de aprobación. Es el mismo número que la vista `pedidos_analitica` de BigQuery, así que esa diferencia queda explicada.

## Features nuevas

Todas se conocen en el momento de predicción. Cada grupo es un `FeatureBuilder` registrado en `FEATURE_BUILDERS`; para sumar uno nuevo basta con escribir la función, registrarla y agregarle un test.

| Grupo | Variables | Cómo se calcula |
|---|---|---|
| Ruta | `max_distance_km`, `customer_region`, `seller_region`, `cross_region` | Distancia en línea recta entre el código postal del cliente y el de cada vendedor (promedio de lat/lng del geolocation), región de cada punta y si cruza regiones |
| Producto | `main_category` | Categoría del ítem más caro del pedido |
| Historial del vendedor | `seller_prev_orders`, `seller_prev_late_rate` | Pedidos del vendedor principal **entregados antes** del momento de predicción y su tasa de retraso (vacía con menos de 5) |

El historial del vendedor es la única feature que mira otros pedidos. Para que no haya fuga solo cuentan los pedidos ya entregados antes del momento de predicción. Hay un test que cambia el resultado de pedidos futuros y comprueba que las features de los pedidos anteriores no cambian.

## Esquema de validación

1. **Tres cohortes por fecha** con `temporal_three_way_split` de `src/validation.py`, las mismas que usa `src/umbral.py` (decisión 9): train (ene 2017 a mar 2018, 61.276 pedidos, 7,8 % de retrasos), validación (mar a may 2018, 15.620, 6,8 %) y test (fines de may a ago 2018, 19.293, 3,5 %).
2. **Elección del modelo:** validación cruzada temporal con 4 tramos sobre train + validación. Cada tramo se predice con un modelo entrenado solo con lo anterior. Gana el mejor PR-AUC medio; si otro candidato más simple queda a menos de 0,01, gana el simple.
3. **Umbral de alerta para comparar:** el 5 % de pedidos con más riesgo en validación (como el nivel "alto" de la decisión 9). El umbral y la calibración definitivos son de la parte 3.
4. **Test:** se calcula recién con el modelo y el umbral ya elegidos. Se informa para los 4 candidatos como diagnóstico; no cambia la elección.

## Resultados

| Modelo | PR-AUC CV | ROC-AUC CV | PR-AUC test | ROC-AUC test |
|---|---|---|---|---|
| Regresión logística, baseline (referencia) | 0,206 | 0,701 | 0,072 | 0,700 |
| **Regresión logística, selección (elegido)** | **0,209** | **0,707** | **0,076** | **0,709** |
| Árboles (HistGradientBoosting), selección | 0,212 | 0,707 | 0,052 | 0,584 |
| Árboles, todas las features | 0,205 | 0,701 | 0,046 | 0,561 |

En test el piso de PR-AUC es 0,035 (la tasa de retraso). Con el 5 % de alertas fijado en validación, el modelo elegido detecta el 20 % de los retrasos del test con 8,1 % de precisión (el baseline: 15 % con 8,7 %).

**Selección de features:** las del baseline sin `purchase_month`, más `max_distance_km`. El mes empeora todos los modelos en validación: la tasa de retraso cambia mucho de un mes a otro (deriva) y el modelo aprende meses que no se repiten. Región, categoría e historial del vendedor no mejoraron la validación, por eso quedan solo en el candidato "todas las features".

## Por qué no elegimos los árboles

En la validación cruzada los árboles empatan con la logística (0,212 contra 0,209), pero en test se caen a ROC-AUC 0,58. El motivo es la deriva de `promised_days`: Olist acortó los plazos prometidos (mediana de 25 días en train, 21 en test). Los árboles aprenden cortes fijos sobre el plazo que dejan de servir cuando cambia; la logística usa una relación lineal y se sostiene. Sin el plazo, los árboles caen a ROC-AUC 0,45 en test, así que ese es el problema. También probamos el plazo relativo a la mediana reciente y la holgura contra el tiempo histórico de la ruta: no lo resuelven.

La regla de empate ya favorecía a la logística antes de mirar el test; el test lo confirma.

## Pendiente

- **Parte 3:** aplicar la calibración y los niveles de riesgo de `src/umbral.py` al modelo principal (`models/modelo_principal.joblib` trae el pipeline, las columnas y el umbral del 5 %).
- **Mathias:** sumar sus features del EDA como builders nuevos. `dias_prometidos`, `mes_compra` y `dia_semana_compra` ya existen (`promised_days`, `purchase_month`, `purchase_dayofweek`). Ideas que todavía no están: tiempo histórico de la ruta estado-estado, carga del vendedor en los últimos días, feriados y fin de mes.
- **Reentrenar con train + validación** antes de la demo. Hoy el modelo guardado se entrena solo con train, para que la calibración y los cortes de la parte 3 se ajusten en validación sin sesgo. Una vez fijados, conviene reentrenar con todo lo anterior al test.
- **Recortar distancias extremas:** llegan a 8.700 km por coordenadas malas en el geolocation. No cambian el resultado, pero conviene limitar las coordenadas a Brasil.
