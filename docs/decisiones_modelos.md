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
| **Regresión logística, selección (elegido)** | **0,209** | **0,707** | **0,073** | **0,708** |
| Árboles (HistGradientBoosting), selección | 0,212 | 0,708 | 0,051 | 0,582 |
| Árboles, todas las features | 0,205 | 0,701 | 0,046 | 0,557 |

En test el piso de PR-AUC es 0,035 (la tasa de retraso). Con el 5 % de alertas fijado en validación, el modelo elegido detecta el 20 % de los retrasos del test con 7,9 % de precisión (el baseline: 16 % con 8,7 %).

Números con las distancias ya recortadas (ver abajo). Antes del recorte el elegido daba PR-AUC 0,076 en test; la diferencia está dentro del ruido y en la validación cruzada no cambia (0,209).

**Selección de features:** las del baseline sin `purchase_month`, más `max_distance_km`. El mes empeora todos los modelos en validación: la tasa de retraso cambia mucho de un mes a otro (deriva) y el modelo aprende meses que no se repiten. Región, categoría e historial del vendedor no mejoraron la validación, por eso quedan solo en el candidato "todas las features".

## Por qué no elegimos los árboles

En la validación cruzada los árboles empatan con la logística (0,212 contra 0,209), pero en test se caen a ROC-AUC 0,58. El motivo es la deriva de `promised_days`: Olist acortó los plazos prometidos (mediana de 25 días en train, 21 en test). Los árboles aprenden cortes fijos sobre el plazo que dejan de servir cuando cambia; la logística usa una relación lineal y se sostiene. Sin el plazo, los árboles caen a ROC-AUC 0,45 en test, así que ese es el problema. También probamos el plazo relativo a la mediana reciente y la holgura contra el tiempo histórico de la ruta: no lo resuelven.

La regla de empate ya favorecía a la logística antes de mirar el test; el test lo confirma.

## Distancias: coordenadas fuera de Brasil

El geolocation trae 42 filas (21 prefijos postales) con coordenadas fuera de Brasil, en Europa o en el océano. Promediadas con las buenas, generaban distancias de hasta 8.700 km. Ahora `_zip_centroids` descarta las coordenadas fuera del rectángulo de Brasil (latitud -33,8 a 5,3; longitud -74 a -34,7) y usa la mediana por prefijo, que no se mueve por un punto suelto. La distancia máxima baja de 8.700 a 3.400 km.

| Cohorte | Pedidos | Distancia cambia más de 1 km | Tenían más de 3.500 km | Quedan sin distancia por el recorte |
|---|---|---|---|---|
| Train | 61.276 | 3.888 | 3 | 0 |
| Validación | 15.620 | 967 | 2 | 0 |
| Test | 19.293 | 1.122 | 2 | 1 |

Conteo por pedido, comparando `max_distance_km` antes del recorte (promedio de todas las coordenadas) y después (mediana de las coordenadas dentro de Brasil): "cambia más de 1 km" es diferencia absoluta estrictamente mayor a 1 km entre pedidos que tienen distancia en las dos versiones. El pedido que se queda sin distancia va solo en la última columna.

Casi todos los cambios son de pocos kilómetros y vienen de usar la mediana en lugar del promedio; solo 7 pedidos tenían distancias imposibles. En total 477 pedidos (0,5 %) no tienen distancia: 476 porque su código postal no está en el geolocation y 1 porque su prefijo solo tenía coordenadas fuera de Brasil. El pipeline la imputa con la mediana.

## Modelo final: reentrenado con train + validación

`python -m src.umbral` (parte 3) fija la calibración de Platt y los cortes de riesgo con el modelo entrenado solo con train, porque necesita predicciones de validación que el modelo no haya visto. Después reentrena el mismo pipeline con train + validación (76.896 pedidos, hasta fines de mayo 2018) y guarda ese modelo en `models/modelo_principal_calibrado.joblib`, con el calibrador y los cortes ya fijados. Con `--sin-reentrenar` se guarda el de train, como antes.

Resultado en test (19.293 pedidos, 3,5 % de retrasos):

| Modelo | ROC-AUC | PR-AUC | Riesgo medio calibrado | Nivel alto: % pedidos | Tasa de retraso en alto | % de retrasos que capta |
|---|---|---|---|---|---|---|
| Entrenado con train | 0,708 | 0,073 | 8,2 % | 8,7 % | 7,9 % | 19,8 % |
| **Reentrenado con train + validación (final)** | **0,708** | **0,074** | **8,3 %** | **9,7 %** | **7,6 %** | **21,1 %** |

Reentrenar no mejora la discriminación en test: los dos modelos ordenan los pedidos igual. Lo usamos igual para la demo porque aprende de los meses más recientes, que son los que más se parecen a los pedidos nuevos. El nivel alto sube de 8,7 % a 9,7 % de los pedidos: el calibrador se ajustó con el modelo de train y el reentrenado da puntajes un poco más altos. Hay que contarlo así en la demo: los cortes se fijan para el 5 %, pero en los meses nuevos el nivel alto marca cerca del 10 %.

**Cómo leer las probabilidades:** las probabilidades calibradas están infladas en los meses nuevos. En test el riesgo medio calibrado es 8,3 % y la tasa real 3,5 %, por dos motivos: el calibrador se ajustó con los puntajes del modelo de train y se aplica al reentrenado, y la tasa de retraso de esos meses cayó a la mitad. Lo confiable es el orden y los niveles: en el nivel alto la tasa de retraso es 2,2 veces la media (lift 2,18).

## Pendiente

- **Mathias:** sumar sus features del EDA como builders nuevos. `dias_prometidos`, `mes_compra` y `dia_semana_compra` ya existen (`promised_days`, `purchase_month`, `purchase_dayofweek`). Ideas que todavía no están: tiempo histórico de la ruta estado-estado, carga del vendedor en los últimos días, feriados y fin de mes.
- **Recalibrar con datos más nuevos** cuando haya pedidos posteriores a agosto 2018: hoy el calibrador y los cortes salen de mar-may 2018, y en los meses nuevos el nivel alto marca casi el doble del 5 % previsto.
- `models/modelo_principal.joblib`, que guarda `src/training.py`, es solo para comparar modelos. El que usa la API es `models/modelo_principal_calibrado.joblib`.
