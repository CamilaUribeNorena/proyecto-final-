# Mi aporte: EDA visual del retraso de pedidos

Notebook: `notebooks/02_eda_graficos.ipynb`

## Objetivo

Mostrar en gráficos **dónde, cuándo y por qué** se retrasan los pedidos, con la misma base que usa el modelo. Es el material de la primera demo y la base de los gráficos del dashboard.

El notebook **no limpia ni redefine nada**. Usa `load_raw_tables()`, `build_target()` y `build_features(anchor="aprobacion")` de `src/`, igual que `01_exploracion_y_target.ipynb`. Si alguien cambia el target o las variables en `src/`, los gráficos se actualizan solos al volver a correrlo.

## Cómo ejecutarlo

1. Los 9 CSV de Kaggle en `data/raw/` (decisión 7).
2. Abrir `notebooks/02_eda_graficos.ipynb`, reiniciar el kernel y ejecutar todo. Tarda unos 20 segundos.
3. Comprobación: tiene que imprimir **96.203** pedidos con target, **96.189** con aprobación y **6,79 %** de retraso. Hay un `assert` que se detiene si la base no coincide con la del notebook 01.

Los gráficos se guardan como PNG en `reports/figuras/` (variable `GUARDAR_FIGURAS`).

## Decisiones de diseño (y por qué)

| Decisión | Por qué |
|---|---|
| Siempre se grafica la **tasa** de retraso, no la cantidad de retrasos | La cantidad depende del volumen. SP tiene más retrasos que AL porque vende 100 veces más, no porque sea peor |
| Línea punteada con la **tasa global (6,79 %)** en todos los gráficos | Lo que importa es cuánto se aleja cada grupo del promedio |
| Se muestra el ***n*** de cada grupo | Una tasa alta con pocos pedidos es ruido |
| Estados con **menos de 300 pedidos** fuera del gráfico 2 (RR, AP, AC, AM, RO, TO) | Con 40 a 274 pedidos, un puñado de casos mueve la tasa varios puntos |
| **Año-mes** en lugar de `purchase_month` | El período tiene enero-agosto dos veces y septiembre-diciembre una vez. Sin año, septiembre parece "vender menos" |
| Se separa **SP del resto** en los gráficos 3 y 4 | SP tiene el 42 % de los pedidos, los plazos más cortos y la tasa más baja. Sin separarlo, mezcla geografía con el efecto que se quiere ver |
| **Sin doble eje Y** (volumen y tasa van en dos paneles) | Con dos escalas, el cruce de las líneas depende de dónde se ponen los ejes y sugiere relaciones que no existen |
| La fecha real de entrega se usa **solo en el anexo A**, como descriptivo | Es información posterior a la aprobación: no puede ser variable del modelo (decisión 4) |

## Hallazgos

1. **El retraso no es estable en el tiempo.** Picos en noviembre de 2017 (12,4 %, Black Friday, volumen +63 %) y en febrero-marzo de 2018 (14,1 % y 19,0 %), estos últimos **sin** aumento de volumen. En junio de 2018 la tasa cae a 1,2 % y ese mes el plazo mediano prometido sube de 22 a 28 días. Refuerza la validación temporal (decisión 5).
2. **La geografía manda.** El Nordeste triplica la tasa global (AL 21,5 %, MA 17,6 %, SE 15,4 %). **RJ** es el caso más accionable: tiene el 12,8 % de los pedidos y el **22,9 % de los retrasos**.
3. **El envío interestatal importa solo fuera de SP.** En bruto: 8,1 % contra 4,5 %. Para clientes de SP no hay diferencia (4,0 % contra 4,7 %). Fuera de SP: 3,5 % con vendedor local contra 8,8 % con vendedor de otro estado. Es una **interacción** entre `seller_other_state` y `customer_state`.
4. **El plazo prometido funciona como margen de seguridad.** Las promesas de 10 días o menos fallan 1 de cada 10 veces en ambas regiones. Desde los 21–25 días, cuanto más plazo, menos retraso. La relación **no es lineal**. Para un plazo igual, SP siempre se retrasa menos. La comparación de medias del notebook 01 (1,1 días de diferencia) subestimaba la relación: dentro de SP la diferencia es de 3 días y en el resto, de 2. La geografía la atenúa.

**Anexo:**
- A. Cuando un pedido llega tarde, la demora mediana es de 7 días; el 10 % supera los 22.
- B. La relación flete/precio no separa (6,5 % a 7,0 % en todos los quintiles). Las horas hasta aprobar el pago separan poco (6,3 % a 8,3 %).
- C. Rareza: los pedidos con 2 o más vendedores se retrasan 1,0 % contra 6,9 %, con el mismo plazo y la misma proporción de SP. Hipótesis abierta: cómo se registra la entrega de pedidos con varios paquetes.

## Implicancias para el modelo (para conversar con quien modela)

- Los hallazgos 3 y 4 describen relaciones **no lineales y con interacción**. La regresión logística (decisión 10) solo las capta si se agregan términos de interacción o tramos. Es un argumento concreto para revisar si un modelo de árboles mejora el PR-AUC, o para agregar `promised_days` en tramos y la interacción región × vendedor de otro estado.
- Los picos temporales sugieren que el modelo puede degradarse en temporadas atípicas. Conviene revisar el desempeño por mes en el conjunto de test.

Esto son propuestas, no decisiones: las define el equipo.

## Guion para la demo (4 minutos, un gráfico por minuto)

1. **Gráfico 1 (cuándo).** "El 6,8 % de los pedidos llega tarde, pero ese promedio esconde meses de 12 y 19 %. En Black Friday el volumen subió 63 % y la red no lo absorbió. En marzo de 2018 fue peor, sin más volumen. En junio la tasa cae a 1 %: ese mes Olist prometió 6 días más."
2. **Gráfico 2 (dónde).** "Los destinos del Nordeste triplican el retraso. Pero el dato para priorizar es Río: 13 % de los pedidos, 23 % de los retrasos."
3. **Gráfico 3 (desde dónde).** "Que el vendedor esté en otro estado duplica el riesgo, pero solo si el cliente no está en SP. La red que llega a SP funciona venga de donde venga."
4. **Gráfico 4 (cuánto margen).** "Las promesas de 10 días o menos fallan una de cada diez veces. El plazo es el margen de seguridad. Cierre: dónde va, de dónde sale y cuánto margen se prometió se conocen al aprobar el pago. Por eso se puede predecir antes de despachar."

## Preguntas probables del jurado

- **"¿Por qué no usaron el conteo de retrasos?"** Porque depende del volumen. La tasa compara grupos de distinto tamaño.
- **"¿Ese pico de marzo no es un error de datos?"** Pasa en las dos regiones (SP 9,4 %, resto 26,0 %) y con volumen normal. Parece un evento operativo real. No tenemos datos para identificar la causa.
- **"¿La diferencia SP contra el resto no se explica solo por distancia?"** En parte. El gráfico 3 muestra que con un vendedor del mismo estado, el resto de Brasil se retrasa incluso menos que SP (3,5 % contra 4,7 %). La distancia pesa, y la red de destino también.
- **"¿El anexo A no es fuga de información?"** Sería fuga si entrara al modelo. Acá solo dimensiona el problema.
- **"¿Qué hicieron con los estados chicos?"** Se excluyen del gráfico de estados (menos de 300 pedidos) pero siguen en el modelo, que recibe `customer_state` completo.

## Limitaciones

- Son asociaciones, no causas. Ningún gráfico prueba que alargar el plazo reduzca el retraso.
- La tasa es solo de pedidos entregados (decisión 2). Si los no entregados fueran en su mayoría retrasos, la tasa real sería algo mayor.
- La región se simplifica en SP y resto. Separar por las 5 macrorregiones de Brasil daría más detalle, a costa de grupos más chicos.

## Pendiente de acordar con el equipo

- **¿Se suben los PNG de `reports/figuras/`?** El `.gitignore` no los excluye. Pesan poco y sirven para el informe y las diapositivas, pero son salidas generadas: se pueden regenerar corriendo el notebook. Si no se suben, hay que agregar `reports/figuras/` al `.gitignore`.
- Si los gráficos del dashboard de Power BI deben replicar estos mismos cortes (tasa, línea global, n mínimo de 300), conviene registrarlo en `docs/decisiones.md` para que el notebook y el dashboard digan lo mismo.
