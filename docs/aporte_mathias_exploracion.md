# Mi aporte: exploración, target y control de variables

## Objetivo de mi parte

Mi trabajo se concentró en entender los datos antes de entrenar modelos. La idea fue comprobar qué representa cada fila, usar el mismo target que el resto del equipo y revisar que las variables no contengan información del futuro.

El orden que seguí fue:

1. cargar los CSV originales;
2. revisar la tabla de pedidos;
3. comprobar la granularidad;
4. entender los estados y las fechas;
5. aplicar el target oficial;
6. explicar por qué antes aparecían tasas de 6,77 % y 8,11 %;
7. construir las variables compartidas;
8. revisar fuga de información;
9. auditar valores faltantes o imposibles;
10. dejar el conjunto preparado para el modelado de mis compañeros.

## 1. Carga de datos

En vez de utilizar una ruta personal de Windows, el notebook usa `load_raw_tables()` de `src/data_loading.py`. Esto permite que cada integrante coloque los CSV de Kaggle en `data/raw/` y ejecute el mismo código.

La tabla principal de pedidos contiene 99.441 filas. Antes de realizar cálculos confirmé que también existen 99.441 `order_id` diferentes. Por lo tanto, en esta tabla una fila representa un pedido y no fue necesario agruparla con `groupby`.

## 2. Revisión inicial

Se revisaron:

- primeras filas y una muestra aleatoria;
- cantidad de filas y columnas;
- tipos de datos;
- valores faltantes;
- duplicados;
- distribución de `order_status`.

Las fechas ya son convertidas al cargarlas. Se usa `errors="coerce"` en las funciones compartidas cuando es necesario: si una fecha no se puede interpretar, pandas la transforma en `NaT` en lugar de detener todo el programa. Esto no corrige el dato; solamente permite detectarlo y decidir qué hacer.

## 3. Target oficial

El target se llama `pedido_retrasado`:

- `1`: el pedido llegó en un día posterior al prometido;
- `0`: llegó el día prometido o antes.

No lo calculé manualmente en el notebook. Utilicé `build_target()` de `src/target.py`, para que todo el equipo aplique exactamente la misma regla.

La función conserva pedidos entregados, exige fechas observables y limita el período de estudio a compras desde enero de 2017 hasta agosto de 2018. El resultado oficial es:

- 96.203 pedidos;
- 89.672 puntuales;
- 6.531 retrasados;
- tasa de retraso: 6,79 %.

La clase positiva es minoritaria. Por eso la exactitud sola no alcanza para evaluar el modelo: un modelo que predijera siempre “puntual” tendría una exactitud alta, pero no detectaría ninguna demora.

## 4. Diferencia entre 6,77 %, 8,11 % y 6,79 %

Las tres cifras no usan exactamente la misma regla o población:

- **8,11 %:** comparación de fecha y hora completas en los 96.470 pedidos entregados observables. Como la fecha prometida está a las 00:00, marca como tardíos pedidos entregados durante ese mismo día.
- **6,77 %:** comparación por día calendario sobre esos mismos 96.470 pedidos. Corrige el problema de la hora.
- **6,79 %:** comparación por día calendario y período oficial enero de 2017-agosto de 2018. Esta es la cifra del proyecto.

El notebook conserva esta comparación como auditoría, pero el target usado por los modelos siempre se obtiene mediante `build_target()`.

## 5. Momento de predicción

El equipo decidió predecir al aprobarse el pago. En el código esto se representa con `anchor="aprobacion"`.

La pregunta del modelo es: **con la información disponible cuando se aprobó el pago, ¿este pedido tiene riesgo de llegar tarde?**

Al construir las variables se excluyeron 14 pedidos sin `order_approved_at`. El conjunto quedó en 96.189 pedidos. Esto es coherente: si no conocemos el momento elegido para predecir, ese registro no puede entrar en esta versión del modelo.

## 6. Variables predictoras

`build_features()` construye una fila por pedido y 16 variables del baseline:

- momento de compra: mes, día de la semana y hora;
- plazo prometido;
- tiempo hasta la aprobación;
- cantidad de artículos y vendedores;
- precio y flete;
- relación entre flete y precio;
- peso y volumen;
- estado del cliente y relación con el estado del vendedor;
- tipo principal de pago y cuotas.

El módulo nuevo `src/feature_engineering.py` no reemplaza esta función. Parte de estas variables y agrega distancia, región, categoría e historial previo del vendedor para comparar modelos.

## 7. Control anti-leakage

La fuga de información aparece cuando el modelo recibe un dato que en la realidad todavía no existiría al momento de predecir.

No se utilizan como predictores:

- fecha de despacho;
- fecha real de entrega;
- estado final del pedido;
- reseñas realizadas después de la compra;
- duración calculada usando la entrega real.

La fecha real de entrega sí se utiliza para crear el target histórico, pero nunca entra en `X`, las variables que recibe el modelo.

## 8. Hallazgo de calidad de datos

Durante la auditoría encontré pedidos con peso, volumen o cuotas iguales a cero. Al volver a las tablas originales se observó que:

- 16 pedidos incluían alguno de 2 productos con medidas faltantes;
- 8 filas de productos tenían peso igual a cero;
- 2 registros de pago tenían cuotas iguales a cero.

Además, `groupby().sum()` convertía algunos grupos formados por valores faltantes en cero. Eso podía hacerle creer al modelo que un producto pesaba literalmente cero.

La corrección aplicada en `src/features.py` hace lo siguiente:

1. convierte medidas físicas menores o iguales a cero en faltantes;
2. mantiene como faltante el total de un pedido si no se conocen todas sus medidas;
3. convierte cuotas menores o iguales a cero en faltantes;
4. deja la imputación dentro del pipeline.

La imputación numérica usa la mediana aprendida únicamente con entrenamiento. De este modo, validación y test no influyen en la preparación del modelo.

Se agregaron pruebas automáticas para asegurar que los valores físicos inválidos y las cuotas en cero continúen representándose como faltantes.

## 9. Qué no hice en el notebook

El notebook no vuelve a programar el target ni las variables. La lógica reutilizable está en `src/`, mientras que el notebook sirve para verla, comprobarla y explicarla.

Tampoco seleccioné el modelo final. El modelado, la validación temporal, la calibración y el umbral están desarrollados en módulos compartidos por otros integrantes. Mi notebook deja documentada la base de datos que esos módulos utilizan.

## 10. Cómo ejecutar mi parte

Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Después se abre `notebooks/01_exploracion_y_target.ipynb`, se reinicia el kernel y se ejecutan todas las celdas en orden.

Antes de abrir un Pull Request se debe comprobar:

- que el notebook termine sin errores;
- que las salidas correspondan al código actual;
- que no aparezcan rutas personales;
- que los CSV no estén incluidos en Git;
- que todas las pruebas pasen.

## Conclusión personal

La parte más importante de este trabajo fue entender que un modelo no empieza al llamar a `.fit()`. Primero hay que definir correctamente qué se quiere predecir, conocer qué representa cada fila, evitar información futura y revisar si los valores tienen sentido.

El resultado de mi parte es un notebook reproducible y una mejora en la calidad de las variables base que también utiliza el modelo principal.
