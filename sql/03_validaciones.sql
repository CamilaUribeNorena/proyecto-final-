-- Validaciones después de cargar. Cada consulta devuelve una fila con
-- el valor esperado y el obtenido; `ok` debe ser TRUE en todas.
-- Correrlas con: python -m sql.cargar_bigquery --validar
-- (o pegarlas en la consola de BigQuery reemplazando PROYECTO).

-- 1. Conteo de filas por tabla (cifras del dataset de Kaggle, versión 2)
SELECT 'orders' AS tabla, 99441 AS esperado, COUNT(*) AS obtenido, COUNT(*) = 99441 AS ok FROM `PROYECTO.olist.orders`
UNION ALL SELECT 'order_items', 112650, COUNT(*), COUNT(*) = 112650 FROM `PROYECTO.olist.order_items`
UNION ALL SELECT 'customers', 99441, COUNT(*), COUNT(*) = 99441 FROM `PROYECTO.olist.customers`
UNION ALL SELECT 'sellers', 3095, COUNT(*), COUNT(*) = 3095 FROM `PROYECTO.olist.sellers`
UNION ALL SELECT 'products', 32951, COUNT(*), COUNT(*) = 32951 FROM `PROYECTO.olist.products`
UNION ALL SELECT 'order_payments', 103886, COUNT(*), COUNT(*) = 103886 FROM `PROYECTO.olist.order_payments`
UNION ALL SELECT 'order_reviews', 99224, COUNT(*), COUNT(*) = 99224 FROM `PROYECTO.olist.order_reviews`
UNION ALL SELECT 'geolocation', 1000163, COUNT(*), COUNT(*) = 1000163 FROM `PROYECTO.olist.geolocation`
UNION ALL SELECT 'category_translation', 71, COUNT(*), COUNT(*) = 71 FROM `PROYECTO.olist.product_category_name_translation`;

-- 2. Llaves primarias únicas (BigQuery no las aplica, hay que verificarlas)
SELECT 'orders.order_id único' AS prueba, COUNT(*) = COUNT(DISTINCT order_id) AS ok FROM `PROYECTO.olist.orders`
UNION ALL SELECT 'customers.customer_id único', COUNT(*) = COUNT(DISTINCT customer_id) FROM `PROYECTO.olist.customers`
UNION ALL SELECT 'products.product_id único', COUNT(*) = COUNT(DISTINCT product_id) FROM `PROYECTO.olist.products`
UNION ALL SELECT 'sellers.seller_id único', COUNT(*) = COUNT(DISTINCT seller_id) FROM `PROYECTO.olist.sellers`
UNION ALL SELECT 'order_items (order_id, order_item_id) único', COUNT(*) = COUNT(DISTINCT CONCAT(order_id, '-', order_item_id)) FROM `PROYECTO.olist.order_items`;

-- 3. Huérfanos: llaves foráneas que no existen en la tabla padre (esperado 0)
SELECT 'orders sin customer' AS prueba, COUNT(*) AS huerfanos, COUNT(*) = 0 AS ok
FROM `PROYECTO.olist.orders` o LEFT JOIN `PROYECTO.olist.customers` c USING (customer_id) WHERE c.customer_id IS NULL
UNION ALL SELECT 'items sin order', COUNT(*), COUNT(*) = 0
FROM `PROYECTO.olist.order_items` i LEFT JOIN `PROYECTO.olist.orders` o USING (order_id) WHERE o.order_id IS NULL
UNION ALL SELECT 'items sin product', COUNT(*), COUNT(*) = 0
FROM `PROYECTO.olist.order_items` i LEFT JOIN `PROYECTO.olist.products` p USING (product_id) WHERE p.product_id IS NULL
UNION ALL SELECT 'items sin seller', COUNT(*), COUNT(*) = 0
FROM `PROYECTO.olist.order_items` i LEFT JOIN `PROYECTO.olist.sellers` s USING (seller_id) WHERE s.seller_id IS NULL
UNION ALL SELECT 'payments sin order', COUNT(*), COUNT(*) = 0
FROM `PROYECTO.olist.order_payments` p LEFT JOIN `PROYECTO.olist.orders` o USING (order_id) WHERE o.order_id IS NULL;

-- 4. Target sin filtro de período: debe coincidir con docs/decisiones.md (96.470 entregados, 6,77 % tarde por día calendario)
SELECT
  COUNT(*) AS entregados,
  ROUND(100 * AVG(CAST(DATE(order_delivered_customer_date) > DATE(order_estimated_delivery_date) AS INT64)), 2) AS pct_tarde_dia_calendario,
  ROUND(100 * AVG(CAST(order_delivered_customer_date > order_estimated_delivery_date AS INT64)), 2) AS pct_tarde_timestamp,
  COUNT(*) = 96470 AS ok_conteo
FROM `PROYECTO.olist.orders`
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL;

-- 4b. Target dentro del período de estudio (decisión 8: 2017-01-01 a 2018-09-01). Comparar con build_target() de src/.
SELECT
  COUNT(*) AS entregados_periodo,
  ROUND(100 * AVG(CAST(DATE(order_delivered_customer_date) > DATE(order_estimated_delivery_date) AS INT64)), 2) AS pct_tarde_periodo,
  MIN(DATE(order_purchase_timestamp)) AS primera_compra,
  MAX(DATE(order_purchase_timestamp)) AS ultima_compra
FROM `PROYECTO.olist.orders`
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL
  AND order_purchase_timestamp >= TIMESTAMP('2017-01-01')
  AND order_purchase_timestamp <  TIMESTAMP('2018-09-01');

-- 5. Tabla analítica: filas y tasa (período de estudio + anchor aprobación, que descarta los pocos pedidos sin order_approved_at)
SELECT COUNT(*) AS filas, ROUND(100 * AVG(pedido_retrasado), 2) AS pct_tarde,
       COUNTIF(n_items IS NULL) AS pedidos_sin_items,
       COUNTIF(main_payment_type IS NULL) AS pedidos_sin_pago
FROM `PROYECTO.olist.pedidos_analitica`;
