-- Tabla analítica: una fila por pedido entregado, con el target y las
-- variables del baseline, calculadas en SQL igual que src/target.py y
-- src/features.py (anchor "aprobacion", el de por defecto).
--
-- Sirve para que cualquiera consulte la tabla sin correr Python, y para
-- verificar que SQL y pandas dan lo mismo (ver sql/03_validaciones.sql).
--
-- Reglas que replica:
--   * Solo order_status = 'delivered' con fecha de entrega y estimada.
--   * Período de estudio (decisión 8): compras desde 2017-01-01 hasta 2018-09-01 exclusive.
--   * pedido_retrasado por DÍA CALENDARIO: DATE(entrega) > DATE(estimada).
--   * Solo pedidos con order_approved_at (anchor aprobación).
--   * Sin fuga: no usa fechas de despacho/entrega ni reseñas como variable.

CREATE OR REPLACE VIEW `PROYECTO.olist.pedidos_analitica` AS
WITH entregados AS (
  SELECT
    order_id,
    customer_id,
    order_purchase_timestamp,
    order_approved_at,
    order_estimated_delivery_date,
    CAST(DATE(order_delivered_customer_date) > DATE(order_estimated_delivery_date) AS INT64) AS pedido_retrasado
  FROM `PROYECTO.olist.orders`
  WHERE order_status = 'delivered'
    AND order_purchase_timestamp >= TIMESTAMP('2017-01-01')
    AND order_purchase_timestamp <  TIMESTAMP('2018-09-01')
    AND order_delivered_customer_date IS NOT NULL
    AND order_estimated_delivery_date IS NOT NULL
    AND order_approved_at IS NOT NULL
),
carrito AS (
  SELECT
    i.order_id,
    COUNT(*) AS n_items,
    COUNT(DISTINCT i.seller_id) AS n_sellers,
    SUM(i.price) AS total_price,
    SUM(i.freight_value) AS total_freight,
    SUM(p.product_weight_g) AS total_weight_g,
    SUM(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS total_volume_cm3,
    ARRAY_AGG(DISTINCT s.seller_state IGNORE NULLS) AS seller_states
  FROM `PROYECTO.olist.order_items` i
  LEFT JOIN `PROYECTO.olist.products` p USING (product_id)
  LEFT JOIN `PROYECTO.olist.sellers` s USING (seller_id)
  GROUP BY i.order_id
),
pagos AS (
  SELECT
    order_id,
    -- tipo de pago con mayor valor (igual que features.py: ordena por payment_value desc)
    ARRAY_AGG(payment_type ORDER BY payment_value DESC LIMIT 1)[OFFSET(0)] AS main_payment_type,
    MAX(payment_installments) AS max_installments
  FROM `PROYECTO.olist.order_payments`
  GROUP BY order_id
)
SELECT
  e.order_id,
  e.order_purchase_timestamp,
  e.order_approved_at AS prediction_time,
  EXTRACT(MONTH FROM e.order_purchase_timestamp) AS purchase_month,
  -- pandas dayofweek: lunes=0 ... domingo=6. BigQuery DAYOFWEEK: domingo=1 ... sábado=7.
  MOD(EXTRACT(DAYOFWEEK FROM e.order_purchase_timestamp) + 5, 7) AS purchase_dayofweek,
  EXTRACT(HOUR FROM e.order_purchase_timestamp) AS purchase_hour,
  DATE_DIFF(DATE(e.order_estimated_delivery_date), DATE(e.order_purchase_timestamp), DAY) AS promised_days,
  TIMESTAMP_DIFF(e.order_approved_at, e.order_purchase_timestamp, SECOND) / 3600.0 AS approval_hours,
  c.customer_state,
  k.n_items,
  k.n_sellers,
  k.total_price,
  k.total_freight,
  SAFE_DIVIDE(k.total_freight, NULLIF(k.total_price, 0)) AS freight_ratio,
  k.total_weight_g,
  k.total_volume_cm3,
  pg.main_payment_type,
  pg.max_installments,
  CASE
    WHEN k.seller_states IS NULL OR ARRAY_LENGTH(k.seller_states) = 0 THEN 'unknown'
    WHEN EXISTS (SELECT 1 FROM UNNEST(k.seller_states) st WHERE st != c.customer_state) THEN 'True'
    ELSE 'False'
  END AS seller_other_state,
  e.pedido_retrasado
FROM entregados e
LEFT JOIN `PROYECTO.olist.customers` c USING (customer_id)
LEFT JOIN carrito k USING (order_id)
LEFT JOIN pagos pg USING (order_id);
