-- Esquema de la base de datos Olist en BigQuery.
-- Dataset: `olist` (crear antes: bq mk --dataset --location=US PROYECTO:olist)
-- Reemplazar PROYECTO por el id del proyecto de Google Cloud, o correr con
-- el script sql/cargar_bigquery.py que lo hace por ti.
--
-- BigQuery no aplica llaves primarias ni foráneas, y el sandbox rechaza
-- declararlas, así que no se declaran: las relaciones están documentadas en
-- sql/README.md y la integridad se verifica con sql/03_validaciones.sql
-- después de cargar. Tampoco se particiona orders: el sandbox borra las
-- particiones con más de 60 días.
--
-- Decisión: los códigos postales son STRING (tienen ceros a la izquierda),
-- igual que en src/data_loading.py.

CREATE TABLE IF NOT EXISTS `PROYECTO.olist.customers` (
  customer_id STRING NOT NULL,
  customer_unique_id STRING NOT NULL,
  customer_zip_code_prefix STRING,
  customer_city STRING,
  customer_state STRING
);

CREATE TABLE IF NOT EXISTS `PROYECTO.olist.sellers` (
  seller_id STRING NOT NULL,
  seller_zip_code_prefix STRING,
  seller_city STRING,
  seller_state STRING
);

CREATE TABLE IF NOT EXISTS `PROYECTO.olist.products` (
  product_id STRING NOT NULL,
  product_category_name STRING,
  product_name_lenght INT64,
  product_description_lenght INT64,
  product_photos_qty INT64,
  product_weight_g FLOAT64,
  product_length_cm FLOAT64,
  product_height_cm FLOAT64,
  product_width_cm FLOAT64
);

CREATE TABLE IF NOT EXISTS `PROYECTO.olist.product_category_name_translation` (
  product_category_name STRING NOT NULL,
  product_category_name_english STRING
);

CREATE TABLE IF NOT EXISTS `PROYECTO.olist.geolocation` (
  geolocation_zip_code_prefix STRING,
  geolocation_lat FLOAT64,
  geolocation_lng FLOAT64,
  geolocation_city STRING,
  geolocation_state STRING
);

CREATE TABLE IF NOT EXISTS `PROYECTO.olist.orders` (
  order_id STRING NOT NULL,
  customer_id STRING NOT NULL,
  order_status STRING,
  order_purchase_timestamp TIMESTAMP,
  order_approved_at TIMESTAMP,
  order_delivered_carrier_date TIMESTAMP,
  order_delivered_customer_date TIMESTAMP,
  order_estimated_delivery_date TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `PROYECTO.olist.order_items` (
  order_id STRING NOT NULL,
  order_item_id INT64 NOT NULL,
  product_id STRING NOT NULL,
  seller_id STRING NOT NULL,
  shipping_limit_date TIMESTAMP,
  price FLOAT64,
  freight_value FLOAT64
);

CREATE TABLE IF NOT EXISTS `PROYECTO.olist.order_payments` (
  order_id STRING NOT NULL,
  payment_sequential INT64 NOT NULL,
  payment_type STRING,
  payment_installments INT64,
  payment_value FLOAT64
);

CREATE TABLE IF NOT EXISTS `PROYECTO.olist.order_reviews` (
  review_id STRING NOT NULL,
  order_id STRING NOT NULL,
  review_score INT64,
  review_comment_title STRING,
  review_comment_message STRING,
  review_creation_date TIMESTAMP,
  review_answer_timestamp TIMESTAMP
);
-- Nota: review_id NO es única en el CSV de Kaggle (hay reseñas repetidas en
-- varios pedidos), por eso no puede ser llave única.
