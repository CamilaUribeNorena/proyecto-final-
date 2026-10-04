# Decisiones del equipo

Registro corto de lo que acordamos y por qué. Cada decisión nueva se agrega abajo con fecha. Si algo cambia, se tacha y se agrega la versión nueva en lugar de borrarla.

| Estado | Significado |
|---|---|
| Acordada | El equipo la tomó y se aplica |
| Pendiente | Hay una propuesta, falta confirmar |

## 1. Alcance: el centro del proyecto es el clasificador de retrasos

- **Estado:** Pendiente de confirmar con el mentor.
- **Decisión:** el entregable principal es un clasificador binario que estima si un pedido va a llegar después del día prometido. El análisis comercial (ventas, categorías, vendedores) queda como anexo descriptivo, no como un segundo modelo.
- **Por qué:** es el problema con target claro, datos suficientes y una acción concreta (priorizar seguimiento logístico). Dos modelos en paralelo dividen el tiempo del equipo.

## 2. Target: `pedido_retrasado` por día calendario

- **Estado:** Acordada.
- **Decisión:** 1 si `order_delivered_customer_date` cae en un día posterior a `order_estimated_delivery_date`. Solo pedidos con estado `delivered` y ambas fechas presentes.
- **Por qué:** la fecha estimada viene siempre a las 00:00. Comparando con la hora, un pedido entregado el mismo día prometido cuenta como tarde. Con los datos de Kaggle: 8,11 % comparando timestamps contra 6,77 % por día calendario (96.470 pedidos entregados). Usamos 6,77 %.
- **Implementación:** una sola función reutilizable en `src/`. Nadie recalcula el target en un notebook.

## 3. Sin fuga de datos

- **Estado:** Acordada.
- **Decisión:** el modelo solo usa información disponible en el momento de predicción. Quedan afuera las fechas de despacho y entrega, el estado final del pedido, las reseñas y cualquier duración calculada con esas fechas.
- **Pendiente de unificar:** el README define el momento de predicción como la aprobación de la compra; el baseline usa la compra (más estricto, no usa `order_approved_at`). Hay que elegir uno y aplicarlo en todos los módulos.

## 4. Validación temporal

- **Estado:** Acordada.
- **Decisión:** se entrena con pedidos más viejos y se evalúa con los más nuevos, ordenados por fecha de compra. Nunca split aleatorio. Las transformaciones se ajustan solo con train.
- **Métricas:** matriz de confusión, precision, recall, F1, ROC-AUC y PR-AUC. El umbral se elige con validación, no con test.

## 5. Flujo de trabajo con Git

- **Estado:** Acordada.
- `main`: solo entregas.
- `developer`: integración. Todo entra por pull request aprobado por Agustina.
- `feature/E#-descripcion`: una rama por tarea, creada desde `developer` actualizado. PR chicos y frecuentes.
- Commits con prefijo: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- La lógica reutilizable va en `src/`; los notebooks solo la usan para explorar y mostrar resultados. Cada notebook tiene un solo dueño para evitar conflictos.

## 6. Datos y archivos fuera de Git

- **Estado:** Acordada.
- Los CSV de Kaggle se descargan a `data/raw/` y no se suben. El `.gitignore` ignora todo `data/` (salvo los `.gitkeep`), y además `*.csv`, `*.zip`, `*.pdf`, `*.docx` y `*.xlsx` en cualquier carpeta, para que nadie suba datos o documentos personales por error.
- Antes de cada commit: revisar `git status` y no usar `git add .` sin mirar.
