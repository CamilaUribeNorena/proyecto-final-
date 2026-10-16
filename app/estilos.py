"""Tokens de diseño y CSS de la app de riesgo (marca DataCraft).

Va separado de la app para que `streamlit_app.py` se lea como composición de
pantalla y no como una hoja de estilos. El color nunca va solo: siempre con
ícono y texto, así se entiende también en escala de grises.
"""

NAVY, CREAM, CREAM_DARK, LINE = "#13294B", "#FBF8F2", "#F1EADC", "#E6DECF"
ORANGE, ORANGE_DARK, INK_SOFT, WHITE = "#E8833A", "#B4541A", "#4A5B78", "#FFFFFF"
LEVEL_COLORS = {"alto": "#C0392B", "medio": "#D69E2E", "bajo": "#2F855A"}
LEVEL_SOFT = {"alto": "#FDECEA", "medio": "#FDF4DC", "bajo": "#E6F4EC"}
LEVEL_TEXT = {"alto": "#8E2A1F", "medio": "#7A5A0F", "bajo": "#22633F"}
UP, UP_SOFT, DOWN, DOWN_SOFT = "#C0392B", "#FDECEA", "#2F855A", "#E6F4EC"
SHADOW_SOFT = "0 1px 2px rgba(19,41,75,.05)"
SHADOW_CARD = "0 1px 2px rgba(19,41,75,.04), 0 8px 24px rgba(19,41,75,.07)"
SHADOW_HOVER = "0 4px 8px rgba(19,41,75,.06), 0 14px 32px rgba(19,41,75,.12)"

CSS = f"""
<style>
h1, h2, h3, .dc-serif {{ font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 1.2rem; max-width: 1280px; }}

/* Cabecera */
.dc-header {{ background: {NAVY}; color: {CREAM}; border-radius: 14px; padding: 18px 24px;
  display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap; margin-bottom: 18px; }}
.dc-logo {{ font-family: Georgia, serif; font-size: 1.7rem; font-weight: 600; color: {CREAM}; }}
.dc-logo span {{ color: {ORANGE}; }}
.dc-sub {{ color: #D9DEE8; font-size: 1rem; }}
.dc-hero {{ font-family: Georgia, serif; font-size: 1.45rem; line-height: 1.35; color: {NAVY}; margin: 4px 0 2px; }}
.dc-hero b {{ color: {ORANGE_DARK}; }}
.dc-muted {{ color: {INK_SOFT}; font-size: 0.9rem; }}

/* Barras de control: filtros globales arriba, segmento + descarga sobre la tabla */
.st-key-dc-toolbar, .st-key-dc-tablebar {{ background: {CREAM_DARK}; border: 1px solid {LINE};
  border-radius: 14px; padding: 12px 18px 14px; box-shadow: {SHADOW_SOFT}; }}
.st-key-dc-toolbar {{ margin-bottom: 14px; }}
.st-key-dc-tablebar {{ padding: 10px 14px; margin-bottom: 10px; }}
.st-key-dc-toolbar [data-testid="stWidgetLabel"] p {{ font-weight: 600; color: {NAVY}; }}
.dc-toolbar-title {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: {INK_SOFT}; margin-bottom: 2px; }}

/* Selector de nivel (radios del segmented control): el activo se ve de lejos */
.st-key-dc-tablebar [role="radiogroup"] button[role="radio"] {{
  border-color: {LINE}; background: {WHITE}; transition: background .15s ease, transform .15s ease; }}
.st-key-dc-tablebar [role="radiogroup"] button[role="radio"]:hover {{ background: {CREAM}; }}
.st-key-dc-tablebar [role="radiogroup"] button[aria-checked="true"],
.st-key-dc-tablebar [role="radiogroup"] button[aria-checked="true"]:hover,
.st-key-dc-tablebar [role="radiogroup"] button[aria-checked="true"]:active {{
  background: {NAVY} !important; border-color: {NAVY} !important; box-shadow: inset 0 -3px 0 {ORANGE}; }}
.st-key-dc-tablebar [role="radiogroup"] button[aria-checked="true"] * {{ color: {CREAM} !important; font-weight: 600; }}
.st-key-dc-tablebar [role="radiogroup"] button:active {{ transform: scale(0.98); }}

/* Tarjetas de nivel */
.dc-card {{ background: {WHITE}; border: 1px solid {LINE}; border-left: 6px solid var(--lvl);
  border-radius: 12px; padding: 14px 16px; height: 100%; box-shadow: {SHADOW_SOFT};
  transition: transform .18s ease, box-shadow .18s ease; }}
.dc-card:hover {{ transform: translateY(-2px); box-shadow: {SHADOW_HOVER}; }}
.dc-card .t {{ font-weight: 600; color: {NAVY}; font-size: 0.95rem; }}
.dc-card .n {{ font-family: Georgia, serif; font-size: 2rem; color: {NAVY}; line-height: 1.2; }}
.dc-card .p {{ color: {INK_SOFT}; font-size: 0.9rem; }}
.dc-card .a {{ color: {NAVY}; font-size: 0.88rem; margin-top: 8px; }}

/* Lift: cuánto concentra retrasos el nivel alto */
.dc-lift {{ background: {WHITE}; border: 1px solid {LINE}; border-top: 5px solid {LEVEL_COLORS["alto"]};
  border-radius: 12px; padding: 14px 16px; box-shadow: {SHADOW_SOFT}; }}
.dc-lift .n {{ font-family: Georgia, serif; font-size: 2.2rem; line-height: 1.1; color: {LEVEL_COLORS["alto"]}; }}
.dc-lift .t {{ font-weight: 600; color: {NAVY}; margin-top: 2px; }}
.dc-lift .s {{ color: {INK_SOFT}; font-size: 0.85rem; margin-top: 6px; }}

/* Inspector de pedido */
.st-key-dc-inspector {{ background: {WHITE}; border: 1px solid {LINE}; border-radius: 14px;
  padding: 18px 18px 14px; box-shadow: {SHADOW_CARD}; gap: 0.6rem; }}
.dc-eyebrow {{ font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: {INK_SOFT}; }}
.dc-hint {{ display: inline-block; margin-top: 6px; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem;
  background: {CREAM_DARK}; color: {INK_SOFT}; border: 1px solid {LINE}; line-height: 1.3; }}
.dc-hint.picked {{ background: #E8EEF9; color: {NAVY}; border-color: #C9D5EC; }}
.dc-oid {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.82rem; color: {NAVY};
  word-break: break-all; background: {CREAM}; border: 1px solid {LINE}; border-radius: 8px; padding: 6px 10px; margin-top: 10px; }}
.dc-level {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 10px; }}
.dc-badge {{ display: inline-block; border-radius: 999px; padding: 3px 12px; font-size: 0.88rem; font-weight: 700;
  color: var(--ink); background: var(--soft); border: 1.5px solid var(--lvl); }}
.dc-score {{ color: {INK_SOFT}; font-size: 0.85rem; }}
.dc-score b {{ font-family: Georgia, serif; font-size: 1.1rem; color: {NAVY}; }}
.dc-facts {{ margin: 12px 0 2px; padding: 2px 12px; background: {CREAM}; border: 1px solid {LINE}; border-radius: 10px; }}
.dc-facts > div {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
  padding: 7px 0; border-bottom: 1px solid {LINE}; }}
.dc-facts > div:last-child {{ border-bottom: 0; }}
.dc-facts small {{ color: {INK_SOFT}; font-size: 0.8rem; }}
.dc-facts b {{ font-family: Georgia, serif; font-size: 1rem; color: {NAVY}; white-space: nowrap; }}

/* Factores: chip + mini barra de impacto */
.dc-sec {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
  margin: 14px 0 6px; padding-bottom: 4px; border-bottom: 1px solid {LINE}; }}
.dc-sec-t {{ font-weight: 700; color: {NAVY}; font-size: 0.92rem; }}
.dc-sec-m {{ font-size: 0.75rem; font-weight: 600; }}
.dc-sec-up {{ color: {UP}; }}
.dc-sec-down {{ color: {DOWN}; }}
.dc-fx {{ padding: 6px 0 4px; }}
.dc-fx-row {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
.dc-chip {{ display: inline-flex; align-items: center; gap: 5px; padding: 2px 10px; border-radius: 999px;
  font-size: 0.82rem; font-weight: 600; white-space: nowrap; }}
.dc-chip i {{ font-style: normal; font-weight: 800; }}
.dc-fx-up .dc-chip {{ background: {UP_SOFT}; color: {LEVEL_TEXT["alto"]}; }}
.dc-fx-down .dc-chip {{ background: {DOWN_SOFT}; color: {LEVEL_TEXT["bajo"]}; }}
.dc-fx-val {{ color: {NAVY}; font-size: 0.88rem; font-weight: 600; text-align: right; }}
.dc-fx-bar {{ height: 6px; margin-top: 6px; border-radius: 999px; background: {CREAM_DARK}; overflow: hidden; }}
.dc-fx-bar i {{ display: block; height: 100%; border-radius: 999px; transition: width .3s ease; }}
.dc-fx-up .dc-fx-bar i {{ background: {UP}; }}
.dc-fx-down .dc-fx-bar i {{ background: {DOWN}; }}
.dc-fx-note {{ color: {INK_SOFT}; font-size: 0.78rem; margin: 8px 0 0; }}

/* Acción sugerida */
.dc-action {{ margin-top: 14px; border-radius: 12px; padding: 12px 14px; background: var(--soft);
  border: 1px solid var(--lvl); border-left-width: 6px; }}
.dc-action .k {{ font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink); }}
.dc-action .t {{ color: {NAVY}; font-weight: 600; margin-top: 4px; line-height: 1.35; }}

/* Estado vacío */
.dc-empty {{ text-align: center; padding: 36px 24px 22px; border: 1px dashed {LINE}; border-radius: 14px; background: {WHITE}; }}
.dc-empty .i {{ font-size: 2rem; }}
.dc-empty .t {{ font-family: Georgia, serif; font-size: 1.25rem; color: {NAVY}; margin-top: 6px; }}
.dc-empty .s {{ color: {INK_SOFT}; margin-top: 4px; }}
</style>
"""
