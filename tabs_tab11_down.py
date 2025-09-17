# tabs_tab11_down.py
import streamlit as st
import pandas as pd
import hashlib, json, datetime
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global

# ========= Config común =========
ETAPAS = [1, 2, 3, 4]
CAMPOS_ETAPA = ["Sal", "[]", "V (mL)", "T", "t ag (h)", "t dec (h)", "V dec (mL)"]

def _new_cols_etapa(n: int): return [f"{n}_{c}" for c in CAMPOS_ETAPA]

DOWN_COLS = (
    ["Síntesis", "VOL ACUO (mL)"]
    + [c for e in ETAPAS for c in _new_cols_etapa(e)]
    + ["Observaciones"]
)

# Tabla de costos (genérica, por síntesis)
COSTOS_COLS = ["Síntesis", "Item", "Cantidad", "Unidad", "Precio unitario", "Moneda", "Subtotal", "Obs"]

def _df_vacio(cols): return pd.DataFrame(columns=cols)

def _hash_rows(rows):
    s = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _ensure_cols(df: pd.DataFrame, cols:list) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

# ========= UI principal =========
def render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.session_state["current_tab"] = "Down"

    # --- Carga de Firestore ---
    datos = cargar_sintesis_global(db) or {}
    rows_down   = datos.get("down_tabla")    or []
    rows_costos = datos.get("down_costos")   or []

    df_down   = _ensure_cols(pd.DataFrame(rows_down)   if rows_down   else _df_vacio(DOWN_COLS),   DOWN_COLS).fillna("")
    df_costos = _ensure_cols(pd.DataFrame(rows_costos) if rows_costos else _df_vacio(COSTOS_COLS), COSTOS_COLS).fillna("")

    # --- Estilo compacto de encabezados ---
    st.markdown("""
    <style>
    div[data-testid="stDataEditorGrid"] thead th{padding:2px 4px!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important;}
    div[data-testid="stDataEditorGrid"] thead svg{display:none!important;}
    </style>
    """, unsafe_allow_html=True)

    # ===================== TABLA: DOWN =====================
    st.subheader("Down")
    colcfg_down = {c: st.column_config.TextColumn(label=c, width="small") for c in DOWN_COLS}
    colcfg_down["Síntesis"]       = st.column_config.TextColumn(label="ID", width="small")
    colcfg_down["VOL ACUO (mL)"]  = st.column_config.TextColumn(label="Vaq (mL)", width="small")
    colcfg_down["Observaciones"]  = st.column_config.TextColumn(label="Obs", width="large")
    for n in ETAPAS:
        colcfg_down[f"{n}_Sal"]        = st.column_config.TextColumn(label=f"{n}_Ag",   width="small")
        colcfg_down[f"{n}_[]"]         = st.column_config.TextColumn(label=f"{n}_[]",   width="small")
        colcfg_down[f"{n}_V (mL)"]     = st.column_config.TextColumn(label=f"{n}_V",    width="small")
        colcfg_down[f"{n}_T"]          = st.column_config.TextColumn(label=f"{n}_T",    width="small")
        colcfg_down[f"{n}_t ag (h)"]   = st.column_config.TextColumn(label=f"{n}_tAg",  width="small")
        colcfg_down[f"{n}_t dec (h)"]  = st.column_config.TextColumn(label=f"{n}_tDec", width="small")
        colcfg_down[f"{n}_V dec (mL)"] = st.column_config.TextColumn(label=f"{n}_Vdec", width="small")

    if "down_hash" not in st.session_state:
        st.session_state["down_hash"] = _hash_rows(df_down.to_dict("records"))

    df_down_edit = st.data_editor(
        df_down, key="down_editor", num_rows="dynamic",
        use_container_width=True, hide_index=True, column_config=colcfg_down
    )

    c1, c2 = st.columns(2)
    if c1.button("➕ Fila (Down)", use_container_width=True):
        df_down_edit = pd.concat([df_down_edit, pd.DataFrame([{c: "" for c in DOWN_COLS}])], ignore_index=True)
    if c2.button("🧹 Limpiar tabla (Down)", use_container_width=True):
        df_down_edit = _df_vacio(DOWN_COLS)

    # ===================== TABLA: COSTOS =====================
    st.subheader("Costos (Down)")
    colcfg_costos = {
        "Síntesis":        st.column_config.TextColumn(label="ID", width="small"),
        "Item":            st.column_config.TextColumn(label="Item", width="large"),
        "Cantidad":        st.column_config.NumberColumn(label="Cantidad", step=0.01, format="%.3f", width="small"),
        "Unidad":          st.column_config.TextColumn(label="Unidad", width="small"),
        "Precio unitario": st.column_config.NumberColumn(label="Precio unitario", step=0.01, format="%.2f", width="small"),
        "Moneda":          st.column_config.TextColumn(label="Moneda", width="small"),
        "Subtotal":        st.column_config.NumberColumn(label="Subtotal", step=0.01, format="%.2f", width="small"),
        "Obs":             st.column_config.TextColumn(label="Obs", width="large"),
    }

    if "costos_hash" not in st.session_state:
        st.session_state["costos_hash"] = _hash_rows(df_costos.to_dict("records"))

    df_costos_edit = st.data_editor(
        df_costos, key="costos_editor", num_rows="dynamic",
        use_container_width=True, hide_index=True, column_config=colcfg_costos
    )

    c3, c4 = st.columns(2)
    if c3.button("➕ Fila (Costos)", use_container_width=True):
        df_costos_edit = pd.concat([df_costos_edit, pd.DataFrame([{c: "" for c in COSTOS_COLS}])], ignore_index=True)
    if c4.button("🧹 Limpiar tabla (Costos)", use_container_width=True):
        df_costos_edit = _df_vacio(COSTOS_COLS)

    # ===================== GUARDADO ÚNICO =====================
    rows_down_new   = df_down_edit.fillna("").to_dict("records")
    rows_costos_new = df_costos_edit.fillna("").to_dict("records")

    changed_down   = _hash_rows(rows_down_new)   != st.session_state["down_hash"]
    changed_costos = _hash_rows(rows_costos_new) != st.session_state["costos_hash"]

    if changed_down or changed_costos:
        payload = {**datos}
        if changed_down:
            payload["down_tabla"]  = rows_down_new
        if changed_costos:
            payload["down_costos"] = rows_costos_new
        try:
            guardar_sintesis_global(db, payload)
            # read-after-write para asegurar persistencia visible
            datos_ref = cargar_sintesis_global(db) or {}
            st.session_state["down_hash"]   = _hash_rows((datos_ref.get("down_tabla")  or []))
            st.session_state["costos_hash"] = _hash_rows((datos_ref.get("down_costos") or []))
            st.toast("Guardado", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Error guardando en Firestore: {e}")
