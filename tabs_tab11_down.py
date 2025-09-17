# tabs_tab11_down.py  (con depuración)
import streamlit as st
import pandas as pd
import hashlib, json, datetime
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global

ETAPAS = [1, 2, 3, 4]
CAMPOS_ETAPA = ["Agente", "V (mL)", "T", "t ag (h)", "t dec (h)", "V dec (mL)"]

def _new_cols_etapa(n:int): return [f"{n}_{c}" for c in CAMPOS_ETAPA]

BASE_COLS = (
    ["Síntesis", "VOL ACUO (mL)"] +
    [c for e in ETAPAS for c in _new_cols_etapa(e)] +
    ["Observaciones"]
)

OLD2NEW = lambda n: {
    f"{n}_Agente":          f"E{n} TIPO DE SAL",
    f"{n}_V (mL)":          f"E{n} VOLUMEN (mL)",
    f"{n}_T":               f"E{n} TEMP (°C)",
    f"{n}_t ag (h)":        f"E{n} t AGIT (h)",
    f"{n}_t dec (h)":       f"E{n} t DECAN (h)",
    f"{n}_V dec (mL)":      f"E{n} FASE ACUO RET (mL)",
}

def _df_vacio(): return pd.DataFrame(columns=BASE_COLS)

def _hash_rows(rows):
    s = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _migrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    for n in ETAPAS:
        mapping = OLD2NEW(n)
        for new_col, old_col in mapping.items():
            if new_col not in df.columns:
                df[new_col] = df[old_col] if old_col in df.columns else ""
    cols_drop = []
    for n in ETAPAS: cols_drop += list(OLD2NEW(n).values())
    return df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")

def render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.session_state["current_tab"] = "Down"

    # --------- CARGA ----------
    datos = cargar_sintesis_global(db) or {}
    # Fallback a posibles nombres antiguos
    raw = (datos.get("down_tabla")
           or datos.get("down")
           or datos.get("down_tab")
           or datos.get("downstream")
           or [])
    df_in = pd.DataFrame(raw) if raw else _df_vacio()

    for c in BASE_COLS:
        if c not in df_in.columns: df_in[c] = ""
    df_in = _migrar_columnas(df_in)
    df_in = df_in[[c for c in BASE_COLS]]

    # --------- DEPURACIÓN ----------
    with st.expander("🔧 Depuración (Down)", expanded=False):
        st.json({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "keys_en_sintesis_global": list(datos.keys()),
            "existe_down_tabla": "down_tabla" in datos,
            "len_down_tabla": len(datos.get("down_tabla") or []),
            "otras_claves_posibles": [k for k in datos.keys() if "down" in k.lower() and k != "down_tabla"],
            "filas_cargadas_editor": len(df_in),
        })
        if not datos.get("down_tabla") and raw:
            st.warning("Se cargó desde una **clave alternativa**. Al guardar se normaliza a 'down_tabla'.")
        if not raw:
            st.info("No se encontraron registros; se muestra tabla vacía.")
        if st.button("Forzar recarga (limpiar estado)"):
            for k in ["down_hash", "down_editor_native"]:
                st.session_state.pop(k, None)
            st.rerun()

    # --------- CONFIG COLUMNAS ----------
    colcfg = {
        "Síntesis": st.column_config.TextColumn(label="ID", width="small"),
        "VOL ACUO (mL)": st.column_config.NumberColumn(label="Vaq (mL)", width="small", step=1),
        "Observaciones": st.column_config.TextColumn(label="Obs", width="large"),
    }
    for n in (1,2,3,4):
        colcfg[f"{n}_Agente"]     = st.column_config.TextColumn(label=f"{n}_Ag", width="small")
        colcfg[f"{n}_V (mL)"]     = st.column_config.NumberColumn(label=f"{n}_V", width="small", step=1)
        colcfg[f"{n}_T"]          = st.column_config.NumberColumn(label=f"{n}_T", width="small", step=1)
        colcfg[f"{n}_t ag (h)"]   = st.column_config.NumberColumn(label=f"{n}_tAg", width="small", step=0.1, format="%.2f")
        colcfg[f"{n}_t dec (h)"]  = st.column_config.NumberColumn(label=f"{n}_tDec", width="small", step=0.1, format="%.2f")
        colcfg[f"{n}_V dec (mL)"] = st.column_config.NumberColumn(label=f"{n}_Vdec", width="small", step=1)

    st.markdown("""
    <style>
    div[data-testid="stDataEditorGrid"] thead th{ padding:2px 4px!important; white-space:nowrap!important; }
    div[data-testid="stDataEditorGrid"] thead svg{ display:none!important; }
    </style>
    """, unsafe_allow_html=True)

    # Hash base
    rows_base = df_in.fillna("").to_dict("records")
    if "down_hash" not in st.session_state:
        st.session_state["down_hash"] = _hash_rows(rows_base)

    # Editor
    df_edit = st.data_editor(
        df_in, num_rows="dynamic", use_container_width=True,
        column_config=colcfg, hide_index=True, key="down_editor_native",
    )

    # --------- GUARDADO ----------
    rows = df_edit.fillna("").to_dict("records")
    h = _hash_rows(rows)
    if h != st.session_state["down_hash"]:
        try:
            payload = {**datos, "down_tabla": rows}   # normaliza a 'down_tabla'
            guardar_sintesis_global(db, payload)
            st.session_state["down_hash"] = h
            st.toast("Guardado", icon="✅")
        except Exception as e:
            st.error(f"Error guardando en Firestore: {e}")

    # Añadir fila
    if st.button("➕ Fila"):
        df_new = pd.concat([df_edit, pd.DataFrame([{c: "" for c in BASE_COLS}])], ignore_index=True)
        rows_new = df_new.fillna("").to_dict("records")
        try:
            guardar_sintesis_global(db, {**datos, "down_tabla": rows_new})
            st.session_state["down_hash"] = _hash_rows(rows_new)
            st.rerun()
        except Exception as e:
            st.error(f"Error guardando nueva fila: {e}")
