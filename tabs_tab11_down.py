# tabs_tab11_down.py
import streamlit as st
import pandas as pd
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global  # persiste en doc global:contentReference[oaicite:0]{index=0}

ETAPAS = [1, 2, 3, 4]

def _cols_etapa(n:int):
    return [
        f"E{n} TIPO DE SAL",
        f"E{n} CONC (g/L)",
        f"E{n} VOLUMEN (mL)",
        f"E{n} TEMP (°C)",
        f"E{n} t AGIT (h)",
        f"E{n} t DECAN (h)",
        f"E{n} FASE ACUO RET (mL)",
    ]

BASE_COLS = ["Síntesis", "VOL ACUO (mL)"] + [c for e in ETAPAS for c in _cols_etapa(e)] + ["Observaciones"]

def _df_vacio():
    return pd.DataFrame(columns=BASE_COLS)

def render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.session_state["current_tab"] = "Down"

    # Cargar desde Firestore
    datos = cargar_sintesis_global(db) or {}
    raw = datos.get("down_tabla") or []
    df_in = pd.DataFrame(raw) if raw else _df_vacio()
    for c in BASE_COLS:
        if c not in df_in.columns:
            df_in[c] = ""
    df_in = df_in[BASE_COLS]

    # Editor nativo (sin títulos ni captions)
    colcfg = {
        "Síntesis": st.column_config.TextColumn(width="small"),
        "VOL ACUO (mL)": st.column_config.NumberColumn(width="small", step=1),
        "Observaciones": st.column_config.TextColumn(width="large"),
    }
    for e in ETAPAS:
        colcfg[f"E{e} TIPO DE SAL"] = st.column_config.TextColumn(width="small")
        colcfg[f"E{e} CONC (g/L)"] = st.column_config.NumberColumn(width="small", step=1)
        colcfg[f"E{e} VOLUMEN (mL)"] = st.column_config.NumberColumn(width="small", step=1)
        colcfg[f"E{e} TEMP (°C)"] = st.column_config.NumberColumn(width="small", step=1)
        colcfg[f"E{e} t AGIT (h)"] = st.column_config.NumberColumn(width="small", step=0.1, format="%.2f")
        colcfg[f"E{e} t DECAN (h)"] = st.column_config.NumberColumn(width="small", step=0.1, format="%.2f")
        colcfg[f"E{e} FASE ACUO RET (mL)"] = st.column_config.NumberColumn(width="small", step=1)

    df_edit = st.data_editor(
        df_in,
        num_rows="dynamic",
        use_container_width=True,
        column_config=colcfg,
        key="down_editor_native",
    )

    # Acciones
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("💾 Guardar"):
            guardar_sintesis_global(db, {**datos, "down_tabla": df_edit.fillna("").to_dict("records")})
            st.toast("Guardado", icon="✅")
    with c2:
        if st.button("➕ Fila"):
            df_new = pd.concat([df_edit, pd.DataFrame([{c: "" for c in BASE_COLS}])], ignore_index=True)
            guardar_sintesis_global(db, {**datos, "down_tabla": df_new.fillna("").to_dict("records")})
            st.rerun()
    with c3:
        up = st.file_uploader("Importar CSV", type=["csv"], label_visibility="collapsed", key="down_csv_up")
        if up is not None:
            tmp = pd.read_csv(up)
            for c in BASE_COLS:
                if c not in tmp.columns:
                    tmp[c] = ""
            tmp = tmp[BASE_COLS]
            guardar_sintesis_global(db, {**datos, "down_tabla": tmp.fillna("").to_dict("records")})
            st.rerun()
    with c4:
        st.download_button("⬇️ CSV", df_edit.to_csv(index=False).encode("utf-8"),
                           file_name="down_tabla.csv", mime="text/csv")
