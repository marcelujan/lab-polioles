# tabs_tab11_down.py
import streamlit as st
import pandas as pd
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global  # persiste en Firestore:contentReference[oaicite:2]{index=2}
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ----- Definición de esquema -----
ETAPA_COLORS = {1: "#ffe6d5", 2: "#d9efff", 3: "#e6f4d7", 4: "#fff89d"}

def _cols_etapa(n: int):
    return [
        f"E{n} TIPO DE SAL",
        f"E{n} CONC (g/L)",
        f"E{n} VOLUMEN (mL)",
        f"E{n} TEMP (°C)",
        f"E{n} t AGIT (h)",
        f"E{n} t DECAN (h)",
        f"E{n} FASE ACUO RET (mL)",  # pedido
    ]

BASE_COLS = (
    ["Síntesis", "VOL ACUO (mL)"]
    + sum((_cols_etapa(i) for i in (1, 2, 3, 4)), [])
    + ["Observaciones"]
)

def _df_vacio(n_rows: int = 0):  # antes 6
    return pd.DataFrame([{c: "" for c in BASE_COLS} for _ in range(n_rows)])

def _load_df(db):
    datos = cargar_sintesis_global(db) or {}
    raw = datos.get("down_tabla")
    df = pd.DataFrame(raw) if raw else _df_vacio()
    # asegurar columnas y orden
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[BASE_COLS]
    return datos, df

def _save_df(db, datos, df: pd.DataFrame):
    payload = {**(datos or {}), "down_tabla": df.fillna("").to_dict(orient="records")}
    guardar_sintesis_global(db, payload)

def render_tab11(db, *_args, **_kwargs):
    # filtro rápido arriba del grid
    q = st.text_input(" ", placeholder="Filtrar rápido…", label_visibility="collapsed", key="down_qf")

    gob = GridOptionsBuilder.from_dataframe(df)
    gob.configure_default_column(editable=True, resizable=True, sortable=True, filter=True)

    # filtros numéricos / texto
    num_cols = [c for c in BASE_COLS if any(k in c for k in ("(mL)", "(°C)", "(g/L)", "(h)"))]
    txt_cols = [c for c in BASE_COLS if c not in num_cols]
    for c in num_cols: gob.configure_column(c, filter="agNumberColumnFilter", type=["numericColumn","customNumericFormat"], valueParser="Number(newValue)")
    for c in txt_cols: gob.configure_column(c, filter="agTextColumnFilter")

    # color por columnas de etapa
    def _style_cols(cols, color):
        for c in cols: gob.configure_column(c, width=120, cellStyle={"backgroundColor": color})
    for i in (1,2,3,4): _style_cols(_cols_etapa(i), ETAPA_COLORS[i])

    gob.configure_column("Síntesis", pinned="left", width=130)
    gob.configure_column("Observaciones", width=260)

    # autoHeight si hay pocas filas; paginar si hay muchas
    rows = len(df)
    if rows <= 20:
        gob.configure_grid_options(domLayout="autoHeight", rowHeight=32, quickFilterText=q or "")
        grid_height = None  # AgGrid ajusta alto al contenido
    else:
        gob.configure_grid_options(domLayout="normal", pagination=True, paginationPageSize=25, quickFilterText=q or "")
        grid_height = 540

    grid = AgGrid(
        df,
        gridOptions=gob.build(),
        update_mode=GridUpdateMode.VALUE_CHANGED,
        fit_columns_on_grid_load=False,
        height=grid_height,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Agregar fila"):
            df_out = pd.concat([pd.DataFrame(grid.data), pd.DataFrame([{c: "" for c in BASE_COLS}])], ignore_index=True)
            _save_df(db, datos, df_out); st.rerun()
    with c2:
        if st.button("💾 Guardar"):
            _save_df(db, datos, pd.DataFrame(grid.data)); st.success("Guardado.")
    with c3:
        if st.button("↺ Recargar"):
            st.rerun()
