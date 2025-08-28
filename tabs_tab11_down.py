# requirements.txt -> streamlit-aggrid>=0.3.5
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global  # :contentReference[oaicite:2]{index=2}

ETAPA_COLORS = {1:"#ffe6d5", 2:"#d9efff", 3:"#e6f4d7", 4:"#efe6ff"}

def _cols_etapa(n):
    return [
        f"E{n} TIPO DE SAL", f"E{n} CONC (g/L)", f"E{n} VOLUMEN (mL)",
        f"E{n} TEMP (°C)", f"E{n} t AGIT (h)", f"E{n} t DECAN (h)",
        f"E{n} FASE ACUO RET (mL)"
    ]

BASE_COLS = ["Síntesis", "VOL ACUO (mL)"] + sum((_cols_etapa(i) for i in [1,2,3,4]), []) + ["Observaciones"]

def _df_vacio(n=6):
    return pd.DataFrame([{c: "" for c in BASE_COLS} for _ in range(n)])

def render_tab11(db, *_):
    datos = cargar_sintesis_global(db) or {}
    df = pd.DataFrame(datos.get("down_tabla") or []) if datos.get("down_tabla") else _df_vacio()
    for c in BASE_COLS:
        if c not in df.columns: df[c] = ""

    g = GridOptionsBuilder.from_dataframe(df[BASE_COLS])
    # ediciones
    for c in BASE_COLS:
        g.configure_column(c, editable=True, resizable=True)

    # tipos numéricos
    for c in [x for x in BASE_COLS if any(k in x for k in ["(mL)","(°C)","(g/L)","(h)"])]:
        g.configure_column(c, type=["numericColumn","customNumericFormat"], valueParser="Number(newValue)")

    # color por columnas de cada etapa + width angosto
    def style_col(cols, color):
        for c in cols: g.configure_column(c, cellStyle={"backgroundColor": color}, width=120)
    for i in [1,2,3,4]:
        style_col(_cols_etapa(i), ETAPA_COLORS[i])
    g.configure_column("Síntesis", pinned="left", width=120)
    g.configure_column("Observaciones", width=220)

    # agregar/eliminar filas
    g.configure_grid_options(rowSelection="single", editable=True)
    grid = AgGrid(
        df, gridOptions=g.build(), update_mode=GridUpdateMode.VALUE_CHANGED,
        enable_enterprise_modules=False, fit_columns_on_grid_load=False, height=520
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("➕ Agregar fila vacía"):
            new = pd.concat([pd.DataFrame([{}], columns=BASE_COLS), pd.DataFrame(grid.data)], ignore_index=True).fillna("")
            guardar_sintesis_global(db, {**datos, "down_tabla": new.to_dict("records")}); st.rerun()
    with col2:
        if st.button("🗑️ Borrar fila seleccionada"):
            sel = grid["selected_rows"]
            if sel:
                left = pd.DataFrame(grid.data)
                left = left.drop(index=int(sel[0]["_selectedRowNodeInfo"]["nodeId"])).reset_index(drop=True)
                guardar_sintesis_global(db, {**datos, "down_tabla": left.to_dict("records")}); st.rerun()
    with col3:
        if st.button("💾 Guardar cambios"):
            guardar_sintesis_global(db, {**datos, "down_tabla": pd.DataFrame(grid.data).to_dict("records")})
            st.success("Guardado.")
