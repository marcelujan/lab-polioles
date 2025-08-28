# tabs_tab11_down.py
import streamlit as st
import pandas as pd
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global  # persiste en Firestore:contentReference[oaicite:2]{index=2}

# Import condicional de AgGrid (opcional)
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    HAS_AGGRID = True
except Exception:
    HAS_AGGRID = False

# ----- Definición de esquema -----
ETAPA_COLORS = {1: "#ffe6d5", 2: "#d9efff", 3: "#e6f4d7", 4: "#efe6ff"}

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

def _df_vacio(n_rows: int = 6) -> pd.DataFrame:
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
    datos, df = _load_df(db)

    if HAS_AGGRID:
        # ===== Vista AG Grid: color por columna + edición =====
        gob = GridOptionsBuilder.from_dataframe(df)

        # configurar columnas editables + tipos
        for c in BASE_COLS:
            gob.configure_column(c, editable=True, resizable=True)

        num_cols = [c for c in BASE_COLS if any(k in c for k in ("(mL)", "(°C)", "(g/L)", "(h)"))]
        for c in num_cols:
            gob.configure_column(c, type=["numericColumn", "customNumericFormat"], valueParser="Number(newValue)")

        # ancho y colores por etapa
        def _style_cols(cols, color):
            for c in cols:
                gob.configure_column(c, width=120, cellStyle={"backgroundColor": color})

        for i in (1, 2, 3, 4):
            _style_cols(_cols_etapa(i), ETAPA_COLORS[i])

        gob.configure_column("Síntesis", pinned="left", width=130)
        gob.configure_column("Observaciones", width=260)

        grid = AgGrid(
            df,
            gridOptions=gob.build(),
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=False,
            height=520,
        )

        b_add, b_save, b_reload = st.columns(3)
        with b_add:
            if st.button("➕ Agregar fila"):
                nuevo = pd.DataFrame([{c: "" for c in BASE_COLS}])
                df_out = pd.concat([pd.DataFrame(grid.data), nuevo], ignore_index=True)
                _save_df(db, datos, df_out)
                st.rerun()
        with b_save:
            if st.button("💾 Guardar"):
                _save_df(db, datos, pd.DataFrame(grid.data))
                st.success("Guardado.")
        with b_reload:
            if st.button("↺ Recargar"):
                st.rerun()

    else:
        # ===== Fallback nativo (sin color por columna) =====
        df_edit = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Síntesis": st.column_config.TextColumn(width="small"),
                "VOL ACUO (mL)": st.column_config.NumberColumn(width="small", step=1),
                "Observaciones": st.column_config.TextColumn(width="medium"),
            },
            key="down_editor",
        )

        b_save, b_reload, b_add = st.columns(3)
        with b_save:
            if st.button("💾 Guardar"):
                _save_df(db, datos, df_edit)
                st.success("Guardado.")
        with b_reload:
            if st.button("↺ Recargar"):
                st.rerun()
        with b_add:
            if st.button("➕ Agregar fila"):
                df_out = pd.concat([df_edit, pd.DataFrame([{c: "" for c in BASE_COLS}])], ignore_index=True)
                _save_df(db, datos, df_out)
                st.rerun()
