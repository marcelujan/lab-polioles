# tabs_tab11_down.py
import streamlit as st
import pandas as pd
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global  # :contentReference[oaicite:2]{index=2}

ETAPAS = [1, 2, 3, 4]

def _columnas_etapa(n):
    base = [
        f"ETAPA {n} – TIPO DE SAL",
        f"ETAPA {n} – CONC. DE SAL. (g/L)",
        f"ETAPA {n} – VOLUMEN (mL)",
        f"ETAPA {n} – TEMPERATURA (°C)",
        f"ETAPA {n} – TIEMPO AGIT. (h)",
        f"ETAPA {n} – TIEMPO DECAN. (h)",
        f"ETAPA {n} – FASE ACUO RETIRADA (mL)",   # 👈 columna pedida
    ]
    return base

COLS = ["VOL ACUO (mL)"]
for e in ETAPAS:
    COLS += _columnas_etapa(e)

def _df_vacio(n_filas=6):
    return pd.DataFrame([{c: ("" if "TIPO" in c else None) for c in COLS} for _ in range(n_filas)])

def render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Down")
    st.session_state["current_tab"] = "Down"

    # Cargar tabla desde Firestore (documento global 'sintesis_global/seleccion')
    datos = cargar_sintesis_global(db) or {}
    registros = datos.get("down_tabla", None)
    if registros:
        try:
            df_in = pd.DataFrame(registros)
            # Asegurar columnas (por si faltan o cambió el orden)
            for c in COLS:
                if c not in df_in.columns:
                    df_in[c] = None
            df_in = df_in[COLS]
        except Exception:
            df_in = _df_vacio()
    else:
        df_in = _df_vacio()

    st.caption("Editá la tabla. Podés agregar/eliminar filas desde el widget.")
    edited = st.data_editor(
        df_in,
        num_rows="dynamic",
        use_container_width=True,
        key="down_tabla_editor",
    )

    col_g, col_r = st.columns([1,1])
    with col_g:
        if st.button("💾 Guardar tabla en Firestore"):
            # Guardar como lista de dicts dentro del mismo documento global
            payload = {**datos, "down_tabla": edited.to_dict(orient="records")}
            guardar_sintesis_global(db, payload)
            st.success("✅ Tabla guardada.")

    with col_r:
        if st.button("↺ Recargar desde Firestore"):
            st.rerun()
