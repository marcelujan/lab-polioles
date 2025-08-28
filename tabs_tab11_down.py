# tabs_tab11_down.py
import streamlit as st
import pandas as pd
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global  # :contentReference[oaicite:2]{index=2}

# ----- Definición de columnas -----
ETAPAS = [1, 2, 3, 4]

def _cols_etapa(n:int):
    # Encabezados “verticales” con \n para hacerlos angostos
    return [
        f"ETAPA {n}\nTIPO\nDE SAL",
        f"ETAPA {n}\nCONC.\n(g/L)",
        f"ETAPA {n}\nVOLUMEN\n(mL)",
        f"ETAPA {n}\nTEMP.\n(°C)",
        f"ETAPA {n}\nTIEMPO\nAGIT. (h)",
        f"ETAPA {n}\nTIEMPO\nDECAN. (h)",
        f"ETAPA {n}\nFASE ACUO\nRETIRADA (mL)",
    ]

BASE_COLS = ["Síntesis", "VOL ACUO (mL)"]
for e in ETAPAS:
    BASE_COLS += _cols_etapa(e)
BASE_COLS += ["Observaciones"]

# Colores suaves por etapa (bandas superiores)
ETAPA_COLORS = {
    1: "#ffe6d5",  # durazno suave
    2: "#d9efff",  # celeste suave
    3: "#e6f4d7",  # verde suave
    4: "#efe6ff",  # lila suave
}

def _df_vacio(n_rows=6) -> pd.DataFrame:
    row = {c: ("" if c in ["Síntesis", "Observaciones"] or "TIPO" in c else None) for c in BASE_COLS}
    return pd.DataFrame([row.copy() for _ in range(n_rows)])

# ----- UI helpers -----
def _bandas_etapas():
    st.markdown(
        """
        <div style="display:flex; gap:8px; margin-bottom:6px;">
          <div style="flex:0 0 160px;"></div>  <!-- Síntesis + VOL ACUO -->
          <div style="flex:1; display:flex; gap:8px;">
            <div style="flex:1; background:#ffe6d5; padding:6px; text-align:center; border-radius:6px;">ETAPA 1</div>
            <div style="flex:1; background:#d9efff; padding:6px; text-align:center; border-radius:6px;">ETAPA 2</div>
            <div style="flex:1; background:#e6f4d7; padding:6px; text-align:center; border-radius:6px;">ETAPA 3</div>
            <div style="flex:1; background:#efe6ff; padding:6px; text-align:center; border-radius:6px;">ETAPA 4</div>
          </div>
          <div style="flex:0 0 120px;"></div> <!-- Observaciones -->
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Down")
    st.session_state["current_tab"] = "Down"

    # --- Cargar desde Firestore ---
    dados = cargar_sintesis_global(db) or {}
    registros = dados.get("down_tabla", None)

    if registros:
        try:
            df_in = pd.DataFrame(registros)
            for c in BASE_COLS:
                if c not in df_in.columns:
                    df_in[c] = None
            df_in = df_in[BASE_COLS]
        except Exception:
            df_in = _df_vacio()
    else:
        df_in = _df_vacio()

    # Banda de colores por etapa (decorativa)
    _bandas_etapas()

    # Config de columnas: numeros angostos y textos angostos
    colcfg = {
        "Síntesis": st.column_config.TextColumn(width="small"),
        "VOL ACUO (mL)": st.column_config.NumberColumn(width="small", step=1),
        "Observaciones": st.column_config.TextColumn(width="medium"),
    }
    for e in ETAPAS:
        colcfg[f"ETAPA {e}\nTIPO\nDE SAL"] = st.column_config.TextColumn(width="small")
        colcfg[f"ETAPA {e}\nCONC.\n(g/L)"] = st.column_config.NumberColumn(width="small", step=1)
        colcfg[f"ETAPA {e}\nVOLUMEN\n(mL)"] = st.column_config.NumberColumn(width="small", step=1)
        colcfg[f"ETAPA {e}\nTEMP.\n(°C)"] = st.column_config.NumberColumn(width="small", step=1)
        colcfg[f"ETAPA {e}\nTIEMPO\nAGIT. (h)"] = st.column_config.NumberColumn(width="small", step=0.1, format="%.2f")
        colcfg[f"ETAPA {e}\nTIEMPO\nDECAN. (h)"] = st.column_config.NumberColumn(width="small", step=0.1, format="%.2f")
        colcfg[f"ETAPA {e}\nFASE ACUO\nRETIRADA (mL)"] = st.column_config.NumberColumn(width="small", step=1)

    st.caption("Editá la tabla. Podés agregar/eliminar filas.")
    df_edit = st.data_editor(
        df_in,
        num_rows="dynamic",
        use_container_width=True,
        column_config=colcfg,
        key="down_tabla_editor_v2",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Guardar en Firestore"):
            payload = {**(dados or {}), "down_tabla": df_edit.to_dict(orient="records")}
            guardar_sintesis_global(db, payload)  # persiste en 'sintesis_global/seleccion':contentReference[oaicite:3]{index=3}
            st.success("✅ Tabla guardada.")
    with c2:
        if st.button("↺ Recargar"):
            st.rerun()
