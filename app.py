import streamlit as st
import pandas as pd
import toml
import json
from datetime import datetime
from datetime import date
from io import BytesIO
import os

# --- CONFIGURACION DE SEGURIDAD ---
config = toml.load("config.toml")
PASSWORD = config["auth"]["password"]

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")
st.title("Laboratorio de Polioles")
st.caption("Versión 2025.04.11")

# Autenticación simple
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    password = st.text_input("Contraseña de acceso", type="password")
    if st.button("Ingresar"):
        if password == PASSWORD:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")
    st.stop()

# Archivo local para guardar datos
DATA_FILE = "muestras_data.json"

# Cargar datos desde archivo siempre que exista
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        st.session_state.muestras = json.load(f)
else:
    st.session_state.muestras = []

# Lista fija de tipos de análisis
tipos_analisis = [
    "Índice de yodo [% p/p I2 abs]",
    "Índice OH [mg KHO/g]",
    "Índice de acidez [mg KOH/g]",
    "Índice de epóxido [mol/100g]",
    "Humedad [%]",
    "PM [g/mol]",
    "Funcionalidad [#]",
    "Viscosidad dinámica [cP]",
    "Densidad [g/mL]"
]

# ---- FORMULARIO PARA AÑADIR NUEVA MUESTRA ----
st.header("Nueva muestra")
with st.form("form_nueva_muestra"):
    nombre_muestra = st.text_input("Nombre de la muestra", "")
    observacion_muestra = st.text_area("Observaciones de la muestra", "")

    st.markdown("### Análisis físico-químicos")

    # Generar tabla siempre con todos los análisis
    df_input = pd.DataFrame([{
        "Tipo": tipo,
        "Valor": 0.0,
        "Fecha": date.today(),
        "Observaciones": ""
    } for tipo in tipos_analisis])

    df_input = st.data_editor(
        df_input,
        num_rows="fixed",
        use_container_width=True,
        key="analisis_editor"
    )

    submitted = st.form_submit_button("Guardar muestra y análisis")

    if submitted and nombre_muestra:
        nueva_muestra = {
            "nombre": nombre_muestra,
            "observacion": observacion_muestra,
            "analisis": []
        }
        for _, row in df_input.iterrows():
            if row["Valor"] and float(row["Valor"]) != 0.0:
                nueva_muestra["analisis"].append({
                    "tipo": row["Tipo"],
                    "valor": row["Valor"],
                    "fecha": str(row["Fecha"]),
                    "observaciones": row["Observaciones"]
                })
        
        if nueva_muestra["analisis"] or True:
            st.session_state.muestras.append(nueva_muestra)
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.muestras, f, ensure_ascii=False, indent=2)
            backup_name = f"muestras_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            with open(backup_name, "w", encoding="utf-8") as f:
                json.dump(st.session_state.muestras, f, ensure_ascii=False, indent=2)
            st.success(f"Muestra '{nombre_muestra}' agregada correctamente.")
        

# ---- VISUALIZAR Y EDITAR ----

# ---- LISTA DE MUESTRAS ----
st.header("Muestras cargadas")

if "editar_idx" not in st.session_state:
    st.session_state.editar_idx = None

if st.session_state.editar_idx is None:
    if st.session_state.muestras:
        for idx, muestra in enumerate(st.session_state.muestras):
            with st.container():
                cols = st.columns([3, 5, 2])
                cols[0].markdown(f"**{muestra['nombre']}**")
                cols[1].markdown(muestra["observacion"])
                if cols[2].button("Editar", key=f"edit_{idx}"):
                    st.session_state.editar_idx = idx
                    st.rerun()
    else:
        st.info("No hay muestras cargadas todavía.")
else:
    idx = st.session_state.editar_idx
    muestra = st.session_state.muestras[idx]
    st.subheader(f"Editar análisis de: {muestra['nombre']}")
    df_edit = pd.DataFrame(muestra["analisis"])

    edited_df = st.data_editor(df_edit, num_rows="dynamic", use_container_width=True, key="editor_detallado")

    if st.button("Guardar cambios"):
        muestra["analisis"] = edited_df.to_dict(orient="records")
        st.session_state.muestras[idx] = muestra
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.muestras, f, ensure_ascii=False, indent=2)
        backup_name = f"muestras_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(backup_name, "w", encoding="utf-8") as f:
            json.dump(st.session_state.muestras, f, ensure_ascii=False, indent=2)
        st.success("Cambios guardados correctamente.")
        st.session_state.editar_idx = None
        st.rerun()

    if st.button("Volver a la lista de muestras"):
        st.session_state.editar_idx = None
        st.rerun()
