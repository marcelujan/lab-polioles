import streamlit as st
import pandas as pd
import toml
import json
from datetime import date, datetime
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

# Cargar datos desde archivo si existe
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

# --- FORMULARIO DE MUESTRAS ---
st.header("Editar o agregar análisis")

nombres_existentes = [m["nombre"] for m in st.session_state.muestras]
opciones = ["Nueva muestra"] + nombres_existentes
seleccion = st.selectbox("Seleccionar muestra", opciones)

if seleccion == "Nueva muestra":
    nombre_muestra = st.text_input("Nombre de nueva muestra", "")
    muestra_existente = None
else:
    nombre_muestra = seleccion
    muestra_existente = next((m for m in st.session_state.muestras if m["nombre"] == seleccion), None)

observacion_muestra = st.text_area("Observaciones de la muestra", muestra_existente["observacion"] if muestra_existente else "")

# Tabla editable de análisis
st.markdown("### Análisis físico-químicos")

analisis_existentes = muestra_existente["analisis"] if muestra_existente else []
df_input = pd.DataFrame(analisis_existentes) if analisis_existentes else pd.DataFrame(columns=["Tipo", "Valor", "Fecha", "Observaciones"])

# Completar columnas si faltan
for col in ["Tipo", "Valor", "Fecha", "Observaciones"]:
    if col not in df_input.columns:
        df_input[col] = "" if col != "Valor" else 0.0
    if col == "Fecha":
        df_input[col] = pd.to_datetime(df_input[col]).dt.date

df_input = st.data_editor(df_input, num_rows="dynamic", use_container_width=True, key="editor_formulario")

if st.button("Guardar muestra"):
    nueva_entrada = {
        "nombre": nombre_muestra,
        "observacion": observacion_muestra,
        "analisis": df_input.to_dict(orient="records")
    }

    # Si la muestra ya existe, actualizarla
    idx = next((i for i, m in enumerate(st.session_state.muestras) if m["nombre"] == nombre_muestra), None)
    if idx is not None:
        st.session_state.muestras[idx] = nueva_entrada
    else:
        st.session_state.muestras.append(nueva_entrada)

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.muestras, f, ensure_ascii=False, indent=2)
    backup_name = f"muestras_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(backup_name, "w", encoding="utf-8") as f:
        json.dump(st.session_state.muestras, f, ensure_ascii=False, indent=2)

    st.success("Muestra guardada correctamente.")
    st.rerun()

# --- TABLA GENERAL DE VISUALIZACIÓN ---
st.header("Muestras cargadas (visualización)")

data_expandida = []
for muestra in st.session_state.muestras:
    for analisis in muestra["analisis"]:
        data_expandida.append({
            "Nombre": muestra["nombre"],
            "Observación muestra": muestra["observacion"],
            "Tipo de análisis": analisis.get("tipo", ""),
            "Valor": analisis.get("valor", ""),
            "Fecha": analisis.get("fecha", ""),
            "Observaciones análisis": analisis.get("observaciones", "")
        })

if data_expandida:
    df_vista = pd.DataFrame(data_expandida)
    st.dataframe(df_vista, use_container_width=True)
    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine="xlsxwriter") as writer:
        df_vista.to_excel(writer, index=False, sheet_name="Muestras")
    st.download_button(
        label="Descargar Excel",
        data=excel_data.getvalue(),
        file_name=f"lab-polioles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No hay muestras cargadas todavía.")
