
import streamlit as st
import pandas as pd
import toml
import json
from datetime import date, datetime
import firebase_admin
from firebase_admin import credentials, firestore

# Inicializar Firebase desde secrets
if "firebase_initialized" not in st.session_state:
    import json
    cred_dict = json.loads(st.secrets["firebase_key"])
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(cred_dict)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
    st.session_state.firebase_initialized = True

db = firestore.client()

# Inicializar Firebase

db = firestore.client()
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
try:
    docs = db.collection("muestras").stream()
    st.session_state.muestras = []
    for doc in docs:
        data = doc.to_dict()
        data["nombre"] = doc.id
        st.session_state.muestras.append(data)
except Exception as e:
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

# Mostrar última entrada de cada tipo de análisis estándar
st.markdown("### Análisis físico-químicos (últimos valores)")

base_rows = []
if muestra_existente:
    for tipo in tipos_analisis:
        filas = [a for a in muestra_existente["analisis"] if a["tipo"] == tipo]
        if filas:
            fila = sorted(filas, key=lambda x: x["fecha"])[-1]
            base_rows.append({
                "Tipo": tipo,
                "Valor": fila["valor"],
                "Fecha": fila["fecha"],
                "Observaciones": fila["observaciones"]
            })
        else:
            base_rows.append({
                "Tipo": tipo,
                "Valor": 0.0,
                "Fecha": date.today(),
                "Observaciones": ""
            })
else:
    for tipo in tipos_analisis:
        base_rows.append({
            "Tipo": tipo,
            "Valor": 0.0,
            "Fecha": date.today(),
            "Observaciones": ""
        })

df_base = pd.DataFrame(base_rows)
df_base_edit = st.data_editor(df_base, num_rows="fixed", use_container_width=True, key="base_editor")

# Análisis nuevos para repetir (vacíos)
st.markdown("### Repeticiones de análisis (opcional)")
df_repeticiones = pd.DataFrame([{"Tipo": "", "Valor": 0.0, "Fecha": date.today(), "Observaciones": ""}])
df_repeticiones_edit = st.data_editor(df_repeticiones, num_rows="dynamic", use_container_width=True, key="repe_editor")

if st.button("Guardar muestra"):
    nueva_entrada = {
        "nombre": nombre_muestra,
        "observacion": observacion_muestra,
        "analisis": []
    }
    for _, row in df_base_edit.iterrows():
        nueva_entrada["analisis"].append({
            "tipo": row["Tipo"],
            "valor": row["Valor"],
            "fecha": str(row["Fecha"]),
            "observaciones": row["Observaciones"]
        })
    for _, row in df_repeticiones_edit.iterrows():
        if row["Tipo"] in tipos_analisis:
            nueva_entrada["analisis"].append({
                "tipo": row["Tipo"],
                "valor": row["Valor"],
                "fecha": str(row["Fecha"]),
                "observaciones": row["Observaciones"]
            })

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
st.header("Muestras cargadas")

data_expandida = []
for i_muestra, muestra in enumerate(st.session_state.muestras):
    for i, analisis in enumerate(muestra.get("analisis", [])):
        data_expandida.append({
            "Nombre": muestra["nombre"],
            "Observación muestra": muestra["observacion"],
            "Tipo de análisis": analisis.get("tipo", ""),
            "Valor": analisis.get("valor", ""),
            "Fecha": analisis.get("fecha", ""),
            "Observaciones análisis": analisis.get("observaciones", ""),
            "Muestra_idx": i_muestra,
            "Analisis_idx": i
        })

if data_expandida:
    df_vista = pd.DataFrame(data_expandida)
    st.dataframe(df_vista.drop(columns=["Muestra_idx", "Analisis_idx"]), use_container_width=True)

    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine="xlsxwriter") as writer:
        df_vista.drop(columns=["Muestra_idx", "Analisis_idx"]).to_excel(writer, index=False, sheet_name="Muestras")
    st.download_button(
        label="Descargar Excel",
        data=excel_data.getvalue(),
        file_name=f"lab-polioles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No hay muestras cargadas todavía.")
