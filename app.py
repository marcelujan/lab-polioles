
import streamlit as st
import pandas as pd
import toml
import json
from datetime import date, datetime
import firebase_admin
from firebase_admin import credentials, firestore
from io import BytesIO
import os

# Inicializar Firebase desde secrets
if "firebase_initialized" not in st.session_state:
    cred_dict = json.loads(st.secrets["firebase_key"])
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(cred_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        st.session_state.firebase_initialized = True

db = firestore.client()

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

# Leer datos de Firestore
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
    "Densidad [g/mL]",
    "Otro análisis"
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

st.markdown("### Análisis físico-químicos")

analisis_existentes = muestra_existente["analisis"] if muestra_existente else []
df_analisis = pd.DataFrame(analisis_existentes)
if not df_analisis.empty:
    df_analisis["Tipo"] = df_analisis["tipo"]
    df_analisis["Valor"] = df_analisis["valor"]
    df_analisis["Fecha"] = pd.to_datetime(df_analisis["fecha"]).dt.date
    df_analisis["Observaciones"] = df_analisis["observaciones"]
    df_analisis = df_analisis[["Tipo", "Valor", "Fecha", "Observaciones"]]
else:
    df_analisis = pd.DataFrame([{"Tipo": "", "Valor": 0.0, "Fecha": date.today(), "Observaciones": ""}])

# Mostrar tabla editable y con eliminación de filas desde el editor
edited = st.data_editor(
    df_analisis,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Tipo": st.column_config.SelectboxColumn("Tipo", options=tipos_analisis)
    },
    key="editor_analisis"
)

if st.button("Guardar muestra"):
    nueva_entrada = {
        "nombre": nombre_muestra,
        "observacion": observacion_muestra,
        "analisis": []
    }
    for _, row in edited.iterrows():
        if row["Tipo"] != "":
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

    db.collection("muestras").document(nombre_muestra).set({
        "observacion": observacion_muestra,
        "analisis": nueva_entrada["analisis"]
    })

    st.success("Muestra guardada correctamente.")
    st.rerun()

# --- VISUALIZACIÓN ---
st.header("Muestras cargadas")

data_expandida = []
for muestra in st.session_state.muestras:
    for analisis in muestra.get("analisis", []):
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
