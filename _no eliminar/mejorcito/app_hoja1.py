import streamlit as st
import pandas as pd
import toml
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import date, datetime
from io import BytesIO
import os

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")
st.title("Laboratorio de Polioles")

# --- Autenticación simple ---
config = toml.load("config.toml")
PASSWORD = config["auth"]["password"]
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False
if not st.session_state.autenticado:
    pwd = st.text_input("Contraseña de acceso", type="password")
    if st.button("Ingresar"):
        if pwd == PASSWORD:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")
    st.stop()

# --- Firebase ---
if "firebase_initialized" not in st.session_state:
    cred_dict = json.loads(st.secrets["firebase_key"])
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(cred_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        st.session_state.firebase_initialized = True
db = firestore.client()

def cargar_muestras():
    try:
        docs = db.collection("muestras").stream()
        return [{**doc.to_dict(), "nombre": doc.id} for doc in docs]
    except:
        return []

def guardar_muestra(nombre, observacion, analisis):
    datos = {
        "observacion": observacion,
        "analisis": analisis
    }
    db.collection("muestras").document(nombre).set(datos)
    backup_name = f"muestras_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(backup_name, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

# --- Añadir muestra ---
st.subheader("Añadir muestra")
muestras = cargar_muestras()
nombres = [m["nombre"] for m in muestras]
opcion = st.selectbox("Seleccionar muestra", ["Nueva muestra"] + nombres)
if opcion == "Nueva muestra":
    nombre_muestra = st.text_input("Nombre de nueva muestra")
    muestra_existente = None
else:
    nombre_muestra = opcion
    muestra_existente = next((m for m in muestras if m["nombre"] == opcion), None)

observacion = st.text_area("Observaciones", value=muestra_existente["observacion"] if muestra_existente else "", height=150)

# --- Nuevo análisis ---
st.subheader("Nuevo análisis")
tipos = [
    "Índice de yodo [% p/p I2 abs]", "Índice OH [mg KHO/g]",
    "Índice de acidez [mg KOH/g]", "Índice de epóxido [mol/100g]",
    "Humedad [%]", "PM [g/mol]", "Funcionalidad [#]",
    "Viscosidad dinámica [cP]", "Densidad [g/mL]", "Otro análisis"
]
df = pd.DataFrame([{"Tipo": "", "Valor": 0.0, "Fecha": date.today(), "Observaciones": ""}])
nuevos_analisis = st.data_editor(df, num_rows="dynamic", use_container_width=True,
    column_config={"Tipo": st.column_config.SelectboxColumn("Tipo", options=tipos)})

if st.button("Guardar análisis"):
    previos = muestra_existente["analisis"] if muestra_existente else []
    nuevos = []
    for _, row in nuevos_analisis.iterrows():
        if row["Tipo"] != "":
            nuevos.append({
                "tipo": row["Tipo"],
                "valor": row["Valor"],
                "fecha": str(row["Fecha"]),
                "observaciones": row["Observaciones"]
            })
    guardar_muestra(nombre_muestra, observacion, previos + nuevos)
    st.success("Análisis guardado.")
    st.rerun()

# --- Visualización ---
st.subheader("Análisis cargados")
muestras = cargar_muestras()
tabla = []
for m in muestras:
    for a in m["analisis"]:
        tabla.append({
            "Nombre": m["nombre"],
            "Tipo": a["tipo"],
            "Valor": a["valor"],
            "Fecha": a["fecha"],
            "Observaciones": a["observaciones"]
        })
df_vista = pd.DataFrame(tabla)
if not df_vista.empty:
    st.dataframe(df_vista, use_container_width=True)

    st.subheader("Eliminar análisis")
    seleccion = st.selectbox("Seleccionar análisis a eliminar", df_vista.index,
        format_func=lambda i: f"{df_vista.at[i, 'Nombre']} – {df_vista.at[i, 'Tipo']} – {df_vista.at[i, 'Fecha']}")
    if st.button("Eliminar análisis"):
        elegido = df_vista.iloc[seleccion]
        for m in muestras:
            if m["nombre"] == elegido["Nombre"]:
                m["analisis"] = [a for a in m["analisis"] if not (
                    a["tipo"] == elegido["Tipo"] and str(a["fecha"]) == elegido["Fecha"]
                )]
                guardar_muestra(m["nombre"], m["observacion"], m["analisis"])
                st.success("Análisis eliminado.")
                st.rerun()

    st.subheader("Exportar")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_vista.to_excel(writer, index=False, sheet_name="Muestras")
    st.download_button("Descargar Excel",
        data=buffer.getvalue(),
        file_name=f"lab-polioles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No hay análisis cargados.")
