
import streamlit as st
import pandas as pd
import toml
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import date, datetime
from io import BytesIO
import os
import matplotlib.pyplot as plt
import zipfile
from tempfile import TemporaryDirectory

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

# --- Autenticación ---
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
    cred_dict["private_key"] = cred_dict["private_key"].replace("\n", "\n")
    cred = credentials.Certificate(cred_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        st.session_state.firebase_initialized = True
db = firestore.client()

# --- Funciones comunes ---
def cargar_muestras():
    try:
        docs = db.collection("muestras").stream()
        return [{**doc.to_dict(), "nombre": doc.id} for doc in docs]
    except:
        return []

def guardar_muestra(nombre, observacion, analisis, espectros=None):
    datos = {
        "observacion": observacion,
        "analisis": analisis
    }
    if espectros is not None:
        datos["espectros"] = espectros
    db.collection("muestras").document(nombre).set(datos)
    backup_name = f"muestras_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(backup_name, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

# --- Simulación de Hoja 3 para mostrar solo la parte de descarga corregida ---
st.title("📦 Exportación de espectros")

muestras = cargar_muestras()

# Crear tabla de espectros simulada
filas = []
for m in muestras:
    for i, e in enumerate(m.get("espectros", [])):
        filas.append({
            "Muestra": m["nombre"],
            "Tipo": e.get("tipo", ""),
            "Archivo": e.get("nombre_archivo", ""),
            "Observaciones": e.get("observaciones", ""),
            "ID": f"{m['nombre']}__{i}"
        })
df_esp_tabla = pd.DataFrame(filas)

# Generar ZIP al inicio (no en respuesta a botón)
if not df_esp_tabla.empty:
    with TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, f"espectros_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
        excel_path = os.path.join(tmpdir, "tabla_espectros.xlsx")

        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df_esp_tabla.drop(columns=["ID"]).to_excel(writer, index=False, sheet_name="Espectros")

        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(excel_path, arcname="tabla_espectros.xlsx")
            for m in muestras:
                for e in m.get("espectros", []):
                    contenido = e.get("contenido")
                    if not contenido:
                        continue
                    carpeta = f"{m['nombre']}"
                    nombre = e.get("nombre_archivo", "espectro")
                    fullpath = os.path.join(tmpdir, carpeta)
                    os.makedirs(fullpath, exist_ok=True)
                    file_path = os.path.join(fullpath, nombre)
                    with open(file_path, "wb") as file_out:
                        if e.get("es_imagen"):
                            file_out.write(bytes.fromhex(contenido))
                        else:
                            file_out.write(contenido.encode("latin1"))
                    zipf.write(file_path, arcname=os.path.join(carpeta, nombre))

        with open(zip_path, "rb") as final_zip:
            st.download_button("📦 Descargar espectros", data=final_zip.read(),
                               file_name=os.path.basename(zip_path),
                               mime="application/zip")
else:
    st.info("No hay espectros cargados.")
