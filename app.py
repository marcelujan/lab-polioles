import streamlit as st
import toml
import json
import firebase_admin
from firebase_admin import credentials, firestore

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")
st.title("🚀 App iniciada correctamente")

st.write("🛡️ Verificando autenticación...")
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

st.write("🔐 Autenticado correctamente")

st.write("🔌 Conectando con Firebase...")
try:
    if "firebase_initialized" not in st.session_state:
        cred_dict = json.loads(st.secrets["firebase_key"])
        cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            st.session_state.firebase_initialized = True
    db = firestore.client()
    st.success("✅ Firebase conectado con éxito.")
except Exception as e:
    st.error(f"❌ Error al conectar con Firebase: {e}")
