import streamlit as st
from firestore_utils import iniciar_firebase, cargar_muestras, guardar_muestra
from auth_utils import registrar_usuario, iniciar_sesion
from ui_utils import mostrar_sector_flotante

from tabs_tab1_lab import render_tab1
from tabs_tab2_datos import render_tab2
from tabs_tab3_espectros import render_tab3
from tabs_tab4_espectros import render_tab4
from tabs_tab5_ftir import render_tab5  # Análisis FTIR con índice OH y comparación
from tabs_tab6_rmn import render_tab6
from tabs_tab7_consola import render_tab7
from tabs_tab8_sugerencias import render_tab8
from tabs_tab9_desarrollos import render_tab9  # Hoja en blanco para pruebas
from tabs_tab10_rmn2d import render_tab10

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")
FIREBASE_API_KEY = st.secrets["firebase_api_key"]

# --- Autenticación ---
if "token" not in st.session_state:
    st.markdown("### Iniciar sesión")
    email = st.text_input("Correo electrónico")
    password = st.text_input("Contraseña", type="password")
    st.warning("Si usás autocompletar, verificá que los campos estén visibles antes de continuar.")

    if st.button("Iniciar sesión"):
        token = iniciar_sesion(email, password)
        if token:
            st.session_state["token"] = token
            st.session_state["user_email"] = email
            st.success("Inicio de sesión exitoso.")
            st.rerun()

    st.markdown("---")
    st.markdown("### ¿No tenés cuenta? Registrate aquí:")
    with st.form("registro"):
        nuevo_email = st.text_input("Nuevo correo")
        nueva_clave = st.text_input("Nueva contraseña", type="password")
        submit_registro = st.form_submit_button("Registrar")
        if submit_registro:
            registrar_usuario(nuevo_email, nueva_clave)
            token = iniciar_sesion(nuevo_email, nueva_clave)
            if token:
                st.session_state["token"] = token
                st.success("Registro e inicio de sesión exitoso.")
                st.rerun()
    st.stop()

# --- Firebase ---
if "firebase_initialized" not in st.session_state:
    st.session_state.db = iniciar_firebase(st.secrets["firebase_key"])
    st.session_state.firebase_initialized = True
db = st.session_state.db

# --- Tabs principales ---
tabs = st.tabs([
    "Lab de polioles",  # 0
    "Análisis de datos",        # 1
    "Carga de espectros",       # 2
    "Análisis de espectros",    # 3
    "FTIR",            # 4
    "RMN",             # 5
    "RMN 2D",                   # 6
    "Consola",                  # 7
    "Sugerencias",              # 8
    "Desarrollos"               # 9
])


with tabs[0]:
    render_tab1(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante)
with tabs[1]:
    render_tab2(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante)
with tabs[2]:
    render_tab3(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante)
with tabs[3]:
    render_tab4(db, cargar_muestras, mostrar_sector_flotante)
with tabs[4]:
    render_tab5(db, cargar_muestras, mostrar_sector_flotante) 
with tabs[5]:
    render_tab6(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante)
with tabs[6]:
    render_tab10(db, cargar_muestras, mostrar_sector_flotante)
with tabs[7]:
    render_tab7(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante)
with tabs[8]:
    render_tab8(db, mostrar_sector_flotante)
with tabs[9]:
    render_tab9(db, cargar_muestras, mostrar_sector_flotante)

from ia_flotante import mostrar_panel_ia
if "user_email" in st.session_state and "db" in st.session_state:
    st.session_state["firebase_db"] = st.session_state.db
    mostrar_panel_ia()
