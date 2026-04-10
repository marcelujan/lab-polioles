import json

import firebase_admin
import streamlit as st
from firebase_admin import credentials

from audit_db import AuditFirestoreClient, apply_audit_mode
from auth_utils import iniciar_sesion
from firestore_utils import cargar_muestras, guardar_muestra
from ia_flotante import mostrar_panel_ia
from tabs_tab10_mc import render_tab10
from tabs_tab11_down import render_tab11
from tabs_tab1_lab import render_tab1
from tabs_tab2_datos import render_tab2
from tabs_tab3_espectros import render_tab3
from tabs_tab4_espectros import render_tab4
from tabs_tab5_ftir import render_tab5
from tabs_tab6_rmn import render_tab6
from tabs_tab7_consola import render_tab7
from tabs_tab8_sugerencias import render_tab8
from tabs_tab9_desarrollos import render_tab9
from ui_utils import mostrar_sector_flotante


if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(st.secrets["firebase_key"]))
    firebase_admin.initialize_app(
        cred,
        {"storageBucket": "laboratorio-polioles.firebasestorage.app"},
    )

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")


TABS = [
    "Lab polioles",
    "Análisis de datos",
    "Carga de espectros",
    "Análisis de espectros",
    "FTIR",
    "RMN",
    "Consola",
    "Sugerencias",
    "Desarrollo",
    "MC",
    "Down",
]


def _clear_runtime_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("_espectros_cache_"):
            st.session_state.pop(key, None)
    st.session_state.pop("db", None)
    st.session_state.pop("firebase_initialized", None)
    st.session_state.pop("firebase_db", None)


if "auth" not in st.session_state:
    st.title("Laboratorio de Polioles")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Correo electrónico")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")

    if submitted:
        auth_ctx = iniciar_sesion(email, password)
        if auth_ctx:
            _clear_runtime_state()
            st.session_state["auth"] = auth_ctx
            st.session_state["user_email"] = auth_ctx.get("email")
            st.rerun()
        else:
            st.error("No fue posible iniciar sesión.")
    st.stop()


auth_ctx = st.session_state.get("auth", {})
audit_mode = not auth_ctx.get("can_use_app", False)
st.session_state["audit_mode"] = audit_mode

if audit_mode:
    if st.session_state.get("firebase_mode") != "audit":
        _clear_runtime_state()
        apply_audit_mode()
        st.session_state.db = AuditFirestoreClient()
        st.session_state.firebase_mode = "audit"
else:
    if st.session_state.get("firebase_mode") != "live":
        _clear_runtime_state()
        st.session_state.db = firebase_admin.firestore.client()
        st.session_state.firebase_mode = "live"

db = st.session_state.db


tabs = st.tabs(TABS)

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
    render_tab7(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante)
with tabs[7]:
    render_tab8(db, mostrar_sector_flotante)
with tabs[8]:
    render_tab9(db, cargar_muestras, mostrar_sector_flotante)
with tabs[9]:
    render_tab10(db, mostrar_sector_flotante)
with tabs[10]:
    render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante)

if not audit_mode and "user_email" in st.session_state and "db" in st.session_state:
    st.session_state["firebase_db"] = st.session_state.db
    mostrar_panel_ia()
