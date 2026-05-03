import json
from functools import wraps

import firebase_admin
import streamlit as st
from firebase_admin import credentials

from auth_utils import iniciar_sesion
from audit_db import get_audit_db, patch_storage_for_audit
from firestore_utils import cargar_muestras, guardar_muestra
from tabs_tab1_lab import render_tab1
from tabs_tab2_datos import render_tab2
from tabs_tab3_espectros import render_tab3
from tabs_tab4_espectros import render_tab4
from tabs_tab5_ftir import render_tab5
from tabs_tab6_rmn import render_tab6
from tabs_tab7_consola import render_tab7
from tabs_tab8_sugerencias import render_tab8
from tabs_tab9_desarrollos import render_tab9
from tabs_tab10_mc import render_tab10
from tabs_tab11_down import render_tab11
from ui_utils import mostrar_sector_flotante


def patch_streamlit_width_args():
    """Compatibilidad temporal para Streamlit >= 1.57.

    Convierte use_container_width=True/False al nuevo argumento width antes de
    llamar a Streamlit, evitando warnings repetidos en todos los tabs.
    """
    nombres_funciones = (
        "dataframe",
        "data_editor",
        "plotly_chart",
        "image",
        "download_button",
        "pyplot",
        "altair_chart",
        "line_chart",
        "bar_chart",
        "area_chart",
        "scatter_chart",
    )

    for nombre in nombres_funciones:
        funcion_original = getattr(st, nombre, None)
        if funcion_original is None or getattr(funcion_original, "_width_arg_patched", False):
            continue

        @wraps(funcion_original)
        def funcion_patcheada(*args, _funcion_original=funcion_original, **kwargs):
            if "use_container_width" in kwargs and "width" not in kwargs:
                kwargs["width"] = "stretch" if kwargs.pop("use_container_width") else "content"
            return _funcion_original(*args, **kwargs)

        funcion_patcheada._width_arg_patched = True
        setattr(st, nombre, funcion_patcheada)


patch_streamlit_width_args()


if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(st.secrets["firebase_key"]))
    firebase_admin.initialize_app(
        cred,
        {"storageBucket": "laboratorio-polioles.firebasestorage.app"},
    )

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")


# ---------- Login ----------
if "auth" not in st.session_state:
    st.title("Laboratorio de Polioles")
    email = st.text_input("Correo electrónico")
    password = st.text_input("Contraseña", type="password")

    if st.button("Ingresar"):
        auth_ctx = iniciar_sesion(email, password)
        if auth_ctx:
            st.session_state["auth"] = auth_ctx
            st.session_state["user_email"] = auth_ctx["email"]
            st.rerun()

    st.stop()


auth_ctx = st.session_state["auth"]
st.session_state["user_email"] = auth_ctx.get("email", "")
st.session_state["audit_mode"] = not auth_ctx.get("can_use_app", False)


# ---------- Base activa ----------
if st.session_state["audit_mode"]:
    patch_storage_for_audit()
    if "audit_db" not in st.session_state:
        st.session_state["audit_db"] = get_audit_db()
    db = st.session_state["audit_db"]
else:
    if "firebase_initialized" not in st.session_state:
        st.session_state["db"] = firebase_admin.firestore.client()
        st.session_state["firebase_initialized"] = True
    db = st.session_state["db"]

st.session_state["firebase_db"] = db


# ---------- Tabs ----------
tabs = st.tabs([
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
    render_tab7(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante)
with tabs[7]:
    render_tab8(db, mostrar_sector_flotante)
with tabs[8]:
    render_tab9(db, cargar_muestras, mostrar_sector_flotante)
with tabs[9]:
    render_tab10(db, mostrar_sector_flotante)
with tabs[10]:
    render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante)

from ia_flotante import mostrar_panel_ia

if "user_email" in st.session_state and "firebase_db" in st.session_state:
    mostrar_panel_ia()
