import json

import firebase_admin
import streamlit as st
from firebase_admin import credentials

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


PUBLIC_SECTIONS = [
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


OWNER_EMAIL = "mlujan1863@gmail.com"


def cerrar_sesion():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()



def render_portada_publica():
    st.title("Laboratorio de Polioles")
    st.caption("Acceso público de auditoría. El uso operativo está restringido.")

    st.info(
        "Cualquier visitante puede revisar la portada y el alcance de la plataforma, "
        "pero las funciones conectadas a Firebase quedan reservadas a cuentas habilitadas manualmente."
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("Qué puede ver un auditor")
        st.markdown(
            """
            - Estructura general de la aplicación.
            - Módulos disponibles y propósito de cada uno.
            - Criterio de acceso y forma de solicitar habilitación.
            """
        )

        st.subheader("Qué no puede hacer un auditor")
        st.markdown(
            """
            - Leer o escribir datos en Firestore.
            - Cargar espectros o editar muestras.
            - Guardar sugerencias, conclusiones u observaciones.
            - Usar herramientas internas de análisis.
            """
        )

    with col2:
        st.subheader("Módulos disponibles")
        for section in PUBLIC_SECTIONS:
            st.markdown(f"- {section}")

        st.subheader("Solicitud de acceso")
        st.write(
            "Las cuentas operativas se crean manualmente por el administrador. "
            "No hay auto-registro desde esta web."
        )



def render_login():
    st.markdown("---")
    st.subheader("Ingreso")
    email = st.text_input("Correo electrónico")
    password = st.text_input("Contraseña", type="password")
    st.warning(
        "Si usás autocompletar, verificá que ambos campos estén visibles antes de continuar."
    )

    if st.button("Iniciar sesión", type="primary"):
        auth_ctx = iniciar_sesion(email, password)
        if auth_ctx:
            st.session_state["auth"] = auth_ctx
            st.session_state["token"] = auth_ctx["id_token"]
            st.session_state["user_email"] = auth_ctx["email"]
            st.success("Inicio de sesión exitoso.")
            st.rerun()



def render_modo_auditoria(auth_ctx):
    st.title("Modo auditoría")
    st.caption(f"Sesión iniciada como {auth_ctx['email']}")

    st.warning(
        "Tu cuenta existe, pero no está habilitada para operar en la aplicación. "
        "No se iniciará la conexión a Firestore."
    )

    st.markdown("### Alcance visible")
    for section in PUBLIC_SECTIONS:
        st.markdown(f"- {section}")

    st.markdown("### Estado del acceso")
    st.code(
        json.dumps(
            {
                "email": auth_ctx["email"],
                "role": auth_ctx.get("role", "auditor"),
                "can_use_app": auth_ctx.get("can_use_app", False),
            },
            indent=2,
            ensure_ascii=False,
        ),
        language="json",
    )

    st.info(
        "Para habilitar el uso real, el administrador debe crear tu cuenta y asignarte "
        "el claim can_use_app=true."
    )


# --- Portada pública ---
auth_ctx = st.session_state.get("auth")
if not auth_ctx:
    render_portada_publica()
    render_login()
    st.stop()

with st.sidebar:
    st.write(f"Sesión: {auth_ctx['email']}")
    if st.button("Cerrar sesión"):
        cerrar_sesion()

# --- Usuario autenticado pero no habilitado ---
if not auth_ctx.get("can_use_app", False):
    render_modo_auditoria(auth_ctx)
    st.stop()

# --- Firebase solo para usuarios habilitados ---
if "firebase_initialized" not in st.session_state:
    st.session_state.db = firebase_admin.firestore.client()
    st.session_state.firebase_initialized = True

db = st.session_state.db

# --- Tabs principales ---
tabs = st.tabs(PUBLIC_SECTIONS)

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

if auth_ctx["email"] == OWNER_EMAIL and "db" in st.session_state:
    st.session_state["firebase_db"] = st.session_state.db
    mostrar_panel_ia()
