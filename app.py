import streamlit as st
st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

from firebase_config import iniciar_firebase
from datetime import datetime
import uuid

# Inicializar Firebase
db = iniciar_firebase()

st.title("🔬 Laboratorio de Polioles")

# --- Función para cargar muestras desde Firebase ---
def cargar_muestras():
    docs = db.collection("muestras").order_by("fecha", direction="DESCENDING").stream()
    return [doc.to_dict() | {"id": doc.id} for doc in docs]

# --- Función para subir una nueva muestra ---
def subir_muestra(nombre, observaciones):
    muestra = {
        "nombre": nombre,
        "observaciones": observaciones,
        "fecha": datetime.now().isoformat()
    }
    db.collection("muestras").document().set(muestra)

# --- Función para editar una muestra ---
def editar_muestra(id, nombre, observaciones):
    db.collection("muestras").document(id).update({
        "nombre": nombre,
        "observaciones": observaciones
    })

# --- Función para eliminar una muestra ---
def eliminar_muestra(id):
    db.collection("muestras").document(id).delete()

# =======================================================
# Sección: Agregar nueva muestra
# =======================================================
st.header("Agregar nueva muestra")
with st.form("form_nueva_muestra"):
    nuevo_nombre = st.text_input("Nombre de la muestra *", max_chars=100)
    nuevas_obs = st.text_area("Observaciones (opcional)", height=100)
    submit_nueva = st.form_submit_button("Guardar")
    if submit_nueva:
        if not nuevo_nombre:
            st.error("El nombre es obligatorio.")
        else:
            subir_muestra(nuevo_nombre, nuevas_obs)
            st.success("Muestra guardada correctamente.")
            st.info("Recargá la página para ver los cambios.")

# =======================================================
# Sección: Mostrar muestras registradas
# =======================================================
st.header("Muestras registradas")
muestras = cargar_muestras()
if not muestras:
    st.info("No hay muestras registradas.")
else:
    for m in muestras:
        with st.container():
            cols = st.columns([3, 1, 1])
            # Columna 1: Datos de la muestra
            with cols[0]:
                st.write("**Nombre:**", m["nombre"])
                st.write("**Fecha:**", m["fecha"][:10])
                st.write("**Observaciones:**", m.get("observaciones", ""))
            # Columna 2: Botón Editar
            with cols[1]:
                if st.button("Editar", key=f"edit_{m['id']}"):
                    st.session_state["editar"] = m["id"]
            # Columna 3: Botón Eliminar
            with cols[2]:
                if st.button("Eliminar", key=f"delete_{m['id']}"):
                    st.session_state["eliminar"] = m["id"]
        
        # Modo edición inline
        if st.session_state.get("editar") == m["id"]:
            with st.form(key=f"form_editar_{m['id']}"):
                nuevo_nombre = st.text_input("Editar nombre", value=m["nombre"])
                nuevas_obs = st.text_area("Editar observaciones", value=m.get("observaciones", ""), height=100)
                guardar = st.form_submit_button("Guardar cambios")
                cancelar = st.form_submit_button("Cancelar")
                if guardar:
                    editar_muestra(m["id"], nuevo_nombre, nuevas_obs)
                    st.success("Muestra actualizada.")
                    st.session_state["editar"] = None
                    st.info("Recargá la página para ver los cambios.")
                if cancelar:
                    st.session_state["editar"] = None
                    st.info("Edición cancelada. Recargá la página para ver los cambios.")
        
        # Confirmación de eliminación inline
        if st.session_state.get("eliminar") == m["id"]:
            st.warning("¿Confirmás que querés eliminar esta muestra?")
            col_del, col_can = st.columns(2)
            with col_del:
                if st.button("Sí", key=f"confirma_{m['id']}"):
                    eliminar_muestra(m["id"])
                    st.success("Muestra eliminada correctamente.")
                    st.session_state["eliminar"] = None
                    st.info("Recargá la página para ver los cambios.")
            with col_can:
                if st.button("Cancelar", key=f"cancela_{m['id']}"):
                    st.session_state["eliminar"] = None
                    st.info("Eliminación cancelada. Recargá la página para ver los cambios.")

