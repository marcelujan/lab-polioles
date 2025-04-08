import streamlit as st
st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

from firebase_config import iniciar_firebase
from datetime import datetime
import uuid

# Inicializar Firebase
db = iniciar_firebase()

# Función para cargar muestras desde Firebase
def cargar_muestras():
    docs = db.collection("muestras").order_by("fecha", direction="DESCENDING").stream()
    return [doc.to_dict() | {"id": doc.id} for doc in docs]

# Función para subir una nueva muestra
def subir_muestra(nombre, observaciones):
    muestra = {
        "nombre": nombre,
        "observaciones": observaciones,
        "fecha": datetime.now().isoformat()
    }
    db.collection("muestras").document().set(muestra)

# Función para editar una muestra
def editar_muestra(id, nombre, observaciones):
    db.collection("muestras").document(id).update({
        "nombre": nombre,
        "observaciones": observaciones
    })

# Función para eliminar una muestra
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
            st.success("Muestra guardada correctamente. Recargá la página para ver los cambios.")
            st.experimental_rerun()

# =======================================================
# Sección: Mostrar muestras registradas
# =======================================================
st.header("Muestras registradas")
muestras = cargar_muestras()
if not muestras:
    st.info("No hay muestras registradas.")
else:
    # Recorrer cada muestra y mostrarla en una fila
    for m in muestras:
        with st.container():
            cols = st.columns([3, 1, 1])
            # Columna 1: Datos de la muestra
            with cols[0]:
                st.write("**Nombre:**", m["nombre"])
                st.write("**Fecha:**", m["fecha"][:10])
                st.write("**Observaciones:**", m.get("observaciones", ""))
            # Columna 2: Botón Editar (para activar modo edición)
            with cols[1]:
                if st.button("Editar", key="edit_" + m["id"]):
                    st.session_state["editar"] = m["id"]
            # Columna 3: Botón Eliminar (para activar confirmación)
            with cols[2]:
                if st.button("Eliminar", key="delete_" + m["id"]):
                    st.session_state["eliminar"] = m["id"]

        # Si la muestra está en modo edición, mostrar formulario inline
        if st.session_state.get("editar") == m["id"]:
            with st.form(key="form_editar_" + m["id"]):
                nuevo_nombre = st.text_input("Editar nombre", value=m["nombre"])
                nuevas_obs = st.text_area("Editar observaciones", value=m.get("observaciones", ""), height=100)
                guardar = st.form_submit_button("Guardar cambios")
                cancelar = st.form_submit_button("Cancelar")
                if guardar:
                    editar_muestra(m["id"], nuevo_nombre, nuevas_obs)
                    st.success("Muestra actualizada. Recargá la página para ver los cambios.")
                    st.session_state["editar"] = None
                    st.experimental_rerun()
                if cancelar:
                    st.session_state["editar"] = None
                    st.info("Edición cancelada. Recargá la página si es necesario.")

        # Si se ha activado el modo de eliminación para esta muestra, pedir confirmación inline
        if st.session_state.get("eliminar") == m["id"]:
            st.warning("¿Confirmá que querés eliminar esta muestra?")
            col_del, col_can = st.columns(2)
            with col_del:
                if st.button("Sí", key="confirma_" + m["id"]):
                    eliminar_muestra(m["id"])
                    st.success("Muestra eliminada. Recargá la página para ver los cambios.")
                    st.session_state["eliminar"] = None
                    st.experimental_rerun()
            with col_can:
                if st.button("Cancelar", key="cancela_" + m["id"]):
                    st.session_state["eliminar"] = None
                    st.info("Eliminación cancelada. Recargá la página si es necesario.")
