import streamlit as st
from firebase_config import iniciar_firebase
from datetime import datetime
import uuid

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

# Inicializar Firebase
db = iniciar_firebase()

# Título
st.title("🔬 Laboratorio de Polioles")

# --- Función para cargar muestras desde Firebase ---
def cargar_muestras():
    docs = db.collection("muestras").order_by("fecha", direction="DESCENDING").stream()
    return [doc.to_dict() | {"id": doc.id} for doc in docs]

# --- Función para subir una nueva muestra ---
def subir_muestra(nombre, observaciones, imagenes):
    nueva_muestra = {
        "nombre": nombre,
        "observaciones": observaciones,
        "fecha": datetime.now().isoformat(),
        "imagenes": []
    }

    # Subida de imágenes
    for img in imagenes:
        img_id = str(uuid.uuid4())
        db.collection("imagenes").document(img_id).set({
            "nombre_muestra": nombre,
            "archivo": img.read(),
            "filename": img.name,
            "tipo": "muestra",
            "fecha": datetime.now().isoformat()
        })
        nueva_muestra["imagenes"].append(img_id)

    doc_ref = db.collection("muestras").document()
    doc_ref.set(nueva_muestra)

# --- Formulario para agregar nueva muestra ---
with st.expander("➕ Agregar nueva muestra"):
    with st.form("form_nueva_muestra"):
        nombre = st.text_input("Nombre de la muestra *", max_chars=100)
        observaciones = st.text_area("Observaciones (100 palabras o más)", height=200)
        imagenes = st.file_uploader("Subir imágenes de la muestra (opcional)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Guardar muestra")
        if submitted:
            if not nombre:
                st.warning("El nombre es obligatorio.")
            else:
                subir_muestra(nombre, observaciones, imagenes)
                st.success("✅ Muestra guardada correctamente.")
                st.experimental_rerun()

# --- Mostrar muestras cargadas ---
st.subheader("📋 Muestras registradas")

muestras = cargar_muestras()
if not muestras:
    st.info("Aún no hay muestras registradas.")
else:
    for m in muestras:
        st.markdown(f"### 🧪 {m['nombre']}")
        st.markdown(m["observaciones"] or "_Sin observaciones_")
        st.caption(f"Fecha de carga: {m['fecha'][:10]}")
