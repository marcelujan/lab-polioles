import streamlit as st
from firebase_config import iniciar_firebase

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")
st.title("🔬 Gestión de Muestras - Firebase + Streamlit")

db = iniciar_firebase()
coleccion = db.collection("muestras")

# -------------------- Cargar muestra --------------------
st.subheader("➕ Agregar nueva muestra")
with st.form("form_muestra"):
    nombre = st.text_input("Nombre")
    tipo = st.selectbox("Tipo", ["Aceite", "Epóxido", "Poliol", "Otro"])
    origen = st.selectbox("Origen", ["Soja", "Vegetal", "Otro", "Desconocido"])
    observaciones = st.text_area("Observaciones")
    submit = st.form_submit_button("Guardar")

    if submit and nombre:
        coleccion.document(nombre).set({
            "tipo": tipo,
            "origen": origen,
            "observaciones": observaciones
        })
        st.success(f"✅ Muestra '{nombre}' guardada.")

# -------------------- Ver todas las muestras --------------------
st.subheader("📋 Muestras cargadas")

docs = coleccion.stream()
muestras = []
for doc in docs:
    data = doc.to_dict()
    data["nombre"] = doc.id
    muestras.append(data)

if muestras:
    st.dataframe(muestras, use_container_width=True)
else:
    st.info("Todavía no hay muestras registradas.")
