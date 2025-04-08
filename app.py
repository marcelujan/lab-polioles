import streamlit as st
st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

from firebase_config import iniciar_firebase
from datetime import datetime
import uuid
import io
import base64
from fpdf import FPDF

def generar_pdf_ficha(muestra: dict) -> io.BytesIO:
    """
    Genera un PDF con la ficha de la muestra.
    Se muestran los campos obligatorios (nombre) y todos los campos de análisis, incluso si están vacíos.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Título
    pdf.set_font("Arial", "B", 16)
    nombre = muestra.get("nombre", "Sin nombre")
    pdf.cell(0, 10, f"Ficha de Muestra: {nombre}", ln=True)
    pdf.ln(5)
    
    # Observaciones
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Observaciones:", ln=True)
    obs = muestra.get("observaciones", "")
    pdf.multi_cell(0, 10, obs if obs else "Sin observaciones")
    pdf.ln(5)
    
    # Fecha de carga
    pdf.cell(0, 10, f"Fecha de carga: {muestra.get('fecha', 'Sin fecha')}", ln=True)
    pdf.ln(5)
    
    # Lista de análisis físico-químicos
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Análisis Físico-Químicos:", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Arial", "", 12)
    propiedades = [
        ("Índice de yodo [% p/p I2 abs]", muestra.get("indice_yodo", "")),
        ("Índice OH [mg KOH/g]", muestra.get("indice_oh", "")),
        ("Índice de acidez [mg KOH/g]", muestra.get("indice_acidez", "")),
        ("Índice de epóxido [mol/100g]", muestra.get("indice_epoxido", "")),
        ("Humedad [%]", muestra.get("humedad", "")),
        ("PM [g/mol]", muestra.get("pm", "")),
        ("Funcionalidad [#]", muestra.get("funcionalidad", "")),
        ("Viscosidad dinámica [cP]", muestra.get("viscosidad", "")),
        ("Densidad [g/mL]", muestra.get("densidad", ""))
    ]
    for prop, valor in propiedades:
        pdf.cell(0, 10, f"{prop}: {valor}", ln=True)
    
    pdf_string = pdf.output(dest="S")
    pdf_bytes = pdf_string.encode("latin1")
    return io.BytesIO(pdf_bytes)

# Inicializar Firebase
db = iniciar_firebase()

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
    # Subida de imágenes, codificadas en base64
    for img in imagenes:
        img_data = img.read()
        encoded_data = base64.b64encode(img_data).decode("utf-8")
        img_id = str(uuid.uuid4())
        db.collection("imagenes").document(img_id).set({
            "nombre_muestra": nombre,
            "archivo": encoded_data,
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
        imagenes = st.file_uploader("Subir imágenes de la muestra (opcional)", 
                                    accept_multiple_files=True, type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Guardar muestra")
        if submitted:
            if not nombre:
                st.warning("El nombre es obligatorio.")
            else:
                subir_muestra(nombre, observaciones, imagenes)
                st.success("✅ Muestra guardada correctamente.")
                st.info("Por favor, recargá la página para ver los cambios.")

# --- Mostrar muestras registradas ---
st.subheader("📋 Muestras registradas")
muestras = cargar_muestras()
if not muestras:
    st.info("Aún no hay muestras registradas.")
else:
    for m in muestras:
        with st.container():
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f"### 🧪 {m['nombre']}")
                st.markdown(m.get("observaciones") or "_Sin observaciones_")
                st.caption(f"📅 Fecha: {m['fecha'][:10]}")
            with col2:
                if st.button("✏️ Editar", key=f"edit_{m['id']}"):
                    st.session_state["edit_id"] = m["id"]
            with col3:
                if st.button("Ver Ficha", key=f"ver_ficha_{m['id']}"):
                    with st.modal(f"Ficha de {m['nombre']}"):
                        st.markdown("### Detalles de la muestra")
                        st.markdown(f"**Nombre:** {m.get('nombre','')}")
                        st.markdown(f"**Observaciones:** {m.get('observaciones','Sin observaciones')}")
                        st.markdown(f"**Fecha de carga:** {m.get('fecha','Sin fecha')}")
                        st.markdown("#### Análisis Físico-Químicos")
                        propiedades = [
                            ("Índice de yodo [% p/p I2 abs]", m.get("indice_yodo", "")),
                            ("Índice OH [mg KOH/g]", m.get("indice_oh", "")),
                            ("Índice de acidez [mg KOH/g]", m.get("indice_acidez", "")),
                            ("Índice de epóxido [mol/100g]", m.get("indice_epoxido", "")),
                            ("Humedad [%]", m.get("humedad", "")),
                            ("PM [g/mol]", m.get("pm", "")),
                            ("Funcionalidad [#]", m.get("funcionalidad", "")),
                            ("Viscosidad dinámica [cP]", m.get("viscosidad", "")),
                            ("Densidad [g/mL]", m.get("densidad", ""))
                        ]
                        for prop, valor in propiedades:
                            st.markdown(f"**{prop}:** {valor}")
                        pdf_bytes = generar_pdf_ficha(m)
                        st.download_button("⬇️ Descargar ficha en PDF", data=pdf_bytes,
                                           file_name=f"{m.get('nombre','Ficha')}.pdf",
                                           mime="application/pdf")
            if st.button("🗑️ Eliminar", key=f"delete_{m['id']}"):
                with st.modal("Confirmar eliminación"):
                    st.warning(f"¿Estás seguro de que querés eliminar la muestra **{m['nombre']}**?", icon="⚠️")
                    col_del, col_can = st.columns(2)
                    with col_del:
                        if st.button("✅ Sí, eliminar"):
                            db.collection("muestras").document(m["id"]).delete()
                            imagenes = db.collection("imagenes").where("nombre_muestra", "==", m["nombre"]).stream()
                            for img in imagenes:
                                db.collection("imagenes").document(img.id).delete()
                            st.success("🗑️ Muestra eliminada correctamente.")
                            st.info("Por favor, recargá la página para ver los cambios.")
                    with col_can:
                        if st.button("❌ Cancelar"):
                            st.info("Eliminación cancelada.")
                            st.info("Por favor, recargá la página para ver los cambios.")

# Modo de edición para una muestra
if st.session_state.get("edit_id"):
    edit_id = st.session_state["edit_id"]
    muestra_edit = None
    for m in cargar_muestras():
        if m["id"] == edit_id:
            muestra_edit = m
            break
    if muestra_edit:
        with st.modal(f"Editar {muestra_edit['nombre']}"):
            with st.form(f"form_editar_{muestra_edit['id']}"):
                nuevo_nombre = st.text_input("Nombre de la muestra", value=muestra_edit["nombre"])
                nuevas_obs = st.text_area("Observaciones", value=muestra_edit["observaciones"], height=200)
                guardar = st.form_submit_button("💾 Guardar cambios")
                cancelar = st.form_submit_button("❌ Cancelar")
                if guardar:
                    db.collection("muestras").document(muestra_edit["id"]).update({
                        "nombre": nuevo_nombre,
                        "observaciones": nuevas_obs
                    })
                    st.success("✅ Muestra actualizada.")
                    st.info("Por favor, recargá la página para ver los cambios.")
                if cancelar:
                    st.info("Edición cancelada. Por favor, recargá la página para ver los cambios.")
