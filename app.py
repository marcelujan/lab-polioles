import streamlit as st
st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

from firebase_config import iniciar_firebase
from datetime import datetime
import uuid
import io
from fpdf import FPDF

# Función para generar PDF (igual que antes)
def generar_pdf_ficha(muestra: dict) -> io.BytesIO:
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

# Inicializamos Firebase
db = iniciar_firebase()

st.title("🔬 Laboratorio de Polioles")

# Función para cargar muestras desde Firebase
def cargar_muestras():
    docs = db.collection("muestras").order_by("fecha", direction="DESCENDING").stream()
    return [doc.to_dict() | {"id": doc.id} for doc in docs]

# Función para subir una nueva muestra
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
        imagenes = st.file_uploader("Subir imágenes de la muestra (opcional)", 
                                    accept_multiple_files=True, type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Guardar muestra")
        if submitted:
            if not nombre:
                st.warning("El nombre es obligatorio.")
            else:
                subir_muestra(nombre, observaciones, imagenes)
                st.success("✅ Muestra guardada correctamente.")
                st.experimental_rerun()  # Forzamos la recarga para actualizar la lista

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
                    # Abrir un modal para ver la ficha
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
                        # Botón para descargar ficha en PDF
                        pdf_bytes = generar_pdf_ficha(m)
                        st.download_button("⬇️ Descargar ficha en PDF", data=pdf_bytes,
                                           file_name=f"{m.get('nombre','Ficha')}.pdf",
                                           mime="application/pdf")
            # Botón de eliminación en un tercer columna o debajo del contenedor
            if st.button("🗑️ Eliminar", key=f"delete_{m['id']}"):
                with st.modal("Confirmar eliminación"):
                    st.warning(f"¿Estás seguro de que querés eliminar la muestra **{m['nombre']}**?", icon="⚠️")
                    col_del, col_can = st.columns(2)
                    with col_del:
                        if st.button("✅ Sí, eliminar"):
                            db.collection("muestras").document(m["id"]).delete()
                            # Eliminar imágenes asociadas
                            imagenes = db.collection("imagenes").where("nombre_muestra", "==", m["nombre"]).stream()
                            for img in imagenes:
                                db.collection("imagenes").document(img.id).delete()
                            st.success("🗑️ Muestra eliminada correctamente.")
                            st.experimental_rerun()
                    with col_can:
                        if st.button("❌ Cancelar"):
                            st.info("Eliminación cancelada.")
                            st.experimental_rerun()
        # Modo de edición inline (si corresponde)
        if st.session_state.get("edit_id") == m["id"]:
            with st.form(f"form_editar_{m['id']}"):
                nuevo_nombre = st.text_input("Nombre de la muestra", value=m["nombre"])
                nuevas_obs = st.text_area("Observaciones", value=m["observaciones"], height=200)
                guardar = st.form_submit_button("💾 Guardar cambios")
                cancelar = st.form_submit_button("❌ Cancelar")
                if guardar:
                    db.collection("muestras").document(m["id"]).update({
                        "nombre": nuevo_nombre,
                        "observaciones": nuevas_obs
                    })
                    st.success("✅ Muestra actualizada.")
                    st.session_state["edit_id"] = None
                    st.experimental_rerun()
                if cancelar:
                    st.session_state["edit_id"] = None
                    st.experimental_rerun()
