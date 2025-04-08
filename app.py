import streamlit as st
from firebase_config import iniciar_firebase
from datetime import datetime
import uuid
import io
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
    
    # Obtenemos el PDF como cadena (dest="S") y lo convertimos a bytes
    pdf_string = pdf.output(dest="S")
    pdf_bytes = pdf_string.encode("latin1")
    return io.BytesIO(pdf_bytes)

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
        imagenes = st.file_uploader("Subir imágenes de la muestra (opcional)", 
                                    accept_multiple_files=True, type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Guardar muestra")
        if submitted:
            if not nombre:
                st.warning("El nombre es obligatorio.")
            else:
                subir_muestra(nombre, observaciones, imagenes)
                st.success("✅ Muestra guardada correctamente.")
                # Usamos un flag en session_state en lugar de llamar directamente a experimental_rerun
                st.session_state["nueva_muestra_guardada"] = True

# Fuera del bloque del formulario, después de procesar la carga
if st.session_state.get("nueva_muestra_guardada"):
    st.session_state["nueva_muestra_guardada"] = False  # Reiniciamos el flag
    try:
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error al recargar la app: {e}")

# --- Mostrar muestras cargadas ---
st.subheader("📋 Muestras registradas")
muestras = cargar_muestras()
if not muestras:
    st.info("Aún no hay muestras registradas.")
else:
    for m in muestras:
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"### 🧪 {m['nombre']}")
                st.markdown(m["observaciones"] or "_Sin observaciones_")
                st.caption(f"📅 Fecha: {m['fecha'][:10]}")
            with col2:
                if st.button("✏️ Editar", key=f"edit_{m['id']}"):
                    st.session_state["edit_id"] = m["id"]
                if st.button("🗑️ Eliminar", key=f"delete_{m['id']}"):
                    st.session_state["delete_id"] = m["id"]

        # Si está en modo edición para esta muestra
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
                    try:
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error al recargar la app: {e}")
                if cancelar:
                    st.session_state["edit_id"] = None
                    try:
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error al recargar la app: {e}")

# --- Confirmación y ejecución de borrado ---
if "delete_id" in st.session_state and st.session_state["delete_id"]:
    id_muestra = st.session_state["delete_id"]
    muestra = db.collection("muestras").document(id_muestra).get().to_dict()
    st.warning(f"¿Estás seguro de que querés eliminar la muestra **{muestra['nombre']}**?", icon="⚠️")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Sí, eliminar"):
            # Eliminar muestra
            db.collection("muestras").document(id_muestra).delete()

            # Eliminar imágenes asociadas
            imagenes = db.collection("imagenes").where("nombre_muestra", "==", muestra["nombre"]).stream()
            for img in imagenes:
                db.collection("imagenes").document(img.id).delete()

            # (Aquí se eliminarían otros datos asociados, como análisis y espectros)
            st.success("🗑️ Muestra eliminada correctamente.")
            st.session_state["delete_id"] = None
            try:
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error al recargar la app: {e}")
    with col2:
        if st.button("❌ Cancelar eliminación"):
            st.session_state["delete_id"] = None
            try:
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error al recargar la app: {e}")

st.markdown("---")
st.subheader("📄 Ficha de Muestra")
# Obtener las muestras nuevamente
muestras = cargar_muestras()
if not muestras:
    st.info("No hay muestras registradas.")
else:
    opciones = {m["nombre"]: m for m in muestras}
    seleccion = st.selectbox("Seleccioná una muestra para ver su ficha", list(opciones.keys()))
    muestra_seleccionada = opciones[seleccion]
    st.markdown("### Detalles de la muestra")
    st.markdown(f"**Nombre:** {muestra_seleccionada.get('nombre', '')}")
    st.markdown(f"**Observaciones:** {muestra_seleccionada.get('observaciones', 'Sin observaciones')}")
    st.markdown(f"**Fecha de carga:** {muestra_seleccionada.get('fecha', 'Sin fecha')}")
    st.markdown("#### Análisis Físico-Químicos")
    propiedades = [
        ("Índice de yodo [% p/p I2 abs]", muestra_seleccionada.get("indice_yodo", "")),
        ("Índice OH [mg KOH/g]", muestra_seleccionada.get("indice_oh", "")),
        ("Índice de acidez [mg KOH/g]", muestra_seleccionada.get("indice_acidez", "")),
        ("Índice de epóxido [mol/100g]", muestra_seleccionada.get("indice_epoxido", "")),
        ("Humedad [%]", muestra_seleccionada.get("humedad", "")),
        ("PM [g/mol]", muestra_seleccionada.get("pm", "")),
        ("Funcionalidad [#]", muestra_seleccionada.get("funcionalidad", "")),
        ("Viscosidad dinámica [cP]", muestra_seleccionada.get("viscosidad", "")),
        ("Densidad [g/mL]", muestra_seleccionada.get("densidad", ""))
    ]
    for prop, valor in propiedades:
        st.markdown(f"**{prop}:** {valor}")
    pdf_bytes = generar_pdf_ficha(muestra_seleccionada)
    st.download_button("⬇️ Descargar ficha en PDF", data=pdf_bytes, file_name=f"{muestra_seleccionada.get('nombre','Ficha')}.pdf", mime="application/pdf")

