import streamlit as st
st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

from firebase_config import iniciar_firebase
from datetime import datetime
import uuid
import io
import base64
import pandas as pd
from fpdf import FPDF

# -----------------------------
# Función para generar PDF de ficha
# -----------------------------
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
    
    # Sección de Análisis Físico-Químicos (mostrará datos si existen en la ficha)
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

# -----------------------------
# Inicializar Firebase
# -----------------------------
db = iniciar_firebase()

# -----------------------------
# Sección 1: Gestión de Muestras
# -----------------------------
st.header("Gestión de Muestras")

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

# Formulario para agregar nueva muestra
st.subheader("Agregar nueva muestra")
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

# Mostrar muestras registradas
st.subheader("Muestras registradas")
muestras = cargar_muestras()
if not muestras:
    st.info("No hay muestras registradas.")
else:
    for m in muestras:
        with st.container():
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.write("Nombre:", m["nombre"])
                st.write("Fecha:", m["fecha"][:10])
                st.write("Observaciones:", m.get("observaciones", ""))
            with cols[1]:
                if st.button("Editar", key=f"edit_{m['id']}"):
                    st.session_state["editar"] = m["id"]
            with cols[2]:
                if st.button("Eliminar", key=f"delete_{m['id']}"):
                    st.session_state["eliminar"] = m["id"]
            # Modo edición inline
            if st.session_state.get("editar") == m["id"]:
                with st.form(key=f"form_editar_{m['id']}"):
                    nuevo_nombre = st.text_input("Nuevo nombre", value=m["nombre"])
                    nuevas_obs = st.text_area("Nuevas observaciones", value=m.get("observaciones", ""), height=100)
                    guardar = st.form_submit_button("Guardar cambios")
                    cancelar = st.form_submit_button("Cancelar")
                    if guardar:
                        editar_muestra(m["id"], nuevo_nombre, nuevas_obs)
                        st.success("Muestra actualizada. Recargá la página para ver los cambios.")
                        st.session_state["editar"] = None
                    if cancelar:
                        st.session_state["editar"] = None
            # Confirmación de eliminación inline
            if st.session_state.get("eliminar") == m["id"]:
                st.warning("Confirmá que querés eliminar esta muestra.")
                col_del, col_can = st.columns(2)
                with col_del:
                    if st.button("Sí", key=f"confirma_{m['id']}"):
                        eliminar_muestra(m["id"])
                        st.success("Muestra eliminada. Recargá la página para ver los cambios.")
                        st.session_state["eliminar"] = None
                with col_can:
                    if st.button("Cancelar", key=f"cancela_{m['id']}"):
                        st.session_state["eliminar"] = None

# Botón para ver la ficha (PDF) de una muestra, en una nueva sección dentro de la misma app
st.markdown("---")
st.header("Ficha de Muestra")
if muestras:
    opciones = {m["nombre"]: m for m in muestras}
    seleccion = st.selectbox("Seleccioná una muestra", list(opciones.keys()))
    muestra_seleccionada = opciones[seleccion]
    st.write("**Nombre:**", muestra_seleccionada.get("nombre", ""))
    st.write("**Observaciones:**", muestra_seleccionada.get("observaciones", "Sin observaciones"))
    st.write("**Fecha de carga:**", muestra_seleccionada.get("fecha", "Sin fecha"))
    pdf_bytes = generar_pdf_ficha(muestra_seleccionada)
    st.download_button("Descargar ficha en PDF", data=pdf_bytes, file_name=f"{muestra_seleccionada.get('nombre','Ficha')}.pdf", mime="application/pdf")
else:
    st.info("No hay muestras registradas.")

# =======================================================
# Sección 2: Análisis Físico-Químicos
# =======================================================
st.header("Análisis Físico-Químicos")

# Función para cargar análisis desde Firebase (cargará documentos de la colección "analisis")
def cargar_analisis():
    docs = db.collection("analisis").order_by("fecha", direction="DESCENDING").stream()
    return [doc.to_dict() | {"id": doc.id} for doc in docs]

# Función para subir un análisis
def subir_analisis(muestra_id, propiedad, valor, fecha, observacion):
    registro = {
        "muestra_id": muestra_id,
        "propiedad": propiedad,
        "valor": valor,
        "fecha": fecha.isoformat(),
        "observacion": observacion
    }
    db.collection("analisis").document().set(registro)

# Formulario para agregar un análisis
st.subheader("Agregar análisis")
if muestras:
    # Seleccionar muestra para la que se cargará el análisis
    opciones_muestras = {m["nombre"]: m["id"] for m in muestras}
    muestra_sel = st.selectbox("Seleccioná una muestra", list(opciones_muestras.keys()))
    propiedad = st.selectbox("Propiedad", [
         "Índice de yodo [% p/p I2 abs]",
         "Índice OH [mg KOH/g]",
         "Índice de acidez [mg KOH/g]",
         "Índice de epóxido [mol/100g]",
         "Humedad [%]",
         "PM [g/mol]",
         "Funcionalidad [#]",
         "Viscosidad dinámica [cP]",
         "Densidad [g/mL]"
    ])
    valor = st.number_input("Valor", min_value=0.0, format="%.2f")
    fecha_input = st.date_input("Fecha", value=datetime.now())
    observacion_analisis = st.text_area("Observación (opcional)", height=50)
    if st.button("Guardar análisis"):
         subir_analisis(opciones_muestras[muestra_sel], propiedad, valor, fecha_input, observacion_analisis)
         st.success("Análisis guardado. Recargá la página para ver los cambios.")
else:
    st.info("No hay muestras disponibles. Primero agregá una muestra.")

# Mostrar análisis cargados
st.subheader("Registros de análisis")
analisis = cargar_analisis()
if analisis:
    df_analisis = pd.DataFrame(analisis)
    st.dataframe(df_analisis)
else:
    st.info("Aún no hay registros de análisis.")
