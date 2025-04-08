import streamlit as st
from firebase_config import iniciar_firebase
import io
from fpdf import FPDF

st.set_page_config(page_title="Ficha de Muestra", layout="wide")

db = iniciar_firebase()

def cargar_muestras():
    docs = db.collection("muestras").order_by("fecha", direction="DESCENDING").stream()
    return [doc.to_dict() | {"id": doc.id} for doc in docs]

def generar_pdf_ficha(muestra: dict) -> io.BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Arial", "B", 16)
    nombre = muestra.get("nombre", "Sin nombre")
    pdf.cell(0, 10, f"Ficha de Muestra: {nombre}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Observaciones:", ln=True)
    obs = muestra.get("observaciones", "")
    pdf.multi_cell(0, 10, obs if obs else "Sin observaciones")
    pdf.ln(5)
    
    pdf.cell(0, 10, f"Fecha de carga: {muestra.get('fecha', 'Sin fecha')}", ln=True)
    pdf.ln(5)
    
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

st.header("Ficha de Muestra")
muestras = cargar_muestras()
if not muestras:
    st.info("No hay muestras registradas.")
else:
    opciones = {m["nombre"]: m for m in muestras}
    seleccion = st.selectbox("Seleccioná una muestra", list(opciones.keys()), key="select_ficha")
    muestra = opciones[seleccion]
    st.write("**Nombre:**", muestra.get("nombre", ""))
    st.write("**Fecha de carga:**", muestra.get("fecha", ""))
    st.write("**Observaciones:**", muestra.get("observaciones", "Sin observaciones"))
    
    pdf_bytes = generar_pdf_ficha(muestra)
    st.download_button("Descargar ficha en PDF", data=pdf_bytes, 
                       file_name=f"{muestra.get('nombre', 'Ficha')}.pdf", 
                       mime="application/pdf")
