# --- Corrección del bloque multiselect ---
import streamlit as st
import pandas as pd

df = pd.DataFrame([
    {"ID": "M1_0", "Nombre": "M1", "Tipo": "Ácido", "Fecha": "2025-01-01"},
    {"ID": "M2_0", "Nombre": "M2", "Tipo": "Viscosidad", "Fecha": "2025-01-01"}
])

seleccion = st.multiselect(
    "Seleccione uno o más análisis para graficar",
    options=df["ID"].tolist(),
    format_func=lambda i: f"{df[df['ID'] == i]['Nombre'].values[0]} - {df[df['ID'] == i]['Tipo'].values[0]} - {df[df['ID'] == i]['Fecha'].values[0]}",
    key="multiselect_analisis"
)

st.write("Seleccionados:", seleccion)
