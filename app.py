import streamlit as st
import pandas as pd
import toml
import json
from datetime import date
from io import BytesIO
import os

# --- CONFIGURACION DE SEGURIDAD ---
config = toml.load("config.toml")
PASSWORD = config["auth"]["password"]

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")
st.title("Laboratorio de Polioles")
st.caption("Versión 2025.04.11")

# Autenticación simple
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    password = st.text_input("Contraseña de acceso", type="password")
    if st.button("Ingresar"):
        if password == PASSWORD:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")
    st.stop()

# Archivo local para guardar datos
DATA_FILE = "muestras_data.json"

# Cargar datos desde archivo siempre que exista
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        st.session_state.muestras = json.load(f)
else:
    st.session_state.muestras = []

# Lista fija de tipos de análisis
tipos_analisis = [
    "Índice de yodo [% p/p I2 abs]",
    "Índice OH [mg KHO/g]",
    "Índice de acidez [mg KOH/g]",
    "Índice de epóxido [mol/100g]",
    "Humedad [%]",
    "PM [g/mol]",
    "Funcionalidad [#]",
    "Viscosidad dinámica [cP]",
    "Densidad [g/mL]"
]

# ---- FORMULARIO PARA AÑADIR NUEVA MUESTRA ----
st.header("Nueva muestra")
with st.form("form_nueva_muestra"):
    nombre_muestra = st.text_input("Nombre de la muestra", "")
    observacion_muestra = st.text_area("Observaciones de la muestra", "")

    st.markdown("### Análisis físico-químicos")

    # Generar tabla siempre con todos los análisis
    df_input = pd.DataFrame([{
        "Tipo": tipo,
        "Valor": 0.0,
        "Fecha": date.today(),
        "Observaciones": ""
    } for tipo in tipos_analisis])

    df_input = st.data_editor(
        df_input,
        num_rows="fixed",
        use_container_width=True,
        key="analisis_editor"
    )

    submitted = st.form_submit_button("Guardar muestra y análisis")

    if submitted and nombre_muestra:
        nueva_muestra = {
            "nombre": nombre_muestra,
            "observacion": observacion_muestra,
            "analisis": []
        }
        for _, row in df_input.iterrows():
            if row["Valor"] and float(row["Valor"]) != 0.0:
                nueva_muestra["analisis"].append({
                    "tipo": row["Tipo"],
                    "valor": row["Valor"],
                    "fecha": str(row["Fecha"]),
                    "observaciones": row["Observaciones"]
                })
        
        if nueva_muestra["analisis"] or True:
            st.session_state.muestras.append(nueva_muestra)
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.muestras, f, ensure_ascii=False, indent=2)
            backup_name = f"muestras_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            with open(backup_name, "w", encoding="utf-8") as f:
                json.dump(st.session_state.muestras, f, ensure_ascii=False, indent=2)
            st.success(f"Muestra '{nombre_muestra}' agregada correctamente.")
        

# ---- VISUALIZAR Y EDITAR ----
st.header("Muestras cargadas")
data_expandida = []
for idx, muestra in enumerate(st.session_state.muestras):
    for analisis in muestra["analisis"]:
        data_expandida.append({
            "Índice": idx,
            "Nombre": muestra["nombre"],
            "Observación de muestra": muestra["observacion"],
            "Tipo de análisis": analisis["tipo"],
            "Valor": analisis["valor"],
            "Fecha": analisis["fecha"],
            "Observaciones de análisis": analisis["observaciones"]
        })

if data_expandida:
    df = pd.DataFrame(data_expandida)
    df_editable = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="editor")

    if st.button("Guardar cambios"):
        nuevas_muestras = {}
        for _, row in df_editable.iterrows():
            idx = int(row["Índice"])
            if idx not in nuevas_muestras:
                nuevas_muestras[idx] = {
                    "nombre": row["Nombre"],
                    "observacion": row["Observación de muestra"],
                    "analisis": []
                }
            nuevas_muestras[idx]["analisis"].append({
                "tipo": row["Tipo de análisis"],
                "valor": row["Valor"],
                "fecha": row["Fecha"],
                "observaciones": row["Observaciones de análisis"]
            })
        st.session_state.muestras = [nuevas_muestras[k] for k in sorted(nuevas_muestras.keys())]
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.muestras, f, ensure_ascii=False, indent=2)
        st.success("Cambios guardados correctamente.")

    def convertir_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Muestras")
        return output.getvalue()

    excel_data = convertir_excel(df_editable)
    st.download_button(
        label="Descargar Excel",
        data=excel_data,
        file_name=f"lab-polioles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No hay muestras cargadas todavía.")
