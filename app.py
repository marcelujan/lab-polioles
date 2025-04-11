import streamlit as st
import pandas as pd
import toml
from datetime import date
from io import BytesIO

# --- CONFIGURACION DE SEGURIDAD ---
config = toml.load("config.toml")
PASSWORD = config["auth"]["password"]

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")
st.title("Laboratorio de Polioles")

# Autenticación simple
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    password = st.text_input("Contraseña de acceso", type="password")
    if st.button("Ingresar"):
        if password == PASSWORD:
            st.session_state.autenticado = True
            st.experimental_rerun()
        else:
            st.error("Contraseña incorrecta")
    st.stop()

# --- INICIO DE LA APP ---
if "muestras" not in st.session_state:
    st.session_state.muestras = []

st.header("Nueva muestra")
with st.form("form_nueva_muestra"):
    nombre_muestra = st.text_input("Nombre de la muestra", "")
    observacion_muestra = st.text_area("Observaciones de la muestra", "")

    st.markdown("### Análisis físico-químicos")
    tipo = st.selectbox("Tipo de análisis", [
        "Índice de yodo [% p/p I2 abs]",
        "Índice OH [mg KHO/g]",
        "Índice de acidez [mg KOH/g]",
        "Índice de epóxido [mol/100g]",
        "Humedad [%]",
        "PM [g/mol]",
        "Funcionalidad [#]",
        "Viscosidad dinámica [cP]",
        "Densidad [g/mL]"
    ])
    valor = st.number_input("Valor", value=0.0, format="%.4f")
    fecha = st.date_input("Fecha", value=date.today())
    observaciones = st.text_input("Observaciones", "")

    agregar_analisis = st.form_submit_button("Agregar muestra")

    if agregar_analisis and nombre_muestra:
        nueva_muestra = {
            "nombre": nombre_muestra,
            "observacion": observacion_muestra,
            "analisis": [{
                "tipo": tipo,
                "valor": valor,
                "fecha": fecha,
                "observaciones": observaciones
            }]
        }
        st.session_state.muestras.append(nueva_muestra)
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
        file_name="muestras_laboratorio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No hay muestras cargadas todavía.")
