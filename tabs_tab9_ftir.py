import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.signal import savgol_filter, find_peaks
from tempfile import TemporaryDirectory
import base64
import os

def render_tab9(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Análisis FTIR")
    st.session_state["current_tab"] = "Análisis FTIR"

    muestras = cargar_muestras(db)
    if not muestras:
        st.warning("No hay muestras disponibles.")
        st.stop()

    # Selección de muestra y espectro
    nombres_muestras = [m["nombre"] for m in muestras]
    muestra_sel = st.selectbox("Seleccionar muestra", nombres_muestras, key="selectbox_ftir_muestra")
    st.session_state["muestra_activa"] = muestra_sel
    muestra = next(m for m in muestras if m["nombre"] == muestra_sel)

    tipos_validos = ["FTIR-Acetato", "FTIR-Cloroformo", "FTIR-ATR"]
    espectros_ftir = [e for e in muestra.get("espectros", []) if e.get("tipo", "") in tipos_validos and not e.get("es_imagen", False)]

    if not espectros_ftir:
        st.warning("La muestra seleccionada no contiene espectros FTIR numéricos.")
        st.stop()

    espectro_sel = st.selectbox("Seleccionar espectro", [e["nombre_archivo"] for e in espectros_ftir])
    espectro = next(e for e in espectros_ftir if e["nombre_archivo"] == espectro_sel)
    extension = os.path.splitext(espectro_sel)[1].lower()

    # Decodificar contenido
    contenido = BytesIO(base64.b64decode(espectro["contenido"]))
    if extension == ".xlsx":
        df = pd.read_excel(contenido)
    else:
        sep_try = [",", ";", "\t", " "]
        for sep in sep_try:
            contenido.seek(0)
            try:
                df = pd.read_csv(contenido, sep=sep, engine="python")
                if df.shape[1] >= 2:
                    break
            except:
                continue
        else:
            st.error("No se pudo leer el archivo.")
            st.stop()

    col_x, col_y = df.columns[:2]
    df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
    df[col_y] = pd.to_numeric(df[col_y], errors="coerce")
    df = df.dropna()

    # Opciones de procesamiento
    st.subheader("Opciones de procesamiento")
    col1, col2 = st.columns(2)
    with col1:
        aplicar_suavizado = st.checkbox("Aplicar suavizado (Savitzky-Golay)", value=False)
    with col2:
        aplicar_normalizacion = st.checkbox("Normalizar intensidad", value=False)

    if aplicar_suavizado:
        window = st.slider("Ventana de suavizado", min_value=3, max_value=51, step=2, value=11)
        poly = st.slider("Orden del polinomio", min_value=1, max_value=5, step=1, value=3)
        df[col_y] = savgol_filter(df[col_y], window_length=window, polyorder=poly)

    if aplicar_normalizacion:
        df[col_y] = (df[col_y] - df[col_y].min()) / (df[col_y].max() - df[col_y].min())

    # Rango de visualización
    st.subheader("Visualización del espectro")
    xmin, xmax = float(df[col_x].min()), float(df[col_x].max())
    ymin, ymax = float(df[col_y].min()), float(df[col_y].max())

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        x_min = st.number_input("X mínimo", value=xmin)
    with col4:
        x_max = st.number_input("X máximo", value=xmax)
    with col5:
        y_min = st.number_input("Y mínimo", value=ymin)
    with col6:
        y_max = st.number_input("Y máximo", value=ymax)

    # Detección de picos
    st.subheader("Detección de picos")
    altura = st.slider("Altura mínima", min_value=0.0, max_value=float(df[col_y].max()), value=float(df[col_y].max()) * 0.2)
    distancia = st.slider("Distancia mínima entre picos", min_value=1, max_value=200, value=20)

    picos, _ = find_peaks(df[col_y], height=altura, distance=distancia)
    df_picos = df.iloc[picos]

    # Graficar
    fig, ax = plt.subplots()
    ax.plot(df[col_x], df[col_y], label="FTIR")
    ax.plot(df_picos[col_x], df_picos[col_y], "ro", label="Picos")
    for i, row in df_picos.iterrows():
        ax.text(row[col_x], row[col_y], f"{row[col_x]:.1f}", fontsize=7, rotation=90, va="bottom")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.legend()
    st.pyplot(fig)

    # Exportar
    buffer_img = BytesIO()
    fig.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
    st.download_button("📷 Descargar gráfico", data=buffer_img.getvalue(), file_name="ftir_procesado.png", mime="image/png")

    excel_out = BytesIO()
    with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="FTIR Procesado")
        df_picos.to_excel(writer, index=False, sheet_name="Picos detectados")
    excel_out.seek(0)
    st.download_button("📥 Descargar resultados", data=excel_out.getvalue(), file_name="ftir_procesado.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    mostrar_sector_flotante(db)
