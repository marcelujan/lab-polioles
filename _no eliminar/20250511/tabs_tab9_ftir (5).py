# Archivo tabs_tab9_ftir actualizado según instrucciones del 11/05/2025
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from scipy.signal import savgol_filter, find_peaks
from datetime import datetime

def render_tab9(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Análisis FTIR")
    st.session_state["current_tab"] = "Análisis FTIR"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    espectros = []
    for m in muestras:
        for e in m.get("espectros", []):
            if e.get("tipo", "").startswith("FTIR") and not e.get("es_imagen", False):
                espectros.append({
                    "muestra": m["nombre"],
                    "tipo": e.get("tipo", ""),
                    "archivo": e.get("nombre_archivo", ""),
                    "contenido": e.get("contenido")
                })

    df_espectros = pd.DataFrame(espectros)
    if df_espectros.empty:
        st.warning("No hay espectros FTIR numéricos disponibles.")
        return

    opciones = df_espectros.apply(lambda row: f"{row['muestra']} – {row['tipo']} – {row['archivo']}", axis=1)
    seleccion = st.multiselect("Seleccionar espectros para comparar", opciones, default=[])
    seleccionados = df_espectros[opciones.isin(seleccion)]
    datos_graficar = []
    for _, row in seleccionados.iterrows():
        try:
            contenido = BytesIO(base64.b64decode(row["contenido"]))
            ext = row["archivo"].split(".")[-1].lower()
            if ext == "xlsx":
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
                    continue
            col_x, col_y = df.columns[:2]
            df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
            df[col_y] = pd.to_numeric(df[col_y], errors="coerce")
            df = df.dropna()
            datos_graficar.append((row["muestra"], row["tipo"], row["archivo"], df))
        except:
            continue

    if not datos_graficar and len(seleccion) > 0:
        st.warning("No se pudieron leer espectros válidos.")
        return

    aplicar_suavizado = st.checkbox("Aplicar suavizado (Savitzky-Golay)", value=False)
    normalizar = st.checkbox("Normalizar intensidad", value=False)
    mostrar_picos = st.checkbox("Mostrar picos detectados automáticamente", value=False)

    st.markdown("###")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x_min = st.number_input("X min", value=1400.0)
    with col2:
        x_max = st.number_input("X max", value=1800.0)
    with col3:
        y_min = st.number_input("Y min", value=0.0)
    with col4:
        y_max = st.number_input("Y max", value=1.0)

    if mostrar_picos:
        colp1, colp2 = st.columns(2)
        with colp1:
            altura_min = st.number_input("Altura mínima", value=0.05)
        with colp2:
            distancia_min = st.number_input("Distancia mínima entre picos", value=5, step=1)

    fig, ax = plt.subplots()
    resumen = pd.DataFrame()

    for muestra, tipo, archivo, df in datos_graficar:
        df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)].copy()
        x = df_filtrado.iloc[:, 0].reset_index(drop=True)
        y = df_filtrado.iloc[:, 1].reset_index(drop=True)

        if aplicar_suavizado and len(y) > 5:
            y = pd.Series(savgol_filter(y, 5, 2))
        if normalizar and y.max() != 0:
            y = y / y.max()

        ax.plot(x, y, label=f"{muestra} – {tipo}")
        resumen[f"{muestra} – {tipo} (X)"] = x
        resumen[f"{muestra} – {tipo} (Y)"] = y

        if mostrar_picos:
            try:
                peaks, _ = find_peaks(y, height=altura_min, distance=distancia_min)
                ax.plot(x[peaks], y[peaks], "x", label=f"Picos {muestra}")
            except:
                pass

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Número de onda [cm⁻¹]")
    ax.set_ylabel("Absorbancia")
    ax.legend()
    st.pyplot(fig)

    # Descargas
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        for muestra, tipo, archivo, df in datos_graficar:
            df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
            df_filtrado.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
    excel_buffer.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("📥 Descargar Excel", data=excel_buffer.getvalue(),
        file_name=f"FTIR_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    st.download_button("📷 Descargar gráfico PNG", data=img_buffer.getvalue(),
        file_name=f"FTIR_{timestamp}.png", mime="image/png")

    mostrar_sector_flotante(db)
