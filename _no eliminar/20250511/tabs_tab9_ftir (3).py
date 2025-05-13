import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import base64
from io import BytesIO
from datetime import datetime

def render_tab9(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Análisis FTIR")
    st.session_state["current_tab"] = "Análisis FTIR"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # --- COMPARACIÓN DE ESPECTROS ---
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

    if not seleccion:
        return

    # --- Opciones de procesamiento ---
    aplicar_suavizado = st.checkbox("Aplicar suavizado (Savitzky-Golay)", value=False)
    normalizar = st.checkbox("Normalizar intensidad", value=False)
    mostrar_picos = st.checkbox("Mostrar picos detectados automáticamente", value=False)

    if mostrar_picos:
        col1, col2 = st.columns(2)
        with col1:
            altura_min = st.number_input("Altura mínima", value=0.05, step=0.01)
        with col2:
            distancia_min = st.number_input("Distancia mínima entre picos", value=10, step=1)

    # --- Rango de visualización manual ---
    st.markdown("### Rango manual editable:")
    all_x = np.concatenate([df.iloc[:, 0].values for _, _, _, df in datos_graficar])
    all_y = np.concatenate([df.iloc[:, 1].values for _, _, _, df in datos_graficar])
    x_min = st.number_input("X min", value=float(np.min(all_x)))
    x_max = st.number_input("X max", value=float(np.max(all_x)))
    y_min = st.number_input("Y min", value=float(np.min(all_y)))
    y_max = st.number_input("Y max", value=float(np.max(all_y)))

    # --- GRAFICAR ---
    fig, ax = plt.subplots()
    resumen = pd.DataFrame()
    for muestra, tipo, archivo, df in datos_graficar:
        df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
        x = df_filtrado.iloc[:, 0].reset_index(drop=True)
        y = df_filtrado.iloc[:, 1].reset_index(drop=True)

        if aplicar_suavizado and len(y) >= 7:
            y = pd.Series(savgol_filter(y, window_length=7, polyorder=2))
        if normalizar and y.max() > 0:
            y = y / y.max()

        ax.plot(x, y, label=f"{muestra} – {tipo}")
        resumen[f"{muestra} – {tipo} (X)"] = x
        resumen[f"{muestra} – {tipo} (Y)"] = y

        if mostrar_picos:
            try:
                indices, _ = find_peaks(y, height=altura_min, distance=distancia_min)
                for idx in indices:
                    ax.plot(x[idx], y[idx], "rx")
                    ax.text(x[idx], y[idx], f"{x[idx]:.0f}", fontsize=6, ha="center", va="bottom")
            except:
                continue

    ax.set_xlabel("Número de onda [cm⁻¹]")
    ax.set_ylabel("Absorbancia")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()
    st.pyplot(fig)

    # --- DESCARGAS ---
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        for muestra, tipo, archivo, df in datos_graficar:
            df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
            df_filtrado.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
    excel_buffer.seek(0)

    st.download_button(
        "📥 Descargar Excel",
        data=excel_buffer.getvalue(),
        file_name=f"FTIR_{now}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        "📷 Descargar gráfico PNG",
        data=img_buffer.getvalue(),
        file_name=f"FTIR_{now}.png",
        mime="image/png"
    )

    mostrar_sector_flotante(db)
