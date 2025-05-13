import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import base64
from io import BytesIO

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
                sep_try = [",", ";", "	", " "]
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

    if not datos_graficar:
        st.warning("No se pudieron leer espectros válidos.")
        return

    # --- Procesamiento opcional ---
    aplicar_suavizado = st.checkbox("Aplicar suavizado (Savitzky-Golay)", value=True)
    normalizar = st.checkbox("Normalizar intensidad", value=True)
    mostrar_picos = st.checkbox("Mostrar picos detectados automáticamente", value=True)

    # --- Rango interactivo X/Y ---
    st.markdown("### Rango de visualización")
    all_x = np.concatenate([df.iloc[:, 0].values for _, _, _, df in datos_graficar])
    all_y = np.concatenate([df.iloc[:, 1].values for _, _, _, df in datos_graficar])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    col1, col2, col3, col4 = st.columns(4)
    with col1: xmin = st.number_input("X min", value=x_min)
    with col2: xmax = st.number_input("X max", value=x_max)
    with col3: ymin = st.number_input("Y min", value=y_min)
    with col4: ymax = st.number_input("Y max", value=y_max)

    # --- Gráfico combinado ---
    fig, ax = plt.subplots()
    resumen = pd.DataFrame()
    for muestra, tipo, archivo, df in datos_graficar:
        df_filtrado = df[(df.iloc[:, 0] >= xmin) & (df.iloc[:, 0] <= xmax)].copy()
        x = df_filtrado.iloc[:, 0].values
        y = df_filtrado.iloc[:, 1].values

        if aplicar_suavizado and len(y) >= 7:
            y = savgol_filter(y, window_length=7, polyorder=2)

        if normalizar and np.max(y) != 0:
            y = y / np.max(y)

        ax.plot(x, y, label=f"{muestra} – {tipo}")
        resumen[f"{muestra} – {tipo} (X)"] = pd.Series(x)
        resumen[f"{muestra} – {tipo} (Y)"] = pd.Series(y)

        if mostrar_picos:
            try:
                picos, _ = find_peaks(y)
                for p in picos:
                    if xmin <= x[p] <= xmax and ymin <= y[p] <= ymax:
                        ax.plot(x[p], y[p], "ro", markersize=3)
                        ax.text(x[p], y[p], f"{x[p]:.1f}", fontsize=6, ha="center", va="bottom")
            except:
                pass

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Número de onda [cm⁻¹]")
    ax.set_ylabel("Absorbancia")
    ax.legend()
    st.pyplot(fig)

    # --- Descargas ---
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        for muestra, tipo, archivo, df in datos_graficar:
            df_filtrado = df[(df.iloc[:, 0] >= xmin) & (df.iloc[:, 0] <= xmax)].copy()
            df_filtrado.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
    excel_buffer.seek(0)
    st.download_button("📥 Descargar Excel", data=excel_buffer.getvalue(), file_name="comparacion_ftir.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    st.download_button("📷 Descargar gráfico PNG", data=img_buffer.getvalue(), file_name="comparacion_ftir.png", mime="image/png")

    mostrar_sector_flotante(db)
