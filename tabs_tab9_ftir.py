# tabs_tab9_ftir.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime

def render_tab9(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Análisis FTIR")
    st.session_state["current_tab"] = "Análisis FTIR"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # --- CARGAR ESPECTROS ---
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
    if seleccionados.empty:
        return

    # --- CHECKBOX OPCIONALES ---
    col_s1, col_s2, col_s3 = st.columns(3)
    aplicar_suavizado = col_s1.checkbox("Aplicar suavizado (Savitzky-Golay)", value=False)
    normalizar_intensidad = col_s2.checkbox("Normalizar intensidad", value=False)
    mostrar_picos = col_s3.checkbox("Mostrar picos detectados automáticamente", value=False)

    if mostrar_picos:
        col_p1, col_p2 = st.columns(2)
        altura_minima = col_p1.number_input("Altura mínima", min_value=0.0, value=0.05, step=0.01)
        distancia_minima = col_p2.number_input("Distancia mínima entre picos", min_value=1, value=5, step=1)

    # --- PROCESAR DATOS ---
    datos_graficar = []
    for _, row in seleccionados.iterrows():
        try:
            contenido = BytesIO(base64.b64decode(row["contenido"]))
            ext = row["archivo"].split(".")[-1].lower()
            if ext == "xlsx":
                df = pd.read_excel(contenido)
            else:
                for sep in [",", ";", "\t", " "]:
                    contenido.seek(0)
                    try:
                        df = pd.read_csv(contenido, sep=sep, engine="python")
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
                else:
                    continue
            df = df.dropna()
            col_x, col_y = df.columns[:2]
            df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
            df[col_y] = pd.to_numeric(df[col_y], errors="coerce")
            df = df.dropna()
            datos_graficar.append((row["muestra"], row["tipo"], row["archivo"], df))
        except:
            continue

    if not datos_graficar:
        return

    # --- RANGO MANUAL ---
    todos_x = np.concatenate([df.iloc[:, 0].values for _, _, _, df in datos_graficar])
    todos_y = np.concatenate([df.iloc[:, 1].values for _, _, _, df in datos_graficar])
    col_x1, col_x2, col_y1, col_y2 = st.columns(4)
    x_min = col_x1.number_input("X min", value=float(np.min(todos_x)))
    x_max = col_x2.number_input("X max", value=float(np.max(todos_x)))
    y_min = col_y1.number_input("Y min", value=float(np.min(todos_y)))
    y_max = col_y2.number_input("Y max", value=float(np.max(todos_y)))

    # --- GRAFICAR ---
    fig, ax = plt.subplots()
    resumen = pd.DataFrame()

    for muestra, tipo, archivo, df in datos_graficar:
        df = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)].copy()
        if df.empty:
            continue
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        # Normalizar
        if normalizar_intensidad:
            y = y / np.max(np.abs(y))

        # Suavizar
        if aplicar_suavizado and len(y) > 7:
            from scipy.signal import savgol_filter
            y = savgol_filter(y, window_length=7, polyorder=3)

        ax.plot(x, y, label=f"{muestra} – {tipo}")
        resumen[f"{muestra} – {tipo} (X)"] = x.reset_index(drop=True)
        resumen[f"{muestra} – {tipo} (Y)"] = y.reset_index(drop=True)

        # Marcar picos si se activa
        if mostrar_picos:
            from scipy.signal import find_peaks
            picos, _ = find_peaks(y, height=altura_minima, distance=distancia_minima)
            for pico in picos:
                ax.axvline(x.iloc[pico], color="gray", linestyle="--", linewidth=0.8)
                ax.text(x.iloc[pico], y.iloc[pico], f"{x.iloc[pico]:.0f}", fontsize=6, ha="center", va="bottom")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Número de onda [cm⁻¹]")
    ax.set_ylabel("Absorbancia")
    ax.legend()
    st.pyplot(fig)

    # --- DESCARGA ---
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        for muestra, tipo, archivo, df in datos_graficar:
            df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
            df_filtrado.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
    buffer_excel.seek(0)

    st.download_button(
        "📥 Descargar Excel",
        data=buffer_excel.getvalue(),
        file_name=f"FTIR_{now}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    buffer_img = BytesIO()
    fig.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        "📷 Descargar gráfico PNG",
        data=buffer_img.getvalue(),
        file_name=f"FTIR_{now}.png",
        mime="image/png"
    )

    mostrar_sector_flotante(db)
