import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
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

    # --- Cargar espectros válidos ---
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

    # --- Selector de espectros ---
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

    if not datos_graficar:
        return

    # --- Opciones de procesamiento ---
    col1, col2, col3 = st.columns(3)
    aplicar_suavizado = col1.checkbox("Aplicar suavizado (Savitzky-Golay)", value=False)
    normalizar = col2.checkbox("Normalizar intensidad", value=False)
    mostrar_picos = col3.checkbox("Mostrar picos detectados automáticamente", value=False)

    altura_min = None
    distancia_min = None
    if mostrar_picos:
        col4, col5 = st.columns(2)
        altura_min = col4.number_input("Altura mínima", min_value=0.0, value=0.0, step=0.1, format="%.2f")
        distancia_min = col5.number_input("Distancia mínima entre picos", min_value=1, value=70, step=1)

    # --- Rango de visualización en un solo renglón ---
    all_x = np.concatenate([df.iloc[:, 0].values for _, _, _, df in datos_graficar])
    all_y = np.concatenate([df.iloc[:, 1].values for _, _, _, df in datos_graficar])
    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X min", value=float(np.min(all_x)))
    x_max = colx2.number_input("X max", value=float(np.max(all_x)))
    y_min = coly1.number_input("Y min", value=float(np.min(all_y)))
    y_max = coly2.number_input("Y max", value=float(np.max(all_y)))

    # --- Gráfico combinado ---
    fig, ax = plt.subplots()
    resumen = pd.DataFrame()

    for muestra, tipo, archivo, df in datos_graficar:
        df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)].copy()
        if df_filtrado.empty:
            continue
        x = df_filtrado.iloc[:, 0].reset_index(drop=True)
        y = df_filtrado.iloc[:, 1].reset_index(drop=True)

        if aplicar_suavizado and len(y) >= 5:
            window = 5 if len(y) < 7 else 7
            if window % 2 == 0: window += 1
            y = pd.Series(savgol_filter(y, window_length=window, polyorder=2)).reset_index(drop=True)

        if normalizar:
            y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y

        label = f"{muestra} – {tipo}"
        ax.plot(x, y, label=label)
        resumen[f"{label} (X)"] = x
        resumen[f"{label} (Y)"] = y

        if mostrar_picos:
            try:
                peaks, _ = find_peaks(y, height=altura_min, distance=distancia_min)
                ax.plot(x.iloc[peaks], y.iloc[peaks], "x", label=f"{label} picos")
            except:
                continue

    ax.set_xlabel("Número de onda [cm⁻¹]")
    ax.set_ylabel("Absorbancia")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()
    st.pyplot(fig)

    # --- Descargas ---
    nombre_base = f"FTIR_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        for muestra, tipo, archivo, df in datos_graficar:
            df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
            df_filtrado.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
    buffer_excel.seek(0)

    st.download_button("📥 Descargar Excel", data=buffer_excel.getvalue(),
                       file_name=f"{nombre_base}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    buffer_img = BytesIO()
    fig.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
    st.download_button("📷 Descargar gráfico PNG", data=buffer_img.getvalue(),
                       file_name=f"{nombre_base}.png", mime="image/png")

    mostrar_sector_flotante(db)
