import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks, peak_widths

def render_tab5(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Análisis FTIR")
    st.session_state["current_tab"] = "Análisis FTIR"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # --- Sección 1: Índice OH espectroscópico ---
    st.subheader("Índice OH espectroscópico")
    espectros_info = []
    for m in muestras:
        for e in m.get("espectros", []):
            tipo = e.get("tipo", "")
            if tipo not in ["FTIR-Acetato", "FTIR-Cloroformo"]:
                continue
            contenido = e.get("contenido")
            es_imagen = e.get("es_imagen", False)
            valor_y_extraido = None
            if contenido and not es_imagen:
                try:
                    extension = e.get("nombre_archivo", "").split(".")[-1].lower()
                    binario = BytesIO(base64.b64decode(contenido))
                    if extension == "xlsx":
                        df = pd.read_excel(binario, header=None)
                    else:
                        for sep in [",", ";", "\t", " "]:
                            binario.seek(0)
                            try:
                                df = pd.read_csv(binario, sep=sep, header=None)
                                if df.shape[1] >= 2:
                                    break
                            except:
                                continue
                        else:
                            df = None
                    if df is not None and df.shape[1] >= 2:
                        df = df.dropna()
                        x_val = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                        y_val = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                        df_limpio = pd.DataFrame({"X": x_val, "Y": y_val}).dropna()
                        objetivo_x = 3548 if tipo == "FTIR-Acetato" else 3611
                        idx = (df_limpio["X"] - objetivo_x).abs().idxmin()
                        valor_y_extraido = df_limpio.loc[idx, "Y"]
                except:
                    valor_y_extraido = None

            espectros_info.append({
                "Muestra": m["nombre"],
                "Tipo": tipo,
                "Fecha": e.get("fecha", ""),
                "Señal": valor_y_extraido,
                "Señal manual 3548": e.get("senal_3548"),
                "Señal manual 3611": e.get("senal_3611"),
                "Peso muestra [g]": e.get("peso_muestra")
            })

    df_oh = pd.DataFrame(espectros_info)
    if not df_oh.empty:
        df_oh["Señal solvente"] = df_oh.apply(lambda row: row["Señal manual 3548"] if row["Tipo"] == "FTIR-Acetato" else row["Señal manual 3611"], axis=1)

        def calcular_indice(row):
            peso = row["Peso muestra [g]"]
            y_graf = row["Señal"]
            y_ref = row["Señal solvente"]
            if not all([peso, y_graf, y_ref]) or peso == 0:
                return "—"
            k = 52.5253 if row["Tipo"] == "FTIR-Acetato" else 66.7324
            return round(((y_graf - y_ref) * k) / peso, 2)

        df_oh["Índice OH"] = df_oh.apply(calcular_indice, axis=1)
        st.dataframe(df_oh[["Muestra", "Tipo", "Fecha", "Señal", "Señal solvente", "Peso muestra [g]", "Índice OH"]], use_container_width=True)

    # --- Sección 2: Comparación de espectros ---
    st.subheader("Comparación de espectros FTIR")
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
    opciones = df_espectros.apply(lambda row: f"{row['muestra']} – {row['tipo']} – {row['archivo']}", axis=1)
    seleccion = st.multiselect("Seleccionar espectros para comparar", opciones, default=[])
    seleccionados = df_espectros[opciones.isin(seleccion)]

    if seleccionados.empty:
        return

    aplicar_suavizado = st.checkbox("Aplicar suavizado (Savitzky-Golay)", value=False)
    normalizar = st.checkbox("Normalizar intensidad", value=False)

    # --- Checkbox y selección para restar espectro ---
    restar_espectro = st.checkbox("Restar espectro", value=False)
    espectro_para_restar = None

    if restar_espectro:
        espectros_referencia = df_espectros.apply(lambda row: f"{row['muestra']} – {row['tipo']} – {row['archivo']}", axis=1).tolist()
        seleccion_resta = st.selectbox("Seleccionar espectro a restar", espectros_referencia, index=0)
        espectro_para_restar = df_espectros[df_espectros.apply(lambda row: f"{row['muestra']} – {row['tipo']} – {row['archivo']}", axis=1) == seleccion_resta]
        if not espectro_para_restar.empty:
            row_ref = espectro_para_restar.iloc[0]
            try:
                contenido_ref = BytesIO(base64.b64decode(row_ref["contenido"]))
                ext_ref = row_ref["archivo"].split(".")[-1].lower()
                if ext_ref == "xlsx":
                    df_ref = pd.read_excel(contenido_ref)
                else:
                    for sep in [",", ";", "\t", " "]:
                        contenido_ref.seek(0)
                        try:
                            df_ref = pd.read_csv(contenido_ref, sep=sep)
                            if df_ref.shape[1] >= 2:
                                break
                        except:
                            continue
                    else:
                        df_ref = None
                if df_ref is not None:
                    df_ref.iloc[:, 0] = pd.to_numeric(df_ref.iloc[:, 0], errors="coerce")
                    df_ref.iloc[:, 1] = pd.to_numeric(df_ref.iloc[:, 1], errors="coerce")
                    df_ref = df_ref.dropna()
                    x_ref = df_ref.iloc[:, 0].values
                    y_ref = df_ref.iloc[:, 1].values
            except:
                x_ref, y_ref = None, None
        else:
            x_ref, y_ref = None, None
    else:
        x_ref, y_ref = None, None

    mostrar_picos = st.checkbox("Mostrar picos detectados automáticamente", value=False)

    if mostrar_picos:
        col1, col2 = st.columns(2)
        altura_min = col1.number_input("Altura mínima", value=0.0, step=0.01)
        distancia_min = col2.number_input("Distancia mínima entre picos", value=70, step=1)

    datos = []
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
                        df = pd.read_csv(contenido, sep=sep)
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
                else:
                    continue
            # 👇 Corrección aquí: convertir antes de eliminar NaN
            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            df = df.dropna()
            datos.append((row["muestra"], row["tipo"], row["archivo"], df))
        except:
            continue

    if not datos:
        return

    all_x = np.concatenate([df.iloc[:, 0].values for _, _, _, df in datos])
    
     # --- Rango de visualización (todo en una fila) ---
    col_x1, col_x2, col_y1, col_y2 = st.columns(4)
    x_min = col_x1.number_input("X min", value=float(np.min(all_x)))
    x_max = col_x2.number_input("X max", value=float(np.max(all_x)))
    y_min = col_y1.number_input("Y min", value=float(np.min([df.iloc[:, 1].min() for _, _, _, df in datos])))
    y_max = col_y2.number_input("Y max", value=float(np.max([df.iloc[:, 1].max() for _, _, _, df in datos])))

    fig, ax = plt.subplots()
    resumen = pd.DataFrame()
    fwhm_rows = []

    for muestra, tipo, archivo, df in datos:
        df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)].copy()
        if df_filtrado.empty:
            continue
        x = df_filtrado.iloc[:, 0].reset_index(drop=True)
        y = df_filtrado.iloc[:, 1].reset_index(drop=True)

        if restar_espectro and x_ref is not None and y_ref is not None:
            try:
                y_interp_ref = np.interp(x, x_ref, y_ref)
                y = y - y_interp_ref
            except:
                st.warning(f"No se pudo restar el espectro de referencia para {row['muestra']}")

        if aplicar_suavizado and len(y) >= 5:
            window = 7 if len(y) % 2 else 7
            y = pd.Series(savgol_filter(y, window_length=window, polyorder=2)).reset_index(drop=True)
        if normalizar and np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))

        label = f"{muestra} – {tipo}"
        ax.plot(x, y, label=label)
        resumen[f"{label} (X)"] = x
        resumen[f"{label} (Y)"] = y

        if mostrar_picos:
            try:
                peaks, props = find_peaks(y, height=altura_min, distance=distancia_min)
                widths, width_heights, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)
                for i, peak in enumerate(peaks):
                    x_fwhm_left = np.interp(left_ips[i], np.arange(len(x)), x)
                    x_fwhm_right = np.interp(right_ips[i], np.arange(len(x)), x)
                    ancho = abs(x_fwhm_right - x_fwhm_left)
                    etiqueta = f"{x.iloc[peak]:.0f} ({ancho:.0f}) cm⁻¹ ⇒ {y.iloc[peak]:.4f}"
                    ax.plot(x.iloc[peak], y.iloc[peak], "x", color="black")
                    ax.text(x.iloc[peak], y.iloc[peak], "   " + etiqueta, fontsize=6, ha="left", va="bottom", rotation=90)
                    fwhm_rows.append({
                        "Muestra": muestra,
                        "Tipo": tipo,
                        "Archivo": archivo,
                        "X pico [cm⁻¹]": round(x.iloc[peak], 2),
                        "Y pico": round(y.iloc[peak], 4),
                        "Ancho FWHM [cm⁻¹]": round(ancho, 2)
                    })
            except:
                continue
 
        # --- Comparación de similitud ---
    comparar_similitud = st.checkbox("Activar comparación de similitud", value=False)
    if comparar_similitud:
        col_sim1, col_sim2, col_sim3, col_sim4 = st.columns([1.2, 1.2, 1.2, 2.4])
        x_comp_min = col_sim1.number_input("X mínimo", value=x_min, step=1.0, key="comp_x_min")
        x_comp_max = col_sim2.number_input("X máximo", value=x_max, step=1.0, key="comp_x_max")
        sombrear = col_sim3.checkbox("Sombrear", value=False)
        modo_similitud = col_sim4.selectbox("Modo de comparación", ["Correlación Pearson", "Comparación de integrales"], label_visibility="collapsed")

        if sombrear:
            ax.axvspan(x_comp_min, x_comp_max, color='gray', alpha=0.2, label="Rango comparado")

    # --- Matriz de similitud ---
    if comparar_similitud and x_comp_min is not None and x_comp_max is not None:
        st.subheader("Matriz de similitud entre espectros")
        vectores = {}
        for muestra, tipo, archivo, df in datos:
            df_filt = df[(df.iloc[:, 0] >= x_comp_min) & (df.iloc[:, 0] <= x_comp_max)].copy()
            if df_filt.empty:
                continue
            x = df_filt.iloc[:, 0].reset_index(drop=True)
            y = df_filt.iloc[:, 1].reset_index(drop=True)
            if aplicar_suavizado and len(y) >= 5:
                window = 7 if len(y) % 2 else 7
                y = pd.Series(savgol_filter(y, window_length=window, polyorder=2)).reset_index(drop=True)
            if normalizar and np.max(np.abs(y)) != 0:
                y = y / np.max(np.abs(y))
            vectores[f"{muestra} – {tipo}"] = (x, y)

        nombres = list(vectores.keys())
        matriz = np.zeros((len(nombres), len(nombres)))
        for i in range(len(nombres)):
            for j in range(len(nombres)):
                xi, yi = vectores[nombres[i]]
                xj, yj = vectores[nombres[j]]
                x_comun = np.linspace(max(xi.min(), xj.min()), min(xi.max(), xj.max()), 500)
                yi_interp = np.interp(x_comun, xi, yi)
                yj_interp = np.interp(x_comun, xj, yj)

                if len(yi_interp) == 0 or len(yj_interp) == 0 or np.isnan(yi_interp).any() or np.isnan(yj_interp).any():
                    simil = 0
                else:
                    if modo_similitud == "Correlación Pearson":
                        if np.std(yi_interp) == 0 or np.std(yj_interp) == 0:
                            simil = 0
                        else:
                            simil = np.corrcoef(yi_interp, yj_interp)[0, 1] * 100
                    else:  # Comparación de integrales
                        area_i = np.trapz(yi_interp, x_comun)
                        area_j = np.trapz(yj_interp, x_comun)
                        if area_i == 0 and area_j == 0:
                            simil = 100
                        elif area_i == 0 or area_j == 0:
                            simil = 0
                        else:
                            simil = (1 - abs(area_i - area_j) / max(abs(area_i), abs(area_j))) * 100

                matriz[i, j] = simil

        df_similitud = pd.DataFrame(matriz, index=nombres, columns=nombres)
     #   df_formateado = df_similitud.applymap(lambda v: f"{v:.2f} %")
      #  st.dataframe(df_formateado.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
       
        # Mostrar la tabla con el gradiente visual (usando los valores originales)
        st.dataframe(
            df_similitud.style
                .format(lambda x: f"{x:.2f} %")
                .background_gradient(cmap="RdYlGn")
                .set_properties(**{"text-align": "center"}),
            use_container_width=True
        )

    if restar_espectro:
        ax.axhline(0, color="black", linestyle="--", linewidth=.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Número de onda [cm⁻¹]")
    ax.set_ylabel("Absorbancia")
    ax.legend()
    st.pyplot(fig)
    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre_base = f"FTIR_{now}"

    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                resumen.to_excel(writer, index=False, sheet_name="Resumen")
                for muestra, tipo, archivo, df in datos:
                    df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
                    df_filtrado.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
                if fwhm_rows:
                    df_fwhm = pd.DataFrame(fwhm_rows)
                    df_fwhm = df_fwhm.sort_values(by="Muestra")
                    df_fwhm.to_excel(writer, index=False, sheet_name="Picos_FWHM")
    buffer_excel.seek(0)
    st.download_button("📥 Descargar Excel", data=buffer_excel.getvalue(), file_name=f"{nombre_base}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    buffer_img = BytesIO()
    fig.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
    st.download_button("📷 Descargar PNG", data=buffer_img.getvalue(), file_name=f"{nombre_base}.png", mime="image/png")

    mostrar_sector_flotante(db, key_suffix="tab5")

