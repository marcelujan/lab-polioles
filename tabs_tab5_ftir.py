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

    # --- Ajuste manual de eje Y ---
    ajustar_y = st.checkbox("Ajuste manual de eje y", value=False)
    ajustes_y = {}

    if ajustar_y:
 #       st.markdown("#### Ajustes verticales por espectro")
        for _, row in seleccionados.iterrows():
            clave = f"{row['muestra']} – {row['tipo']} – {row['archivo']}"
            ajustes_y[clave] = st.number_input(f"Ajuste Y para {clave}", value=0.0, step=0.1)
    else:
        for _, row in seleccionados.iterrows():
            clave = f"{row['muestra']} – {row['tipo']} – {row['archivo']}"
            ajustes_y[clave] = 0.0


    # --- Checkbox y selección para restar espectro ---
    restar_espectro = st.checkbox("Restar espectro", value=False)
    ajuste_y_ref = 0.0
    espectro_para_restar = None

    if restar_espectro:
        espectros_referencia = df_espectros.apply(lambda row: f"{row['muestra']} – {row['tipo']} – {row['archivo']}", axis=1).tolist()
        seleccion_resta = st.selectbox("Seleccionar espectro a restar", espectros_referencia, index=0)
        ajuste_y_ref = st.number_input("Ajuste Y para espectro de referencia", value=0.0, step=0.1)

        espectro_para_restar = df_espectros[df_espectros.apply(lambda row: f"{row['muestra']} – {row['tipo']} – {row['archivo']}", axis=1) == seleccion_resta]
        if not espectro_para_restar.empty:
            row_ref = espectro_para_restar.iloc[0]
            try:
                contenido_ref = BytesIO(base64.b64decode(row_ref["contenido"]))
                ext_ref = row_ref["archivo"].split(".")[-1].lower()
                if ext_ref == "xlsx":
                    df_ref = pd.read_excel(contenido_ref, header=None)
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
                    df_ref = df_ref.iloc[:, :2]  # Asegura solo 2 columnas
                    df_ref.columns = ["x", "y"]
                    df_ref = df_ref.astype(str)
                    df_ref = df_ref.apply(pd.to_numeric, errors="coerce")
                    df_ref = df_ref.dropna().astype(float)
                    x_ref = df_ref.iloc[:, 0].values
                    y_ref = df_ref.iloc[:, 1].values + ajuste_y_ref  # Aplica el ajuste de Y
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
                df = pd.read_excel(contenido, header=None)
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
            df = df.iloc[:, :2]  # Asegura solo 2 columnas
            df.columns = ["x", "y"]
            df = df.astype(str)  # Limpia espacios
            df = df.apply(pd.to_numeric, errors="coerce")  # Convierte a numérico
            df = df.dropna().reset_index(drop=True)  
            try:
                df["x"] = df["x"].astype(float)
                df["y"] = df["y"].astype(float)
            except Exception as e:
                st.error(f"❌ Error al convertir muestra {row['muestra']} – {row['archivo']}: {e}")
                st.stop()

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

    # Guardar configuración de comparación de similitud
    comparar_similitud = st.checkbox("Activar comparación de similitud", value=False)
    x_comp_min, x_comp_max, sombrear, modo_similitud = None, None, False, None

    if comparar_similitud:
        col_sim1, col_sim2, col_sim3, col_sim4 = st.columns([1.2, 1.2, 1.2, 2.4])
        x_comp_min = col_sim1.number_input("X mínimo", value=x_min, step=1.0, key="comp_x_min")
        x_comp_max = col_sim2.number_input("X máximo", value=x_max, step=1.0, key="comp_x_max")
        sombrear = col_sim3.checkbox("Sombrear", value=False)
        modo_similitud = col_sim4.selectbox("Modo de comparación", ["Correlación Pearson", "Comparación de integrales"], label_visibility="collapsed")


    fig, ax = plt.subplots()
    resumen = pd.DataFrame()
    fwhm_rows = []

    for muestra, tipo, archivo, df in datos:
        df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)].copy()
        if df_filtrado.empty:
            continue
        # Ordenar espectro a graficar
        x = df_filtrado.iloc[:, 0].values
        y = df_filtrado.iloc[:, 1].values
        orden = np.argsort(x)
        x = x[orden]
        y = y[orden]

        # Aplicar ajuste de eje Y personalizado
        clave = f"{muestra} – {tipo} – {archivo}"
        ajuste_y = ajustes_y.get(clave, 0.0)
        y = y + ajuste_y

        # Interpolar y restar si corresponde
        if restar_espectro and x_ref is not None and y_ref is not None:
            try:
                # Asegurar que x_ref esté ordenado
                x_ref_ord, y_ref_ord = zip(*sorted(zip(x_ref, y_ref)))
                x_ref_arr = np.array(x_ref_ord).astype(float)
                y_ref_arr = np.array(y_ref_ord).astype(float) + ajuste_y_ref

                # Filtrar x para que esté dentro del dominio de x_ref
                mascara_valida = (x >= x_ref_arr.min()) & (x <= x_ref_arr.max())
                x = x[mascara_valida]
                y = y[mascara_valida]

                # Interpolar y restar
                y_interp_ref = np.interp(x, x_ref_arr, y_ref_arr)
                y = y - y_interp_ref

            except Exception as e:
                st.warning(f"No se pudo restar el espectro de referencia para {row['muestra']}. Error: {e}")


        # Convertir a Series para el resto del procesamiento
        x = pd.Series(x)
        y = pd.Series(y)


        if aplicar_suavizado and len(y) >= 5:
            window = 7 if len(y) % 2 else 7
            y = pd.Series(savgol_filter(y, window_length=window, polyorder=2)).reset_index(drop=True)
        if normalizar and np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))

        label = f"{muestra} – {tipo}"
        ax.plot(x, y, label=label)
        x = x.astype(float)
        resumen[f"{label} (X)"] = x
        y = y.astype(float)
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
 
    # Aplicar sombreado en el gráfico si está activado
    if sombrear and x_comp_min is not None and x_comp_max is not None:
        ax.axvspan(x_comp_min, x_comp_max, color='gray', alpha=0.2, label="Rango comparado")

    ax.axhline(0, color="black", linestyle="--", linewidth=.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Número de onda [cm⁻¹]")
    ax.set_ylabel("Absorbancia")
    ax.legend()
    st.pyplot(fig)
    

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

       
        # Mostrar la tabla con el gradiente visual (usando los valores originales)
        st.dataframe(
            df_similitud.style
                .format(lambda x: f"{x:.2f} %")
                .background_gradient(cmap="RdYlGn")
                .set_properties(**{"text-align": "center"}),
            use_container_width=True
        )

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

