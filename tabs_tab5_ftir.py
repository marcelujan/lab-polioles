import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from numpy import interp

def obtener_ids_espectros(nombre):
    return [doc.id for doc in firestore.Client().collection("muestras").document(nombre).collection("espectros").list_documents()]

def obtener_espectros_para_muestra(db, nombre):
    clave = f"_espectros_cache_{nombre}"
    if clave not in st.session_state:
        ref = db.collection("muestras").document(nombre).collection("espectros")
        docs = ref.stream()
        st.session_state[clave] = [doc.to_dict() for doc in docs]
    return st.session_state[clave]


def render_tabla_calculos_ftir(db, datos_plotly):
    mostrar = st.checkbox("Mostrar Tabla de cálculos FTIR", value=False, key="mostrar_tabla_calculos_ftir")
    if not mostrar:
        return

    st.subheader("Tabla de cálculos FTIR")
    sombrear = st.checkbox("Sombrear Tabla de cálculos", value=False, key="sombrear_tabla_calculos_ftir")

    filas_totales = []
    claves_renderizadas = []

    for muestra, tipo, archivo, df in datos_plotly:
        clave = f"{muestra}/{archivo}"
        claves_renderizadas.append((muestra, archivo))
        doc_ref = db.collection("tablas_ftir_calculos").document(muestra).collection("archivos").document(archivo)
        doc = doc_ref.get()
        if doc.exists:
            filas = doc.to_dict().get("filas", [])
        else:
            filas = []

        for fila in filas:
            fila["Muestra"] = muestra
            fila["Tipo"] = tipo
            fila["Archivo"] = archivo
        filas_totales.extend(filas)

    df_tabla = pd.DataFrame(filas_totales)
    columnas = ["Muestra", "Tipo", "Archivo", "X min", "X max", "Área", "Grupo funcional", "Observaciones"]

    if df_tabla.empty:
        df_tabla = pd.DataFrame(columns=columnas)

    editada = st.data_editor(
        df_tabla,
        column_order=columnas,
        use_container_width=True,
        key="tabla_calculos_ftir",
        num_rows="dynamic"
    )

    # Botón para calcular área
    if st.button("Recalcular áreas FTIR", key="recalc_area_ftir"):
        nuevas_filas = []
        for _, row in editada.iterrows():
            try:
                x0 = float(row["X min"])
                x1 = float(row["X max"])
                muestra = row["Muestra"]
                archivo = row["Archivo"]
                df = next((df for m, t, a, df in datos_plotly if m == muestra and a == archivo), None)
                if df is not None:
                    df_filt = df[(df["x"] >= x0) & (df["x"] <= x1)]
                    area = np.trapz(df_filt["y"], df_filt["x"])
                    row["Área"] = round(area, 6)
            except:
                continue
            nuevas_filas.append(row)
        editada = pd.DataFrame(nuevas_filas)

    # Guardar por muestra y archivo
    for (muestra, archivo) in claves_renderizadas:
        df_filtrado = editada[(editada["Muestra"] == muestra) & (editada["Archivo"] == archivo)]
        columnas_guardar = ["X min", "X max", "Área", "Grupo funcional", "Observaciones"]
        filas_guardar = df_filtrado[columnas_guardar].to_dict(orient="records")
        doc_ref = db.collection("tablas_ftir_calculos").document(muestra).collection("archivos").document(archivo)
        doc_ref.set({"filas": filas_guardar})

    # Sombrear en gráfico si aplica
    if sombrear:
        for _, row in editada.iterrows():
            try:
                x0 = float(row["X min"])
                x1 = float(row["X max"])
                st.session_state.setdefault("fig_extra_shapes", []).append({
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": x0,
                    "x1": x1,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": "rgba(0, 100, 250, 0.1)",
                    "line": {"width": 0}
                })
            except:
                continue


def render_tabla_bibliografia_ftir(db):
    st.subheader("Tabla bibliográfica FTIR")
    delinear = st.checkbox("Delinear Tabla bibliográfica", value=False, key="delinear_biblio_ftir")

    ruta = "tablas_ftir_bibliografia/default"
    doc_ref = db.document(ruta)
    doc = doc_ref.get()

    if doc.exists:
        filas = doc.to_dict().get("filas", [])
        df_biblio = pd.DataFrame(filas)
    else:
        df_biblio = pd.DataFrame([{
            "Grupo funcional": "",
            "X pico [cm⁻¹]": 0.0,
            "X min": 0.0,
            "X max": 0.0,
            "Comentarios": ""
        }])

    editada = st.data_editor(
        df_biblio,
        num_rows="dynamic",
        use_container_width=True,
        key="tabla_biblio_ftir"
    )

    # Guardar en firebase
    if st.button("Guardar bibliografía FTIR", key="guardar_biblio_ftir"):
        doc_ref.set({"filas": editada.to_dict(orient="records")})
        st.success("Bibliografía guardada correctamente.")

    if delinear:
        st.session_state["fig_extra_shapes"] = []
        for _, row in editada.iterrows():
            try:
                x0 = float(row["X min"])
                x1 = float(row["X max"])
                st.session_state["fig_extra_shapes"].append({
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": x0,
                    "x1": x1,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": "rgba(255, 0, 0, 0.1)",
                    "line": {"width": 0}
                })
            except:
                continue

    return editada if delinear else pd.DataFrame([])

def render_deconvolucion_ftir(preprocesados, x_min, x_max, y_min, y_max):
    st.subheader("🔍 Deconvolución FTIR")
    if not preprocesados:
        st.info("No hay espectros preprocesados para deconvolución.")
        return

    if not st.checkbox("Activar deconvolución", key="activar_deconv"):
        return

    col1, col2, col3, col4 = st.columns(4)
    checkboxes = {}
    claves_disponibles = list(preprocesados.keys())

    for i, clave in enumerate(claves_disponibles):
        with [col1, col2, col3, col4][i % 4]:
            checkboxes[clave] = st.checkbox(clave, value=False, key=f"deconv_{clave}")

    # Inicializar acumulador
    if "resultados_totales" not in st.session_state:
        st.session_state["resultados_totales"] = {}

    for clave in claves_disponibles:
        if not checkboxes.get(clave):
            continue

        try:
            df = preprocesados.get(clave)
            if df is None or df.empty:
                continue

            x = df["x"].values
            y = df["y"].values

            def multi_gaussian(x, *params):
                y_fit = np.zeros_like(x)
                for i in range(0, len(params), 3):
                    amp, cen, wid = params[i:i+3]
                    y_fit += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
                return y_fit

            n_gauss = st.slider(f"Nº de gaussianas para {clave}", 1, 10, 3, key=f"gauss_{clave}")
            p0 = []
            for i in range(n_gauss):
                p0 += [y.max() / n_gauss, x.min() + i * (np.ptp(x) / n_gauss), 10]

            popt, _ = curve_fit(multi_gaussian, x, y, p0=p0, maxfev=10000)
            y_fit = multi_gaussian(x, *popt)

            fig, ax = plt.subplots()
            ax.plot(x, y, label="Original", color="black")
            ax.plot(x, y_fit, "--", label="Ajuste", color="orange")

            resultados = []
            colores = plt.cm.get_cmap("tab10")
            for i in range(n_gauss):
                amp, cen, wid = popt[3*i:3*i+3]
                gauss = amp * np.exp(-(x - cen)**2 / (2 * wid**2))
                area = amp * wid * np.sqrt(2*np.pi)
                ax.plot(x, gauss, ":", label=f"Pico {i+1}", color=colores(i))
                resultados.append({
                    "Pico": i+1,
                    "Centro (cm⁻¹)": round(cen, 2),
                    "Amplitud": round(amp, 2),
                    "Anchura σ": round(wid, 2),
                    "Área": round(area, 2)
                })

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Número de onda [cm⁻¹]")
            ax.set_ylabel("Absorbancia")
            ax.legend()
            st.pyplot(fig)

            rmse = np.sqrt(np.mean((y - y_fit) ** 2))
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            st.markdown(f"""**{clave}**  
**RMSE:** {rmse:.4f} &nbsp;&nbsp;&nbsp;&nbsp; **R²:** {r2:.4f}""")

            df_result = pd.DataFrame(resultados)
            st.session_state["resultados_totales"][clave] = df_result
            st.dataframe(df_result, use_container_width=True)

            buf_excel = BytesIO()
            with pd.ExcelWriter(buf_excel, engine="xlsxwriter") as writer:
                df_result.to_excel(writer, index=False, sheet_name="Deconvolucion")
            buf_excel.seek(0)
            st.download_button("📅 Descargar parámetros", data=buf_excel.getvalue(),
                               file_name=f"deconv_{clave.replace(' – ', '_')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"dl_{clave}")

        except Exception as e:
            if "Optimal parameters not found" in str(e):
                st.warning(f"""
⚠️ No se pudo ajustar **{clave}** porque el optimizador no encontró parámetros adecuados.  
👉 Sugerencia: probá ajustar el rango X o el número de gaussianas.
""")
            else:
                st.warning(f"❌ Error al ajustar {clave}: {e}")

    # --- Exportar todas las deconvoluciones seleccionadas ---
    if st.button("📦 Exportar TODO en Excel"):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            for clave, df in st.session_state["resultados_totales"].items():
                nombre_hoja = clave.replace(" – ", "_")[:31]
                df.to_excel(writer, index=False, sheet_name=nombre_hoja)
        buffer.seek(0)
        st.download_button("📁 Descargar TODO", data=buffer.getvalue(),
                           file_name="FTIR_Deconvoluciones.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_total_deconv")



def render_grafico_combinado_ftir(fig, datos_plotly, aplicar_suavizado, normalizar,
                                   offset_vertical, ajustes_y, restar_espectro,
                                   x_ref, y_ref, x_min, x_max, y_min, y_max,
                                   mostrar_picos=False, altura_min=0.01, distancia_min=5):
    for i, (muestra, tipo, archivo, df) in enumerate(datos_plotly):
        clave = f"{muestra} – {tipo} – {archivo}"
        df_filtrado = df[(df["x"] >= x_min) & (df["x"] <= x_max)].copy()
        if df_filtrado.empty:
            continue
        x = df_filtrado["x"].values
        y = df_filtrado["y"].values

        if aplicar_suavizado and len(y) >= 7:
            y = savgol_filter(y, window_length=7, polyorder=2)
        if normalizar and np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))
        if offset_vertical:
            y = y + i * 0.2
        y = y + ajustes_y.get(clave, 0.0)

        if restar_espectro and x_ref is not None and y_ref is not None:
            mascara_valida = (x >= np.min(x_ref)) & (x <= np.max(x_ref))
            x = x[mascara_valida]
            y = y[mascara_valida]
            y_interp = np.interp(x, x_ref, y_ref)
            y = y - y_interp

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            name=clave,
            hovertemplate="x=%{x}<br>y=%{y}<extra></extra>"
        ))

    if mostrar_picos:
        from scipy.signal import find_peaks
        try:
            peaks, _ = find_peaks(y, height=altura_min, distance=distancia_min)
            for p in peaks:
                fig.add_trace(go.Scatter(
                    x=[x[p]],
                    y=[y[p]],
                    mode="markers+text",
                    marker=dict(color="black", size=6),
                    text=[f"{x[p]:.2f}"],
                    textposition="top center",
                    showlegend=False
                ))
        except Exception as e:
            st.warning(f"⚠️ Error al detectar picos en {clave}: {e}")



    if "fig_extra_shapes" in st.session_state:
        fig.update_layout(shapes=st.session_state["fig_extra_shapes"])

    fig.update_layout(
        xaxis_title="Número de onda [cm⁻¹]",
        yaxis_title="Absorbancia",
        margin=dict(l=10, r=10, t=30, b=10),
        height=500,
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max] if not normalizar else None),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5
        )
    )


    st.plotly_chart(fig, use_container_width=True)

def generar_preprocesados_ftir(datos_plotly, aplicar_suavizado, normalizar,
                                offset_vertical, ajustes_y, restar_espectro,
                                x_ref, y_ref, x_min, x_max):
    preprocesados = {}

    for i, (muestra, tipo, archivo, df) in enumerate(datos_plotly):
        clave = f"{muestra} – {tipo} – {archivo}"
        df_filtrado = df[(df["x"] >= x_min) & (df["x"] <= x_max)].copy()
        if df_filtrado.empty:
            continue
        x = df_filtrado["x"].values
        y = df_filtrado["y"].values.astype(float)

        if aplicar_suavizado and len(y) >= 7:
            y = savgol_filter(y, window_length=7, polyorder=2)
        if normalizar and np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))
        if offset_vertical:
            y = y + i * 0.2
        y = y + ajustes_y.get(clave, 0.0)

        if restar_espectro and x_ref is not None and y_ref is not None:
            mascara_valida = (x >= np.min(x_ref)) & (x <= np.max(x_ref))
            x = x[mascara_valida]
            y = y[mascara_valida]
            y_interp = interp(x, x_ref, y_ref)
            y = y - y_interp

        df_pre = pd.DataFrame({"x": x, "y": y})
        preprocesados[clave] = df_pre

    return preprocesados

def exportar_resultados_ftir(preprocesados, resumen=None, fwhm_rows=None, x_min=None, x_max=None):
    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
        if resumen is not None:
            resumen.to_excel(writer, index=False, sheet_name="Resumen")
        for clave, df in preprocesados.items():
            if x_min is not None and x_max is not None:
                df_filtrado = df[(df["x"] >= x_min) & (df["x"] <= x_max)]
            else:
                df_filtrado = df
            nombre_hoja = clave.replace(" – ", "_")[:31]
            df_filtrado.to_excel(writer, index=False, sheet_name=nombre_hoja)
        if fwhm_rows:
            df_fwhm = pd.DataFrame(fwhm_rows)
            df_fwhm.to_excel(writer, index=False, sheet_name="Picos_FWHM")
    buffer_excel.seek(0)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre_base = f"FTIR_{now}.xlsx"
    st.download_button("📥 Descargar Excel", data=buffer_excel.getvalue(),
                       file_name=nombre_base,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def exportar_figura_plotly_png(fig, nombre_base="FTIR"):
    buffer_img = BytesIO()
    fig.write_image(buffer_img, format="png", width=1200, height=600, scale=3)
    buffer_img.seek(0)
    nombre_archivo = f"{nombre_base}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    st.download_button("📷 Descargar PNG", data=buffer_img.getvalue(),
                       file_name=nombre_archivo, mime="image/png")

def render_controles_preprocesamiento(datos_plotly):
#    st.markdown("### Preprocesamiento y visualización")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    aplicar_suavizado = col1.checkbox("Suavizado SG", value=False, key="suavizado_ftir")
    normalizar = col2.checkbox("Normalizar", value=False, key="normalizar_ftir")
    mostrar_picos = col3.checkbox("Detectar picos", value=False, key="picos_ftir")  # No usado acá, solo flag
    restar_espectro = col4.checkbox("Restar espectro", value=False, key="restar_ftir")
    ajuste_y_manual = col5.checkbox("Ajuste manual Y", value=False, key="ajuste_y_ftir")
    offset_vertical = col6.checkbox("Superposición vertical", value=False, key="offset_y_ftir")

    # Rango XY automático
    todos_x = np.concatenate([df["x"].values for _, _, _, df in datos_plotly])
    todos_y = np.concatenate([df["y"].values for _, _, _, df in datos_plotly])
    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X min", value=float(np.min(todos_x)))
    x_max = colx2.number_input("X max", value=float(np.max(todos_x)))
    y_min = coly1.number_input("Y min", value=float(np.min(todos_y)))
    y_max = coly2.number_input("Y max", value=float(np.max(todos_y)))

    # Ajustes Y individuales
    ajustes_y = {}
    if ajuste_y_manual:
        st.markdown("#### Ajuste Y individual por espectro")
        for i, (muestra, tipo, archivo, df) in enumerate(datos_plotly):
            clave = f"{muestra} – {tipo} – {archivo}"
            ajustes_y[clave] = st.number_input(f"{clave}", step=0.1, value=0.0, key=f"ajuste_y_{clave}")
    else:
        for i, (muestra, tipo, archivo, df) in enumerate(datos_plotly):
            clave = f"{muestra} – {tipo} – {archivo}"
            ajustes_y[clave] = 0.0

    # Resta de espectro de referencia
    x_ref, y_ref = None, None
    if restar_espectro:
        claves_validas = [f"{m} – {t} – {a}" for m, t, a, _ in datos_plotly]
        espectro_ref = st.selectbox("Seleccionar espectro a restar", claves_validas, key="ref_ftir")
        ajuste_y_ref = st.number_input("Ajuste Y referencia", value=0.0, step=0.1, key="ajuste_ref_ftir")

        for m, t, a, df in datos_plotly:
            if espectro_ref == f"{m} – {t} – {a}":
                x_ref = df["x"].values
                y_ref = df["y"].values + ajuste_y_ref
                break

    return {
        "suavizado": aplicar_suavizado,
        "normalizar": normalizar,
        "mostrar_picos": mostrar_picos,
        "restar": restar_espectro,
        "ajuste_y_manual": ajuste_y_manual,
        "offset_vertical": offset_vertical,
        "ajustes_y": ajustes_y,
        "x_ref": x_ref,
        "y_ref": y_ref,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max
    }

def seleccionar_espectros_validos(db, muestras):
    tipos_validos = ["FTIR-Acetato", "FTIR-Cloroformo", "FTIR-ATR"]
    espectros_dict = {}

    for m in muestras:
        nombre = m["nombre"]
        for e in obtener_espectros_para_muestra(db, nombre):
            tipo = e.get("tipo", "")
            if tipo in tipos_validos and not e.get("es_imagen", False):
                archivo = e.get("nombre_archivo", "Sin nombre")
                espectros_dict[(nombre, archivo)] = {
                    "contenido": e.get("contenido"),
                    "tipo": tipo,
                    "archivo": archivo,
                    "muestra": nombre
                }

    if not espectros_dict:
        st.info("No hay espectros FTIR válidos para mostrar.")
        return []

    # --- Selector de muestra y espectros ---
    muestras_disponibles = sorted(set(k[0] for k in espectros_dict.keys()))
    muestras_sel = st.multiselect("Seleccionar muestras", opciones_muestras)
    if not muestras_sel:
        return
    archivos_disp = [k[1] for k in espectros_dict.keys() if k[0] == muestra_sel]
    archivos_sel = st.multiselect("Seleccionar espectros de esa muestra", archivos_disp, key="archivos_ftir")

    # --- Leer archivos seleccionados ---
    datos_plotly = []
    for archivo in archivos_sel:
        clave = (muestra_sel, archivo)
        e = espectros_dict[clave]
        contenido = BytesIO(base64.b64decode(e["contenido"]))
        ext = archivo.split(".")[-1].lower()

        try:
            if ext == "xlsx":
                df = pd.read_excel(contenido, header=None)
            else:
                for sep in [",", ";", "\t", " "]:
                    contenido.seek(0)
                    try:
                        df = pd.read_csv(contenido, sep=sep, header=None)
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
                else:
                    df = None

            if df is not None and df.shape[1] >= 2:
                df = df.iloc[:, :2]
                df.columns = ["x", "y"]
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                datos_plotly.append((e["muestra"], e["tipo"], e["archivo"], df))
        except Exception as ex:
            st.warning(f"Error al cargar {archivo}: {ex}")

    if not datos_plotly:
        st.info("Seleccioná espectros válidos para graficar.")

    return datos_plotly

def calcular_indice_oh_auto(db, muestras):
    espectros_info = []

    for m in muestras:
        for e in obtener_espectros_para_muestra(db, m["nombre"]):
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

    if df_oh.empty:
        return df_oh

    df_oh["Señal solvente"] = df_oh.apply(
        lambda row: row["Señal manual 3548"] if row["Tipo"] == "FTIR-Acetato" else row["Señal manual 3611"],
        axis=1
    )

    def calcular_indice(row):
        peso = row["Peso muestra [g]"]
        y_graf = row["Señal"]
        y_ref = row["Señal solvente"]
        if not all([peso, y_graf, y_ref]) or peso == 0:
            return np.nan
        k = 52.5253 if row["Tipo"] == "FTIR-Acetato" else 66.7324
        return round(((y_graf - y_ref) * k) / peso, 2)

    df_oh["Índice OH"] = df_oh.apply(calcular_indice, axis=1)
    df_oh["Índice OH"] = pd.to_numeric(df_oh["Índice OH"], errors="coerce")

    return df_oh[["Muestra", "Tipo", "Fecha", "Señal", "Señal solvente", "Peso muestra [g]", "Índice OH"]]

def calculadora_indice_oh_manual():
    st.subheader("Calculadora manual de Índice OH")

    datos_oh = pd.DataFrame([
        {"Tipo": "FTIR-Acetato [3548 cm⁻¹]", "Señal": 0.0000, "Señal solvente": 0.0000, "Peso muestra [g]": 0.0000},
        {"Tipo": "FTIR-Cloroformo A [3611 cm⁻¹]", "Señal": 0.0000, "Señal solvente": 0.0000, "Peso muestra [g]": 0.0000},
        {"Tipo": "FTIR-Cloroformo D [3611 cm⁻¹]", "Señal": 0.0000, "Señal solvente": 0.0000, "Peso muestra [g]": 0.0000},
        {"Tipo": "FTIR-Cloroformo E [3611 cm⁻¹]", "Señal": 0.0000, "Señal solvente": 0.0000, "Peso muestra [g]": 0.0000}
    ])

    col1, col2 = st.columns([4, 1])

    with col1:
        edited_input = st.data_editor(
            datos_oh,
            column_order=["Tipo", "Señal", "Señal solvente", "Peso muestra [g]"],
            column_config={"Tipo": st.column_config.TextColumn(disabled=True)},
            use_container_width=True,
            hide_index=True,
            key="editor_oh_calculadora",
            num_rows="fixed"
        )

    resultados = []
    for _, row in edited_input.iterrows():
        try:
            y = float(row["Señal"])
            y_ref = float(row["Señal solvente"])
            peso = float(row["Peso muestra [g]"])
            if peso > 0:
                k = 52.5253 if "Acetato" in row["Tipo"] else 66.7324
                indice = round(((y - y_ref) * k) / peso, 2)
            else:
                indice = "—"
        except:
            indice = "—"
        resultados.append({"Índice OH": indice})

    with col2:
        st.dataframe(pd.DataFrame(resultados), use_container_width=True, hide_index=True)

def obtener_ids_espectros(nombre):
    return [doc.id for doc in firestore.Client().collection("muestras").document(nombre).collection("espectros").list_documents()]

def obtener_espectros_para_muestra(db, nombre):
    clave = f"_espectros_cache_{nombre}"
    if clave not in st.session_state:
        ref = db.collection("muestras").document(nombre).collection("espectros")
        docs = ref.stream()
        st.session_state[clave] = [doc.to_dict() for doc in docs]
    return st.session_state[clave]

# Nueva versión que retorna todos los valores clave
def render_comparacion_espectros_ftir(db, muestras):
#    st.subheader("Comparación de espectros FTIR")
    tipos_validos = ["FTIR-Acetato", "FTIR-Cloroformo", "FTIR-ATR"]
    espectros_dict = {}

    for m in muestras:
        nombre = m["nombre"]
        for e in obtener_espectros_para_muestra(db, nombre):
            tipo = e.get("tipo", "")
            if tipo in tipos_validos and not e.get("es_imagen", False):
                archivo = e.get("nombre_archivo", "Sin nombre")
                clave = (nombre, archivo)
                espectros_dict[clave] = {
                    "contenido": e.get("contenido"),
                    "tipo": tipo,
                    "archivo": archivo,
                    "muestra": nombre
                }

    archivos_disponibles = [
        f"{muestra} – {archivo}" for (muestra, archivo) in espectros_dict.keys()
    ]
    archivos_sel = st.multiselect("Seleccionar espectros", archivos_disponibles, key="archivos_ftir")

    datos_plotly = []
    for item in archivos_sel:
        muestra, archivo = item.split(" – ", 1)
        clave = (muestra, archivo)
        e = espectros_dict[clave]
        contenido = BytesIO(base64.b64decode(e["contenido"]))
        ext = archivo.split(".")[-1].lower()
        try:
            if ext == "xlsx":
                df = pd.read_excel(contenido, header=None)
            else:
                for sep in [",", ";", "\t", " "]:
                    contenido.seek(0)
                    try:
                        df = pd.read_csv(contenido, sep=sep, header=None)
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
                else:
                    df = None
            if df is not None and df.shape[1] >= 2:
                df = df.iloc[:, :2]
                df.columns = ["x", "y"]
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                datos_plotly.append((e["muestra"], e["tipo"], e["archivo"], df))
        except Exception as ex:
            st.warning(f"Error al cargar {archivo}: {ex}")

    if not datos_plotly:
        st.info("Seleccioná espectros válidos para graficar.")
        return [], None, {}, None, None, None, None, None, None


#    st.markdown("### Preprocesamiento y visualización")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    aplicar_suavizado = col1.checkbox("Suavizado SG", value=False, key="suavizado_ftir")
    normalizar = col2.checkbox("Normalizar", value=False, key="normalizar_ftir")
    mostrar_picos = col3.checkbox("Detectar picos", value=False, key="picos_ftir")
    restar_espectro = col4.checkbox("Restar espectro", value=False, key="restar_ftir")
    ajuste_y_manual = col5.checkbox("Ajuste manual Y", value=False, key="ajuste_y_ftir")
    offset_vertical = col6.checkbox("Superposición vertical", value=False, key="offset_y_ftir")

    todos_x = np.concatenate([df["x"].values for _, _, _, df in datos_plotly])
    todos_y = np.concatenate([df["y"].values for _, _, _, df in datos_plotly])
    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X min", value=float(np.min(todos_x)))
    x_max = colx2.number_input("X max", value=float(np.max(todos_x)))
    y_min = coly1.number_input("Y min", value=float(np.min(todos_y)))
    y_max = coly2.number_input("Y max", value=float(np.max(todos_y)))

    altura_min = 0.01
    distancia_min = 5
    if mostrar_picos:
        colp1, colp2 = st.columns(2)
        altura_min = colp1.number_input("Altura mínima", value=0.05, step=0.01, key="altura_min_ftir")
        distancia_min = colp2.number_input("Distancia mínima entre picos", value=5, step=1, key="distancia_min_ftir")

    ajustes_y = {}
    if ajuste_y_manual:
        st.markdown("#### Ajuste Y individual por espectro")
        for i, (muestra, tipo, archivo, df) in enumerate(datos_plotly):
            clave = f"{muestra} – {tipo} – {archivo}"
            ajustes_y[clave] = st.number_input(f"{clave}", step=0.1, value=0.0, key=f"ajuste_y_{clave}")
    else:
        for i, (muestra, tipo, archivo, df) in enumerate(datos_plotly):
            clave = f"{muestra} – {tipo} – {archivo}"
            ajustes_y[clave] = 0.0

    x_ref, y_ref = None, None
    if restar_espectro:
        claves_validas = [f"{m} – {t} – {a}" for m, t, a, _ in datos_plotly]
        espectro_ref = st.selectbox("Seleccionar espectro a restar", claves_validas, key="ref_ftir")
        ajuste_y_ref = st.number_input("Ajuste Y referencia", value=0.0, step=0.1, key="ajuste_ref_ftir")

        for m, t, a, df in datos_plotly:
            if espectro_ref == f"{m} – {t} – {a}":
                df_ref = df.copy()
                x_ref = df_ref["x"].values
                y_ref = df_ref["y"].values + ajuste_y_ref
                break

    render_tabla_bibliografia_ftir(db)

    fig = go.Figure()
    render_grafico_combinado_ftir(
        fig, datos_plotly, aplicar_suavizado, normalizar,
        offset_vertical, ajustes_y, restar_espectro,
        x_ref, y_ref, x_min, x_max, y_min, y_max,
        mostrar_picos, altura_min, distancia_min
    )

    preprocesados = generar_preprocesados_ftir(datos_plotly, aplicar_suavizado, normalizar, offset_vertical, ajustes_y, restar_espectro, x_ref, y_ref, x_min, x_max)

    return datos_plotly, fig, preprocesados, x_ref, y_ref, x_min, x_max, y_min, y_max


def render_tab5(db, cargar_muestras, mostrar_sector_flotante):
#    st.title("Análisis FTIR")
    st.session_state["current_tab"] = "Análisis FTIR"
    opciones_muestras = sorted([m["nombre"] for m in cargar_muestras(db)])
    muestras_sel = st.multiselect("Seleccionar muestras", opciones_muestras)
    if not muestras_sel:
        st.stop()
    muestras = [m for m in cargar_muestras(db) if m["nombre"] in muestras_sel]
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()


    # 1. Gráfica FTIR (internamente llama todo)
    datos_plotly, fig, preprocesados, x_ref, y_ref, x_min, x_max, y_min, y_max = render_comparacion_espectros_ftir(db, muestras)
 
    if not datos_plotly:
        return  # Evita errores si no se seleccionaron espectros


    # 2. Deconvolución FTIR (opcional)
    if st.checkbox("Mostrar deconvolución", value=False, key="activar_deconv_ftir"):
        render_deconvolucion_ftir(preprocesados, x_min, x_max, y_min, y_max)

    if st.checkbox("Mostrar opciones de exportación", value=False):
        exportar_resultados_ftir(
            preprocesados=preprocesados,
            resumen=None,  # o df_resumen si lo tuvieras
            fwhm_rows=None,  # o fwhm_data si lo calculás
            x_min=x_min, x_max=x_max
        )
        exportar_figura_plotly_png(fig, nombre_base="FTIR")


    # 3. Calcular índice OH espectroscópico ---
    st.subheader("Índice OH espectroscópico")
    espectros_info = []
    for m in muestras:
        for e in obtener_espectros_para_muestra(db, m["nombre"]):
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
                return None
            k = 52.5253 if row["Tipo"] == "FTIR-Acetato" else 66.7324
            return round(((y_graf - y_ref) * k) / peso, 2)

        df_oh["Índice OH"] = df_oh.apply(calcular_indice, axis=1)
        df_oh["Índice OH"] = pd.to_numeric(df_oh["Índice OH"], errors="coerce")
        st.dataframe(df_oh[["Muestra", "Tipo", "Fecha", "Señal", "Señal solvente", "Peso muestra [g]", "Índice OH"]], use_container_width=True)


    # 4. Calculadora manual de índice OH
    calculadora_indice_oh_manual()

    # Sector flotante final
    mostrar_sector_flotante(db, key_suffix="tab5")
