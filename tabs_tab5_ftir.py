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
from collections import defaultdict
from scipy.signal import find_peaks


GRUPOS_FUNCIONALES = ["Formiato", "Cloroformo", "C=C olefínicos", "Glicerol medio", "Glicerol extremos", "Metil-Éster", "Eter", "Ester", "Ácido carboxílico", "OH", "Epóxido", "C=C", "Alfa-C=O","Alfa-C-OH", "Alfa-C=C", "C=C-Alfa-C=C", "Beta-carbonilo", "Alfa-epóxido", "Epóxido-alfa-epóxido", "CH2", "CH3", "SO3-"]

def obtener_ids_espectros(nombre):
    return [doc.id for doc in firestore.Client().collection("muestras").document(nombre).collection("espectros").list_documents()]

def obtener_espectros_para_muestra(db, nombre, force=False):
    clave = f"_espectros_cache_{nombre}"
    if force or clave not in st.session_state:
        ref = db.collection("muestras").document(nombre).collection("espectros")
        docs = ref.stream()
        st.session_state[clave] = [doc.to_dict() for doc in docs]
    return st.session_state[clave]

def render_tabla_calculos_ftir(db, datos_plotly, mostrar=True, sombrear=False):
    if not datos_plotly:
        return

    filas_totales = []
    claves_guardado = []

    for muestra, tipo, archivo, df in datos_plotly:
        clave = f"{muestra}__{archivo}"
        doc_ref = db.collection("tablas_ftir").document(clave)
        doc = doc_ref.get()
        filas = doc.to_dict().get("filas", []) if doc.exists else []

        for fila in filas:
            fila["Muestra"] = muestra
            fila["Archivo"] = archivo
        filas_totales.extend(filas)
        claves_guardado.append((muestra, archivo, df))

    columnas = ["Muestra", "Grupo funcional", "D pico", "X min", "X max", "Área", "Observaciones", "Archivo"]

    df_tabla = pd.DataFrame(filas_totales, columns=columnas)
    df_tabla["Observaciones"] = df_tabla["Observaciones"].astype(str)
    df_tabla["Muestra"] = df_tabla["Muestra"].astype(str)
    df_tabla["Archivo"] = df_tabla["Archivo"].astype(str)
    for col in ["D pico", "X min", "X max", "Área"]:
        df_tabla[col] = pd.to_numeric(df_tabla[col], errors="coerce")

    if mostrar:
        if not filas_totales:
            st.info("No hay datos previos. Podés agregar una nueva fila manualmente.")
            opciones_muestras = list(set([m for m, _, _, _ in datos_plotly]))
            opciones_archivos = list(set([a for _, _, a, _ in datos_plotly]))
            col1, col2 = st.columns(2)
            with col1:
                muestra_nueva = st.selectbox("Muestra para nueva fila", opciones_muestras, key="muestra_nueva_ftir")
            with col2:
                archivo_nuevo = st.selectbox("Archivo para nueva fila", opciones_archivos, key="archivo_nuevo_ftir")

            filas_totales = [{
                "Muestra": muestra_nueva,
                "Grupo funcional": "",
                "D pico": None,
                "X min": None,
                "X max": None,
                "Área": None,
                "Observaciones": "",
                "Archivo": archivo_nuevo
            }]
        st.markdown("**📊 Tabla de Cálculos FTIR**")
        key_editor = f"tabla_calculos_ftir_{'sombreado' if sombrear else 'normal'}"
        editada = st.data_editor(
            df_tabla,
            num_rows="dynamic",
            key=key_editor,
            column_order=columnas,
            use_container_width=True,
            column_config={
                "Grupo funcional": st.column_config.SelectboxColumn("Grupo funcional", options=GRUPOS_FUNCIONALES),
                "D pico": st.column_config.NumberColumn("δ pico [cm⁻¹]", format="%.2f"),
                "X min": st.column_config.NumberColumn("X min", format="%.2f"),
                "X max": st.column_config.NumberColumn("X max", format="%.2f"),
                "Área": st.column_config.NumberColumn("🔴Área", disabled=True, format="%.2f"),
                "Observaciones": st.column_config.TextColumn("Observaciones"),
                "Muestra": st.column_config.TextColumn("Muestra"),
                "Archivo": st.column_config.TextColumn("Archivo"),
            }
        )

        if st.button("🔴Recalcular áreas FTIR", key="recalc_area_ftir_global"):
            for i, row in editada.iterrows():
                try:
                    x0 = float(row["X min"])
                    x1 = float(row["X max"])
                    muestra = row["Muestra"]
                    archivo = row["Archivo"]
                    df = next((df for m, a, df in claves_guardado if m == muestra and a == archivo), None)
                    if df is not None:
                        df_filt = df[(df["x"] >= min(x0, x1)) & (df["x"] <= max(x0, x1))].copy()
                        df_filt = df_filt.sort_values("x")
                        area = np.trapz(df_filt["y"], df_filt["x"])
                        editada.at[i, "Área"] = round(area, 2)
                except:
                    continue

            for muestra, archivo, _ in claves_guardado:
                df_guardar = editada[(editada["Muestra"] == muestra) & (editada["Archivo"] == archivo)]
                columnas_guardar = ["Grupo funcional", "D pico", "X min", "X max", "Área", "Observaciones"]
                filas_guardar = df_guardar[columnas_guardar].dropna(subset=["X min", "X max"]).to_dict(orient="records")

                doc_ref = db.collection("tablas_ftir").document(f"{muestra}__{archivo}")
                doc_ref.set({"filas": filas_guardar})

                if not filas_guardar:
                    st.warning(f"⚠️ No se guardaron filas para **{muestra} – {archivo}** (faltan X min o X max).")

            st.success("Todas las áreas fueron recalculadas y guardadas correctamente.")
    else:
        editada = df_tabla

    if sombrear:
        st.session_state["shapes_calculos_ftir"] = []
        for _, row in editada.iterrows():
            try:
                x0 = float(row["X min"])
                x1 = float(row["X max"])
                st.session_state["shapes_calculos_ftir"].append({
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": min(x0, x1),
                    "x1": max(x0, x1),
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": "rgba(0, 0, 255, 0.1)",
                    "line": {"width": 0}
                })
            except:
                continue
    else:
        st.session_state["shapes_calculos_ftir"] = []


def render_tabla_bibliografia_ftir(db, mostrar=True, delinear=False):
    st.session_state["shapes_biblio_ftir"] = []
    st.session_state["annots_biblio_ftir"] = []

    ruta = "tablas_ftir_bibliografia/default"
    doc_ref = db.document(ruta)
    doc = doc_ref.get()
    filas = doc.to_dict().get("filas", []) if doc.exists else []

    columnas = ["Grupo funcional", "X min", "δ pico", "X max", "Tipo de muestra", "Observaciones"]
    df_biblio = pd.DataFrame(filas) if filas else pd.DataFrame([dict.fromkeys(columnas, "")])

    # Solo mostrar editor si el usuario quiere ver la tabla
    if mostrar:
        st.markdown("**📚 Tabla bibliográfica FTIR**")
        key_editor = f"tabla_calculos_ftir_local_{'sombreado' if delinear else 'limpio'}"
        editada = st.data_editor(
            df_biblio,
            column_order=columnas,
            use_container_width=True,
            key=key_editor,
            num_rows="dynamic",
            column_config={
                "Grupo funcional": st.column_config.SelectboxColumn("Grupo funcional", options=GRUPOS_FUNCIONALES),
                "X min": st.column_config.NumberColumn("X min", format="%d"),
                "δ pico": st.column_config.NumberColumn("δ pico", format="%d"),
                "X max": st.column_config.NumberColumn("X max", format="%d"),
                "Tipo de muestra": st.column_config.TextColumn("Tipo de muestra"),
                "Observaciones": st.column_config.TextColumn("Observaciones"),
            }
        )

        if st.button("Guardar bibliografía FTIR", key="guardar_biblio_ftir"):
            doc_ref.set({"filas": editada.to_dict(orient="records")})
            st.success("Bibliografía guardada correctamente.")
    else:
        editada = df_biblio

    # Dibujar aunque no se muestre la tabla
    if delinear:
        for _, row in editada.iterrows():
            try:
                x0 = float(row["δ pico"])
                grupo = str(row.get("Grupo funcional", "")).strip()
                obs = str(row.get("Observaciones", "")).strip()

                if obs:
                    texto = f"{grupo} – {obs[:30]}"  # combinamos ambas, truncando observaciones
                else:
                    texto = grupo
                st.session_state["shapes_biblio_ftir"].append({
                    "type": "line",
                    "xref": "x",
                    "yref": "paper",
                    "x0": x0,
                    "x1": x0,
                    "y0": 0,
                    "y1": 0.5,
                    "line": {"color": "black", "width": 1, "dash": "dot"}
                })
                st.session_state["annots_biblio_ftir"].append({
                    "xref": "x",
                    "yref": "paper",
                    "x": x0,
                    "y": 0.65,
                    "text": texto,
                    "showarrow": False,
                    "font": {"color": "black", "size": 10},
                    "textangle": -90
                })
            except:
                continue

    return editada if mostrar else pd.DataFrame([])


def render_deconvolucion_ftir(preprocesados, x_min, x_max, y_min, y_max, activar_deconv):
    if not activar_deconv or not preprocesados:
        return

    # Selección de espectros
    col1, col2, col3, col4 = st.columns(4)
    claves_disponibles = list(preprocesados.keys())
    checkboxes = {}
    for i, clave in enumerate(claves_disponibles):
        with [col1, col2, col3, col4][i % 4]:
            checkboxes[clave] = st.checkbox(clave, value=False, key=f"deconv_{clave}")

    # Inicializar acumulador si no existe
    if "resultados_totales" not in st.session_state:
        st.session_state["resultados_totales"] = {}

    ref_nombre = st.session_state.get("espectro_ref_nombre", "")

    for clave in claves_disponibles:
        if not checkboxes.get(clave):
            continue
        if clave == ref_nombre:
            continue  # ⛔ no deconvolucionar el espectro restado
        try:
            df = preprocesados.get(clave)
            if df is None or df.empty:
                continue

            x = df["x"].values.astype(float)
            y = df["y"].values.astype(float)

            # Limitar al rango de usuario
            mask = (x >= x_min) & (x <= x_max)
            x = x[mask]
            y = y[mask]
            if len(x) < 10:
                st.warning(f"⚠️ Muy pocos puntos para {clave} en el rango definido.")
                continue

            def multi_gaussian(x, *params):
                y_fit = np.zeros_like(x, dtype=float)
                for i in range(0, len(params), 3):
                    amp, cen, wid = params[i:i+3]
                    y_fit += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
                return y_fit

            # Parámetros iniciales y ajuste
            n_gauss = st.slider(f"Nº de gaussianas para {clave}", 1, 10, 3, key=f"gauss_{clave}")
            p0 = []
            for i in range(n_gauss):
                p0 += [y.max() / n_gauss, x.min() + i * (np.ptp(x) / n_gauss), 10]

            popt, _ = curve_fit(multi_gaussian, x, y, p0=p0, maxfev=10000)
            y_fit = multi_gaussian(x, *popt)

            # Plot con Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Original", line=dict(color="black")))
            fig.add_trace(go.Scatter(x=x, y=y_fit, mode="lines", name="Ajuste", line=dict(dash="dash", color="orange")))

            resultados = []
            for i in range(n_gauss):
                amp, cen, wid = popt[3*i:3*i+3]
                gauss = amp * np.exp(-(x - cen)**2 / (2 * wid**2))
                area = amp * wid * np.sqrt(2*np.pi)
                fwhm = 2.355 * abs(wid)
                fig.add_trace(go.Scatter(
                    x=x, y=gauss, mode="lines", name=f"Pico {i+1}", line=dict(dash="dot")
                ))
                resultados.append({
                    "Pico": i+1,
                    "Centro (cm⁻¹)": round(cen, 2),
                    "Amplitud": round(amp, 2),
                    "Anchura σ": round(wid, 2),
                    "FWHM": round(fwhm, 2),
                    "Área": round(area, 2)
                })

            fig.update_layout(
                title=f"Deconvolución – {clave}",
                xaxis_title="Número de onda [cm⁻¹]",
                yaxis_title="Absorbancia",
                height=500,
                xaxis=dict(range=[x_max, x_min]),
                yaxis=dict(range=[y_min, y_max]),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Métricas de ajuste
            rmse = np.sqrt(np.mean((y - y_fit)**2))
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            st.markdown(f"**{clave}**  **RMSE:** {rmse:.4f} &nbsp;&nbsp;&nbsp;&nbsp; **R²:** {r2:.4f}")

            # Mostrar y guardar resultados
            df_result = pd.DataFrame(resultados)
            st.session_state["resultados_totales"][clave] = df_result
            st.dataframe(df_result, use_container_width=True)

            # Botón de descarga individual
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


def render_tabla_similitud_ftir_matriz(preprocesados, x_min, x_max, tipo_comparacion, sombrear_similitud):
    # --- Sombreado opcional en el gráfico ---
    if sombrear_similitud:
        st.session_state["shapes_similitud_ftir"] = [{
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": min(x_min, x_max),
            "x1": max(x_min, x_max),
            "y0": 0,
            "y1": 1,
            "fillcolor": "rgba(0, 200, 0, 0.1)",
            "line": {"width": 0}
        }]
    else:
        st.session_state["shapes_similitud_ftir"] = []

    # --- Comparación ---
    st.markdown("**🔍 Matriz de similitud entre espectros**")

    nombre_repetidos = defaultdict(int)
    vectores = {}
    etiquetas_map = {}

    for clave, df in preprocesados.items():
        nombre_base = clave.split(" – ")[0].strip()
        nombre_repetidos[nombre_base] += 1
        nombre_final = f"{nombre_base} #{nombre_repetidos[nombre_base]}"
        etiquetas_map[clave] = nombre_final

        df_filt = df[(df["x"] >= min(x_min, x_max)) & (df["x"] <= max(x_min, x_max))].copy()
        if not df_filt.empty:
            vectores[nombre_final] = (df_filt["x"].values, df_filt["y"].values)

    etiquetas = list(vectores.keys())
    muestra_ref = st.selectbox("🔹 Muestra de referencia", etiquetas, index=0, key="simil_ref")

    # Reordenar con muestra de referencia al inicio
    nombres = etiquetas.copy()
    if muestra_ref in nombres:
        nombres.remove(muestra_ref)
        nombres = [muestra_ref] + nombres

    matriz = np.zeros((len(nombres), len(nombres)))

    for i in range(len(nombres)):
        for j in range(len(nombres)):
            xi, yi = vectores[nombres[i]]
            xj, yj = vectores[nombres[j]]

            x_comun = np.linspace(max(xi.min(), xj.min()), min(xi.max(), xj.max()), 500)
            try:
                yi_interp = np.interp(x_comun, xi, yi)
                yj_interp = np.interp(x_comun, xj, yj)

                if len(yi_interp) == 0 or len(yj_interp) == 0:
                    simil = 0
                else:
                    if tipo_comparacion == "Pearson (correlación)":
                        if np.std(yi_interp) == 0 or np.std(yj_interp) == 0:
                            simil = 0
                        else:
                            simil = np.corrcoef(yi_interp, yj_interp)[0, 1] * 100
                    else:  # Comparación por área integrada
                        area_i = np.trapz(yi_interp, x_comun)
                        area_j = np.trapz(yj_interp, x_comun)
                        if area_i == 0 and area_j == 0:
                            simil = 100
                        elif area_i == 0 or area_j == 0:
                            simil = 0
                        else:
                            simil = (1 - abs(area_i - area_j) / max(abs(area_i), abs(area_j))) * 100
            except:
                simil = 0

            matriz[i, j] = simil

    df_similitud = pd.DataFrame(matriz, index=nombres, columns=nombres)

    # --- Mostrar tabla como heatmap ---
    st.dataframe(
        df_similitud.style
            .format(lambda x: f"{x:.2f} %")
            .background_gradient(cmap="RdYlGn")
            .set_properties(**{"text-align": "center"}),
        use_container_width=True
    )


def render_grafico_combinado_ftir(fig, datos_plotly, aplicar_suavizado, normalizar,
                                    ajustes_y, restar_espectro,
                                    x_ref, y_ref, x_min, x_max, y_min, y_max,
                                    mostrar_picos, altura_min, distancia_min):
    for i, (muestra, tipo, archivo, df) in enumerate(datos_plotly):
        clave = f"{muestra} – {tipo} – {archivo}"
        df_filtrado = df[(df["x"] >= x_min) & (df["x"] <= x_max)]
        if df_filtrado.empty:
            continue
        x = df_filtrado["x"].values
        y = df_filtrado["y"].values
        if aplicar_suavizado and len(y) >= 7:
            y = savgol_filter(y, window_length=7, polyorder=2)
        if normalizar and np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))
        y = y + ajustes_y.get(clave, 0.0)

        # x e y ya están filtrados, suavizados, normalizados, ajustados, etc.
        x = df_filtrado["x"].values
        y_data = y.copy()  # y ya tiene suavizado, normalizado, offset, etc.

        if x_ref is not None and y_ref is not None:
            try:
                # Ordenar referencia por eje x ascendente
                idx_ref = np.argsort(x_ref)
                x_ref_sorted = x_ref[idx_ref]
                y_ref_sorted = y_ref[idx_ref]

                # Interpolación segura con exclusión de NaN
                y_interp = np.interp(x, x_ref_sorted, y_ref_sorted, left=np.nan, right=np.nan)
                mask_validos = ~np.isnan(y_interp)
                x = x[mask_validos]
                y_data = y_data[mask_validos]
                y_interp = y_interp[mask_validos]
                y_data = y_data - y_interp
            except Exception as e:
                st.warning(f"Error en interpolación: {e}")


        fig.add_trace(go.Scatter(x=x, y=y_data, mode="lines", name=archivo, hovertemplate="x=%{x}<br>y=%{y}<extra></extra>"))

    if mostrar_picos:

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

    shapes = []
    annotations = []

    for k in ["shapes_calculos_ftir", "shapes_similitud_ftir", "shapes_biblio_ftir"]:
        shapes.extend(st.session_state.get(k, []))

    for k in ["annots_biblio_ftir"]:
        annotations.extend(st.session_state.get(k, []))

    fig.update_layout(shapes=shapes, annotations=annotations)
  
    fig.update_layout(
        xaxis_title="Número de onda [cm⁻¹]",
        yaxis_title="Absorbancia",
        margin=dict(l=10, r=10, t=30, b=10),
        height=500,
        xaxis=dict(range=[x_max, x_min]),
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


def render_graficos_individuales_ftir(preprocesados, x_min, x_max, y_min, y_max,
                                      aplicar_suavizado, normalizar, ajustes_y, restar_espectro,
                                      mostrar_picos, altura_min, distancia_min):

    for clave, df in preprocesados.items():
        df_filtrado = df[(df["x"] >= x_min) & (df["x"] <= x_max)]
        if df_filtrado.empty:
            continue
        x = df_filtrado["x"].values
        y = df_filtrado["y"].values

        if aplicar_suavizado and len(y) >= 7:
            y = savgol_filter(y, window_length=7, polyorder=2)
        if normalizar and np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))
        y = y + ajustes_y.get(clave, 0.0)

        if restar_espectro and st.session_state.get("y_ref_interp") is not None:
            y_ref_interp = np.interp(x, st.session_state["x_ref_interp"], st.session_state["y_ref_interp"], left=np.nan, right=np.nan)
            mask_validos = ~np.isnan(y_ref_interp)
            x = x[mask_validos]
            y = y[mask_validos] - y_ref_interp[mask_validos]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=clave))

        # Picos si corresponde
        if mostrar_picos:
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

        # Aplicar sombreado/delineado si están en session_state
        shapes = []
        annotations = []
        for k in ["shapes_calculos_ftir", "shapes_similitud_ftir", "shapes_biblio_ftir"]:
            shapes.extend(st.session_state.get(k, []))
        for k in ["annots_biblio_ftir"]:
            annotations.extend(st.session_state.get(k, []))

        fig.update_layout(
            title=clave,
            xaxis_title="Número de onda [cm⁻¹]",
            yaxis_title="Absorbancia",
            shapes=shapes,
            annotations=annotations,
            xaxis=dict(range=[x_max, x_min]),
            yaxis=dict(range=[y_min, y_max] if not normalizar else None),
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)


def generar_preprocesados_ftir(datos_plotly, aplicar_suavizado, normalizar,
                                ajustes_y, restar_espectro,
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
        y = y + ajustes_y.get(clave, 0.0)

        if restar_espectro and x_ref is not None and y_ref is not None:
            idx_ref = np.argsort(x_ref)
            x_ref_sorted = x_ref[idx_ref]
            y_ref_sorted = y_ref[idx_ref] 
                      
            y_interp = np.interp(x, x_ref_sorted, y_ref_sorted)
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
                df_filtrado = df[(df["x"] >= x_min) & (df["x"] <= x_max)].copy()
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
    mostrar_picos = col3.checkbox("Detectar picos", value=False, key="picos_ftir")
    restar_espectro = col4.checkbox("Restar espectro", value=False, key="restar_ftir")
    ajuste_y_manual = col5.checkbox("Ajuste manual Y", value=False, key="ajuste_y_ftir")
    mostrar_grafico_vertical = col6.checkbox("Superposición vertical", value=False, key="vertical_plot_ftir")

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
        st.session_state["espectro_ref_nombre"] = espectro_ref 
        ajuste_y_ref = st.number_input("Ajuste Y referencia", value=0.0, step=0.1, key="ajuste_ref_ftir")

        for m, t, a, df in datos_plotly:
            if espectro_ref == f"{m} – {t} – {a}":
                x_ref = df["x"].values
                y_ref = df["y"].values.astype(float)

                # Aplicar mismo suavizado y normalización que al resto
                if aplicar_suavizado and len(y_ref) >= 7:
                    y_ref = savgol_filter(y_ref, window_length=7, polyorder=2)
                if normalizar and np.max(np.abs(y_ref)) != 0:
                    y_ref = y_ref / np.max(np.abs(y_ref))

                y_ref = y_ref + ajuste_y_ref
                break
    return {
        "suavizado": aplicar_suavizado,
        "normalizar": normalizar,
        "mostrar_picos": mostrar_picos,
        "restar": restar_espectro,
        "ajuste_y_manual": ajuste_y_manual,
        "ajustes_y": ajustes_y,
        "x_ref": x_ref,
        "y_ref": y_ref,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "mostrar_grafico_vertical": mostrar_grafico_vertical,
    }





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
                "Observaciones": e.get("observaciones", ""),
                "Fecha": e.get("fecha", ""),
                "Señal": valor_y_extraido,
                "Señal manual 3548": e.get("senal_3548"),
                "Señal manual 3611": e.get("senal_3611"),
                "Peso muestra [g]": e.get("peso_muestra"),
                "Archivo": e.get("nombre_archivo", ""),
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
    df_oh = df_oh.sort_values(by=["Muestra", "Fecha", "Observaciones"])

    # Eliminar duplicados solo en las columnas automáticas
    df_oh = df_oh.drop_duplicates(subset=[
        "Muestra", "Tipo", "Observaciones", "Fecha", "Señal", "Señal manual 3548",
        "Señal manual 3611", "Peso muestra [g]", "Archivo", "Señal solvente", "Índice OH"
    ])

    return df_oh[["Muestra", "Tipo", "Observaciones", "Fecha","Señal", "Señal solvente", "Peso muestra [g]", "Índice OH"]]


def calculadora_indice_oh_manual():
    #st.subheader("Calculadora manual de Índice OH")

    datos_oh = pd.DataFrame([
        {"Tipo": "FTIR-Acetato [3548 cm⁻¹]", "Señal": 0.0000, "Señal solvente": 0.0000, "Peso muestra [g]": 0.0000},
        {"Tipo": "FTIR-Cloroformo A [3611 cm⁻¹]", "Señal": 0.0000, "Señal solvente": 0.0000, "Peso muestra [g]": 0.0000},
        {"Tipo": "FTIR-Cloroformo B [3611 cm⁻¹]", "Señal": 0.0000, "Señal solvente": 0.0000, "Peso muestra [g]": 0.0000},
        {"Tipo": "FTIR-Cloroformo C [3611 cm⁻¹]", "Señal": 0.0000, "Señal solvente": 0.0000, "Peso muestra [g]": 0.0000}
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


def render_comparacion_espectros_ftir(db, muestras):
    #st.subheader("Comparación de espectros FTIR")
    tipos_validos = ["FTIR-Acetato", "FTIR-Cloroformo", "FTIR-ATR"]
    espectros_dict = {}

    # --- Leer espectros válidos ---
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
                    "muestra": nombre,
                    "fecha": e.get("fecha", "Sin fecha"),
                    "peso_muestra": e.get("peso_muestra"),
                    "observaciones": e.get("observaciones", "")
                }

    if not muestras:
        return [], None, {}, None, None, None, None, None, None

    if not espectros_dict:
        st.info("No hay espectros FTIR válidos para mostrar.")
        return [], None, {}, None, None, None, None, None, None

    # --- Filtros globales ---
    tipos_unicos = sorted(set(e["tipo"] for e in espectros_dict.values()))
    fechas_unicas = sorted(set(e.get("fecha", "Sin fecha") for e in espectros_dict.values()))
    pesos_validos = [
        e.get("peso_muestra") for e in espectros_dict.values()
        if isinstance(e.get("peso_muestra"), (int, float)) and e.get("peso_muestra") is not None
    ]

    # --- Mostrar filtros globales ---
    tipos_seleccionados = st.multiselect(
        "Tipos de espectro", tipos_unicos, default=tipos_unicos, key="tipos_global"
    )
    fechas_seleccionadas = st.multiselect(
        "Fechas", fechas_unicas, default=fechas_unicas, key="fechas_global"
    )
    if pesos_validos:
        peso_min_val = float(min(pesos_validos))
        peso_max_val = float(max(pesos_validos))
        
        # Evitar que min_value y max_value sean iguales
        if peso_min_val == peso_max_val:
            peso_max_val = peso_min_val + 0.01  # Agregar un pequeño incremento
            
        peso_min, peso_max = st.slider(
            "Peso muestra", 
            min_value=peso_min_val,
            max_value=peso_max_val,
            value=(peso_min_val, peso_max_val),
            step=0.01,
            key="peso_global"
        )
    else:
        peso_min, peso_max = None, None
        st.info("No hay pesos definidos en los espectros.")

    # --- Selector de espectros filtrado ---
    archivos_disp = []
    archivos_dict_displayname = {}  # Para mapear display_name → (m, archivo)

    for (m, archivo), e in espectros_dict.items():
        if e["tipo"] not in tipos_seleccionados:
            continue
        if e.get("fecha", "Sin fecha") not in fechas_seleccionadas:
            continue
        peso = e.get("peso_muestra")
        if isinstance(peso, (int, float)):
            if not (peso_min <= peso <= peso_max):
                continue

        # Construir display_name
        if e.get("observaciones", ""):
            display_name = f"{archivo} – {e['observaciones'][:20]}"
        else:
            display_name = archivo

        archivos_disp.append(display_name)
        archivos_dict_displayname[display_name] = (m, archivo)  # para poder decodificar después

    archivos_disp = sorted(set(archivos_disp))

    archivos_sel = st.multiselect(
        "Seleccionar espectros", archivos_disp, key="archivos_ftir"
    )

    # --- Leer archivos seleccionados ---
    datos_plotly = []
    for item in archivos_sel:
        muestra, archivo = archivos_dict_displayname[item]
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

    # --- Resto de tu flujo ---
    controles = render_controles_preprocesamiento(datos_plotly)

    altura_min = 0.02
    distancia_min = 20
    if controles["mostrar_picos"]:
        colp1, colp2 = st.columns(2)
        altura_min = colp1.number_input("Altura mínima para detección de picos", value=0.02, step=0.01)
        distancia_min = colp2.number_input("Distancia mínima entre picos", value=20, step=1)

    col1, col2 = st.columns(2)
    with col1:
        mostrar_calculos = st.checkbox("📊 Tabla de Cálculos FTIR", key="mostrar_tabla_calculos_ftir")
    with col2:
        sombrear_calculos = st.checkbox("Sombrear Cálculos FTIR", key="sombrear_tabla_calculos_ftir")
    with col1:
        mostrar_biblio = st.checkbox("📚 Tabla Bibliográfica FTIR", key="mostrar_tabla_biblio_ftir")
    with col2:
        delinear_biblio = st.checkbox("Delinear Bibliografía FTIR", key="delinear_tabla_biblio_ftir")
    with col1:
        mostrar_similitud = st.checkbox("🔍 Tabla de Similitud FTIR", key="mostrar_tabla_similitud_ftir")
    with col2:
        sombrear_similitud = st.checkbox("Sombrear Similitud FTIR", key="sombrear_tabla_similitud_ftir")

    preprocesados = generar_preprocesados_ftir(
        datos_plotly, controles["suavizado"], controles["normalizar"],
        controles["ajustes_y"], controles["restar"],
        controles["x_ref"], controles["y_ref"],
        controles["x_min"], controles["x_max"]
    )

    render_tabla_calculos_ftir(db, datos_plotly, mostrar=mostrar_calculos, sombrear=sombrear_calculos)
    render_tabla_bibliografia_ftir(db, mostrar=mostrar_biblio, delinear=delinear_biblio)

    if mostrar_similitud:
        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.2])
        x_min = col1.number_input("X min", value=1000.0, step=1.0, key="simil_xmin")
        x_max = col2.number_input("X max", value=1100.0, step=1.0, key="simil_xmax")
        tipo = col4.selectbox("Modo", ["Pearson (correlación)", "Área integrada (RMSE relativo)"],index=1, label_visibility="collapsed")

        render_tabla_similitud_ftir_matriz(
            preprocesados=preprocesados,
            x_min=x_min,
            x_max=x_max,
            tipo_comparacion=tipo,
            sombrear_similitud=sombrear_similitud
        )

    fig = go.Figure()
    render_grafico_combinado_ftir(
        fig, datos_plotly,
        controles["suavizado"], controles["normalizar"],
        controles["ajustes_y"], controles["restar"],
        controles["x_ref"], controles["y_ref"],
        controles["x_min"], controles["x_max"],
        controles["y_min"], controles["y_max"],
        controles["mostrar_picos"],
        altura_min, distancia_min
    )
    
    if controles["mostrar_grafico_vertical"]:
        offset_vertical = st.slider(
            "Separación vertical entre espectros", min_value=0.0, max_value=5.0, value=1.0, step=0.1
        )

        fig_vertical = go.Figure()
        for i, (muestra, tipo, archivo, df) in enumerate(datos_plotly):
            y_offset = offset_vertical * i
            df_filtrado = df[(df["x"] >= controles["x_min"]) & (df["x"] <= controles["x_max"])].copy()
            x = df_filtrado["x"].values
            y = df_filtrado["y"].values.astype(float)

            if controles["suavizado"] and len(y) >= 7:
                y = savgol_filter(y, window_length=7, polyorder=2)
            if controles["normalizar"] and np.max(np.abs(y)) != 0:
                y = y / np.max(np.abs(y))
            y = y + y_offset

            fig_vertical.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=f"{muestra} – {tipo} – {archivo}",
                hovertemplate="x=%{x}<br>y=%{y}<extra></extra>"
            ))

        fig_vertical.update_layout(
            title="Superposición vertical",
            xaxis_title="Número de onda [cm⁻¹]",
            yaxis_title="Absorbancia desplazada",
            xaxis=dict(range=[controles["x_max"], controles["x_min"]]),  # eje invertido
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_vertical, use_container_width=True)

    return (
        datos_plotly,
        fig,
        preprocesados,
        controles["x_ref"], controles["y_ref"],
        controles["x_min"], controles["x_max"],
        controles["y_min"], controles["y_max"]
    )


def render_tab5(db, cargar_muestras, mostrar_sector_flotante):
#    st.title("Análisis FTIR")
    st.session_state["current_tab"] = "Análisis FTIR"
    datos_plotly = []
    opciones_muestras = sorted([m["nombre"] for m in cargar_muestras(db)])
    muestras_sel = st.multiselect("Seleccionar muestras", opciones_muestras)
    muestras = [m for m in cargar_muestras(db) if m["nombre"] in muestras_sel]

    if muestras_sel and muestras:
        # 1. Gráfica FTIR (internamente llama todo)
        datos_plotly, fig, preprocesados, x_ref, y_ref, x_min, x_max, y_min, y_max = render_comparacion_espectros_ftir(db, muestras)

        # --- Gráficos individuales FTIR ---
        mostrar_individuales = st.checkbox("Gráficos individuales FTIR", key="mostrar_individuales_ftir")
        if datos_plotly and mostrar_individuales:
            render_graficos_individuales_ftir(
                preprocesados=preprocesados,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                aplicar_suavizado=st.session_state.get("activar_suavizado", False),
                normalizar=st.session_state.get("activar_normalizar", False),
                ajustes_y=st.session_state.get("ajustes_y", {}),
                restar_espectro=st.session_state.get("activar_resta", False),
                mostrar_picos=st.session_state.get("mostrar_picos", False),
                altura_min=st.session_state.get("altura_min", 0.02),
                distancia_min=st.session_state.get("distancia_min", 50),
            )

        # --- Deconvolución FTIR ---
        activar_deconv = st.checkbox("Deconvolución de espectros FTIR", value=False, key="activar_deconv_ftir")
        if datos_plotly and activar_deconv:
            render_deconvolucion_ftir(
                preprocesados=preprocesados,  # DEBE SER este, con resta aplicada
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                activar_deconv=True
            )

            if st.checkbox("Mostrar opciones de exportación", value=False):
                exportar_resultados_ftir(
                    preprocesados=preprocesados,
                    resumen=None,
                    fwhm_rows=None,
                    x_min=x_min, x_max=x_max
                )
                exportar_figura_plotly_png(fig, nombre_base="FTIR")


    # 3. Índice OH espectroscópico
    if st.checkbox("Índice OH espectroscópico", value=False):
        doc_ref = db.document("tablas_indice_oh/manual")
        doc = doc_ref.get()
        
        # leer solo al iniciar
        if doc.exists and "df_oh_editado" not in st.session_state:
            filas_guardadas = doc.to_dict().get("filas", [])
            st.session_state["df_oh_editado"] = pd.DataFrame(filas_guardadas)
        elif "df_oh_editado" not in st.session_state:
            df_resultado = calcular_indice_oh_auto(db, cargar_muestras(db)).reset_index(drop=True)
            if not df_resultado.empty:
                df_resultado["X"] = None
                df_resultado["Curva"] = ""
                st.session_state["df_oh_editado"] = df_resultado

        colA, colB = st.columns(2)
        if colA.button("🔄 Refrescar espectros"):
            # limpia caché de espectros para ver lo nuevo
            for k in list(st.session_state.keys()):
                if k.startswith("_espectros_cache_"):
                    st.session_state.pop(k)

        if colB.button("🔁 Recalcular y fusionar"):
            # 1) recalc auto con muestras actuales
            for k in list(st.session_state.keys()):
                if k.startswith("_espectros_cache_"):
                    st.session_state.pop(k)
            df_auto = calcular_indice_oh_auto(db, cargar_muestras(db)).reset_index(drop=True)

            # 2) traer lo manual guardado
            doc_ref = db.document("tablas_indice_oh/manual")
            filas_guardadas = doc_ref.get().to_dict().get("filas", []) if doc_ref.get().exists else []
            df_manual = pd.DataFrame(filas_guardadas)

            # 3) fusionar por claves y conservar columnas manuales (X, Curva)
            claves = ["Muestra","Tipo","Observaciones","Fecha"]
            if not df_manual.empty:
                keep = df_manual[claves + ["X","Curva"]].drop_duplicates()
                df_final = df_auto.merge(keep, on=claves, how="left")
                # agregar filas manuales que no estén en auto
                faltantes = keep.merge(df_auto[claves], on=claves, how="left", indicator=True)
                faltantes = faltantes[faltantes["_merge"]=="left_only"].drop(columns=["_merge"])
                df_final = pd.concat([df_final, faltantes], ignore_index=True)
            else:
                df_final = df_auto

            st.session_state["df_oh_editado"] = df_final
            doc_ref.set({"filas": df_final.to_dict(orient="records")})
            st.success("Tabla actualizada y fusionada sin perder tus datos manuales.")


        df_editado = st.data_editor(
            st.session_state.get("df_oh_editado", pd.DataFrame()),
            column_config={
                "X": st.column_config.NumberColumn("X", format="%.3f"),
                "Curva": st.column_config.TextColumn("Curva"),
            },
            column_order=[
                "Muestra", "Tipo", "Observaciones", "Fecha",
                "Señal", "Señal solvente", "Peso muestra [g]",
                "Índice OH", "X", "Curva"
            ],
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="editor_xy_manual"
        )

        if st.button("Guardar curvas"):
            st.session_state["df_oh_editado"] = df_editado
            doc_ref.set({"filas": df_editado.to_dict(orient="records")})
            st.success("Datos guardados")

        # filtrar datos válidos
        df_filtrado = df_editado[
            df_editado["X"].notna() & df_editado["Índice OH"].notna()
        ]


        if not df_filtrado.empty:
            # generar un checkbox para cada curva
            st.markdown("**Mostrar curvas**")
            claves_curvas = df_filtrado["Curva"].fillna("Sin curva").unique()
            col1, col2, col3, col4, col5 = st.columns(5)
            mostrar_curvas = {}

            for i, curva in enumerate(claves_curvas):
                col = [col1, col2, col3, col4, col5][i % 5]
                mostrar_curvas[curva] = col.checkbox(curva, value=True, key=f"mostrar_{curva}")

            # calcular valores automáticos de rango una vez definidos los checkboxes
            curvas_activas = [c for c, v in mostrar_curvas.items() if v]
            df_visibles = df_filtrado[df_filtrado["Curva"].isin(curvas_activas)]

            if not df_visibles.empty:
                x_min_default = df_visibles["X"].min()
                x_max_default = df_visibles["X"].max()
                y_min_default = df_visibles["Índice OH"].min()
                y_max_default = df_visibles["Índice OH"].max()
            else:
                x_min_default = 0.0
                x_max_default = 1000.0
                y_min_default = 0.0
                y_max_default = 100.0

            # controles de rango de ejes
            st.markdown("**Rango de los ejes**")
            colx1, colx2, coly1, coly2 = st.columns(4)

            rango_x_min = colx1.number_input("X mínimo", value=x_min_default, step=10.0, format="%.1f")
            rango_x_max = colx2.number_input("X máximo", value=x_max_default, step=10.0, format="%.1f")
            rango_y_min = coly1.number_input("Y mínimo", value=y_min_default, step=1.0, format="%.1f")
            rango_y_max = coly2.number_input("Y máximo", value=y_max_default, step=1.0, format="%.1f")

            # ahora si armar la figura
            fig_plotly = go.Figure()

            for curva, grupo in df_filtrado.groupby("Curva" if "Curva" in df_filtrado else ""):
                if not mostrar_curvas.get(curva, True):
                    continue
                grupo_ordenado = grupo.sort_values("X")
                fig_plotly.add_trace(
                    go.Scatter(
                        x=grupo_ordenado["X"],
                        y=grupo_ordenado["Índice OH"],
                        mode="lines+markers",
                        name=curva or "Sin curva",
                        hovertemplate="X=%{x:.2f}<br>Índice OH=%{y:.2f}<extra></extra>"
                    )
                )

            fig_plotly.update_layout(
                xaxis_title="tiempo",
                yaxis_title="Índice OH",
                xaxis=dict(range=[rango_x_min, rango_x_max]),
                yaxis=dict(range=[rango_y_min, rango_y_max]),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                height=600,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_plotly, use_container_width=True)

            # SEGUNDO gráfico (Plotly con ajuste Y, opcional)
            if st.checkbox("Ajuste manual Y"):
                fig_plotly2 = go.Figure()
                ajustes_y = {}

                st.markdown("**Ajustes Y individuales**")
                claves_curvas = df_filtrado["Curva"].fillna("Sin curva").unique()
                col1, col2, col3, col4, col5 = st.columns(5)

                for i, curva in enumerate(claves_curvas):
                    grupo = df_filtrado[df_filtrado["Curva"] == curva]
                    x0_match = grupo[grupo["X"] == 0.0]
                    if not x0_match.empty:
                        y0 = x0_match["Índice OH"].iloc[0]
                        valor_por_defecto = -y0
                    else:
                        valor_por_defecto = 0.0

                    col = [col1, col2, col3, col4, col5][i % 5]
                    ajustes_y[curva] = col.number_input(
                        curva, value=valor_por_defecto, step=0.1, format="%.2f", key=f"ajuste_y_{curva}"
                    )

                for curva, grupo in df_filtrado.groupby("Curva" if "Curva" in df_filtrado else ""):
                    if not mostrar_curvas.get(curva, True):
                       continue
                    grupo_ordenado = grupo.sort_values("X")
                    offset = ajustes_y.get(curva or "Sin curva", 0.0)
                    fig_plotly2.add_trace(
                        go.Scatter(
                            x=grupo_ordenado["X"],
                            y=grupo_ordenado["Índice OH"] + offset,
                            mode="lines+markers",
                            name=curva or "Sin curva",
                            hovertemplate="X=%{x:.2f}<br>Índice OH ajustado=%{y:.2f}<extra></extra>"
                        )
                    )

                y_min_segundo = -10.0  # piso fijo del eje Y
                fig_plotly2.update_layout(
                    xaxis_title="tiempo",
                    yaxis_title="Índice OH",
                    xaxis=dict(range=[rango_x_min, rango_x_max]),
                    yaxis=dict(range=[y_min_segundo, rango_y_max]),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    height=600,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                # línea discontinua negra en y = 0 a todo el ancho del gráfico
                fig_plotly2.add_shape(
                    type="line", xref="paper", yref="y",
                    x0=0, x1=1, y0=0, y1=0,
                    line=dict(color="black", dash="dash")
                )
                
                st.plotly_chart(fig_plotly2, use_container_width=True)

    # 4. Calculadora manual de Índice OH
    if st.checkbox("Calculadora manual de Índice OH espectroscópico", value=False):
        calculadora_indice_oh_manual()
