# --- Hoja 6: Análisis RMN ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import base64
import os

# --- Configuraciones globales ---
GRUPOS_FUNCIONALES = ["Formiato", "Cloroformo", "C=C olefínicos", "Glicerol medio", "Glicerol extremos", "Metil-Éster", "Eter", "Ester", "Ácido carboxílico", "OH", "Epóxido", "C=C", "Alfa-C=O", "Alfa-C=C", "Vecino a alfa-carbonilo", "Alfa-epóxido", "CH2", "CH3"]

def render_tab6(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    # --- Cargar muestras y espectros ---
    muestras = cargar_muestras(db)
    if not muestras:
        st.warning("No hay muestras disponibles.")
        st.stop()

    espectros = []
    for m in muestras:
        nombre = m["nombre"]
        docs = db.collection("muestras").document(nombre).collection("espectros").stream()
        for i, doc in enumerate(docs):
            e = doc.to_dict()
            tipo = (e.get("tipo") or "").upper()
            if "RMN" in tipo:
                espectros.append({
                    "muestra": nombre,
                    "tipo": tipo,
                    "archivo": e.get("nombre_archivo", "sin nombre"),
                    "contenido": e.get("contenido"),
                    "mascaras": e.get("mascaras", []),
                    "id": f"{nombre}__{i}"
                })

    df_total = pd.DataFrame(espectros)
    if df_total.empty:
        st.warning("No hay espectros RMN disponibles.")
        st.stop()

    muestras_sel = st.multiselect("Seleccionar muestras", sorted(df_total["muestra"].unique()))
    df_filtrado = df_total[df_total["muestra"].isin(muestras_sel)]

    opciones = [
        f"{row['muestra']} – {row['archivo']}" for _, row in df_filtrado.iterrows()
    ]
    ids_map = dict(zip(opciones, df_filtrado["id"]))
    seleccion = st.multiselect("Seleccionar espectros", opciones)

    df_sel = df_filtrado[df_filtrado["id"].isin([ids_map.get(s) for s in seleccion])]

    df_rmn1h = df_sel[df_sel["tipo"] == "RMN 1H"]
    if not df_rmn1h.empty:
        st.markdown("## 🧪 RMN 1H")
        render_rmn_plot(df_rmn1h, tipo="RMN 1H", key_sufijo="rmn1h", db=db)

    df_rmn13c = df_sel[df_sel["tipo"] == "RMN 13C"]
    if not df_rmn13c.empty:
        st.markdown("## 🧪 RMN 13C")
        render_rmn_plot(df_rmn13c, tipo="RMN 13C", key_sufijo="rmn13c", db=db)

    imagenes_sel = df_sel[df_sel["archivo"].str.lower().str.endswith((".png", ".jpg", ".jpeg"))]
    if not imagenes_sel.empty:
        st.markdown("## 🧪 RMN Imágenes")
        render_imagenes(imagenes_sel)

def render_rmn_plot(df, tipo="RMN 1H", key_sufijo="rmn1h", db=None):
    if df.empty:
        st.info(f"No hay espectros disponibles para {tipo}.")
        return

    # --- Filtros estilo FTIR ---
    col1, col2, col3, col4 = st.columns(4)
    normalizar = col1.checkbox("Normalizar intensidad", key=f"norm_{key_sufijo}")
    mostrar_picos = col2.checkbox("Mostrar picos detectados", key=f"picos_{key_sufijo}")
    restar_espectro = col3.checkbox("Restar espectro de fondo", key=f"resta_{key_sufijo}")
    ajuste_y_manual = col4.checkbox("Ajuste manual eje Y", key=f"ajuste_y_{key_sufijo}")

    ajustes_y = {}
    if ajuste_y_manual:
        st.markdown("#### Ajustes verticales por espectro")
        for _, row in df.iterrows():
            clave = row["archivo"]
            ajustes_y[clave] = st.number_input(f"Ajuste Y para {clave}", value=0.0, step=0.1, key=f"ajuste_val_{clave}")
    else:
        for _, row in df.iterrows():
            ajustes_y[row["archivo"]] = 0.0

    seleccion_resta = None
    if restar_espectro:
        opciones_restar = [f"{row['muestra']} – {row['archivo']}" for _, row in df.iterrows()]
        seleccion_resta = st.selectbox("Seleccionar espectro a restar:", opciones_restar, key=f"sel_resta_{key_sufijo}")

    # --- Rango de visualización ---
    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X mínimo", value=0.0, key=f"x_min_{key_sufijo}")
    x_max = colx2.number_input("X máximo", value=9.0 if tipo == "RMN 1H" else 200.0, key=f"x_max_{key_sufijo}")
    y_min = coly1.number_input("Y mínimo", value=0.0, key=f"y_min_{key_sufijo}")
    y_max = coly2.number_input("Y máximo", value=80.0 if tipo == "RMN 1H" else 1.5, key=f"y_max_{key_sufijo}")

    # --- Decodificar espectro de fondo si aplica ---
    espectro_resta = None
    if restar_espectro and seleccion_resta:
        id_resta = seleccion_resta.split(" – ")[-1].strip()
        fila_resta = df[df["archivo"] == id_resta].iloc[0] if id_resta in set(df["archivo"]) else None
        if fila_resta is not None:
            try:
                espectro_resta = decodificar_csv_o_excel(fila_resta["contenido"], fila_resta["archivo"])
                if espectro_resta is not None:
                    espectro_resta.columns = ["x", "y"]
                    espectro_resta.dropna(inplace=True)
            except:
                espectro_resta = None
                espectro_resta.columns = ["x", "y"]
                espectro_resta.dropna(inplace=True)

    # --- Parámetros de picos ---
    if mostrar_picos:
        colp1, colp2 = st.columns(2)
        altura_min = colp1.number_input("Altura mínima", value=0.05, step=0.01, key=f"altura_min_{key_sufijo}")
        distancia_min = colp2.number_input("Distancia mínima entre picos", value=5, step=1, key=f"distancia_min_{key_sufijo}")

    # --- Sección reorganizada: Checkboxes de tablas y sombreado ---
    col_tabla, col_sombra = st.columns(2)

    with col_tabla:
        nombre_tabla_dt2 = f"🧮 Tabla de Cálculos D/T2 (FAMAF) {tipo}"
        mostrar_tabla_dt2 = st.checkbox(nombre_tabla_dt2, value=False, key=f"mostrar_dt2_{key_sufijo}")

        nombre_tabla_senales = f"📈 Tabla de Cálculos {tipo}"
        mostrar_tabla_senales = st.checkbox(nombre_tabla_senales, value=False, key=f"mostrar_senales_{key_sufijo}")

        nombre_tabla_biblio = f"📚 Tabla Bibliográfica {tipo[-3:]}"  # 1H o 13C
        mostrar_tabla_biblio = st.checkbox(nombre_tabla_biblio, value=False, key=f"mostrar_biblio_{tipo.lower()}_{key_sufijo}")

    with col_sombra:
        nombre_sombra_dt2 = f"Sombrear Tabla de Cálculos D/T2 (FAMAF) {tipo}"
        aplicar_sombra_dt2 = st.checkbox(nombre_sombra_dt2, value=False, key=f"sombra_dt2_{key_sufijo}")

        nombre_sombra_senales = f"Sombrear Tabla de Cálculos {tipo}"
        aplicar_sombra_senales = st.checkbox(nombre_sombra_senales, value=False, key=f"sombra_senales_{key_sufijo}")

        nombre_sombra_biblio = f"Sombrear Tabla Bibliográfica {tipo[-3:]}"
        aplicar_sombra_biblio = st.checkbox(nombre_sombra_biblio, value=False, key=f"sombra_biblio_{key_sufijo}")

# --- Tabla de Cálculo D/T2 ---
    if mostrar_tabla_dt2:
        columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                         "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]

        filas_guardadas = []
        for _, row in df.iterrows():
            muestra = row["muestra"]
            archivo = row["archivo"]
            doc = db.collection("muestras").document(muestra).collection("dt2").document(tipo.lower())
            data = doc.get().to_dict()
            if data and "filas" in data:
                filas_guardadas.extend([f for f in data["filas"] if f.get("Archivo") == archivo])

        df_dt2 = pd.DataFrame(filas_guardadas)
        for col in columnas_dt2:
            if col not in df_dt2.columns:
                df_dt2[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
        df_dt2 = df_dt2[columnas_dt2]

        ### Cálculo D/T2"
        st.markdown("**🧮 Tabla de Cálculos D/T2 (FAMAF)**")
        with st.form(f"form_dt2_{key_sufijo}"):
            df_dt2_edit = st.data_editor(
                df_dt2,
                column_config={
                    "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
                    "δ pico": st.column_config.NumberColumn(format="%.2f"),
                    "X min": st.column_config.NumberColumn(format="%.2f"),
                    "X max": st.column_config.NumberColumn(format="%.2f"),
                    "Área": st.column_config.NumberColumn(format="%.2f", label="🔴Área", disabled=True),
                    "D": st.column_config.NumberColumn(format="%.2e"),
                    "T2": st.column_config.NumberColumn(format="%.3f"),
                    "Xas min": st.column_config.NumberColumn(format="%.2f"),
                    "Xas max": st.column_config.NumberColumn(format="%.2f"),
                    "Has": st.column_config.NumberColumn(format="%.2f"),
                    "Área as": st.column_config.NumberColumn(format="%.2f", label="🔴Área as", disabled=True),
                    "H": st.column_config.NumberColumn(format="%.2f", label="🔴H" if tipo == "RMN 1H" else "🔴C", disabled=True),
                    "Observaciones": st.column_config.TextColumn(),
                    "Archivo": st.column_config.TextColumn(disabled=True),
                    "Muestra": st.column_config.TextColumn(disabled=True),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",
                key=f"tabla_dt2_{key_sufijo}"
            )
            etiqueta_boton_dt2 = "🔴 Recalcular 'Área', 'Área as' y 'H'" if tipo == "RMN 1H" else "🔴 Recalcular 'Área', 'Área as' y 'C'"
            recalcular = st.form_submit_button(etiqueta_boton_dt2)

        if recalcular:
            for i, row in df_dt2_edit.iterrows():
                try:
                    muestra = row["Muestra"]
                    archivo = row["Archivo"]
                    x_min = float(row["X min"])
                    x_max = float(row["X max"])
                    xas_min = float(row["Xas min"]) if row["Xas min"] not in [None, ""] else None
                    xas_max = float(row["Xas max"]) if row["Xas max"] not in [None, ""] else None
                    has = float(row["Has"]) if row["Has"] not in [None, ""] else None

                    espectros = db.collection("muestras").document(muestra).collection("espectros").stream()
                    espectro = next((e.to_dict() for e in espectros if e.to_dict().get("nombre_archivo") == archivo), None)
                    if not espectro:
                        continue

                    contenido = BytesIO(base64.b64decode(espectro["contenido"]))
                    extension = os.path.splitext(archivo)[1].lower()
                    if extension == ".xlsx":
                        df_esp = pd.read_excel(contenido)
                    else:
                        for sep in [",", ";", "	", " "]:
                            contenido.seek(0)
                            try:
                                df_esp = pd.read_csv(contenido, sep=sep)
                                if df_esp.shape[1] >= 2:
                                    break
                            except:
                                continue
                        else:
                            continue

                    col_x, col_y = df_esp.columns[:2]
                    df_esp[col_x] = pd.to_numeric(df_esp[col_x], errors="coerce")
                    df_esp[col_y] = pd.to_numeric(df_esp[col_y], errors="coerce")
                    df_esp = df_esp.dropna()

                    df_main = df_esp[(df_esp[col_x] >= min(x_min, x_max)) & (df_esp[col_x] <= max(x_min, x_max))]
                    area = np.trapz(df_main[col_y], df_main[col_x]) if not df_main.empty else None
                    df_dt2_edit.at[i, "Área"] = round(area, 2) if area else None

                    if xas_min is not None and xas_max is not None:
                        df_as = df_esp[(df_esp[col_x] >= min(xas_min, xas_max)) & (df_esp[col_x] <= max(xas_min, xas_max))]
                        area_as = np.trapz(df_as[col_y], df_as[col_x]) if not df_as.empty else None
                        df_dt2_edit.at[i, "Área as"] = round(area_as, 2) if area_as else None

                        if area and area_as and has and area_as != 0:
                            h_calc = (area * has) / area_as
                            df_dt2_edit.at[i, "H"] = round(h_calc, 2)

                except Exception as e:
                    st.warning(f"⚠️ Error en fila {i}: {e}")

            filas_actualizadas = df_dt2_edit.to_dict(orient="records")
            for muestra in df_dt2_edit["Muestra"].unique():
                filas_m = [f for f in filas_actualizadas if f["Muestra"] == muestra]
                doc = db.collection("muestras").document(muestra).collection("dt2").document(tipo.lower())
                doc.set({"filas": filas_m})

    # --- Tabla de Cálculo de señales ---
    if mostrar_tabla_senales:
        columnas_senales = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2", "Xas min", "Xas max", "Cas", "Área as", "C", "Observaciones", "Archivo"]
        tipo_doc = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
        doc_ref = db.collection("tablas_integrales").document(tipo_doc)
        if not doc_ref.get().exists:
            doc_ref.set({"filas": []})

        filas_guardadas = doc_ref.get().to_dict().get("filas", [])
        combinaciones = {(row["muestra"], row["archivo"]) for _, row in df.iterrows()}
        filas_activas = [f for f in filas_guardadas if (f.get("Muestra"), f.get("Archivo")) in combinaciones]

        df_senales = pd.DataFrame(filas_activas)
        for col in columnas_senales:
            if col not in df_senales.columns:
                df_senales[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
        df_senales = df_senales[columnas_senales]

        ### Cálculo de señales"
        st.markdown("**📈 Tabla de Cálculos**")
        with st.form(f"form_senales_{key_sufijo}"):
            df_senales_edit = st.data_editor(
                df_senales,
                column_config={
                    "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
                    "δ pico": st.column_config.NumberColumn(format="%.2f"),
                    "X min": st.column_config.NumberColumn(format="%.2f"),
                    "X max": st.column_config.NumberColumn(format="%.2f"),
                    "Área": st.column_config.NumberColumn(format="%.2f", label="🔴Área", disabled=True),
                    "D": st.column_config.NumberColumn(format="%.2e"),
                    "T2": st.column_config.NumberColumn(format="%.3f"),
                    "Xas min": st.column_config.NumberColumn(format="%.2f"),
                    "Xas max": st.column_config.NumberColumn(format="%.2f"),
                    "Cas": st.column_config.NumberColumn(format="%.2f"),
                    "Área as": st.column_config.NumberColumn(format="%.2f", label="🔴Área as", disabled=True),
                    "C": st.column_config.NumberColumn(format="%.2f", label="🔴H" if tipo == "RMN 1H" else "🔴C",disabled=True                    ),
                    "Observaciones": st.column_config.TextColumn(),
                    "Archivo": st.column_config.TextColumn(disabled=True),
                    "Muestra": st.column_config.TextColumn(disabled=True),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",
                key=f"tabla_senales_{key_sufijo}"
            )
            texto_boton = "🔴 Recalcular 'Área', 'Área as' y 'H'" if tipo == "RMN 1H" else "🔴 Recalcular 'Área', 'Área as' y 'C'"
            recalcular = st.form_submit_button(texto_boton)

        if recalcular:
            for i, row in df_senales_edit.iterrows():
                try:
                    muestra = row["Muestra"]
                    archivo = row["Archivo"]
                    x_min = float(row["X min"])
                    x_max = float(row["X max"])
                    xas_min = float(row["Xas min"]) if row["Xas min"] not in [None, ""] else None
                    xas_max = float(row["Xas max"]) if row["Xas max"] not in [None, ""] else None
                    has = float(row["Has"]) if row["Has"] not in [None, ""] else None

                    espectros = db.collection("muestras").document(muestra).collection("espectros").stream()
                    espectro = next((e.to_dict() for e in espectros if e.to_dict().get("nombre_archivo") == archivo), None)
                    if not espectro:
                        continue

                    contenido = BytesIO(base64.b64decode(espectro["contenido"]))
                    extension = os.path.splitext(archivo)[1].lower()
                    if extension == ".xlsx":
                        df_esp = pd.read_excel(contenido)
                    else:
                        for sep in [",", ";", "	", " "]:
                            contenido.seek(0)
                            try:
                                df_esp = pd.read_csv(contenido, sep=sep)
                                if df_esp.shape[1] >= 2:
                                    break
                            except:
                                continue
                        else:
                            continue

                    col_x, col_y = df_esp.columns[:2]
                    df_esp[col_x] = pd.to_numeric(df_esp[col_x], errors="coerce")
                    df_esp[col_y] = pd.to_numeric(df_esp[col_y], errors="coerce")
                    df_esp = df_esp.dropna()

                    df_main = df_esp[(df_esp[col_x] >= min(x_min, x_max)) & (df_esp[col_x] <= max(x_min, x_max))]
                    area = np.trapz(df_main[col_y], df_main[col_x]) if not df_main.empty else None
                    df_senales_edit.at[i, "Área"] = round(area, 2) if area else None

                    if xas_min is not None and xas_max is not None:
                        df_as = df_esp[(df_esp[col_x] >= min(xas_min, xas_max)) & (df_esp[col_x] <= max(xas_min, xas_max))]
                        area_as = np.trapz(df_as[col_y], df_as[col_x]) if not df_as.empty else None
                        df_senales_edit.at[i, "Área as"] = round(area_as, 2) if area_as else None

                        if area and area_as and has and area_as != 0:
                            h_calc = (area * has) / area_as
                            df_senales_edit.at[i, "H"] = round(h_calc, 2)

                except Exception as e:
                    st.warning(f"⚠️ Error en fila {i}: {e}")

            doc_ref.set({"filas": df_senales_edit.to_dict(orient="records")})
            st.success("✅ Datos recalculados y guardados correctamente.")
            st.rerun()

    # --- Tabla Bibliográfica de señales pico δ (RMN 1H o RMN 13C) ---
    if mostrar_tabla_biblio:
        doc_biblio = db.collection("configuracion_global").document("tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c")
        if not doc_biblio.get().exists:
            doc_biblio.set({"filas": []})

        filas_biblio = doc_biblio.get().to_dict().get("filas", [])
        columnas_biblio = ["Grupo funcional", "X min", "δ pico", "X max", "Tipo de muestra", "Observaciones"]
        df_biblio = pd.DataFrame(filas_biblio)

        for col in columnas_biblio:
            if col not in df_biblio.columns:
                df_biblio[col] = "" if col in ["Grupo funcional", "Tipo de muestra", "Observaciones"] else None
        df_biblio = df_biblio[columnas_biblio]
        st.markdown(f"**📚 Tabla Bibliográfica {tipo[-3:]}**")
        df_biblio_edit = st.data_editor(
            df_biblio,
            column_config={
                "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
                "X min": st.column_config.NumberColumn(format="%.2f"),
                "δ pico": st.column_config.NumberColumn(format="%.2f"),
                "X max": st.column_config.NumberColumn(format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            key=f"tabla_biblio_{key_sufijo}"
        )

        colb1, colb2 = st.columns([1, 1])
        with colb1:
            if st.button(f"🔴 Actualizar Tabla Bibliográfica {tipo[-3:]}"):
                doc_biblio.set({"filas": df_biblio_edit.to_dict(orient="records")})
                st.success("✅ Datos bibliográficos actualizados.")
        with colb2:
            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                df_biblio_edit.to_excel(writer, index=False, sheet_name=f"Bibliografía {tipo[-3:]}")
            buffer_excel.seek(0)
            st.download_button(
                f"📥 Descargar tabla {tipo[-3:]}",
                data=buffer_excel.getvalue(),
                file_name=f"tabla_bibliografica_rmn{tipo[-3:].lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # --- Sombreados por D/T2 ---
    check_d_por_espectro = {}
    check_t2_por_espectro = {}

    if aplicar_sombra_dt2:
        st.markdown("**Espectros para aplicar Sombreados por D/T2**")
        for _, row in df.iterrows():
            archivo = row["archivo"]
            col_d, col_t2 = st.columns([1, 1])
            check_d_por_espectro[archivo] = col_d.checkbox(f"D – {archivo}", key=f"chk_d_{archivo}_{key_sufijo}")
            check_t2_por_espectro[archivo] = col_t2.checkbox(f"T2 – {archivo}", key=f"chk_t2_{archivo}_{key_sufijo}")


# --- Trazado ---
    fig = go.Figure()
    for _, row in df.iterrows():
        archivo_actual = row["archivo"]
        muestra_actual = row["muestra"]
        df_esp = decodificar_csv_o_excel(row["contenido"], row["archivo"])
        if df_esp is not None:
            col_x, col_y = df_esp.columns[:2]
            y_data = df_esp[col_y].copy()
            y_data = y_data + ajustes_y.get(row["archivo"], 0.0)
            if espectro_resta is not None:
                df_esp = df_esp.rename(columns={col_x: "x", col_y: "y"}).dropna()
                espectro_resta_interp = np.interp(df_esp["x"], espectro_resta["x"], espectro_resta["y"])
                y_data = df_esp["y"] - espectro_resta_interp
            if normalizar:
                y_data = y_data + ajustes_y.get(row["archivo"], 0.0)
            if normalizar:
                y_data = y_data / y_data.max() if y_data.max() != 0 else y_data
            x_vals = df_esp["x"] if "x" in df_esp.columns else df_esp[col_x]
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_data,
                mode='lines',
                name=row["archivo"]
            ))

            # Detección de picos
            if mostrar_picos:
                from scipy.signal import find_peaks
                try:
                    peaks, _ = find_peaks(y_data, height=altura_min, distance=distancia_min)
                    for p in peaks:
                        fig.add_trace(go.Scatter(
                            x=[x_vals.iloc[p]],
                            y=[y_data.iloc[p]],
                            mode="markers+text",
                            marker=dict(color="black", size=6),
                            text=[f"{x_vals.iloc[p]:.2f}"],
                            textposition="top center",
                            showlegend=False
                        ))
                except:
                    st.warning(f"⚠️ No se pudieron detectar picos en {row['archivo']}.")

    fig.update_layout(
        xaxis_title="[ppm]",
        yaxis_title="Intensidad",
        xaxis=dict(range=[x_max, x_min]),
        yaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None),
        template="simple_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )


    # Sombreado D/T2 (1H o 13C si hay datos)
    if aplicar_sombra_dt2:
        doc_dt2 = db.collection("muestras").document(muestra_actual).collection("dt2").document("datos")
        if doc_dt2.get().exists:
            filas_dt2 = doc_dt2.get().to_dict().get("filas", [])
            for f in filas_dt2:
                if f.get("Archivo") == archivo_actual:
                    if check_d_por_espectro.get(archivo_actual) and f.get("X min") and f.get("X max"):
                        fig_indiv.add_vrect(
                            x0=min(f["X min"], f["X max"]),
                            x1=max(f["X min"], f["X max"]),
                            fillcolor="rgba(255,0,0,0.1)",
                            line_width=0,
                            annotation_text="D",
                            annotation_position="top left"
                        )
                    if check_t2_por_espectro.get(archivo_actual) and f.get("Xas min") and f.get("Xas max"):
                        fig_indiv.add_vrect(
                            x0=min(f["Xas min"], f["Xas max"]),
                            x1=max(f["Xas min"], f["Xas max"]),
                            fillcolor="rgba(0,0,255,0.1)",
                            line_width=0,
                            annotation_text="T2",
                            annotation_position="top right"
                        )


           # Aplicar sombreado por Cálculo de señales si está activo
        if aplicar_sombra_senales:
            tipo_doc_senales = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
            doc_senales = db.collection("tablas_integrales").document(tipo_doc_senales)
            if doc_senales.get().exists:
                filas_senales = doc_senales.get().to_dict().get("filas", [])
                for f in filas_senales:
                    if f.get("Archivo") == archivo_actual:
                        x1 = f.get("X min")
                        x2 = f.get("X max")
                        if x1 is not None and x2 is not None:
                            fig.add_vrect(
                                x0=min(x1, x2),
                                x1=max(x1, x2),
                                fillcolor="rgba(0,255,0,0.1)",
                                layer="below",
                                line_width=0,
                                annotation_text=f.get("δ pico", ""),
                                annotation_position="top"
                            )

        # Aplicar sombreado por bibliografía si está activo
        if aplicar_sombra_biblio:
            doc_biblio = db.collection("configuracion_global").document("tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c")
            if doc_biblio.get().exists:
                filas_biblio = doc_biblio.get().to_dict().get("filas", [])
                for f in filas_biblio:
                    delta = f.get("δ pico")
                    if delta is not None:
                        (fig if 'fig' in locals() else fig_indiv).add_vline(
                            x=delta,
                            line=dict(color="black", dash="dot"),
                            annotation_text=f"δ = {delta:.2f}",
                            annotation_position="top right"
                        )


    st.plotly_chart(fig, use_container_width=True)

    # --- Gráficos individuales con sombreados ---
    mostrar_indiv = st.checkbox("Gráficos individuales", key=f"chk_indiv_{key_sufijo}")
    if mostrar_indiv:
        for _, row in df.iterrows():
            archivo_actual = row["archivo"]
            muestra_actual = row["muestra"]
            df_esp = decodificar_csv_o_excel(row["contenido"], row["archivo"])
            if df_esp is None:
                continue

            col_x, col_y = df_esp.columns[:2]
            y_data = df_esp[col_y].copy() + ajustes_y.get(archivo_actual, 0.0)
            if normalizar:
                y_data = y_data / y_data.max() if y_data.max() != 0 else y_data
            x_vals = df_esp[col_x]

            fig_indiv = go.Figure()
            fig_indiv.add_trace(go.Scatter(x=x_vals, y=y_data, mode='lines', name=archivo_actual))

            # Sombreado D/T2 (1H o 13C si hay datos)
            if aplicar_sombra_dt2:
                doc_dt2 = db.collection("muestras").document(muestra_actual).collection("dt2").document("datos")
                if doc_dt2.get().exists:
                    filas_dt2 = doc_dt2.get().to_dict().get("filas", [])
                    for f in filas_dt2:
                        if f.get("Archivo") == archivo_actual:
                            if check_d_por_espectro.get(archivo_actual) and f.get("X min") and f.get("X max"):
                                fig_indiv.add_vrect(
                                    x0=min(f["X min"], f["X max"]),
                                    x1=max(f["X min"], f["X max"]),
                                    fillcolor="rgba(255,0,0,0.1)",
                                    line_width=0,
                                    annotation_text="D",
                                    annotation_position="top left"
                                )
                            if check_t2_por_espectro.get(archivo_actual) and f.get("Xas min") and f.get("Xas max"):
                                fig_indiv.add_vrect(
                                    x0=min(f["Xas min"], f["Xas max"]),
                                    x1=max(f["Xas min"], f["Xas max"]),
                                    fillcolor="rgba(0,0,255,0.1)",
                                    line_width=0,
                                    annotation_text="T2",
                                    annotation_position="top right"
                                )

            # Sombreado por señales
            if aplicar_sombra_senales:
                tipo_doc_senales = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
                doc_senales = db.collection("tablas_integrales").document(tipo_doc_senales)
                if doc_senales.get().exists:
                    filas_senales = doc_senales.get().to_dict().get("filas", [])
                    for f in filas_senales:
                        if f.get("Archivo") == archivo_actual:
                            x1 = f.get("X min")
                            x2 = f.get("X max")
                            if x1 is not None and x2 is not None:
                                fig_indiv.add_vrect(
                                    x0=min(x1, x2),
                                    x1=max(x1, x2),
                                    fillcolor="rgba(0,255,0,0.1)", line_width=0,
                                    annotation_text=f.get("δ pico", ""), annotation_position="top"
                                )

            # Aplicar sombreado por bibliografía si está activo
            if aplicar_sombra_biblio:
                doc_biblio = db.collection("configuracion_global").document("tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c")
                if doc_biblio.get().exists:
                    filas_biblio = doc_biblio.get().to_dict().get("filas", [])
                    for f in filas_biblio:
                        delta = f.get("δ pico")
                        if delta is not None:
                            (fig if 'fig' in locals() else fig_indiv).add_vline(
                                x=delta,
                                line=dict(color="black", dash="dot"),
                                annotation_text=f"δ = {delta:.2f}",
                                annotation_position="top right"
                            )


            fig_indiv.update_layout(
                title=f"{archivo_actual}",
                xaxis_title="[ppm]",
                yaxis_title="Intensidad",
                xaxis=dict(range=[x_max, x_min]),
                yaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None),
                height=400,
                template="simple_white"
            )

            st.plotly_chart(fig_indiv, use_container_width=True)


def decodificar_csv_o_excel(contenido_base64, archivo):
    try:
        contenido = BytesIO(base64.b64decode(contenido_base64))
        ext = os.path.splitext(archivo)[1].lower()
        if ext == ".xlsx":
            return pd.read_excel(contenido)
        else:
            for sep in [",", ";", "\t", " "]:
                contenido.seek(0)
                try:
                    df = pd.read_csv(contenido, sep=sep)
                    if df.shape[1] >= 2:
                        return df
                except:
                    continue
    except Exception as e:
        st.warning(f"Error al decodificar {archivo}: {e}")
    return None

def render_imagenes(df):
    #st.markdown("## 🧪 RMN Imágenes")
    imagenes_disponibles = df[df["archivo"].str.lower().str.endswith((".png", ".jpg", ".jpeg"))]

    if imagenes_disponibles.empty:
        st.info("No hay imágenes seleccionadas.")
    else:
        for _, row in imagenes_disponibles.iterrows():
            st.markdown(f"**{row['archivo']}** – {row['muestra']}")
            try:
                from PIL import Image
                from io import BytesIO
                import base64

                image_data = BytesIO(base64.b64decode(row["contenido"]))
                image = Image.open(image_data)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"❌ No se pudo mostrar la imagen: {e}")


















    def buscar_tablas_ocultas_en_muestras(db):
        st.markdown("### 🔍 Búsqueda de tablas ocultas en Firestore")
        muestras = db.collection("muestras").stream()

        for m in muestras:
            nombre_muestra = m.id
            st.markdown(f"#### 📁 Muestra: `{nombre_muestra}`")
            encontrado = False

            subcolecciones = db.collection("muestras").document(nombre_muestra).collections()
            for subcol in subcolecciones:
                nombre_subcol = subcol.id
                documentos = subcol.stream()
                for doc in documentos:
                    datos = doc.to_dict()
                    if "filas" in datos and isinstance(datos["filas"], list):
                        st.success(f"✅ {nombre_subcol}/{doc.id} → {len(datos['filas'])} filas")
                        encontrado = True
                    else:
                        # Detección alternativa
                        if any(isinstance(v, list) and all(isinstance(x, dict) for x in v) for v in datos.values()):
                            st.warning(f"⚠️ {nombre_subcol}/{doc.id} contiene listas tipo tabla (sin clave 'filas')")
                            encontrado = True

            if not encontrado:
                st.info("🕵️ Sin tablas encontradas en esta muestra.")
            st.markdown("---")


    # 👇 Añadir esto en la interfaz principal donde quieras mostrar la opción
    if st.checkbox("🔎 Buscar datos ocultos en Firebase", value=False):
        buscar_tablas_ocultas_en_muestras(db)
