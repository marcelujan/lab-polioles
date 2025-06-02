# --- Hoja 6: Análisis RMN ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import base64
import os

# --- Configuraciones globales ---
GRUPOS_FUNCIONALES = ["Formiato", "Cloroformo", "C=C olefínicos", "Glicerol medio", "Glicerol extremos", "Metil-Éster", "Eter", "Ester", "Ácido carboxílico", "OH", "Epóxido", "C=C", "Alfa-C=O","Alfa-C-OH", "Alfa-C=C", "Vecino a alfa-carbonilo", "Alfa-epóxido", "CH2", "CH3"]

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

    superposicion_vertical = st.checkbox("📊 Superposición vertical de espectros", key=f"offset_{key_sufijo}")

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

        nombre_sombra_biblio = f"Delinear Tabla Bibliográfica {tipo[-3:]}"
        aplicar_sombra_biblio = st.checkbox(nombre_sombra_biblio, value=False, key=f"sombra_biblio_{key_sufijo}")

    # --- Tabla de Cálculo D/T2 ---
    if mostrar_tabla_dt2:
        if tipo == "RMN 1H":
            columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                            "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]
        else:
            columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                            "Xas min", "Xas max", "Cas", "Área as", "C", "Observaciones", "Archivo"]

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
        df_dt2 = df_dt2.sort_values(by=["Archivo", "X max"])

        if df_dt2.empty:
            st.warning("⚠️ La tabla D/T2 está vacía. Seleccioná una muestra y archivo para crear una fila inicial.")

            muestras_disponibles = sorted(set(df["muestra"]))
            archivos_disponibles = sorted(set(df["archivo"]))

            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                muestra_nueva = st.selectbox("📌 Muestra", muestras_disponibles, key="muestra_nueva_dt2")
            with col2:
                archivo_nuevo = st.selectbox("📁 Archivo", archivos_disponibles, key="archivo_nuevo_dt2")
            with col3:
                if st.button("➕ Crear fila inicial D/T2"):
                    fila_vacia = {col: None for col in columnas_dt2}
                    fila_vacia["Muestra"] = muestra_nueva
                    fila_vacia["Archivo"] = archivo_nuevo
                    df_dt2 = pd.DataFrame([fila_vacia])

        ### Cálculo D/T2"
        st.markdown("**🧮 Tabla de Cálculos D/T2 (FAMAF)**")
        with st.form(f"form_dt2_{key_sufijo}"):
            col_config = {
                "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
                "δ pico": st.column_config.NumberColumn(format="%.2f"),
                "X min": st.column_config.NumberColumn(format="%.2f"),
                "X max": st.column_config.NumberColumn(format="%.2f"),
                "Área": st.column_config.NumberColumn(format="%.2f", label="🔴Área", disabled=True),
                "D": st.column_config.NumberColumn(format="%.2e"),
                "T2": st.column_config.NumberColumn(format="%.3f"),
                "Xas min": st.column_config.NumberColumn(format="%.2f"),
                "Xas max": st.column_config.NumberColumn(format="%.2f"),
                "Área as": st.column_config.NumberColumn(format="%.2f", label="🔴Área as", disabled=True),
                "Observaciones": st.column_config.TextColumn(),
                "Archivo": st.column_config.TextColumn(disabled=False),
                "Muestra": st.column_config.TextColumn(disabled=False),
            }

            if tipo == "RMN 1H":
                col_config["Has"] = st.column_config.NumberColumn(format="%.2f")
                col_config["H"] = st.column_config.NumberColumn(format="%.2f", label="🔴H", disabled=True)
            else:
                col_config["Cas"] = st.column_config.NumberColumn(format="%.2f")
                col_config["C"] = st.column_config.NumberColumn(format="%.2f", label="🔴C", disabled=True)

            df_dt2_edit = st.data_editor(
                df_dt2,
                column_config=col_config,
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

                    has_o_cas_key = "Has" if tipo == "RMN 1H" else "Cas"
                    h_o_c_key = "H" if tipo == "RMN 1H" else "C"
                    has_or_cas = float(row[has_o_cas_key]) if row[has_o_cas_key] not in [None, ""] else None

                    # Buscar espectro correspondiente
                    espectros = db.collection("muestras").document(muestra).collection("espectros").stream()
                    espectro = next((e.to_dict() for e in espectros if e.to_dict().get("nombre_archivo") == archivo), None)
                    if not espectro:
                        continue

                    contenido = BytesIO(base64.b64decode(espectro["contenido"]))
                    extension = os.path.splitext(archivo)[1].lower()
                    if extension == ".xlsx":
                        df_esp = pd.read_excel(contenido)
                    else:
                        for sep in [",", ";", "\t", " "]:
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

                        if area and area_as and has_or_cas and area_as != 0:
                            resultado = (area * has_or_cas) / area_as
                            df_dt2_edit.at[i, h_o_c_key] = round(resultado, 2)

                except Exception as e:
                    st.warning(f"⚠️ Error en fila {i}: {e}")

            filas_actualizadas = df_dt2_edit.to_dict(orient="records")

            for muestra in df_dt2_edit["Muestra"].unique():
                filas_m = [f for f in filas_actualizadas if f["Muestra"] == muestra]

                doc = db.collection("muestras").document(muestra).collection("dt2").document(tipo.lower())

                # Recuperar las filas previas (guardadas) si existen
                filas_previas = []
                doc_data = doc.get().to_dict()
                if doc_data and "filas" in doc_data:
                    filas_previas = doc_data["filas"]

                # Determinar combinaciones activas a actualizar (por archivo)
                archivos_actualizados = set(f["Archivo"] for f in filas_m if f.get("Archivo"))

                # Conservar filas previas que NO estén siendo actualizadas
                filas_conservadas = [
                    f for f in filas_previas
                    if f.get("Archivo") not in archivos_actualizados
                ]

                # Guardar combinación: conservadas + nuevas
                filas_finales = filas_conservadas + filas_m
                doc.set({"filas": filas_finales})

            st.success("✅ Datos D/T2 recalculados y guardados correctamente.")
            st.rerun()

    # --- Tabla de Cálculo de señales (versión con fila por defecto) ---
    if mostrar_tabla_senales:
        if tipo == "RMN 1H":
            columnas_senales = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                                "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]
        else:
            columnas_senales = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                                "Xas min", "Xas max", "Cas", "Área as", "C", "Observaciones", "Archivo"]

        tipo_doc = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
        doc_ref = db.collection("tablas_integrales").document(tipo_doc)
        doc_data = doc_ref.get().to_dict() or {}
        filas_guardadas = doc_data.get("filas", [])

        # Combos reales de muestra + archivo activos
        combinaciones_real = {(row["muestra"], row["archivo"]) for _, row in df.iterrows()}

        # Agrupar filas existentes por combinacion
        agrupadas = {}
        for fila in filas_guardadas:
            clave = (fila.get("Muestra"), fila.get("Archivo"))
            if clave in combinaciones_real:
                agrupadas.setdefault(clave, []).append(fila)

        # Generar una fila vacía por combinación activa si no hay ninguna fila
        filas_totales = []
        for (m, a) in combinaciones_real:
            filas = agrupadas.get((m, a), [])
            if not filas:
                fila_vacia = {col: None for col in columnas_senales}
                fila_vacia["Muestra"] = m
                fila_vacia["Archivo"] = a
                filas = [fila_vacia]
            filas_totales.extend(filas)

            # Solo agregamos nueva fila si NO hay ninguna vacía ya presente
            import math
            hay_fila_vacia = any(
                all(
                    f.get(campo) in [None, ""] or (isinstance(f.get(campo), float) and math.isnan(f.get(campo)))
                    for campo in ["δ pico", "X min", "X max"]
                )
                for f in filas
            )

            if not hay_fila_vacia:
                nueva = {col: None for col in columnas_senales}
                nueva["Muestra"] = m
                nueva["Archivo"] = a
                filas_totales.append(nueva)

        df_senales = pd.DataFrame(filas_totales)
        for col in columnas_senales:
            if col not in df_senales.columns:
                df_senales[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
        df_senales = df_senales[columnas_senales]
        df_senales = df_senales.sort_values(by=["Archivo", "X max"])


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
                    "Área as": st.column_config.NumberColumn(format="%.2f", label="🔴Área as", disabled=True),
                    "C" if tipo != "RMN 1H" else "H": st.column_config.NumberColumn(format="%.2f", label="🔴H" if tipo == "RMN 1H" else "🔴C", disabled=True),
                    "Cas" if tipo != "RMN 1H" else "Has": st.column_config.NumberColumn(format="%.2f"),
                    "Observaciones": st.column_config.TextColumn(),
                    "Archivo": st.column_config.TextColumn(disabled=False),
                    "Muestra": st.column_config.TextColumn(disabled=False),
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
                    if not row.get("Muestra") or not row.get("Archivo") or row.get("X min") in [None, ""] or row.get("X max") in [None, ""]:
                        continue  # ignorar filas vacías o mal definidas
                    muestra = row["Muestra"]
                    archivo = row["Archivo"]
                    x_min = float(row["X min"])
                    x_max = float(row["X max"])
                    xas_min = float(row["Xas min"]) if row["Xas min"] not in [None, ""] else None
                    xas_max = float(row["Xas max"]) if row["Xas max"] not in [None, ""] else None

                    has_o_cas_key = "Has" if tipo == "RMN 1H" else "Cas"
                    h_o_c_key = "H" if tipo == "RMN 1H" else "C"
                    has_or_cas = float(row[has_o_cas_key]) if row[has_o_cas_key] not in [None, ""] else None

                    # Buscar espectro correspondiente
                    espectros = db.collection("muestras").document(muestra).collection("espectros").stream()
                    espectro = next((e.to_dict() for e in espectros if e.to_dict().get("nombre_archivo") == archivo), None)
                    if not espectro:
                        continue

                    contenido = BytesIO(base64.b64decode(espectro["contenido"]))
                    extension = os.path.splitext(archivo)[1].lower()
                    if extension == ".xlsx":
                        df_esp = pd.read_excel(contenido)
                    else:
                        for sep in [",", ";", "\t", " "]:
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

                        if area and area_as and has_or_cas and area_as != 0:
                            resultado = (area * has_or_cas) / area_as
                            df_senales_edit.at[i, h_o_c_key] = round(resultado, 2)

                except Exception as e:
                    st.warning(f"⚠️ Error en fila {i}: {e}")

            # Obtener filas recalculadas actuales
            filas_actualizadas_raw = df_senales_edit.to_dict(orient="records")

            # Validar y filtrar filas que tengan Muestra y Archivo válidos
            filas_actualizadas = []
            for fila in filas_actualizadas_raw:
                muestra = fila.get("Muestra")
                archivo = fila.get("Archivo")
                if not muestra or not archivo:
                    continue  # ignorar filas incompletas
                filas_actualizadas.append(fila)

            if not filas_actualizadas:
                st.warning("⚠️ No se guardó ninguna fila porque faltan datos en 'Muestra' o 'Archivo'.")
            else:
                # Recuperar filas previas
                tipo_doc = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
                doc_ref = db.collection("tablas_integrales").document(tipo_doc)
                doc_data = doc_ref.get().to_dict()
                filas_previas = doc_data.get("filas", []) if doc_data else []

                combinaciones_actualizadas = {(f["Muestra"], f["Archivo"]) for f in filas_actualizadas}

                filas_conservadas = [
                    f for f in filas_previas
                    if (f.get("Muestra"), f.get("Archivo")) not in combinaciones_actualizadas
                ]

                filas_finales = filas_conservadas + filas_actualizadas
                doc_ref.set({"filas": filas_finales})
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

        # Sombreado por Cálculo de señales
        if aplicar_sombra_senales:
            tipo_doc_senales = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
            doc_senales = db.collection("tablas_integrales").document(tipo_doc_senales)
            if doc_senales.get().exists:
                filas_senales = doc_senales.get().to_dict().get("filas", [])
                for f in filas_senales:
                    if f.get("Archivo") != archivo_actual:
                        continue

                    x1 = f.get("X min")
                    x2 = f.get("X max")
                    grupo = f.get("Grupo funcional")
                    valor = f.get("H") if tipo == "RMN 1H" else f.get("C")

                    if x1 is None or x2 is None:
                        continue

                    fig.add_vrect(
                        x0=min(x1, x2),
                        x1=max(x1, x2),
                        fillcolor="rgba(0,255,0,0.3)",
                        layer="below",
                        line_width=0
                    )

                    fig.add_vline(x=x1, line=dict(color="black", width=1), layer="above")
                    fig.add_vline(x=x2, line=dict(color="black", width=1), layer="above")

                    if grupo not in [None, ""] or valor not in [None, ""]:
                        partes = []
                        if grupo not in [None, ""]:
                            partes.append(f"{grupo}")
                        if valor not in [None, ""]:
                            partes.append(f"{valor:.2f} {'H' if tipo == 'RMN 1H' else 'C'}")
                        etiqueta = " = ".join(partes) if len(partes) == 2 else " ".join(partes)

                        fig.add_annotation(
                            x=(x1 + x2) / 2,
                            y=y_max * 0.98,
                            text=etiqueta,
                            showarrow=False,
                            font=dict(size=10, color="black"),
                            textangle=270,
                            xanchor="center",
                            yanchor="top"
                        )


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

        # Aplicar sombreado por bibliografía si está activo
    if aplicar_sombra_biblio:
        doc_biblio = db.collection("configuracion_global").document(
            "tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c"
        )
        if doc_biblio.get().exists:
            filas_biblio = doc_biblio.get().to_dict().get("filas", [])
            if y_max is None:
                y_max = 1.05 * max([trace.y.max() for trace in fig.data if hasattr(trace, "y")])

            for f in filas_biblio:
                delta = f.get("δ pico")
                grupo = f.get("Grupo funcional")

                if delta is not None:
                    # Línea vertical negra punteada visible en todo el alto
                    fig.add_shape(
                        type="line",
                        x0=delta, x1=delta,
                        y0=0,
                        y1=y_max,
                        line=dict(color="black", dash="dot", width=1),
                        layer="above"
                    )

                    # Etiqueta más alta, sin tocar la curva
                    texto = grupo if grupo not in [None, ""] else f"δ = {delta:.2f}"
                    fig.add_annotation(
                        x=delta,
                        y=y_max * 0.95,
                        text=texto,
                        showarrow=False,
                        textangle=270,
                        font=dict(size=10, color="black"),
                        xanchor="center",
                        yanchor="top"
                    )



    fig.update_layout(
        xaxis_title="[ppm]",
        yaxis_title="Intensidad",
        xaxis=dict(range=[x_max, x_min]),
        yaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None),
        template="simple_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )


    # Sombreado D/T2 en gráfico combinado
    if aplicar_sombra_dt2:
        for _, row in df.iterrows():
            muestra_actual = row["muestra"]
            archivo_actual = row["archivo"]
            doc_dt2 = db.collection("muestras").document(muestra_actual).collection("dt2").document(tipo.lower())
            if doc_dt2.get().exists:
                filas_dt2 = doc_dt2.get().to_dict().get("filas", [])
                for f in filas_dt2:
                    if f.get("Archivo") != archivo_actual:
                        continue

                    x1 = f.get("X min")
                    x2 = f.get("X max")
                    d_val = f.get("D")
                    t2_val = f.get("T2")

                    tiene_d = d_val not in [None, ""]
                    tiene_t2 = t2_val not in [None, ""]

                    mostrar_d = check_d_por_espectro.get(archivo_actual) and tiene_d
                    mostrar_t2 = check_t2_por_espectro.get(archivo_actual) and tiene_t2

                    if not (mostrar_d or mostrar_t2) or x1 is None or x2 is None:
                        continue

                    # Texto centrado con valores
                    partes = []
                    if mostrar_d:
                        partes.append(f"D = {float(d_val):.2e}")
                    if mostrar_t2:
                        partes.append(f"T2 = {float(t2_val):.3f}")
                    etiqueta = "   ".join(partes)

                    color = "rgba(128,128,255,0.3)" if mostrar_d and mostrar_t2 else (
                        "rgba(255,0,0,0.3)" if mostrar_d else "rgba(0,0,255,0.3)"
                    )

                    fig.add_vrect(
                        x0=min(x1, x2),
                        x1=max(x1, x2),
                        fillcolor=color,
                        line_width=0
                    )

                    fig.add_annotation(
                        x=(x1 + x2) / 2,
                        y=y_max * 0.98,  # o un valor más ajustado si lo querés más bajo
                        text=etiqueta,
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        textangle=270,
                        xanchor="center",
                        yanchor="top"
                    )


    st.plotly_chart(fig, use_container_width=True)

    # --- Gráfico con desplazamiento vertical estilo stacked ---
    if superposicion_vertical:
        offset_auto = round((y_max - y_min) / (len(df) + 1), 2) if (y_max is not None and y_min is not None and y_max > y_min) else 1.0
        offset_manual = st.slider(
            "Separación entre espectros (offset)",
            min_value=0.1,
            max_value=30.0,
            value=offset_auto,
            step=0.1,
            key=f"offset_val_{key_sufijo}"
        )
        fig_offset = go.Figure()
        step_offset = offset_manual

        for i, (_, row) in enumerate(df.iterrows()):
            df_esp = decodificar_csv_o_excel(row["contenido"], row["archivo"])
            if df_esp is None:
                continue
            col_x, col_y = df_esp.columns[:2]
            y_data = df_esp[col_y].copy()
            if normalizar:
                y_data = y_data / y_data.max() if y_data.max() != 0 else y_data

            offset = step_offset * i
            fig_offset.add_trace(go.Scatter(
                x=df_esp[col_x],
                y=y_data + offset,
                mode='lines',
                name=row["archivo"]
            ))

        fig_offset.update_layout(
            xaxis_title="[ppm]",
            yaxis_title="Offset + Intensidad",
            xaxis=dict(range=[x_max, x_min]),
            height=500,
            showlegend=True,
            template="simple_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_offset, use_container_width=True)

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

            # Sombreado D/T2 (1H o 13C si hay datos) - INDIVIDUALES
            if aplicar_sombra_dt2:
                doc_dt2 = db.collection("muestras").document(muestra_actual).collection("dt2").document(tipo.lower())
                if doc_dt2.get().exists:
                    filas_dt2 = doc_dt2.get().to_dict().get("filas", [])
                    for f in filas_dt2:
                        if f.get("Archivo") != archivo_actual:
                            continue

                        x1 = f.get("X min")
                        x2 = f.get("X max")
                        d_val = f.get("D")
                        t2_val = f.get("T2")

                        tiene_d = d_val not in [None, ""]
                        tiene_t2 = t2_val not in [None, ""]

                        mostrar_d = check_d_por_espectro.get(archivo_actual) and tiene_d
                        mostrar_t2 = check_t2_por_espectro.get(archivo_actual) and tiene_t2

                        if not (mostrar_d or mostrar_t2) or x1 is None or x2 is None:
                            continue

                        # Etiqueta compuesta
                        partes = []
                        if mostrar_d:
                            partes.append(f"D = {float(d_val):.2e}")
                        if mostrar_t2:
                            partes.append(f"T2 = {float(t2_val):.3f}")
                        etiqueta = "   ".join(partes)

                        color = "rgba(128,128,255,0.3)" if mostrar_d and mostrar_t2 else (
                            "rgba(255,0,0,0.3)" if mostrar_d else "rgba(0,0,255,0.3)"
                        )

                        fig_indiv.add_vrect(
                            x0=min(x1, x2),
                            x1=max(x1, x2),
                            fillcolor=color,
                            line_width=0
                        )

                        fig_indiv.add_vline(
                            x=x1,
                            line=dict(color="black", width=1),
                            layer="above"
                        )
                        fig_indiv.add_vline(
                            x=x2,
                            line=dict(color="black", width=1),
                            layer="above"
                        )

                        fig_indiv.add_annotation(
                            x=(x1 + x2) / 2,
                            y=y_max * 0.98,
                            text=etiqueta,
                            showarrow=False,
                            font=dict(size=10, color="black"),
                            textangle=270,
                            xanchor="center",
                            yanchor="top"
                        )


            # Sombreado por señales - INDIVIDUALES
            if aplicar_sombra_senales:
                tipo_doc_senales = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
                doc_senales = db.collection("tablas_integrales").document(tipo_doc_senales)
                if doc_senales.get().exists:
                    filas_senales = doc_senales.get().to_dict().get("filas", [])
                    for f in filas_senales:
                        if f.get("Archivo") != archivo_actual:
                            continue

                        x1 = f.get("X min")
                        x2 = f.get("X max")
                        grupo = f.get("Grupo funcional")
                        valor = f.get("H") if tipo == "RMN 1H" else f.get("C")

                        # Validación obligatoria para evitar errores en min/max
                        if x1 is None or x2 is None:
                            continue

                        # Sombrear siempre que haya Xmin y Xmax
                        fig_indiv.add_vrect(
                            x0=min(x1, x2),
                            x1=max(x1, x2),
                            fillcolor="rgba(0,255,0,0.3)",
                            layer="below",
                            line_width=0
                        )

                        fig_indiv.add_vline(
                            x=x1,
                            line=dict(color="black", width=1),
                            layer="above"
                        )
                        fig_indiv.add_vline(
                            x=x2,
                            line=dict(color="black", width=1),
                            layer="above"
                        )

                        # Mostrar etiqueta si hay grupo y/o valor
                        if grupo not in [None, ""] or valor not in [None, ""]:
                            partes = []
                            if grupo not in [None, ""]:
                                partes.append(f"{grupo}")
                            if valor not in [None, ""]:
                                partes.append(f"{valor:.2f} {'H' if tipo == 'RMN 1H' else 'C'}")
                            etiqueta = " = ".join(partes) if len(partes) == 2 else " ".join(partes)

                            fig_indiv.add_annotation(
                                x=(x1 + x2) / 2,
                                y=y_max * 0.98,
                                text=etiqueta,
                                showarrow=False,
                                font=dict(size=10, color="black"),
                                textangle=270,
                                xanchor="center",
                                yanchor="top"
                            )


            # Aplicar sombreado por bibliografía si está activo - INDIVIDUALES
            if aplicar_sombra_biblio:
                doc_biblio = db.collection("configuracion_global").document(
                    "tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c"
                )
                if doc_biblio.get().exists:
                    filas_biblio = doc_biblio.get().to_dict().get("filas", [])
                    for f in filas_biblio:
                        delta = f.get("δ pico")
                        grupo = f.get("Grupo funcional")

                        if delta is not None:
                            fig_indiv.add_shape(
                                type="line",
                                x0=delta, x1=delta,
                                y0=0,
                                y1=y_max,
                                line=dict(color="black", dash="dot", width=1),
                                layer="above"
                            )

                            texto = grupo if grupo not in [None, ""] else f"δ = {delta:.2f}"
                            fig_indiv.add_annotation(
                                x=delta,
                                y=y_max * 0.95,
                                text=texto,
                                showarrow=False,
                                textangle=270,
                                font=dict(size=10, color="black"),
                                xanchor="center",
                                yanchor="top"
                            )


            fig_indiv.update_layout(
                title=f"{archivo_actual}",
                xaxis_title="[ppm]",
                yaxis_title="Intensidad",
                xaxis=dict(range=[x_max, x_min]),
                yaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None),
                height=500,
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
