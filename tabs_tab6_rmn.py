# tabs_tab6_rmn.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from io import BytesIO
from datetime import datetime
import os
import base64
import zipfile
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt #solo para pruebas

# --- Configuraciones globales ---
GRUPOS_FUNCIONALES = ["Formiato", "Cloroformo", "C=C olefínicos", "Glicerol medio", "Glicerol extremos", "Metil-Éster", "Eter", "Ester", "Ácido carboxílico", "OH", "Epóxido", "C=C", "Alfa-C=O", "Alfa-C=C", "Vecino a alfa-carbonilo", "Alfa-epóxido", "CH2", "CH3"]

def obtener_ids_espectros(nombre):
    return [doc.id for doc in firestore.Client().collection("muestras").document(nombre).collection("espectros").list_documents()]

def render_tab6(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Análisis RMN")
    st.session_state["current_tab"] = "Análisis RMN"

    # --- Selección de muestras ---
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # Filtrar solo muestras con espectros RMN
    muestras_con_rmn = []
    for m in muestras:
        espectros = db.collection("muestras").document(m["nombre"]).collection("espectros").stream()
        for e in espectros:
            tipo = (e.to_dict().get("tipo") or "").upper()
            if tipo in ["RMN 1H", "RMN 13C", "RMN-LF 1H"]:
                muestras_con_rmn.append(m["nombre"])
                break  # No hace falta revisar más espectros de esta muestra

    muestras_con_rmn = sorted(set(muestras_con_rmn))
    muestras_sel = st.multiselect("Seleccionar muestras con espectros RMN", muestras_con_rmn, default=[])

    if not muestras_sel:
        st.warning("Seleccioná al menos una muestra.")
        st.stop()

    # --- Selección de espectros por muestra ---
    espectros_sel = {}
    for muestra in muestras_sel:
        espectros_ref = db.collection("muestras").document(muestra).collection("espectros").stream()
        espectros = [doc.to_dict() for doc in espectros_ref]
        nombres_archivos = [e.get("nombre_archivo", "sin nombre") for e in espectros]
        espectros_sel[muestra] = st.multiselect(f"Espectros para {muestra}", nombres_archivos, default=nombres_archivos[:1])

    st.divider()

    # ==============================
    # === SECCIÓN RMN 1H ===========
    # ==============================
    st.subheader("🔬 RMN 1H")

    # --- Máscara D/T2 ---
    activar_mascara = st.checkbox("Máscara D/T2", value=False)
    usar_mascara = {}
    if activar_mascara:
        st.markdown("Activar sombreado individual por muestra:")
        cols = st.columns(len(muestras_sel))
        for idx, muestra in enumerate(muestras_sel):
            with cols[idx]:
                usar_mascara[muestra] = st.checkbox(muestra, key=f"chk_mask_{muestra}", value=False)

    # --- Cálculo D/T2 ---
    activar_calculo_dt2 = st.checkbox("Cálculo D/T2", value=False)
    if activar_calculo_dt2:
        columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                        "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]
        filas_guardadas = []

        # Cargar filas desde Firebase
        for muestra in muestras_sel:
            doc = db.collection("muestras").document(muestra).collection("dt2").document("datos")
            data = doc.get().to_dict()
            if data and "filas" in data:
                filas_guardadas.extend(data["filas"])

        df_dt2 = pd.DataFrame(filas_guardadas)
        for col in columnas_dt2:
            if col not in df_dt2.columns:
                df_dt2[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
        df_dt2 = df_dt2[columnas_dt2]

        with st.form("form_dt2"):
            df_dt2_edit = st.data_editor(
                df_dt2,
                column_config={
                    "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
                    "δ pico": st.column_config.NumberColumn(format="%.2f"),
                    "X min": st.column_config.NumberColumn(format="%.2f"),
                    "X max": st.column_config.NumberColumn(format="%.2f"),
                    "Área": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "D": st.column_config.NumberColumn(format="%.2e"),
                    "T2": st.column_config.NumberColumn(format="%.3f"),
                    "Xas min": st.column_config.NumberColumn(format="%.2f"),
                    "Xas max": st.column_config.NumberColumn(format="%.2f"),
                    "Has": st.column_config.NumberColumn(format="%.2f"),
                    "Área as": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "H": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "Observaciones": st.column_config.TextColumn(),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",
                key="tabla_dt2"
            )
            recalcular = st.form_submit_button("🔴Recalcular 'Área', 'Área as' y 'H'")

        if recalcular:
            for i, row in df_dt2_edit.iterrows():
                try:
                    nombre_muestra = row["Muestra"]
                    archivo = row["Archivo"]
                    x_min = float(row["X min"])
                    x_max = float(row["X max"])
                    xas_min = float(row["Xas min"]) if row["Xas min"] not in [None, ""] else None
                    xas_max = float(row["Xas max"]) if row["Xas max"] not in [None, ""] else None
                    has = float(row["Has"]) if row["Has"] not in [None, ""] else None

                    # Buscar espectro
                    espectros = db.collection("muestras").document(nombre_muestra).collection("espectros").stream()
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

                    # Área principal
                    df_main = df_esp[(df_esp[col_x] >= min(x_min, x_max)) & (df_esp[col_x] <= max(x_min, x_max))]
                    area = np.trapz(df_main[col_y], df_main[col_x]) if not df_main.empty else None
                    df_dt2_edit.at[i, "Área"] = round(area, 2) if area else None

                    # Área as
                    if xas_min is not None and xas_max is not None:
                        df_as = df_esp[(df_esp[col_x] >= min(xas_min, xas_max)) & (df_esp[col_x] <= max(xas_min, xas_max))]
                        area_as = np.trapz(df_as[col_y], df_as[col_x]) if not df_as.empty else None
                        df_dt2_edit.at[i, "Área as"] = round(area_as, 2) if area_as else None

                        if area and area_as and has and area_as != 0:
                            h_calc = (area * has) / area_as
                            df_dt2_edit.at[i, "H"] = round(h_calc, 2)

                except Exception as e:
                    st.warning(f"⚠️ Error en fila {i}: {e}")

            # Guardar en Firebase por muestra
            filas_actualizadas = df_dt2_edit.to_dict(orient="records")
            for muestra in muestras_sel:
                filas_m = [f for f in filas_actualizadas if f["Muestra"] == muestra]
                doc = db.collection("muestras").document(muestra).collection("dt2").document("datos")
                doc.set({"filas": filas_m})

            st.success("✅ Recalculado y guardado correctamente.")
            st.rerun()


    # --- Señales pico bibliografía ---
    activar_picos = st.checkbox("Señales Pico Bibliografía", value=False)
    if activar_picos:
        editar_tabla_biblio = st.checkbox("Editar Tabla Bibliográfica", value=False)
        if editar_tabla_biblio:
            st.data_editor(
                pd.DataFrame(columns=["Grupo funcional", "X min", "δ pico", "X max", "Tipo de muestra", "Observaciones"]),
                use_container_width=True,
                num_rows="dynamic"
            )

    # --- Control de ejes del gráfico ---
    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X mínimo", value=0.0)
    x_max = colx2.number_input("X máximo", value=10.0)
    y_min = coly1.number_input("Y mínimo", value=0.0)
    y_max = coly2.number_input("Y máximo", value=100.0)

    # --- Gráfico combinado ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Espectros RMN 1H")
    ax.set_xlabel("[ppm]")
    ax.set_ylabel("Señal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color="black", linewidth=0.7)
    st.pyplot(fig)

    # ==============================
    # === Cálculo de señales =======
    # ==============================
    activar_calculo_senales = st.checkbox("Cálculo de señales", value=False)
    if activar_calculo_senales:
        columnas_integral = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                             "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]

        doc_ref = db.collection("tablas_integrales").document("rmn1h")
        if not doc_ref.get().exists:
            doc_ref.set({"filas": []})

        filas_actuales = doc_ref.get().to_dict().get("filas", [])
        df_integral = pd.DataFrame(filas_actuales)

        for col in columnas_integral:
            if col not in df_integral.columns:
                df_integral[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
        df_integral = df_integral[columnas_integral]

        with st.form("form_integral"):
            df_integral_edit = st.data_editor(
                df_integral,
                column_config={
                    "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
                    "δ pico": st.column_config.NumberColumn(format="%.2f"),
                    "X min": st.column_config.NumberColumn(format="%.2f"),
                    "X max": st.column_config.NumberColumn(format="%.2f"),
                    "Área": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "D": st.column_config.NumberColumn(format="%.2e"),
                    "T2": st.column_config.NumberColumn(format="%.3f"),
                    "Xas min": st.column_config.NumberColumn(format="%.2f"),
                    "Xas max": st.column_config.NumberColumn(format="%.2f"),
                    "Has": st.column_config.NumberColumn(format="%.2f"),
                    "Área as": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "H": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "Observaciones": st.column_config.TextColumn(),
                    "Archivo": st.column_config.TextColumn(),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",
                key="tabla_integral"
            )

            recalcular = st.form_submit_button("🔴Recalcular 'Área', 'Área as' y 'H'")

        if recalcular:
            for i, row in df_integral_edit.iterrows():
                try:
                    muestra = row.get("Muestra")
                    archivo = row.get("Archivo")
                    x_min = float(row.get("X min"))
                    x_max = float(row.get("X max"))
                    xas_min = float(row.get("Xas min")) if row.get("Xas min") not in [None, ""] else None
                    xas_max = float(row.get("Xas max")) if row.get("Xas max") not in [None, ""] else None
                    has = float(row.get("Has")) if row.get("Has") not in [None, ""] else None

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
                    df_integral_edit.at[i, "Área"] = round(area, 2) if area else None

                    if xas_min is not None and xas_max is not None:
                        df_as = df_esp[(df_esp[col_x] >= min(xas_min, xas_max)) & (df_esp[col_x] <= max(xas_min, xas_max))]
                        area_as = np.trapz(df_as[col_y], df_as[col_x]) if not df_as.empty else None
                        df_integral_edit.at[i, "Área as"] = round(area_as, 2) if area_as else None

                        if area and area_as and has and area_as != 0:
                            h_calc = (area * has) / area_as
                            df_integral_edit.at[i, "H"] = round(h_calc, 2)

                except Exception as e:
                    st.warning(f"⚠️ Error en fila {i}: {e}")

            doc_ref.set({"filas": df_integral_edit.to_dict(orient="records")})
            st.success("✅ Datos recalculados y guardados correctamente.")
            st.rerun()


    # ==============================
    # === RMN 13C ==================
    # ==============================
    st.subheader("🧪 RMN 13C")
    st.info("Aquí se graficarán los espectros RMN 13C seleccionados.")

    # ==============================
    # === Imágenes RMN ============
    # ==============================
    st.subheader("🖼️ Espectros imagen")
    st.info("Aquí se mostrarán las imágenes de espectros RMN.")