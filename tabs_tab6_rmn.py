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
        espectros_validos = []
        for doc in db.collection("muestras").document(muestra).collection("espectros").stream():
            espectro = doc.to_dict()
            tipo = (espectro.get("tipo") or "").upper()
            if tipo in ["RMN 1H", "RMN 13C", "RMN-LF 1H"]:
                espectros_validos.append(espectro.get("nombre_archivo", "sin nombre"))

        espectros_sel[muestra] = st.multiselect(
            f"Espectros para {muestra}",
            options=espectros_validos,
            default=[]
        )


    st.divider()

    # ==============================
    # === SECCIÓN RMN 1H ===========
    # ==============================
    st.subheader("🔬 RMN 1H")

    # --- Máscara D/T2 ---
    activar_mascara = st.checkbox("Máscara D/T2", value=False, key="chk_global_mascara_dt2")
    usar_mascara = {}
    if activar_mascara:
        st.markdown("Activar sombreado individual por muestra:")
        cols = st.columns(len(muestras_sel))
        for idx, muestra in enumerate(muestras_sel):
            with cols[idx]:
                usar_mascara[muestra] = st.checkbox(
                    label=muestra,
                    key=f"chk_mask_{muestra}_{idx}",
                    value=False
                )


    # --- Cálculo D/T2 ---
    espectros_activos = {m: archivos for m, archivos in espectros_sel.items() if archivos}
    activar_calculo_dt2 = st.checkbox("Cálculo D/T2", value=False, key="chk_calc_dt2")
    if activar_calculo_dt2:
        columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                        "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]
        filas_guardadas = []

        # Cargar filas desde Firebase
        for muestra in muestras_sel:
            if muestra not in espectros_activos:
                continue
            doc = db.collection("muestras").document(muestra).collection("dt2").document("datos")
            data = doc.get().to_dict()
            if data and "filas" in data:
                for fila in data["filas"]:
                    if fila.get("Archivo") in espectros_activos[muestra]:
                        filas_guardadas.append(fila)


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
  #  activar_picos = st.checkbox("Señales Pico Bibliográfica", value=False, key="chk_deltas")
    col_pico, col_editar = st.columns([1, 1])
    with col_pico:
        activar_picos = st.checkbox("Señales Pico Bibliográfica", value=False, key="chk_deltas")
    with col_editar:
        editar_tabla_biblio = st.checkbox("Editar Tabla Bibliográfica", value=False, key="chk_editar_biblio")

    # Cargar y preparar la tabla
    if activar_picos:
        doc_biblio = db.collection("configuracion_global").document("tabla_editable_rmn1h")
        if not doc_biblio.get().exists:
            doc_biblio.set({"filas": []})

        filas_biblio = doc_biblio.get().to_dict().get("filas", [])
        columnas_biblio = ["Grupo funcional", "X min", "δ pico", "X max", "Tipo de muestra", "Observaciones"]
        df_biblio = pd.DataFrame(filas_biblio)

        for col in columnas_biblio:
            if col not in df_biblio.columns:
                df_biblio[col] = "" if col in ["Grupo funcional", "Tipo de muestra", "Observaciones"] else None
        df_biblio = df_biblio[columnas_biblio]

        if editar_tabla_biblio:
            df_biblio_edit = st.data_editor(
                df_biblio,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="editor_tabla_biblio",
                column_config={
                    "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
                    "X min": st.column_config.NumberColumn(format="%.2f"),
                    "δ pico": st.column_config.NumberColumn(format="%.2f"),
                    "X max": st.column_config.NumberColumn(format="%.2f"),
                }
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔴Actualizar Tabla Bibliográfica"):
                    doc_biblio.set({"filas": df_biblio_edit.to_dict(orient="records")})
                    st.success("✅ Tabla bibliográfica actualizada.")
                    st.rerun()
            with col2:
                buffer_excel = BytesIO()
                with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                    df_biblio_edit.to_excel(writer, index=False, sheet_name="Tabla_Bibliográfica")
                buffer_excel.seek(0)
                st.download_button(
                    "📅 Descargar tabla",
                    data=buffer_excel.getvalue(),
                    file_name="tabla_bibliografica_rmn1h.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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

    colores = plt.cm.tab10.colors
    graficado = False

    for idx, (muestra, archivos) in enumerate(espectros_activos.items()):
        for archivo in archivos:
            espectros = db.collection("muestras").document(muestra).collection("espectros").stream()
            espectro = next((e.to_dict() for e in espectros if e.to_dict().get("nombre_archivo") == archivo), None)
            if not espectro:
                continue

            try:
                contenido = BytesIO(base64.b64decode(espectro["contenido"]))
                extension = os.path.splitext(archivo)[1].lower()
                if extension == ".xlsx":
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

                col_x, col_y = df.columns[:2]
                df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
                df[col_y] = pd.to_numeric(df[col_y], errors="coerce")
                df = df.dropna()

                color = colores[idx % len(colores)]
                ax.plot(df[col_x], df[col_y], label=f"{muestra} - {archivo}", color=color)
                graficado = True

                # Mostrar máscaras si están activadas
                if activar_mascara and usar_mascara.get(muestra, False):
                    for mascara in espectro.get("mascaras", []):
                        x0 = mascara.get("x_min")
                        x1 = mascara.get("x_max")
                        d = mascara.get("difusividad")
                        t2 = mascara.get("t2")
                        obs = mascara.get("observacion", "")
                        sub_df = df[(df[col_x] >= min(x0, x1)) & (df[col_x] <= max(x0, x1))]
                        if sub_df.empty:
                            continue
                        ax.axvspan(x0, x1, color=color, alpha=0.2)
                        if d and t2:
                            ax.text((x0+x1)/2, ax.get_ylim()[1]*0.9,
                                    f"D={d:.1e}\nT2={t2:.2f}", ha="center", va="center", fontsize=6, rotation=90)

            except Exception as e:
                st.warning(f"No se pudo graficar {archivo}: {e}")

    # Mostrar leyenda solo si se graficó algo
    if graficado:
        ax.legend()
    else:
        ax.text((x_max + x_min)/2, (y_max + y_min)/2, "No se han graficado espectros.\nVerificá la selección.",
                ha="center", va="center", fontsize=10, color="red")

    # Graficar líneas δ pico
    if "fig" in locals() and not df_biblio.empty:
            for _, row in df_biblio.iterrows():
                try:
                    delta = float(row["δ pico"])
                    etiqueta = str(row["Grupo funcional"])
                    ax.axvline(x=delta, linestyle="dashed", color="black", linewidth=1)
                    ax.text(
                        delta, ax.get_ylim()[1], etiqueta,
                        rotation=90, va="bottom", ha="center",
                        fontsize=6, color="black"
                    )
                except:
                    continue
    st.pyplot(fig)


    # ==============================
    # === Cálculo de señales =======
    # ==============================
    espectros_activos = {m: archivos for m, archivos in espectros_sel.items() if archivos}
    activar_calculo_senales = st.checkbox("Cálculo de señales", value=False, key="chk_calc_senales")
    if activar_calculo_senales:
        columnas_integral = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                             "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]

        doc_ref = db.collection("tablas_integrales").document("rmn1h")
        if not doc_ref.get().exists:
            doc_ref.set({"filas": []})

        filas_total = doc_ref.get().to_dict().get("filas", [])
        filas_actuales = [
            f for f in filas_total
            if f.get("Muestra") in espectros_activos
            and f.get("Archivo") in espectros_activos.get(f.get("Muestra"), [])
        ]
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