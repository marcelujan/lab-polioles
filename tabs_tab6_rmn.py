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

    # --- SELECTOR UNIFICADO ---
    # Cargar muestras desde Firebase
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # Cargar espectros RMN válidos
    espectros_rmn = []
    for m in muestras:
        nombre = m["nombre"]
        espectros = db.collection("muestras").document(nombre).collection("espectros").stream()
        for i, doc in enumerate(espectros):
            e = doc.to_dict()
            tipo = (e.get("tipo") or "").upper()
            if "RMN" in tipo:
                espectros_rmn.append({
                    "muestra": nombre,
                    "tipo": tipo,
                    "archivo": e.get("nombre_archivo", "sin nombre"),
                    "contenido": e.get("contenido"),
                    "fecha": e.get("fecha"),
                    "mascaras": e.get("mascaras", []),
                    "es_imagen": e.get("es_imagen", False),
                    "id": f"{nombre}__{i}"
                })

    # Crear DataFrame consolidado
    df_total = pd.DataFrame(espectros_rmn)
    if df_total.empty:
        st.warning("No hay espectros RMN disponibles.")
        st.stop()

    # Selección de muestras
    muestras_disp = sorted(df_total["muestra"].unique())
    muestras_sel = st.multiselect("Seleccionar muestras", muestras_disp, default=[])

    # Filtrar por muestras seleccionadas
    df_filtrado = df_total[df_total["muestra"].isin(muestras_sel)]

    # Selector de espectros por ID legible
    espectros_info = [
        {"id": row["id"], "nombre": f"{row['muestra']} – {row['archivo']}"}
        for _, row in df_filtrado.iterrows()
    ]

    ids_disponibles = [e["id"] for e in espectros_info]
    ids_legibles = {e["id"]: e["nombre"] for e in espectros_info}

    ids_sel = st.multiselect(
        "Seleccionar espectros a visualizar:",
        options=ids_disponibles,
        format_func=lambda i: ids_legibles.get(i, i)
    )

    # DataFrame final con espectros seleccionados
    df_sel = df_filtrado[df_filtrado["id"].isin(ids_sel)]


    # ==============================
    # === SECCIÓN RMN 1H ===========
    # ==============================
    st.subheader("🔬 RMN 1H")
    df_rmn1h = df_sel[df_sel["tipo"] == "RMN 1H"]

    if df_rmn1h.empty:
        return  # o st.stop() si no querés seguir con el resto de esta sección


    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X mínimo", value=0.0)
    x_max = colx2.number_input("X máximo", value=9.0)
    y_min = coly1.number_input("Y mínimo", value=0.0)
    y_max = coly2.number_input("Y máximo", value=80.0)

    activar_mascara = st.checkbox("Máscara D/T2", value=False, key="chk_mascara_rmn1h")
    cols = st.columns(len(df_rmn1h)) if activar_mascara else []

    # Generar gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel("[ppm]")
    ax.set_ylabel("Señal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color="black", linewidth=0.7)

    graficado = False
    colores = plt.cm.tab10.colors
    col_idx = 0  # índice seguro para st.columns()

    for idx, row in df_rmn1h.iterrows():
        muestra = row["muestra"]
        archivo = row["archivo"]
        contenido = row["contenido"]
        mascaras = row.get("mascaras", [])

        try:
            contenido_bin = BytesIO(base64.b64decode(contenido))
            contenido_bin.seek(0)
            extension = os.path.splitext(archivo)[1].lower()
            if extension == ".xlsx":
                df = pd.read_excel(contenido_bin)
            else:
                for sep in [",", ";", "\t", " "]:
                    contenido_bin.seek(0)
                    try:
                        df = pd.read_csv(contenido_bin, sep=sep)
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
                else:
                    st.warning(f"⚠️ {archivo}: no se pudo leer con separadores comunes.")
                    continue

            if df.shape[1] < 2:
                st.warning(f"⚠️ {archivo}: el archivo no tiene al menos 2 columnas.")
                continue

            col_x, col_y = df.columns[:2]
            df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
            df[col_y] = pd.to_numeric(df[col_y], errors="coerce")
            df = df.dropna()

            color = colores[idx % len(colores)]
            ax.plot(df[col_x], df[col_y], label=f"{archivo}", color=color)
            graficado = True

            if activar_mascara and col_idx < len(cols):
                with cols[col_idx]:
                    key_chk = f"chk_masc_{row['muestra']}_{archivo}"
                    mostrar_mascara = st.checkbox(archivo, key=key_chk, value=False)

                if mostrar_mascara:
                    for mascara in mascaras:
                        x0 = mascara.get("x_min")
                        x1 = mascara.get("x_max")
                        d = mascara.get("difusividad")
                        t2 = mascara.get("t2")
                        if x0 is not None and x1 is not None:
                            ax.axvspan(x0, x1, color=color, alpha=0.2)
                            y_etiqueta = min(50, ax.get_ylim()[1] * 0.95)
                            if d and t2:
                                ax.text(
                                    (x0 + x1) / 2, y_etiqueta,
                                    f"D={d:.1e} T2={t2:.3f}",
                                    ha="center", va="center", fontsize=6, color="black", rotation=90
                                )

                col_idx += 1

        except Exception as e:
            st.warning(f"No se pudo graficar {archivo}: {e}")

    if graficado:
        ax.legend()
    else:
        ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, "No se han graficado espectros RMN 1H.",
                ha="center", va="center", fontsize=10, color="red")


    # --- Cálculo D/T2 desde df_sel ---
    activar_calculo_dt2 = st.checkbox("Cálculo D/T2", value=False, key="chk_calc_dt2_dfsel")

    if activar_calculo_dt2:
        columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                        "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]

        # Cargar desde Firebase para espectros seleccionados
        filas_guardadas = []
        for _, row in df_sel.iterrows():
            muestra = row["muestra"]
            archivo = row["archivo"]
            doc = db.collection("muestras").document(muestra).collection("dt2").document("datos")
            data = doc.get().to_dict()
            if data and "filas" in data:
                filas_guardadas.extend([f for f in data["filas"] if f.get("Archivo") == archivo])

        df_dt2 = pd.DataFrame(filas_guardadas)
        for col in columnas_dt2:
            if col not in df_dt2.columns:
                df_dt2[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
        df_dt2 = df_dt2[columnas_dt2]

        with st.form("form_dt2_dfsel"):
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
                key="tabla_dt2_dfsel"
            )
            recalcular = st.form_submit_button("🔴Recalcular 'Área', 'Área as' y 'H'")

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

                        if area and area_as and has and area_as != 0:
                            h_calc = (area * has) / area_as
                            df_dt2_edit.at[i, "H"] = round(h_calc, 2)

                except Exception as e:
                    st.warning(f"⚠️ Error en fila {i}: {e}")

            # Guardar en Firebase agrupado por muestra
            filas_actualizadas = df_dt2_edit.to_dict(orient="records")
            for muestra in df_dt2_edit["Muestra"].unique():
                filas_m = [f for f in filas_actualizadas if f["Muestra"] == muestra]
                doc = db.collection("muestras").document(muestra).collection("dt2").document("datos")
                doc.set({"filas": filas_m})

                # Actualizar el campo "mascaras" en cada espectro correspondiente
                for fila in filas_m:
                    archivo = fila.get("Archivo")
                    mascaras_actualizadas = [
                        {
                            "x_min": f.get("X min"),
                            "x_max": f.get("X max"),
                            "difusividad": f.get("D"),
                            "t2": f.get("T2")
                        }
                        for f in filas_m if f.get("Archivo") == archivo and f.get("X min") is not None and f.get("X max") is not None
                    ]

                    # Buscar documento de espectro por nombre de archivo
                    espectros = db.collection("muestras").document(muestra).collection("espectros").stream()
                    for doc_esp in espectros:
                        e_dict = doc_esp.to_dict()
                        if e_dict.get("nombre_archivo") == archivo:
                            doc_esp.reference.update({"mascaras": mascaras_actualizadas})
                            break

            st.success("✅ Datos recalculados y guardados correctamente.")
            st.rerun()

    #  --- Señales Pico Bibliografía desde df_sel ---
    col_bib1, col_bib2 = st.columns([1, 1])
    activar_picos = editar_tabla_biblio = False
    with col_bib1:
        activar_picos = st.checkbox("Señales Pico Bibliográfica", value=False, key="chk_deltas_biblio_dfsel")
    if activar_picos:
        with col_bib2:
            editar_tabla_biblio = st.checkbox("Editar Tabla Bibliográfica", value=False, key="chk_editar_biblio_dfsel")

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
                key="editor_tabla_biblio_dfsel",
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
                    "📥 Descargar tabla",
                    data=buffer_excel.getvalue(),
                    file_name="tabla_bibliografica_rmn1h.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        # Trazado de líneas δ-pico si hay gráfico existente
        if "ax" in locals() and not df_biblio.empty:
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
                except Exception as e:
                    st.warning(f"⚠️ Error al trazar δ pico: {e}")

    st.pyplot(fig)

    # --- Cálculo de señales desde df_sel ---
    activar_calculo_senales = st.checkbox("Cálculo de señales", value=False, key="chk_calc_senales_dfsel")

    if activar_calculo_senales:
        columnas_integral = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                            "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]

        doc_ref = db.collection("tablas_integrales").document("rmn1h")
        if not doc_ref.get().exists:
            doc_ref.set({"filas": []})

        filas_total = doc_ref.get().to_dict().get("filas", [])

        # Extraer combinaciones válidas de muestra + archivo seleccionadas
        combinaciones_activas = {(row["muestra"], row["archivo"]) for _, row in df_sel.iterrows()}

        # Filtrar filas existentes para solo las muestras activas
        filas_actuales = [
            f for f in filas_total
            if (f.get("Muestra"), f.get("Archivo")) in combinaciones_activas
        ]

        df_integral = pd.DataFrame(filas_actuales)

        # Si no hay datos, permitir agregar fila nueva manualmente
        if df_integral.empty:
            st.warning("⚠️ No hay datos previos guardados para estas muestras.")
            muestra_nueva = st.selectbox("Seleccionar muestra para comenzar", sorted({m for m, _ in combinaciones_activas}))
            archivos_disp = sorted({a for m, a in combinaciones_activas if m == muestra_nueva})
            archivo_nuevo = st.selectbox("Seleccionar archivo", archivos_disp)

            df_integral = pd.DataFrame([{
                "Muestra": muestra_nueva,
                "Grupo funcional": "",
                "δ pico": None,
                "X min": None,
                "X max": None,
                "Área": None,
                "D": None,
                "T2": None,
                "Xas min": None,
                "Xas max": None,
                "Has": None,
                "Área as": None,
                "H": None,
                "Observaciones": "",
                "Archivo": archivo_nuevo,
            }])

        # Asegurar todas las columnas estén presentes
        for col in columnas_integral:
            if col not in df_integral.columns:
                df_integral[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
        df_integral = df_integral[columnas_integral]

        with st.form("form_integral_dfsel"):
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
                key="tabla_integral_dfsel"
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
















 
























# --- Zona RMN 13C ---
    st.subheader("🧪 RMN 13C")
    df_rmn13C = df_sel[(df_sel["tipo"] == "RMN 13C") & (~df_sel["es_imagen"])].copy()
    if df_rmn13C.empty:
        st.info("No hay espectros RMN 13C numéricos seleccionados.")
    else:
        fig13, ax13 = plt.subplots()
        for _, row in df_rmn13C.iterrows():
            try:
                contenido = BytesIO(base64.b64decode(row["contenido"]))
                extension = os.path.splitext(row["archivo"])[1].lower()
                if extension == ".xlsx":
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
                        raise ValueError("No se pudo leer el archivo.")

                col_x, col_y = df.columns[:2]
                df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
                df[col_y] = pd.to_numeric(df[col_y], errors="coerce")
                df = df.dropna()

                ax13.plot(df[col_x], df[col_y], label=f"{row['muestra']}")
            except:
                st.warning(f"No se pudo graficar espectro: {row['archivo']}")

        ax13.set_xlabel("[ppm]")
        ax13.set_ylabel("Señal")
        ax13.legend()
        st.pyplot(fig13)

        # Botón para descargar imagen del gráfico RMN 13C
        buffer_img13 = BytesIO()
        fig13.savefig(buffer_img13, format="png", dpi=300, bbox_inches="tight")
        st.download_button("📷 Descargar gráfico RMN 13C", data=buffer_img13.getvalue(), file_name="grafico_rmn13c.png", mime="image/png")

    # --- Zona Imágenes ---
    st.subheader("🖼️ Espectros imagen")
    df_rmn_img = df_sel[df_sel["es_imagen"]]
    if df_rmn_img.empty:
        st.info("No hay espectros RMN en formato imagen seleccionados.")
    else:
        for _, row in df_rmn_img.iterrows():
            try:
                imagen = BytesIO(base64.b64decode(row["contenido"]))
                st.image(imagen, caption=f"{row['muestra']} – {row['archivo']} ({row['fecha']})", use_container_width=True)
            except:
                st.warning(f"No se pudo mostrar imagen: {row['archivo']}")

        # Botón para descargar ZIP con todas las imágenes mostradas
        with TemporaryDirectory() as tmpdir:
            import hashlib

            # Construir nombre base del ZIP desde la primera fila válida
            primera_fila = df_rmn_img.iloc[0].to_dict() if not df_rmn_img.empty else {}
            muestra = primera_fila.get("muestra", "Desconocida")
            tipo = primera_fila.get("tipo", "RMN")
            fecha = primera_fila.get("fecha", datetime.now().strftime("%Y-%m-%d"))
            archivo = primera_fila.get("archivo", "archivo")
            peso = primera_fila.get("peso") or primera_fila.get("peso_muestra") or "?"

            nombre_zip = f"{muestra} — {tipo} — {fecha} — {archivo} — {peso} g"
            nombre_zip = nombre_zip.replace(" ", "_").replace("—", "-").replace(":", "-").replace("/", "-").replace("\\", "-")
            zip_path = os.path.join(tmpdir, f"{nombre_zip}.zip")

            with zipfile.ZipFile(zip_path, "w") as zipf:
                for idx, row in df_rmn_img.iterrows():
                    contenido = row.get("contenido")
                    if not contenido:
                        continue

                    muestra = row.get("muestra", "muestra").replace(" ", "")[:8]
                    archivo = row.get("archivo", "espectro.xlsx")
                    fecha = row.get("fecha", "fecha").replace(" ", "_")
                    tipo = row.get("tipo", "tipo").replace(" ", "_")

                    # Parte significativa del nombre base
                    base_archivo = os.path.splitext(archivo)[0][:25].replace(" ", "_").replace("/", "-").replace("\\", "-")

                    # Hash único para evitar duplicados
                    contenido_bytes = base64.b64decode(contenido)
                    hash_id = hashlib.sha1(contenido_bytes + str(idx).encode()).hexdigest()[:6]

                    # Nombre de archivo final seguro y único
                    nombre_final = f"{muestra}_{tipo}_{fecha}_idx{idx}_{hash_id}.xlsx"
                    nombre_final = nombre_final.replace("—", "-").replace(":", "-").replace("/", "-").replace("\\", "-")

                    ruta = os.path.join(tmpdir, nombre_final)
                    try:
                        with open(ruta, "wb") as f:
                            f.write(contenido_bytes)
                        zipf.write(ruta, arcname=nombre_final)
                    except Exception as e:
                        st.warning(f"⚠️ No se pudo agregar {archivo}: {e}")

            # Botón de descarga final
            with open(zip_path, "rb") as final_zip:
                st.download_button(
                    "📦 Descargar imágenes RMN",
                    data=final_zip.read(),
                    file_name=os.path.basename(zip_path),
                    mime="application/zip"
                )

    mostrar_sector_flotante(db, key_suffix="tab6")