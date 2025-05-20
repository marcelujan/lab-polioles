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

GRUPOS_FUNCIONALES = [
    "Glicerol medio", "Glicerol extremos", "OH", "C=C",
    "Epóxido", "Éter", "Ester", "Ácido carboxílico", "Formiato"
]

def obtener_ids_espectros(nombre):
    return [doc.id for doc in firestore.Client().collection("muestras").document(nombre).collection("espectros").list_documents()]

def obtener_espectros_para_muestra(db, nombre):
    clave = f"_espectros_cache_{nombre}"
    if clave not in st.session_state:
        ref = db.collection("muestras").document(nombre).collection("espectros")
        docs = ref.stream()
        st.session_state[clave] = [doc.to_dict() for doc in docs]
    return st.session_state[clave]

def graficar_mascaras(df, col_x, col_y, lista_mascaras, ax, color):
    """Grafica máscaras sobre un espectro y devuelve las máscaras válidas y advertencias."""
    filas = []
    advertencias = []
    for mascara in lista_mascaras:
        x0 = mascara.get("x_min")
        x1 = mascara.get("x_max")
        d = mascara.get("difusividad")
        t2 = mascara.get("t2")
        obs = mascara.get("observacion", "")

        sub_df = df[(df[col_x] >= min(x0, x1)) & (df[col_x] <= max(x0, x1))]
        if sub_df.empty:
            advertencias.append(f"⚠️ Sin datos en rango {x0}–{x1} ppm")
            continue

        area = np.trapz(sub_df[col_y], sub_df[col_x])
        ax.axvspan(x0, x1, color=color, alpha=0.3)

        if d and t2:
            ax.text((x0+x1)/2, max(df[col_y])*0.9,
                    f"D={d:.1e}     T2={t2:.3f}", ha="center", va="center", fontsize=6, color="black", rotation=90)

        filas.append({
            "x_min": x0, "x_max": x1, "D": d, "T2": t2, "Área": area, "Obs": obs
        })

    return filas, advertencias

def render_tab6(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    global GRUPOS_FUNCIONALES
    st.title("Análisis RMN")
    st.session_state["current_tab"] = "Análisis RMN"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # --- Filtrar muestras y espectros ---
    espectros_rmn = []
    for m in muestras:
        espectros = obtener_espectros_para_muestra(db, m["nombre"])
        for i, e in enumerate(espectros):
            tipo = e.get("tipo", "").upper()
            if "RMN" in tipo:
                espectros_rmn.append({
                    "muestra": m["nombre"],
                    "tipo": tipo,
                    "es_imagen": e.get("es_imagen", False),
                    "archivo": e.get("nombre_archivo", ""),
                    "contenido": e.get("contenido"),
                    "fecha": e.get("fecha"),
                    "mascaras": e.get("mascaras", []),
                    "id": f"{m['nombre']}__{i}"
                })

    df_rmn1H = pd.DataFrame(espectros_rmn)

    st.subheader("Filtrar espectros")
    if df_rmn1H.empty or "muestra" not in df_rmn1H.columns:
        st.warning("No hay espectros RMN disponibles.")
        st.stop()
    muestras_disp = sorted(df_rmn1H["muestra"].unique())
    muestras_sel = st.multiselect("Muestras", muestras_disp, default=[])
    st.session_state["muestra_activa"] = muestras_sel[0] if len(muestras_sel) == 1 else None

    df_filtrado = df_rmn1H[df_rmn1H["muestra"].isin(muestras_sel)]

    espectros_info = [
        {"id": row["id"], "nombre": f"{row['muestra']} – {row['archivo']}"}
        for _, row in df_filtrado.iterrows()
    ]

    seleccionados = st.multiselect(
        "Seleccionar espectros a visualizar:",
        options=[e["id"] for e in espectros_info],
        format_func=lambda i: next(e["nombre"] for e in espectros_info if e["id"] == i)
    )

    df_sel = df_filtrado[df_filtrado["id"].isin(seleccionados)]

    # --- Zona RMN 1H ---
    st.subheader("🔬 RMN 1H")
    df_rmn1H = df_sel[(df_sel["tipo"] == "RMN 1H") & (~df_sel["es_imagen"])].copy()
    if df_rmn1H.empty:
        st.info("No hay espectros RMN 1H numéricos seleccionados.")
    else:
        activar_mascara = st.checkbox("Máscara D/T2", value=False)
        usar_mascara = {}
        if activar_mascara:
            st.markdown("<div style='margin-left: 2rem; padding-bottom: 0.5rem;'>", unsafe_allow_html=True)
            col_checkboxes = st.columns(len(df_rmn1H))
            for idx, (_, row) in enumerate(df_rmn1H.iterrows()):
                with col_checkboxes[idx]:
                    usar_mascara[row["id"]] = st.checkbox(
                        f"{row['muestra']}",
                        value=False,
                        key=f"chk_mask_{row['id']}_{idx}"
                    )
            st.markdown("</div>", unsafe_allow_html=True)
            
        activar_calculos = st.checkbox("Cálculos D/T2", value=False)

    # --- Rango de visualización (todo en una fila) ---
    if not df_rmn1H.empty:
        all_x = []
        all_y = []
        for _, row in df_rmn1H.iterrows():
            try:
                contenido = BytesIO(base64.b64decode(row["contenido"]))
                extension = os.path.splitext(row["archivo"])[1].lower()
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
                all_x.extend(df[col_x].values.tolist())
                all_y.extend(df[col_y].values.tolist())
            except:
                continue

        if all_x and all_y:
            colx1, colx2, coly1, coly2 = st.columns(4)
            x_min = colx1.number_input("X mínimo", value=float(np.min(all_x)))
            x_max = colx2.number_input("X máximo", value=float(np.max(all_x)))
            y_min = coly1.number_input("Y mínimo", value=float(np.min(all_y)))
            y_max = coly2.number_input("Y máximo", value=float(np.max(all_y)))
        else:
            x_min, x_max, y_min, y_max = None, None, None, None


        colores = plt.cm.tab10.colors
        fig, ax = plt.subplots()


        # Graficar todos los espectros seleccionados
        for idx, (_, row) in enumerate(df_rmn1H.iterrows()):
            color = colores[idx % len(colores)]
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

                ax.plot(df[col_x], df[col_y], label=f"{row['muestra']}", color=color)
                if usar_mascara.get(row["id"], False):
                    mascaras = row.get("mascaras", [])
                    graficar_mascaras(df, col_x, col_y, mascaras, ax, color)
            except Exception as e:
                st.warning(f"No se pudo graficar espectro: {row['archivo']}")


        # Cálculos D/T2 sólo si el checkbox está activado
        if activar_calculos:
            columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2", 
                            "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]

            doc_dt2 = db.collection("tablas_dt2").document("cuantificable")
            doc_data = doc_dt2.get().to_dict() or {}
            filas_guardadas = doc_data.get("filas", [])
            df_dt2 = pd.DataFrame(filas_guardadas)

            for col in columnas_dt2:
                if col not in df_dt2.columns:
                    df_dt2[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
            df_dt2 = df_dt2[columnas_dt2]

            with st.form("form_edicion_dt2"):
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
                        "Área as": st.column_config.NumberColumn(format="%.2f", label="🔴Área as", disabled=True),
                        "Has": st.column_config.NumberColumn(format="%.2f"),
                        "H": st.column_config.NumberColumn(format="%.2f", label="🔴H", disabled=True),
                        "Observaciones": st.column_config.TextColumn(),
                    },
                    hide_index=True,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="tabla_dt2_cuantificable"
                )

                recalcular = st.form_submit_button("🔴Recalcular 'Área', 'Área as' y 'H'")


                if recalcular:
                    for i, row in df_dt2_edit.iterrows():
                        try:
                            nombre_muestra = row.get("Muestra")
                            archivo = row.get("Archivo")
                            x_min = float(row.get("X min"))
                            x_max = float(row.get("X max"))
                            try:
                                xas_min = float(row.get("Xas min")) if row.get("Xas min") not in [None, ""] else None
                                xas_max = float(row.get("Xas max")) if row.get("Xas max") not in [None, ""] else None
                                has = float(row.get("Has")) if row.get("Has") not in [None, ""] else None
                            except ValueError:
                                xas_min = xas_max = has = None

                            espectros_muestra = df_rmn1H[df_rmn1H["muestra"] == nombre_muestra]
                            if espectros_muestra.empty:
                                continue

                            if not archivo or archivo not in list(espectros_muestra["archivo"]):
                                archivo = espectros_muestra.iloc[0]["archivo"]
                                df_dt2_edit.at[i, "Archivo"] = archivo

                            espectro_row = espectros_muestra[espectros_muestra["archivo"] == archivo].iloc[0]
                            contenido = BytesIO(base64.b64decode(espectro_row["contenido"]))
                            extension = os.path.splitext(archivo)[1].lower()

                            if extension == ".xlsx":
                                df_espectro = pd.read_excel(contenido)
                            else:
                                for sep in [",", ";", "\t", " "]:
                                    contenido.seek(0)
                                    try:
                                        df_espectro = pd.read_csv(contenido, sep=sep)
                                        if df_espectro.shape[1] >= 2:
                                            break
                                    except:
                                        continue
                                else:
                                    continue

                            col_x, col_y = df_espectro.columns[:2]
                            df_espectro[col_x] = pd.to_numeric(df_espectro[col_x], errors="coerce")
                            df_espectro[col_y] = pd.to_numeric(df_espectro[col_y], errors="coerce")
                            df_espectro = df_espectro.dropna().sort_values(by=col_x)

                            # Cálculo área total
                            df_sub = df_espectro[(df_espectro[col_x] >= min(x_min, x_max)) & (df_espectro[col_x] <= max(x_min, x_max))]
                            area = np.trapz(df_sub[col_y], df_sub[col_x]) if not df_sub.empty else None
                            df_dt2_edit.at[i, "Área"] = round(area, 2) if area is not None else None

                            # Cálculo área as y H
                            if xas_min is not None and xas_max is not None:
                                df_sub_as = df_espectro[(df_espectro[col_x] >= min(xas_min, xas_max)) & (df_espectro[col_x] <= max(xas_min, xas_max))]
                                area_as = np.trapz(df_sub_as[col_y], df_sub_as[col_x]) if not df_sub_as.empty else None
                                df_dt2_edit.at[i, "Área as"] = round(area_as, 2) if area_as is not None else None

                                if area_as and has and area_as != 0 and area:
                                    h_calc = (area * has) / area_as
                                    df_dt2_edit.at[i, "H"] = round(h_calc, 2)

                        except Exception as e:
                            st.warning(f"⚠️ Error en fila {i}: {e}")
                            continue

                    doc_dt2.set({"filas": df_dt2_edit.to_dict(orient="records")})
                    st.success("✅ Área, Área as y H recalculadas correctamente")
                    st.rerun()




        ax.set_xlabel("[ppm]")
        ax.set_ylabel("Señal")
        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

        ax.legend()
        
        trazar_deltas = st.session_state.get("mostrar_deltas", False)
        
        # --- Leer datos de tabla y preparar valores δ pico si corresponde ---
        tabla_path_rmn1h = "tabla_editable_rmn1h"
        doc_ref = db.collection("configuracion_global").document(tabla_path_rmn1h)

        # Crear documento si no existe
        if not doc_ref.get().exists:
            doc_ref.set({"filas": []})

        # Obtener el documento
        doc_tabla = doc_ref.get()
        columnas_rmn1h = ["Tipo de muestra", "Grupo funcional", "X min", "δ pico", "X max", "Observaciones"]
        filas_rmn1h = doc_tabla.to_dict().get("filas", [])

        df_rmn1h_tabla = pd.DataFrame(filas_rmn1h)
        for col in columnas_rmn1h:
            if col not in df_rmn1h_tabla.columns:
                df_rmn1h_tabla[col] = "" if col in ["Tipo de muestra", "Grupo funcional", "Observaciones"] else np.nan
        df_rmn1h_tabla = df_rmn1h_tabla[columnas_rmn1h]


        
        if trazar_deltas and not df_rmn1h_tabla.empty:
            for _, row in df_rmn1h_tabla.iterrows():
                try:
                    delta = float(row["δ pico"])
                    etiqueta = str(row.get("Grupo funcional", "")).strip()
                    if etiqueta and not np.isnan(delta):
                        ax.axvline(x=delta, color="gray", linestyle="dashed", linewidth=1)
                        ax.text(
                            delta,
                            ax.get_ylim()[1]*0.95,
                            etiqueta,
                            rotation=90,
                            va="top", ha="center",
                            fontsize=7, color="gray"
                        )
                except:
                    continue







        st.pyplot(fig)

        # Botón para descargar imagen del gráfico RMN 1H
        buffer_img = BytesIO()
        fig.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
        st.download_button("📷 Descargar gráfico RMN 1H", data=buffer_img.getvalue(), file_name="grafico_rmn1h.png", mime="image/png")            
       

        # --- Mostrar tabla editable ---
        df_edit_rmn1h = st.data_editor(
            df_rmn1h_tabla,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="editor_tabla_rmn1h",
            column_config={
                "X min": st.column_config.NumberColumn(format="%.2f"),
                "δ pico": st.column_config.NumberColumn(format="%.2f"),
                "X max": st.column_config.NumberColumn(format="%.2f")
            }
        )


        # --- BOTONES en la misma fila ---
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔴Actualizar Tabla Bibliográfica"):
                doc_ref.set({"filas": df_edit_rmn1h.to_dict(orient="records")})
                st.success("✅ Tabla bibliográfica actualizada")
                st.rerun()

        with col2:
            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                df_edit_rmn1h.to_excel(writer, index=False, sheet_name="Tabla_Bibliográfica_RMN1H")
            buffer_excel.seek(0)
            st.download_button(
                "📥 Descargar Tabla Bibliográfica",
                data=buffer_excel.getvalue(),
                file_name="tabla_bibliografica_rmn1h.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # --- Checkbox para activar las líneas δ pico ---
        st.session_state["mostrar_deltas"] = st.checkbox("Señales δ pico", value=st.session_state.get("mostrar_deltas", False))



        # --- Trazar líneas verticales para 'δ pico' si se activa ---
       # trazar_deltas = st.checkbox("Señales δ pico", value=False)






        # --- Formulario de edición y botón limpio ---
        activar_edicion = st.checkbox("Cálculo de señales", value=False)

        if activar_edicion:
            columnas_integral = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                                "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]

            doc_ref = db.collection("tablas_integrales").document("rmn1h")
            if not doc_ref.get().exists:
                filas_iniciales = [{"Muestra": m, "Grupo funcional": ""} for m in muestras_sel]
                doc_ref.set({"filas": filas_iniciales})

            # Obtener y preparar datos actuales
            filas_actuales = doc_ref.get().to_dict().get("filas", [])
            df_integral = pd.DataFrame(filas_actuales)

            for col in columnas_integral:
                if col not in df_integral.columns:
                    df_integral[col] = "" if col == "Observaciones" else None
            df_integral["Observaciones"] = df_integral["Observaciones"].astype(str)
            df_integral = df_integral[columnas_integral]

            with st.form("form_edicion_integral"):
                df_integral_edit = st.data_editor(
                    df_integral,
                    column_config={
                        "Muestra": st.column_config.SelectboxColumn(options=muestras_sel, required=True),
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
                        "H": st.column_config.NumberColumn(format="%.2f", label="🔴H", disabled=True),
                        "Observaciones": st.column_config.TextColumn(),
                        "Archivo": st.column_config.TextColumn(),
                    },
                    hide_index=True,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="tabla_integral_edicion"
                )

                recalcular = st.form_submit_button("🔴Recalcular 'Área', 'Área as' y 'H'")

            if recalcular:
                doc_ref.set({"filas": df_integral_edit.to_dict(orient="records")})
                filas_actualizadas = doc_ref.get().to_dict().get("filas", [])
                df_final = pd.DataFrame(filas_actualizadas)

                for col in columnas_integral:
                    if col not in df_final.columns:
                        df_final[col] = "" if col == "Observaciones" else None
                df_final["Observaciones"] = df_final["Observaciones"].astype(str)
                df_final = df_final[columnas_integral]

                for i, row in df_final.iterrows():
                    try:
                        nombre_muestra = row.get("Muestra", None)
                        x_min = float(row.get("X min", None))
                        x_max = float(row.get("X max", None))
                        espectros_muestra = df_rmn1H[df_rmn1H["muestra"] == nombre_muestra]
                        if espectros_muestra.empty:
                            continue

                        archivo = row.get("Archivo", "")
                        if not archivo or archivo not in list(espectros_muestra["archivo"]):
                            archivo = espectros_muestra.iloc[0]["archivo"]
                            df_final.at[i, "Archivo"] = archivo

                        espectro_row = espectros_muestra[espectros_muestra["archivo"] == archivo].iloc[0]
                        contenido = BytesIO(base64.b64decode(espectro_row["contenido"]))
                        extension = os.path.splitext(archivo)[1].lower()

                        if extension == ".xlsx":
                            df_espectro = pd.read_excel(contenido)
                        else:
                            for sep in [",", ";", "\t", " "]:
                                contenido.seek(0)
                                try:
                                    df_espectro = pd.read_csv(contenido, sep=sep)
                                    if df_espectro.shape[1] >= 2:
                                        break
                                except:
                                    continue
                            else:
                                continue

                        col_x, col_y = df_espectro.columns[:2]
                        df_espectro[col_x] = pd.to_numeric(df_espectro[col_x], errors="coerce")
                        df_espectro[col_y] = pd.to_numeric(df_espectro[col_y], errors="coerce")
                        df_espectro = df_espectro.dropna()
                        df_espectro = df_espectro.sort_values(by=col_x)

                        df_sub = df_espectro[(df_espectro[col_x] >= min(x_min, x_max)) & (df_espectro[col_x] <= max(x_min, x_max))]
                        area = np.trapz(df_sub[col_y], df_sub[col_x]) if not df_sub.empty else np.nan
                        df_final.at[i, "Área"] = round(area, 2) if not np.isnan(area) else None

                        try:
                            has = float(row.get("Has", None))
                            xas_min = float(row.get("Xas min", None))
                            xas_max = float(row.get("Xas max", None))
                        except (TypeError, ValueError):
                            continue

                        df_sub_as = df_espectro[(df_espectro[col_x] >= min(xas_min, xas_max)) & (df_espectro[col_x] <= max(xas_min, xas_max))]
                        area_as = np.trapz(df_sub_as[col_y], df_sub_as[col_x]) if not df_sub_as.empty else np.nan
                        df_final.at[i, "Área as"] = round(area_as, 2) if not np.isnan(area_as) else None

                        if not np.isnan(area) and not np.isnan(area_as) and area_as != 0:
                            h_calc = (area * has) / area_as
                            df_final.at[i, "H"] = round(h_calc, 2)

                    except Exception as e:
                        st.warning(f"⚠️ Error en fila {i}: {e}")
                        continue

                doc_ref.set({"filas": df_final.to_dict(orient="records")})
                st.rerun()


            # ---- Mostrar botón de descarga siempre con últimos datos guardados ----
            doc_ref = db.collection("tablas_integrales").document("rmn1h")
            filas_guardadas = doc_ref.get().to_dict().get("filas", [])
            df_export = pd.DataFrame(filas_guardadas)
            if not df_export.empty:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    df_export.to_excel(writer, index=False, sheet_name="Integrales_RMN")

                st.download_button(
                    label="📥 Descargar Tabla Excel",
                    data=excel_buffer.getvalue(),
                    file_name="Tabla_RMN1H_YYYY-MM-DD_HH-MM-SS.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )






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
            zip_path = os.path.join(tmpdir, f"imagenes_rmn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for _, row in df_rmn_img.iterrows():
                    nombre = row["archivo"]
                    contenido = row["contenido"]
                    if not contenido:
                        continue
                    try:
                        img_bytes = base64.b64decode(contenido)
                        ruta = os.path.join(tmpdir, nombre)
                        with open(ruta, "wb") as f:
                            f.write(img_bytes)
                        zipf.write(ruta, arcname=nombre)
                    except:
                        continue
            with open(zip_path, "rb") as final_zip:
                st.download_button("📦 Descargar imágenes RMN", data=final_zip.read(), file_name=os.path.basename(zip_path), mime="application/zip")

    mostrar_sector_flotante(db, key_suffix="tab6")
