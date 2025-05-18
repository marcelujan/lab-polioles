# tabs_tab6_rmn.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64
from io import BytesIO
from datetime import datetime
from scipy.signal import find_peaks
from numpy import trapezoid as trapz

import os
import zipfile
from tempfile import TemporaryDirectory


def obtener_ids_espectros(nombre):
    return [doc.id for doc in firestore.Client().collection("muestras").document(nombre).collection("espectros").list_documents()]

def obtener_espectros_para_muestra(db, nombre):
    clave = f"_espectros_cache_{nombre}"
    if clave not in st.session_state:
        ref = db.collection("muestras").document(nombre).collection("espectros")
        docs = ref.stream()
        st.session_state[clave] = [doc.to_dict() for doc in docs]
    return st.session_state[clave]

def render_tab9(db, cargar_muestras, mostrar_sector_flotante):
    st.session_state["current_tab"] = "Análisis RMN 1H"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # --- Cargar espectros desde Firebase ---
    espectros_rmn = []
    for m in muestras:
        espectros = db.collection("muestras").document(m["nombre"]).collection("espectros").stream()
        for i, e in enumerate(espectros):
            datos = e.to_dict()
            tipo = datos.get("tipo", "")
            if not tipo.startswith("RMN"):
                continue
            espectros_rmn.append({
                "muestra": m["nombre"],
                "tipo": tipo,
                "es_imagen": datos.get("es_imagen", False),
                "archivo": datos.get("nombre_archivo", ""),
                "contenido": datos.get("contenido"),
                "fecha": datos.get("fecha"),
                "mascaras": datos.get("mascaras", []),
                "id": f"{m['nombre']}__{i}"
            })

    st.success(f"{len(espectros_rmn)} espectros RMN 1H recuperados.")

    # --- Tabla editable de señales ---
    st.subheader("👁️‍🗨️ Edición manual de señales detectadas")
    editar = st.checkbox("Edición de señales", value=False)

    columnas_tabla = [
        "Muestra", "Tipo", "δ pico", "X min", "X max", "Área", "D", "T2",
        "Xas min", "Xas max", "Has", "H", "Observaciones", "Archivo"
    ]

    if "tabla_rmn1h" not in st.session_state:
        st.session_state.tabla_rmn1h = pd.DataFrame(columns=columnas_tabla)

    if editar:
        tabla = st.data_editor(
            st.session_state.tabla_rmn1h,
            column_order=columnas_tabla,
            use_container_width=True,
            hide_index=True,
            key="editor_signales_rmn1h",
            num_rows="dynamic"
        )

        # Calcular H automáticamente para cada fila
        def calcular_H(row):
            try:
                xas_min = float(row["Xas min"])
                xas_max = float(row["Xas max"])
                has = float(row["Has"])
                x_min = float(row["X min"])
                x_max = float(row["X max"])
                area = float(row["Área"])
                if xas_max > xas_min and x_max > x_min:
                    area_asignada = area * ((xas_max - xas_min) / (x_max - x_min))
                    if area_asignada != 0:
                        return round((area / area_asignada) * has, 2)
            except:
                return "—"
            return "—"

        tabla["H"] = tabla.apply(calcular_H, axis=1)
        st.session_state.tabla_rmn1h = tabla
        st.dataframe(tabla, use_container_width=True)
    else:
        st.dataframe(st.session_state.tabla_rmn1h, use_container_width=True)

    # --- Decodificación base64 y preprocesamiento ---
    datos_rmn = []
    for espectro in espectros_rmn:
        if espectro["es_imagen"]:
            continue
        try:
            binario = BytesIO(base64.b64decode(espectro["contenido"]))
            ext = espectro["archivo"].split(".")[-1].lower()
            if ext == "xlsx":
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
                df = df.iloc[:, :2]
                df.columns = ["x", "y"]
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                espectro["df"] = df
                datos_rmn.append(espectro)
        except Exception as e:
            st.warning(f"No se pudo decodificar '{espectro['archivo']}': {e}")

    if not datos_rmn:
        st.warning("No se encontraron espectros válidos.")
        st.stop()

    st.success(f"{len(datos_rmn)} espectros RMN decodificados correctamente.")

    df_rmn = pd.DataFrame(datos_rmn)
    muestras_disp = sorted(df_rmn["muestra"].unique())
    muestras_sel = st.multiselect("Muestras", muestras_disp, default=muestras_disp[:1])
    df_filtrado = df_rmn[df_rmn["muestra"].isin(muestras_sel)]

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

    mostrar_linea_base = st.checkbox("Mostrar línea base (y = 0)", value=True)

    # --- Generación de tabla editable con anotaciones ---
    columnas_tabla = [
        "Muestra", "Tipo", "δ pico", "X min", "X max", "Área", "D", "T2",
        "Xas min", "Xas max", "Has", "H", "Observaciones", "Archivo"
    ]

    if "tabla_rmn1h" not in st.session_state:
        filas = []
        for _, row in df_sel.iterrows():
            df = row["df"].sort_values(by="x")
            x = df["x"].values
            y = df["y"].values
            if x[0] < x[-1]:
                x = x[::-1]
                y = y[::-1]
            peaks, _ = find_peaks(y, height=np.mean(y) + np.std(y))
            for p in peaks:
                if p <= 0 or p >= len(x)-1:
                    continue
                delta = x[p]
                left = max(0, p - 20)
                right = min(len(x) - 1, p + 20)
                area = trapz(y[left:right], x[left:right])
                filas.append({
                    "Muestra": row["muestra"],
                    "Tipo": row["tipo"],
                    "δ pico": round(delta, 2),
                    "X min": round(x[left], 2),
                    "X max": round(x[right], 2),
                    "Área": round(area, 2),
                    "D": "", "T2": "",
                    "Xas min": round(x[left], 2),
                    "Xas max": round(x[right], 2),
                    "Has": 1, "H": 1,
                    "Observaciones": "",
                    "Archivo": row["archivo"]
                })
        st.session_state.tabla_rmn1h = pd.DataFrame(filas)

    st.subheader("📝 Edición manual de señales")
    st.data_editor(
        st.session_state.tabla_rmn1h,
        use_container_width=True,
        hide_index=True,
        key="editor_tabla_rmn1h"
    )

   # --- Tabla editable de señales ---
    st.subheader("📝 Tabla editable de señales detectadas")
    columnas_tabla = [
        "Muestra", "Tipo", "δ pico", "X min", "X max", "Área", "D", "T2",
        "Xas min", "Xas max", "Has", "H", "Observaciones", "Archivo"
    ]

    if "tabla_rmn1h" not in st.session_state:
        tabla = []
        for _, row in df_sel.iterrows():
            df = row["df"].sort_values(by="x")
            x = df["x"].values
            y = df["y"].values
            if x[0] < x[-1]:
                x = x[::-1]
                y = y[::-1]
            peaks, _ = find_peaks(y, height=np.mean(y) + np.std(y))
            for i, p in enumerate(peaks):
                if i == 0 or i == len(peaks) - 1:
                    continue
                delta = x[p]
                x_min = round(x[max(0, p - 20)], 2)
                x_max = round(x[min(len(x)-1, p + 20)], 2)
                mask = (x >= x_min) & (x <= x_max)
                area = trapz(y[mask], x[mask])
                tabla.append({
                    "Muestra": row["muestra"],
                    "Tipo": row["tipo"],
                    "δ pico": round(delta, 2),
                    "X min": x_min,
                    "X max": x_max,
                    "Área": round(area, 2),
                    "D": "",
                    "T2": "",
                    "Xas min": x_min,
                    "Xas max": x_max,
                    "Has": 1,
                    "H": 1,
                    "Observaciones": "",
                    "Archivo": row["archivo"]
                })
        st.session_state["tabla_rmn1h"] = pd.DataFrame(tabla)

    st.data_editor(
        st.session_state["tabla_rmn1h"],
        column_order=columnas_tabla,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="editor_rmn1h"
    )

    # --- Visualización con anotaciones enriquecidas ---
    st.subheader("Visualización de espectros con anotaciones")
    fig, ax = plt.subplots()
    resumen = []
    tabla_auto = []

    for _, row in df_sel.iterrows():
        df = row["df"].sort_values(by="x")
        x = df["x"].values
        y = df["y"].values

        if x[0] < x[-1]:
            x = x[::-1]
            y = y[::-1]

        peaks, _ = find_peaks(y, height=np.mean(y) + np.std(y))
        integrales = []

        for i, p in enumerate(peaks):
            if i == 0 or i == len(peaks) - 1:
                continue
            left = max(0, p - 20)
            right = min(len(x) - 1, p + 20)
            area = trapz(y[left:right], x[left:right])
            delta = x[p]

            if 0.5 <= delta <= 1.5:
                grupo = "CH₃"
            elif 1.5 < delta <= 3.0:
                grupo = "CH₂ / CH"
            elif 3.0 < delta <= 4.5:
                grupo = "CH electronegativo"
            elif 5.0 < delta <= 6.5:
                grupo = "Vinílico"
            elif 6.5 < delta <= 9.0:
                grupo = "Aromático"
            elif 9.0 < delta <= 10.0:
                grupo = "Aldehído"
            elif 10.0 < delta <= 12.0:
                grupo = "Ácido / OH"
            else:
                grupo = "Desconocido"

            anotacion = f"{grupo}\nδ={delta:.2f} ppm\nÁrea={area:.2f}"
            ax.plot(x[p], y[p], "x")
            ax.text(x[p], y[p], anotacion, fontsize=6, ha="left", va="bottom", rotation=90)

            integrales.append({
                "δ (ppm)": round(delta, 2),
                "Área": round(area, 2),
                "Grupo funcional": grupo
            })

            tabla_auto.append({
                "Muestra": row["muestra"],
                "Tipo": row["tipo"],
                "δ pico": round(delta, 2),
                "X min": round(x[p]-0.05, 2),
                "X max": round(x[p]+0.05, 2),
                "Área": round(area, 2),
                "D": "",
                "T2": "",
                "Xas min": round(x[p]-0.05, 2),
                "Xas max": round(x[p]+0.05, 2),
                "Has": 1,
                "H": 1,
                "Observaciones": "",
                "Archivo": row["archivo"]
            })

        ax.plot(x, y, label=f"{row['muestra']} – {row['archivo']}")

        for i in integrales:
            resumen.append({
                "Muestra": row["muestra"],
                "Archivo": row["archivo"],
                **i
            })

    ax.set_xlabel("δ (ppm)")
    ax.set_ylabel("Intensidad")
    ax.invert_xaxis()
    ax.legend()
    st.pyplot(fig)

    if resumen:
        st.subheader("Tabla de integrales y anotaciones")
        df_resumen = pd.DataFrame(resumen)
        st.dataframe(df_resumen, use_container_width=True)

    # --- Tabla editable integrada ---
    st.subheader("📝 Edición de señales")
    editar = st.checkbox("Editar señales detectadas", value=True)

    columnas_tabla = [
        "Muestra", "Tipo", "δ pico", "X min", "X max", "Área", "D", "T2",
        "Xas min", "Xas max", "Has", "H", "Observaciones", "Archivo"
    ]

    if "tabla_rmn1h" not in st.session_state:
        st.session_state.tabla_rmn1h = pd.DataFrame(tabla_inicial)

    if editar:
        def calcular_H(row):
            try:
                xas_min = float(row["Xas min"])
                xas_max = float(row["Xas max"])
                has = float(row["Has"])
                x_min = float(row["X min"])
                x_max = float(row["X max"])
                area = float(row["Área"])
                if xas_max > xas_min and x_max > x_min:
                    area_asignada = area * ((xas_max - xas_min) / (x_max - x_min))
                    if area_asignada != 0:
                        return round((area / area_asignada) * has, 2)
            except:
                return "—"
            return "—"

        tabla = st.data_editor(
            st.session_state.tabla_rmn1h,
            column_order=columnas_tabla,
            use_container_width=True,
            hide_index=True,
            key="editor_tabla_rmn1h",
            num_rows="dynamic"
        )

        tabla["H"] = tabla.apply(calcular_H, axis=1)
        st.session_state.tabla_rmn1h = tabla
        st.dataframe(tabla, use_container_width=True)
    else:
        st.dataframe(st.session_state.tabla_rmn1h, use_container_width=True)

    # Mostrar tabla editable con datos detectados
    if st.checkbox("Editar señales detectadas", value=True):
        df_tabla = pd.DataFrame(tabla_auto)
        st.data_editor(
            df_tabla,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )


    # --- Controles de preprocesamiento ---
    aplicar_suavizado = st.checkbox("Aplicar suavizado (Savitzky-Golay)", value=False)
    normalizar = st.checkbox("Normalizar intensidad", value=False)

    # --- Preparar figura ---
    fig, ax = plt.subplots()
    y_mins, y_maxs = [], []

    for _, row in df_sel.iterrows():

        df = row["df"].copy()
        x = df["x"]
        y = df["y"]

    # Suavizado
    if aplicar_suavizado and len(y) >= 5:
        y = savgol_filter(y, window_length=7, polyorder=2)

    # Normalización
    if normalizar and np.max(np.abs(y)) != 0:
        y = y / np.max(np.abs(y))

    # Graficar espectro
    label = f"{row['muestra']} – {row['tipo']}"
    ax.plot(x, y, label=label)
    y_mins.append(y.min())
    y_maxs.append(y.max())

    # Dibujar máscaras si existen
    for mascara in row.get("mascaras", []):
        try:
            x_ini = float(mascara.get("x_min", np.nan))
            x_fin = float(mascara.get("x_max", np.nan))
            if not np.isnan(x_ini) and not np.isnan(x_fin):
                ax.axvspan(x_ini, x_fin, color="red", alpha=0.2)
        except Exception as e:
            st.warning(f"Máscara inválida para {label}: {e}")

    if y_mins and y_maxs:
        ax.set_ylim(min(y_mins), max(y_maxs))

    ax.set_xlabel("Desplazamiento químico δ (ppm)")
    ax.set_ylabel("Intensidad")
    ax.invert_xaxis()
    ax.legend()
    st.pyplot(fig)





    # --- Zona RMN 1H ---
    st.subheader("🔬 RMN 1H")
    df_rmn1H = df_sel[(df_sel["tipo"] == "RMN 1H") & (~df_sel["es_imagen"])].copy()
    if df_rmn1H.empty:
        st.info("No hay espectros RMN 1H numéricos seleccionados.")
    else:
        st.markdown("**Máscara D/T2:**")
        usar_mascara = {}
        colores = plt.cm.tab10.colors
        fig, ax = plt.subplots()

        # Mostrar checkbox para cada espectro
        for idx, (_, row) in enumerate(df_rmn1H.iterrows()):
            usar_mascara[row["id"]] = st.checkbox(
                f"{row['muestra']} – {row['archivo']}",
                value=False,
                key=f"chk_mask_{row['id']}_{idx}"
            )

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

            except Exception as e:
                st.warning(f"No se pudo graficar espectro: {row['archivo']}")


        # Solo si hay máscaras activadas se muestra la sección de asignación y se calculan áreas
        filas_mascaras = []
        mapa_mascaras = {}
        if any(usar_mascara.values()):
            st.markdown("**Asignación para cuantificación**")
            df_asignacion = pd.DataFrame([{"H": 1.0, "X mínimo": 4.8, "X máximo": 5.6}])
            df_asignacion_edit = st.data_editor(df_asignacion, hide_index=True, num_rows="fixed", use_container_width=True, key="asignacion")
            h_config = {
                "H": float(df_asignacion_edit.iloc[0]["H"]),
                "Xmin": float(df_asignacion_edit.iloc[0]["X mínimo"]),
                "Xmax": float(df_asignacion_edit.iloc[0]["X máximo"])}

            for idx, (_, row) in enumerate(df_rmn1H.iterrows()):
                if not usar_mascara.get(row['id'], False):
                    continue
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

                    # Calcular área de asignación H
                    df_h = df[(df[col_x] >= h_config["Xmin"]) & (df[col_x] <= h_config["Xmax"])]
                    integracion_h = np.trapz(df_h[col_y], df_h[col_x]) if not df_h.empty else np.nan
                    nuevas_mascaras = []
                    for j, mascara in enumerate(row.get("mascaras", [])):
                        x0 = mascara.get("x_min")
                        x1 = mascara.get("x_max")
                        d = mascara.get("difusividad")
                        t2 = mascara.get("t2")
                        obs = mascara.get("observacion", "")

                        sub_df = df[(df[col_x] >= min(x0, x1)) & (df[col_x] <= max(x0, x1))]
                        area = np.trapz(sub_df[col_y], sub_df[col_x]) if not sub_df.empty else 0
                        h = (area * h_config["H"]) / integracion_h if integracion_h else np.nan

                        ax.axvspan(x0, x1, color=color, alpha=0.3)
                        if d and t2:
                            ax.text((x0+x1)/2, max(df[col_y])*0.9,
                                    f"D={d:.1e}     T2={t2:.3f}", ha="center", va="center", fontsize=6, color="black", rotation=90)
                        nuevas_mascaras.append({
                            "difusividad": d,
                            "t2": t2,
                            "x_min": x0,
                            "x_max": x1,
                            "observacion": obs
                        })
                        filas_mascaras.append({
                            "ID espectro": row["id"],
                            "Muestra": row["muestra"],
                            "Archivo": row["archivo"],
                            "D [m2/s]": d,
                            "T2 [s]": t2,
                            "Xmin [ppm]": round(x0, 2),
                            "Xmax [ppm]": round(x1, 2),
                            "Área": round(area, 2),
                            "H": round(h, 2) if not np.isnan(h) else "—",
                            "Observación": obs
                        })
                    mapa_mascaras[row["id"]] = nuevas_mascaras
                except:
                    continue

            df_editable = pd.DataFrame(filas_mascaras)
            df_editable_display = st.data_editor(
                df_editable,
                column_config={"D [m2/s]": st.column_config.NumberColumn(format="%.2e"),
                               "Xmin [ppm]": st.column_config.NumberColumn(format="%.2f"),
                               "Xmax [ppm]": st.column_config.NumberColumn(format="%.2f"),
                               "Área": st.column_config.NumberColumn(format="%.2f"),
                               "H": st.column_config.NumberColumn(format="%.2f"),
                               "T2 [s]": st.column_config.NumberColumn(format="%.3f")},
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",
                key="editor_mascaras"
            )

            for i, row in df_editable_display.iterrows():
                id_esp = row["ID espectro"]
                idx = int(id_esp.split("__")[1])
                for m in muestras:
                    if m["nombre"] == id_esp.split("__")[0]:
                        espectros = m.get("espectros", [])
                        if idx < len(espectros):
                            espectros[idx]["mascaras"] = mapa_mascaras.get(id_esp, [])
                            guardar_muestra(db, m["nombre"], m.get("observacion", ""), m.get("analisis", []), espectros)

            st.caption(f"*Asignación: {int(h_config['H'])} H = integral entre x = {h_config['Xmin']} y x = {h_config['Xmax']}")
       
        ax.set_xlabel("[ppm]")
        ax.set_ylabel("Señal")
        ax.legend()
        
        st.pyplot(fig)

        if filas_mascaras:
            df_editable = pd.DataFrame(filas_mascaras)
            st.subheader("🧾 Tabla de máscaras aplicadas")
            st.dataframe(df_editable, use_container_width=True)


        # --- Tabla nueva debajo del gráfico RMN 1H ---
        tabla_path_rmn1h = "tabla_editable_rmn1h"
        doc_ref = db.collection("configuracion_global").document(tabla_path_rmn1h)

        # Crear documento si no existe
        if not doc_ref.get().exists:
            doc_ref.set({"filas": []})

        # Obtener el documento actualizado
        doc_tabla = doc_ref.get()
        columnas_rmn1h = ["Tipo de muestra", "Grupo funcional", "X min", "X pico", "X max", "Observaciones"]
        filas_rmn1h = doc_tabla.to_dict().get("filas", [])

        df_rmn1h_tabla = pd.DataFrame(filas_rmn1h)
        for col in columnas_rmn1h:
            if col not in df_rmn1h_tabla.columns:
                df_rmn1h_tabla[col] = "" if col in ["Tipo de muestra", "Grupo funcional", "Observaciones"] else np.nan
        df_rmn1h_tabla = df_rmn1h_tabla[columnas_rmn1h]  # asegurar orden

        df_edit_rmn1h = st.data_editor(
            df_rmn1h_tabla,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="editor_tabla_rmn1h",
            column_config={
                "X min": st.column_config.NumberColumn(format="%.2f"),
                "X pico": st.column_config.NumberColumn(format="%.2f"),
                "X max": st.column_config.NumberColumn(format="%.2f")})

        # Guardar si hay cambios
        if not df_edit_rmn1h.equals(df_rmn1h_tabla):
            doc_ref.set({"filas": df_edit_rmn1h.to_dict(orient="records")})

            # Botón de descarga de tabla de máscaras
            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                df_editable_display.drop(columns=["ID espectro"]).to_excel(writer, index=False, sheet_name="Mascaras_RMN1H")
            buffer_excel.seek(0)
            st.download_button("📁 Descargar máscaras D/T2", data=buffer_excel.getvalue(), file_name="mascaras_rmn1h.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Botón para descargar imagen del gráfico RMN 1H
        buffer_img = BytesIO()
        fig.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
        st.download_button("📷 Descargar gráfico RMN 1H", data=buffer_img.getvalue(), file_name="grafico_rmn1h.png", mime="image/png")            


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

    mostrar_sector_flotante(db, key_suffix="tab9")
