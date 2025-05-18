# tabs_tab6_rmn.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from datetime import datetime
import os
import base64
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

    df_rmn = pd.DataFrame(espectros_rmn)

    st.subheader("Filtrar espectros")
    if df_rmn.empty or "muestra" not in df_rmn.columns:
        st.warning("No hay espectros RMN disponibles.")
        st.stop()
    muestras_disp = sorted(df_rmn["muestra"].unique())
    muestras_sel = st.multiselect("Muestras", muestras_disp, default=[])
    st.session_state["muestra_activa"] = muestras_sel[0] if len(muestras_sel) == 1 else None

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

                    # Calcular área de referencia H
                    df_h = df[(df[col_x] >= h_config["Xmin"]) & (df[col_x] <= h_config["Xmax"])]
                    integracion_h = np.trapz(df_h[col_y], df_h[col_x]) if not df_h.empty else np.nan

                    # Aplicar máscaras y graficar
                    filas, advertencias = graficar_mascaras(df, col_x, col_y, row.get("mascaras", []), ax, color)

                    for f in filas:
                        h = (f["Área"] * h_config["H"]) / integracion_h if integracion_h else np.nan
                        filas_mascaras.append({
                            "ID espectro": row["id"],
                            "Muestra": row["muestra"],
                            "Archivo": row["archivo"],
                            "D [m2/s]": f["D"],
                            "T2 [s]": f["T2"],
                            "Xmin [ppm]": round(f["x_min"], 2),
                            "Xmax [ppm]": round(f["x_max"], 2),
                            "Área": round(f["Área"], 2),
                            "H": round(h, 2) if not np.isnan(h) else "—",
                            "Observación": f["Obs"]
                        })

                    for advertencia in advertencias:
                        st.warning(f"{row['muestra']} – {row['archivo']}: {advertencia}")

                    mapa_mascaras[row["id"]] = row.get("mascaras", [])

                except Exception as e:
                    st.warning(f"No se pudo procesar espectro: {row['archivo']}")


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

    mostrar_sector_flotante(db, key_suffix="tab6")
