# tabs_tab4_espectros.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import base64
import os
import zipfile
from tempfile import TemporaryDirectory


def obtener_espectros_para_muestra(db, nombre):
    ref = db.collection("muestras").document(nombre).collection("espectros")
    docs = ref.stream()
    return [doc.to_dict() for doc in docs]

def render_tab4(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Análisis de espectros")
    st.session_state["current_tab"] = "Análisis de espectros"
    
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    espectros_info = []
    for m in muestras:
        espectros = obtener_espectros_para_muestra(db, m["nombre"])
        for e in espectros:
            espectros_info.append({
                "Muestra": m["nombre"],
                "Tipo": e.get("tipo", ""),
                "Nombre archivo": e.get("nombre_archivo", ""),
                "Fecha": e.get("fecha", ""),
                "Observaciones": e.get("observaciones", ""),
                "Contenido": e.get("contenido"),
                "Es imagen": e.get("es_imagen", False)
            })

            espectros_info.append({
                "Muestra": m["nombre"],
                "Tipo": e.get("tipo", ""),
                "Nombre archivo": e.get("nombre_archivo", ""),
                "Fecha": e.get("fecha", ""),
                "Observaciones": e.get("observaciones", ""),
                "Contenido": e.get("contenido"),
                "Es imagen": e.get("es_imagen", False)
            })
    df_esp = pd.DataFrame(espectros_info)
    if df_esp.empty:
        st.warning("No hay espectros cargados.")
        st.stop()

    # Filtros de búsqueda de espectros
    st.subheader("Filtrar espectros")
    muestras_disp = df_esp["Muestra"].unique().tolist()
    tipos_disp = df_esp["Tipo"].unique().tolist()
    muestras_sel = st.multiselect("Muestras", muestras_disp, default=[])

    st.session_state["muestra_activa"] = muestras_sel[0] if len(muestras_sel) == 1 else None
    tipos_sel = st.multiselect("Tipo de espectro", tipos_disp, default=[])
    st.session_state["tipo_espectro_activo"] = tipos_sel[0] if len(tipos_sel) == 1 else None

    df_filtrado = df_esp[df_esp["Muestra"].isin(muestras_sel) & df_esp["Tipo"].isin(tipos_sel)]

    # Generar nombres para cada espectro disponible
    espectros_info = []
    for idx, row in df_filtrado.iterrows():
        fecha = row.get("Fecha", "Sin fecha")
        observaciones = row.get("Observaciones", "Sin observaciones")
        if len(observaciones) > 80:
            observaciones = observaciones[:77] + "..."
        ext = os.path.splitext(row["Nombre archivo"])[1].lower().strip(".")
        nombre = f"{row['Muestra']} – {row['Tipo']} – {fecha} – {observaciones} ({ext})"
        espectros_info.append({"identificador": idx, "nombre": nombre})

    # Selección de espectros a visualizar
    if espectros_info:
        seleccionados_nombres = st.multiselect("Seleccionar espectros a visualizar:", [e["nombre"] for e in espectros_info], default=[])
        seleccionados_idx = [e["identificador"] for e in espectros_info if e["nombre"] in seleccionados_nombres]
        df_filtrado = df_filtrado.loc[seleccionados_idx]
    else:
        st.warning("No hay espectros disponibles para seleccionar.")

    # Separar espectros numéricos y espectros en imagen
    df_datos = df_filtrado[~df_filtrado["Es imagen"]]
    df_imagenes = df_filtrado[df_filtrado["Es imagen"]]

    # Gráfico combinado para espectros numéricos
    if not df_datos.empty:
        st.subheader("Gráfico combinado de espectros numéricos")
        fig, ax = plt.subplots()
        rango_x = [float("inf"), float("-inf")]
        rango_y = [float("inf"), float("-inf")]
        data_validos = []

        # Decodificación y limpieza de cada archivo
        for _, row in df_datos.iterrows():
            try:
                extension = os.path.splitext(row["Nombre archivo"])[1].lower()
                contenido = BytesIO(base64.b64decode(row["Contenido"]))
                if extension == ".xlsx":
                    df = pd.read_excel(contenido)
                else:
                    for sep in [",", ";", "\t", " "]:
                        contenido.seek(0)
                        try:
                            df = pd.read_csv(contenido, sep=sep, engine="python")
                            if df.shape[1] >= 2:
                                break
                        except:
                            continue
                    else:
                        continue
                col_x, col_y = df.columns[:2]

                # Conversión forzada a numérico (corrige errores de coma como decimal)
                for col in [col_x, col_y]:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna()
                if df.empty:
                    continue
                data_validos.append((row["Muestra"], row["Tipo"], df[col_x], df[col_y]))

                # Ajustar rango global automático
                rango_x[0] = min(rango_x[0], df[col_x].min())
                rango_x[1] = max(rango_x[1], df[col_x].max())
                rango_y[0] = min(rango_y[0], df[col_y].min())
                rango_y[1] = max(rango_y[1], df[col_y].max())
            except:
                continue

        # Graficar solo si hay datos válidos
        if not data_validos:
            st.warning("No se pudo graficar ningún espectro válido.")
        else:
            # Selección de rangos de visualización
            col1, col2, col3, col4 = st.columns(4)
            x_min = col1.number_input("X mínimo", value=rango_x[0])
            x_max = col2.number_input("X máximo", value=rango_x[1])
            y_min = col3.number_input("Y mínimo", value=rango_y[0])
            y_max = col4.number_input("Y máximo", value=rango_y[1])

            # Graficar todos los espectros seleccionados
            for muestra, tipo, x, y in data_validos:
                x_filtrado = x[(x >= x_min) & (x <= x_max)]
                y_filtrado = y[(x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)]
                if not y_filtrado.empty:
                    ax.plot(x_filtrado[:len(y_filtrado)], y_filtrado, label=f"{muestra} – {tipo}")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.legend()
            st.pyplot(fig)

            # Exportar resumen y hojas individuales en Excel
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                resumen = pd.DataFrame()
                for muestra, tipo, x, y in data_validos:
                    x_filtrado = x[(x >= x_min) & (x <= x_max)]
                    y_filtrado = y[(x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)]
                    df_tmp = pd.DataFrame({f"X_{muestra}_{tipo}": x_filtrado[:len(y_filtrado)], f"Y_{muestra}_{tipo}": y_filtrado})
                    df_tmp.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
                    resumen = df_tmp if resumen.empty else pd.concat([resumen, df_tmp], axis=1)
                resumen.to_excel(writer, index=False, sheet_name="Resumen")
            excel_buffer.seek(0)
            st.download_button("\U0001F4E5 Exportar resumen a Excel", data=excel_buffer.getvalue(),
                               file_name=f"espectros_resumen_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Mostrar imágenes cargadas
    if not df_imagenes.empty:
        st.subheader("Imágenes de espectros")
        for _, row in df_imagenes.iterrows():
            try:
                imagen = BytesIO(base64.b64decode(row["Contenido"]))
                st.image(imagen, caption=f"{row['Muestra']} – {row['Tipo']} – {row['Fecha']}", use_container_width=True)
            except:
                st.warning(f"No se pudo mostrar la imagen: {row['Nombre archivo']}")

    # Descarga agrupada de imágenes seleccionadas + info TXT
    if not df_imagenes.empty and not df_imagenes[df_imagenes["Muestra"].isin(muestras_sel) & df_imagenes["Tipo"].isin(tipos_sel)].empty:
        st.subheader("Descargar imágenes seleccionadas")

    if st.button("\U0001F4E5 Descargar imágenes", key="descargar_imagenes"):
        seleccionadas = df_imagenes[df_imagenes["Muestra"].isin(muestras_sel) & df_imagenes["Tipo"].isin(tipos_sel)]
        with TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, f"imagenes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for _, row in seleccionadas.iterrows():
                    carpeta = row["Muestra"]
                    os.makedirs(os.path.join(tmpdir, carpeta), exist_ok=True)

                    # Guardar imagen
                    nombre_img = row["Nombre archivo"]
                    path_img = os.path.join(tmpdir, carpeta, nombre_img)
                    with open(path_img, "wb") as f:
                        f.write(base64.b64decode(row["Contenido"]))
                    zipf.write(path_img, arcname=os.path.join(carpeta, nombre_img))

                    # Guardar archivo TXT con metadatos
                    nombre_txt = os.path.splitext(nombre_img)[0] + ".txt"
                    path_txt = os.path.join(tmpdir, carpeta, nombre_txt)
                    with open(path_txt, "w", encoding="utf-8") as f:
                        f.write(f"Nombre del archivo: {nombre_img}\n")
                        f.write(f"Tipo de espectro: {row['Tipo']}\n")
                        f.write(f"Fecha: {row['Fecha']}\n")
                        f.write(f"Observaciones: {row['Observaciones']}\n")
                    zipf.write(path_txt, arcname=os.path.join(carpeta, nombre_txt))

            # Leer el ZIP y preparar para descarga
            with open(zip_path, "rb") as final_zip:
                zip_bytes = final_zip.read()
                
        st.download_button("\U0001F4E6 Descargar ZIP de imágenes", data=zip_bytes,
                           file_name=os.path.basename(zip_path), mime="application/zip")

    mostrar_sector_flotante(db, key_suffix="tab4")
