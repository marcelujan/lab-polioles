# tabs_tab3_espectros.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from datetime import datetime, date
import os
import base64
import json
from tempfile import TemporaryDirectory
import zipfile
from firebase_admin import storage
import plotly.graph_objects as go

def obtener_ids_espectros(nombre):
    return [doc.id for doc in firestore.Client().collection("muestras").document(nombre).collection("espectros").list_documents()]

def obtener_espectros_para_muestra(db, nombre):
    clave = f"_espectros_cache_{nombre}"
    if clave not in st.session_state:
        ref = db.collection("muestras").document(nombre).collection("espectros")
        docs = ref.stream()
        st.session_state[clave] = [doc.to_dict() for doc in docs]
    return st.session_state[clave]

def render_tab3(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Carga de espectros")
    st.session_state["current_tab"] = "Carga de espectros"
    muestras = cargar_muestras(db)
    nombres_muestras = [m["nombre"] for m in muestras]

    st.subheader("Subir nuevo espectro")
    nombre_sel = st.selectbox("Seleccionar muestra", nombres_muestras)
    st.session_state["muestra_activa"] = nombre_sel

    tipos_espectro_base = ["FTIR-Acetato", "FTIR-Cloroformo", "FTIR-ATR", "RMN 1H", "RMN 1H D", "RMN 1H T2", "RMN 1H LF", "RMN 13C"]
    if "tipos_espectro" not in st.session_state:
        st.session_state.tipos_espectro = tipos_espectro_base.copy()
    tipo_espectro = st.selectbox("Tipo de espectro", st.session_state.tipos_espectro)

    senal_3548 = senal_3611 = peso_muestra = None
    mascaras_rmn1h = []

    if tipo_espectro == "FTIR-Acetato":
        st.markdown("**Datos manuales opcionales para FTIR-Acetato:**")
        senal_3548 = st.number_input("Señal de Acetato a 3548 cm⁻¹", step=0.0001, format="%.4f")
        peso_muestra = st.number_input("Peso de la muestra [g]", step=0.0001, format="%.4f")

    elif tipo_espectro == "FTIR-Cloroformo":
        st.markdown("**Datos manuales opcionales para FTIR-Cloroformo:**")
        senal_3611 = st.number_input("Señal de Cloroformo a 3611 cm⁻¹", step=0.0001, format="%.4f")
        peso_muestra = st.number_input("Peso de la muestra [g]", step=0.0001, format="%.4f")

    elif tipo_espectro == "RMN 1H":
        st.markdown("**Máscaras D/T2 (opcional):**")
        n_mascaras = st.number_input("Cantidad de conjuntos D, T2, Xmin, Xmax", min_value=0, max_value=30, step=1, value=0)
        for i in range(n_mascaras):
            st.markdown(f"Máscara {i+1}:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                d = st.number_input(f"D [m2/s] {i+1}", key=f"d_{i}", format="%.2e")
            with col2:
                t2 = st.number_input(f"T2 [s] {i+1}", key=f"t2_{i}", format="%.3f")
            with col3:
                xmin = st.number_input(f"Xmin [ppm] {i+1}", key=f"xmin_{i}")
            with col4:
                xmax = st.number_input(f"Xmax [ppm] {i+1}", key=f"xmax_{i}")
            mascaras_rmn1h.append({"difusividad": d, "t2": t2, "x_min": xmin, "x_max": xmax})

    elif tipo_espectro == "RMN 1H T2":
        st.markdown("**Subir los 4 archivos requeridos para RMN 1H T2**")
        ppmAxis_file = st.file_uploader("Archivo ppmAxis.dat", type=["dat"], key="ppmAxis")
        T2axis_file = st.file_uploader("Archivo T2axis.dat", type=["dat"], key="T2axis")
        T2_proy_file = st.file_uploader("Archivo T2_proy.dat", type=["dat"], key="T2proy")
        ILT2D_file   = st.file_uploader("Archivo ILT2D.dat", type=["dat"], key="ILT2D")


        if ppmAxis_file and T2axis_file and T2_proy_file and ILT2D_file:
            try:
                ppmAxis = np.loadtxt(ppmAxis_file)
                T2axis = np.loadtxt(T2axis_file)
                T2_proy = np.loadtxt(T2_proy_file)
                ILT2D = np.loadtxt(ILT2D_file)
                z = ILT2D.T
                fecha_espectro = st.date_input("Fecha del espectro", value=date.today())
                observaciones = st.text_area("Observaciones", key="obs_rmn1h_t2")

                st.markdown("### Vista previa ILT2D")

                nivel = st.number_input("Nivel de contorno para vista previa", min_value=0.01, max_value=1.0, value=0.1, format="%.2f")

                fig2d = go.Figure()
                fig2d.add_trace(go.Contour(
                    x=ppmAxis,
                    y=T2axis,
                    z=z,
                    colorscale="Viridis",
                    contours=dict(
                        coloring="lines",
                        start=nivel,
                        end=nivel,
                        size=0.1,
                        showlabels=False
                    ),
                    line=dict(width=1.5),
                    showscale=False
                ))
                fig2d.update_layout(
                    xaxis=dict(
                        autorange=False,
                        range=[9, 0],
                        title="ppm"
                    ),
                    yaxis=dict(
                        type="log",
                        autorange=False,
                        range=[np.log10(T2axis.min()), np.log10(T2axis.max())],
                        title="T2 (s)"
                    ),
                    height=500
                )


                st.plotly_chart(fig2d, use_container_width=True)

                st.markdown("### Vista previa curva de decaimiento T2")

                fig1d = go.Figure()
                fig1d.add_trace(go.Scatter(
                    x=T2axis,
                    y=T2_proy,
                    mode="lines",
                    name="Proyección T2"
                ))
                fig1d.update_layout(
                    xaxis_title="T2 (s)",
                    yaxis_title="Intensidad",
                    xaxis_type="log",
                    height=400
                )
                st.plotly_chart(fig1d, use_container_width=True)

            except Exception as e:
                st.warning(f"Error generando vista previa: {e}")


        if st.button("Guardar espectro RMN 1H T2"):
            if not (ppmAxis_file and T2axis_file and T2_proy_file and ILT2D_file):
                st.warning("Debes subir los 4 archivos obligatorios.")
            else:
                bucket = storage.bucket()
                fecha_str = fecha_espectro.strftime("%Y-%m-%d")
                resumen_obs = observaciones.replace("\n", " ").strip()[:30].replace(" ", "_")

                archivos_urls = {}
                for nombre_arch, fileobj in [
                    ("ppmAxis", ppmAxis_file),
                    ("T2axis", T2axis_file),
                    ("T2_proy", T2_proy_file),
                    ("ILT2D", ILT2D_file)
                ]:
                    nombre_final = f"{nombre_sel}_{tipo_espectro}_{fecha_str}_{nombre_arch}_{resumen_obs}.dat"
                    blob = bucket.blob(f"espectros/{nombre_final}")
                    blob.upload_from_string(fileobj.getvalue(), content_type="text/plain")
                    blob.make_public()
                    archivos_urls[nombre_arch] = blob.public_url

                nuevo = {
                    "tipo": tipo_espectro,
                    "observaciones": observaciones.strip(),
                    "archivo_original": "",  # no aplica
                    "nombre_archivo": f"{nombre_sel}_{tipo_espectro}_{fecha_str}",
                    "url_archivo": None,
                    "archivos": archivos_urls,
                    "es_imagen": False,
                    "fecha": str(fecha_espectro)
                }

                ref = db.collection("muestras").document(nombre_sel).collection("espectros")
                ref.document().set(nuevo)

                st.success("Espectro RMN 1H T2 guardado correctamente.")
                st.rerun()

    nuevo_tipo = st.text_input("¿Agregar nuevo tipo de espectro?", "")
    if nuevo_tipo and nuevo_tipo not in st.session_state.tipos_espectro:
        st.session_state.tipos_espectro.append(nuevo_tipo)
        tipo_espectro = nuevo_tipo

    observaciones = st.text_area("Observaciones")
    fecha_espectro = st.date_input("Fecha del espectro", value=date.today())
    archivo = st.file_uploader("Archivo del espectro", type=["xlsx", "csv", "txt", "png", "jpg", "jpeg"])

    if archivo:
        nombre_archivo = archivo.name
        extension = os.path.splitext(nombre_archivo)[1].lower()
        es_imagen = extension in [".png", ".jpg", ".jpeg"]
        st.markdown("### Vista previa")
        if es_imagen:
            st.image(archivo, use_container_width=True)
        else:
            try:
                if extension == ".xlsx":
                    df_esp = pd.read_excel(archivo)
                else:
                    df_esp = pd.read_csv(archivo, sep=None, engine="python")
                if df_esp.shape[1] >= 2:
                    col_x, col_y = df_esp.columns[:2]
                    min_x, max_x = float(df_esp[col_x].min()), float(df_esp[col_x].max())
                    x_range = st.slider("Rango eje X", min_value=min_x, max_value=max_x, value=(min_x, max_x))
                    df_filtrado = df_esp[(df_esp[col_x] >= x_range[0]) & (df_esp[col_x] <= x_range[1])]
                    fig, ax = plt.subplots()
                    ax.plot(df_filtrado[col_x], df_filtrado[col_y])
                    ax.set_xlabel(col_x)
                    ax.set_ylabel(col_y)
                    st.pyplot(fig)
                else:
                    st.warning("El archivo debe tener al menos dos columnas.")
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")


        # Generar nuevo nombre de archivo basado en muestra, tipo, fecha y resumen de observaciones
        extension = os.path.splitext(archivo.name)[1].lower().strip(".")
        resumen_obs = observaciones.replace("\n", " ").strip()[:30].replace(" ", "_")
        fecha_str = fecha_espectro.strftime("%Y-%m-%d")
        nombre_sin_ext = f"{nombre_sel}_{tipo_espectro}_{fecha_str}-{resumen_obs}"
        nombre_generado = f"{nombre_sin_ext}.{extension}"

        # Mostrar nombre final antes de guardar
        st.markdown(f"**🆔 Nuevo nombre asignado al archivo para su descarga:** `{nombre_generado}`")

    if st.button("Guardar espectro") and archivo:
        extension = os.path.splitext(archivo.name)[1].lower().strip(".")
        resumen_obs = observaciones.replace("\n", " ").strip()[:30].replace(" ", "_")
        fecha_str = fecha_espectro.strftime("%Y-%m-%d")
        nombre_sin_ext = f"{nombre_sel}_{tipo_espectro}_{fecha_str}-{resumen_obs}"
        nombre_generado = f"{nombre_sin_ext}.{extension}"

        if tipo_espectro == "RMN 1H D":
            # subir SOLO RMN 1H D a storage
            bucket = storage.bucket()
            blob = bucket.blob(f"espectros/{nombre_generado}")
            blob.upload_from_string(archivo.getvalue(), content_type=archivo.type)
            blob.make_public()
            url_publica = blob.public_url

            nuevo = {
                "tipo": tipo_espectro,
                "observaciones": observaciones.strip(),
                "archivo_original": archivo.name,
                "nombre_archivo": nombre_generado,
                "url_archivo": url_publica,
                "es_imagen": False,
                "fecha": str(fecha_espectro), 
                "peso_muestra": None,   
                "mascaras": []  
            }

        else:
            # resto de tipos va como siempre a Firestore con base64
            nuevo = {
                "tipo": tipo_espectro,
                "observaciones": observaciones.strip(), 
                "archivo_original": archivo.name, 
                "nombre_archivo": nombre_generado,
                "contenido": base64.b64encode(archivo.getvalue()).decode("utf-8"),
                "es_imagen": archivo.type.startswith("image/"),
                "fecha": str(fecha_espectro),
                "senal_3548": senal_3548,
                "senal_3611": senal_3611,
                "peso_muestra": peso_muestra,
                "mascaras": mascaras_rmn1h if tipo_espectro == "RMN 1H" else []
            }

        ref = db.collection("muestras").document(nombre_sel).collection("espectros")
        ref.document().set(nuevo)
        st.success("Espectro guardado.")
        st.rerun()

    filas = []
    filas_mascaras = []
    for m in muestras:
        espectros = obtener_espectros_para_muestra(db, m["nombre"])
        for i, e in enumerate(espectros):
            fila = {
                "Muestra": m["nombre"],
                "Tipo": e.get("tipo", ""),
                "Archivo": e.get("nombre_archivo", ""),
                "Fecha": e.get("fecha", ""),
                "Peso": e.get("peso_muestra", ""), 
                "Observaciones": e.get("observaciones", ""),
                "ID": f"{m['nombre']}__{i}"
            }
            if e.get("mascaras"):
                fila["Máscaras"] = json.dumps(e["mascaras"])
                for j, mascara in enumerate(e["mascaras"]):
                    filas_mascaras.append({
                        "Muestra": m["nombre"],
                        "Archivo": e.get("nombre_archivo", ""),
                        "Máscara N°": j+1,
                        "D [m2/s]": mascara.get("difusividad"),
                        "T2 [s]": mascara.get("t2"),
                        "Xmin [ppm]": mascara.get("x_min"),
                        "Xmax [ppm]": mascara.get("x_max")
                    })
            else:
                fila["Máscaras"] = ""
            filas.append(fila)

    df_esp_tabla = pd.DataFrame(filas)
    df_esp_tabla = df_esp_tabla.sort_values(by=["Muestra", "Fecha", "Observaciones"]).reset_index(drop=True)
    df_mascaras = pd.DataFrame(filas_mascaras)

    # Mostrar resumen solo si el usuario lo solicita
    if st.checkbox("Espectros cargados"):
        if not df_esp_tabla.empty:
            columnas_resumen = ["Muestra", "Tipo", "Fecha", "Peso", "Observaciones", "Archivo"]
            columnas_presentes = [col for col in columnas_resumen if col in df_esp_tabla.columns]
            st.data_editor(
                df_esp_tabla[columnas_presentes],
                use_container_width=True,
                hide_index=True,
                disabled=True,
                key="tabla_resumen_esp"
            )
        else:
            st.info("No hay espectros cargados.")


    # Editar, eliminar, descargar 
    if not df_esp_tabla.empty:
        if st.checkbox("Editar espectros"):
            columnas_visibles = ["ID", "Muestra", "Tipo", "Fecha", "Peso", "Observaciones"]
            df_edit = df_esp_tabla[columnas_visibles].copy()

            df_editor = st.data_editor(
                df_edit,
                column_config={
                    "ID": st.column_config.TextColumn("ID", disabled=True),
                    "Muestra": st.column_config.TextColumn(disabled=True),
                    "Tipo": st.column_config.TextColumn(disabled=True),
                    "Fecha": st.column_config.TextColumn(disabled=True),
                    "Peso": st.column_config.NumberColumn("Peso [g]", format="%.4f"),
                    "Observaciones": st.column_config.TextColumn("Observaciones")
                },
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="editor_obs_peso"
            )

            if st.button("💾 Guardar cambios"):
                cambios = 0
                for i, row in df_editor.iterrows():
                    original = df_esp_tabla[df_esp_tabla["ID"] == row["ID"]].iloc[0]
                    nombre, idx = row["ID"].split("__")
                    idx = int(idx)
                    espectros = obtener_espectros_para_muestra(db, nombre)
                    docs = list(db.collection("muestras").document(nombre).collection("espectros").list_documents())
                    if docs and idx < len(docs):
                        espectro_id = docs[idx].id
                        cambios_doc = {}
                        if row["Observaciones"] != original["Observaciones"]:
                            cambios_doc["observaciones"] = row["Observaciones"]
                        if not pd.isna(row["Peso"]) and row["Peso"] != original["Peso"]:
                            cambios_doc["peso_muestra"] = row["Peso"]
                        if cambios_doc:
                            db.collection("muestras").document(nombre).collection("espectros").document(espectro_id).update(cambios_doc)
                            cambios += 1
                st.success(f"{cambios} espectro(s) actualizado(s).")
                st.rerun()

        def descripcion_espectro(i):
            fila = df_esp_tabla[df_esp_tabla['ID'] == i].iloc[0]
            peso = fila.get("Peso", "—")
            return f"{fila['Muestra']} — {fila['Tipo']} — {fila['Fecha']} — {fila['Archivo']} — {peso} g"
        if st.checkbox("Eliminar espectro cargado"):
            seleccion = st.selectbox(
                "Seleccionar espectro a eliminar",
                df_esp_tabla["ID"],
                format_func=descripcion_espectro
            )        
            st.markdown("")
            confirmar = st.checkbox(
                f"Confirmar eliminación del espectro: {df_esp_tabla[df_esp_tabla['ID'] == seleccion]['Archivo'].values[0]}",
                key="chk_eliminar_esp"
            )
            if st.button("Eliminar espectro"):
                if confirmar:
                    nombre, idx = seleccion.split("__")
                    for m in muestras:
                        if m["nombre"] == nombre:
                            espectros = obtener_espectros_para_muestra(db, nombre)
                            docs = list(db.collection("muestras").document(nombre).collection("espectros").list_documents())
                            idx = int(idx)
                            if docs and idx < len(docs):
                                espectro_id = docs[idx].id
                            else:
                                st.warning(f"No hay espectros disponibles para la muestra '{nombre}'")
                                return
                            db.collection("muestras").document(nombre).collection("espectros").document(espectro_id).delete()
                            st.success("Espectro eliminado.")
                            st.rerun()
                else:
                    st.warning("Debes confirmar la eliminación marcando la casilla.")



        if st.button("📦 Preparar descarga"):  # Preparar descarga de espectros (Excel y ZIP)
            with TemporaryDirectory() as tmpdir:
                # Construir nombre único del ZIP usando la primera fila visible
                primera_fila = df_esp_tabla.iloc[0].to_dict() if not df_esp_tabla.empty else {}
                muestra = primera_fila.get("muestra", "Desconocida")
                tipo = primera_fila.get("tipo", "RMN")
                fecha = primera_fila.get("fecha", datetime.now().strftime("%Y-%m-%d"))
                archivo = primera_fila.get("archivo", "archivo")
                peso = primera_fila.get("peso") or primera_fila.get("peso_muestra") or "?"

                nombre_zip = f"{muestra} — {tipo} — {fecha} — {archivo} — {peso} g".replace(" ", "_").replace("—", "-")
                nombre_zip = nombre_zip.replace(":", "-").replace("/", "-").replace("\\", "-")
                zip_path = os.path.join(tmpdir, f"{nombre_zip}.zip")

                # Exportar tabla de espectros a Excel
                excel_path = os.path.join(tmpdir, "tabla_espectros.xlsx")
                with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
                    df_esp_tabla.drop(columns=["ID"]).to_excel(writer, index=False, sheet_name="Espectros")
                    if not df_mascaras.empty:
                        df_mascaras.to_excel(writer, index=False, sheet_name="Mascaras_RMN1H")

                # Preparar archivo ZIP
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    zipf.write(excel_path, arcname="tabla_espectros.xlsx")

                    for m in muestras:
                        espectros = obtener_espectros_para_muestra(db, m["nombre"])
                        for i, e in enumerate(espectros):
                            contenido = e.get("contenido")
                            if not contenido:
                                continue

                            # Campos base
                            nombre_original = e.get("nombre_archivo", f"espectro_{i}.xlsx")
                            muestra_abrev = m["nombre"].replace(" ", "")[:8]
                            fecha = e.get("fecha", "fecha").replace(" ", "_")
                            tipo = e.get("tipo", "tipo").replace(" ", "_")
                            contenido_bytes = base64.b64decode(contenido)

                            # Generar hash único con contenido + índice
                            import hashlib
                            hash_id = hashlib.sha1(contenido_bytes + str(i).encode()).hexdigest()[:6]

                            # Crear nombre corto y único para el archivo
                            nombre_archivo = os.path.splitext(nombre_original)[0][:25]
                            nombre_archivo = nombre_archivo.replace(" ", "_").replace("/", "-").replace("\\", "-")
                            nombre_final = f"{muestra_abrev}_{tipo}_{fecha}_idx{i}_{hash_id}.xlsx"
                            nombre_final = nombre_final.replace("—", "-").replace(" ", "_").replace(":", "-")

                            # Guardar archivo temporal
                            file_path = os.path.join(tmpdir, nombre_final)
                            with open(file_path, "wb") as f:
                                f.write(contenido_bytes)

                            # Agregar al ZIP (sin carpetas internas)
                            zipf.write(file_path, arcname=nombre_final)

                # Guardar ZIP en sesión para descarga
                with open(zip_path, "rb") as final_zip:
                    zip_bytes = final_zip.read()
                    st.session_state["zip_bytes"] = zip_bytes
                    st.session_state["zip_name"] = os.path.basename(zip_path)

        # Botón de descarga del ZIP preparado
        if "zip_bytes" in st.session_state:
            st.download_button("📦 Descargar espectros", data=st.session_state["zip_bytes"],
                            file_name=st.session_state["zip_name"],
                            mime="application/zip")

    else:
        st.info("No hay espectros cargados.")
