# --- HOJA 4 ---
with tab4:
    st.title("Análisis de espectros")

    # --- Sincronización de selección usando session_state ---
    if "muestras_sel" not in st.session_state:
        st.session_state.muestras_sel = []
    if "tipos_sel" not in st.session_state:
        st.session_state.tipos_sel = []

    muestras_sel = st.multiselect("Muestras", muestras_disp, default=st.session_state.muestras_sel, key="muestras_sel")
    tipos_sel = st.multiselect("Tipo de espectro", tipos_disp, default=st.session_state.tipos_sel, key="tipos_sel")



    muestras = cargar_muestras()
    if not muestras:
        st.info("No hay muestras cargadas con espectros.")
        st.stop()

    espectros_info = []
    for m in muestras:
        for e in m.get("espectros", []):
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

    st.subheader("Filtrar espectros")
    muestras_disp = df_esp["Muestra"].unique().tolist()
    tipos_disp = df_esp["Tipo"].unique().tolist()
    muestras_sel = st.multiselect("Muestras", muestras_disp, default=[])
    tipos_sel = st.multiselect("Tipo de espectro", tipos_disp, default=[])

    df_filtrado = df_esp[df_esp["Muestra"].isin(muestras_sel) & df_esp["Tipo"].isin(tipos_sel)]
    df_datos = df_filtrado[~df_filtrado["Es imagen"]]
    df_imagenes = df_filtrado[df_filtrado["Es imagen"]]

    if not df_datos.empty:
        st.subheader("Gráfico combinado de espectros numéricos")

        import matplotlib.pyplot as plt
        import pandas as pd
        from io import BytesIO
        import base64

        fig, ax = plt.subplots()
        rango_x = [float("inf"), float("-inf")]
        rango_y = [float("inf"), float("-inf")]

        data_validos = []

        for _, row in df_datos.iterrows():
            try:
                extension = os.path.splitext(row["Nombre archivo"])[1].lower()
                if extension == ".xlsx":
                    binario = BytesIO(base64.b64decode(row["Contenido"]))
                    df = pd.read_excel(binario)
                else:
                    contenido = BytesIO(base64.b64decode(row["Contenido"]))
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
                        continue

                col_x, col_y = df.columns[:2]
                for col in [col_x, col_y]:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.dropna()
                if df.empty:
                    continue

                data_validos.append((row["Muestra"], row["Tipo"], df[col_x], df[col_y]))

                rango_x[0] = min(rango_x[0], df[col_x].min())
                rango_x[1] = max(rango_x[1], df[col_x].max())
                rango_y[0] = min(rango_y[0], df[col_y].min())
                rango_y[1] = max(rango_y[1], df[col_y].max())
            except:
                continue

        if not data_validos:
            st.warning("No se pudo graficar ningún espectro válido.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x_min = st.number_input("X mínimo", value=rango_x[0])
            with col2:
                x_max = st.number_input("X máximo", value=rango_x[1])
            with col3:
                y_min = st.number_input("Y mínimo", value=rango_y[0])
            with col4:
                y_max = st.number_input("Y máximo", value=rango_y[1])

            for muestra, tipo, x, y in data_validos:
                x_filtrado = x[(x >= x_min) & (x <= x_max)]
                y_filtrado = y[(x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)]
                if not y_filtrado.empty:
                    ax.plot(x_filtrado[:len(y_filtrado)], y_filtrado, label=f"{muestra} – {tipo}")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.legend()
            
            st.pyplot(fig)

            # Exportar Excel con resumen y hojas individuales
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                resumen = pd.DataFrame()
                for muestra, tipo, x, y in data_validos:
                    x_filtrado = x[(x >= x_min) & (x <= x_max)]
                    y_filtrado = y[(x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)]
                    df_tmp = pd.DataFrame({f"X_{muestra}_{tipo}": x_filtrado[:len(y_filtrado)],
                                           f"Y_{muestra}_{tipo}": y_filtrado})
                    df_tmp.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
                    if resumen.empty:
                        resumen = df_tmp.copy()
                    else:
                        resumen = pd.concat([resumen, df_tmp], axis=1)
                resumen.to_excel(writer, index=False, sheet_name="Resumen")
            excel_buffer.seek(0)

            st.download_button(
                "📥 Exportar resumen a Excel",
                data=excel_buffer.getvalue(),
                file_name=f"espectros_resumen_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


    if not df_imagenes.empty:
        st.subheader("Imágenes de espectros")
        for _, row in df_imagenes.iterrows():
            try:
                imagen = BytesIO(base64.b64decode(row["Contenido"]))
                st.image(imagen, caption=f"{row['Muestra']} – {row['Tipo']} – {row['Fecha']}", use_container_width=True)
            except:
                st.warning(f"No se pudo mostrar la imagen: {row['Nombre archivo']}")


    if not df_imagenes.empty and not df_imagenes[df_imagenes["Muestra"].isin(muestras_sel) & df_imagenes["Tipo"].isin(tipos_sel)].empty:
        st.subheader("Descargar imágenes seleccionadas")
    
    if st.button("📥 Descargar imágenes", key="descargar_imagenes"):
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
    
                        # Crear .txt de observaciones
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
    
            st.download_button("📦 Descargar ZIP de imágenes",
                               data=zip_bytes,
                               file_name=os.path.basename(zip_path),
                               mime="application/zip")

# --- HOJA 5 ---
with tab5:
    st.title("Índice OH")

    # --- Reutilizar selección sincronizada en Hoja 5 ---
    muestras_sel = st.session_state.get("muestras_sel", [])
    tipos_sel = st.session_state.get("tipos_sel", [])

    st.write("Muestras seleccionadas:", muestras_sel)
    st.write("Tipos de espectro seleccionados:", tipos_sel)


    st.info("Aquí podrás calcular y visualizar el Índice OH de tus espectros subidos.")
    st.warning("Funcionalidad en construcción...")
# --- HOJA 7 ---
with tab7:
    st.title("Sugerencias y comentarios")

    sugerencias_ref = db.collection("sugerencias")

    st.subheader("Dejar una sugerencia")
    comentario = st.text_area("Escribí tu sugerencia o comentario aquí:")
    if st.button("Enviar sugerencia"):
        if comentario.strip():
            sugerencias_ref.add({
                "comentario": comentario.strip(),
                "fecha": datetime.now().isoformat()
            })
            st.success("Gracias por tu comentario.")
            st.rerun()
        else:
            st.warning("El comentario no puede estar vacío.")


    st.subheader("Comentarios recibidos")

    docs = sugerencias_ref.order_by("fecha", direction=firestore.Query.DESCENDING).stream()
    sugerencias = [{"id": doc.id, **doc.to_dict()} for doc in docs]

    for s in sugerencias:
        st.markdown(f"**{s['fecha'][:19].replace('T',' ')}**")
        st.markdown(s["comentario"])
        if st.button("Eliminar", key=f"del_{s['id']}"):
            sugerencias_ref.document(s["id"]).delete()
            st.success("Comentario eliminado.")
            st.rerun()



# --- HOJA 6 ---
with tab6:
    st.title("Consola")

    muestras = cargar_muestras()
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    for muestra in muestras:
        with st.expander(f"📁 {muestra['nombre']}"):
            st.markdown(f"📝 **Observación:** {muestra.get('observacion', '—')}")

            analisis = muestra.get("analisis", [])
            if analisis:
                st.markdown("📊 **Análisis cargados:**")
                for a in analisis:
                    st.markdown(f"- {a['tipo']}: {a['valor']} ({a['fecha']})")

                import pandas as pd
                from io import BytesIO
                df_analisis = pd.DataFrame(analisis)
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_analisis.to_excel(writer, index=False, sheet_name="Análisis")
                st.download_button("⬇️ Descargar análisis",
                    data=buffer.getvalue(),
                    file_name=f"analisis_{muestra['nombre']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            espectros = muestra.get("espectros", [])
            if espectros:
                st.markdown("🧪 **Espectros cargados:**")
                for e in espectros:
                    etiqueta = f"{e['tipo']} ({e['fecha']})"
                    if e.get("es_imagen", False):
                        st.markdown(f"🖼️ {etiqueta}")
                    else:
                        st.markdown(f"📈 {etiqueta}")

                import zipfile, base64, os
                from tempfile import TemporaryDirectory

                if st.button(f"⬇️ Descargar espectros ZIP", key=f"zip_{muestra['nombre']}"):
                    with TemporaryDirectory() as tmpdir:
                        zip_path = os.path.join(tmpdir, f"espectros_{muestra['nombre']}.zip")
                        with zipfile.ZipFile(zip_path, "w") as zipf:
                            for e in espectros:
                                contenido = e.get("contenido")
                                if not contenido:
                                    continue
                                nombre = e.get("nombre_archivo", "espectro")
                                ruta = os.path.join(tmpdir, nombre)
                                with open(ruta, "wb") as f:
                                    if e.get("es_imagen"):
                                        f.write(bytes.fromhex(contenido))
                                    else:
                                        f.write(base64.b64decode(contenido))
                                zipf.write(ruta, arcname=nombre)

                        with open(zip_path, "rb") as final_zip:
                            st.download_button("📦 Descargar ZIP de espectros",
                                data=final_zip.read(),
                                file_name=f"espectros_{muestra['nombre']}.zip",
                                mime="application/zip",
                                key=f"dl_zip_{muestra['nombre']}")
    st.markdown("---")
    if st.button("Cerrar sesión"):
        st.session_state.autenticado = False
        st.rerun()
