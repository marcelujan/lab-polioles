
# ... (código anterior intacto)
# Actualización de etiquetas y exportación ZIP en Hoja 3

# --- HOJA 3: Carga de espectros ---
with tab3:
    st.title("Carga de espectros")

    muestras = cargar_muestras()
    nombres_muestras = [m["nombre"] for m in muestras]

    st.subheader("Subir nuevo espectro")
    nombre_sel = st.selectbox("Seleccionar muestra", nombres_muestras)
    tipo_espectro = st.selectbox("Tipo de espectro", ["FTIR", "LF-RMN", "RMN 1H", "UV-Vis", "DSC", "Otro espectro"])
    observaciones = st.text_area("Observaciones")
    archivo = st.file_uploader("Archivo del espectro", type=["xlsx", "csv", "txt", "png", "jpg", "jpeg"])

    if archivo:
        nombre_archivo = archivo.name
        extension = os.path.splitext(nombre_archivo)[1].lower()
        es_imagen = extension in [".png", ".jpg", ".jpeg"]

        st.markdown("### Vista previa")
        if es_imagen:
            st.image(archivo, use_column_width=True)
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

    if st.button("Guardar espectro") and archivo:
        espectros = next((m for m in muestras if m["nombre"] == nombre_sel), {}).get("espectros", [])
        nuevo = {
            "tipo": tipo_espectro,
            "observaciones": observaciones,
            "nombre_archivo": archivo.name,
            "contenido": archivo.getvalue().decode("latin1") if not es_imagen else archivo.getvalue().hex(),
            "es_imagen": es_imagen,
        }
        espectros.append(nuevo)

        for m in muestras:
            if m["nombre"] == nombre_sel:
                m["espectros"] = espectros
                guardar_muestra(m["nombre"], m.get("observacion", ""), m.get("analisis", []), espectros)
                st.success("Espectro guardado.")
                st.rerun()

    st.subheader("Espectros cargados")
    filas = []
    for m in muestras:
        for i, e in enumerate(m.get("espectros", [])):
            filas.append({
                "Muestra": m["nombre"],
                "Tipo": e.get("tipo", ""),
                "Archivo": e.get("nombre_archivo", ""),
                "Observaciones": e.get("observaciones", ""),
                "ID": f"{m['nombre']}__{i}"
            })
    df_esp_tabla = pd.DataFrame(filas)
    if not df_esp_tabla.empty:
        st.dataframe(df_esp_tabla.drop(columns=["ID"]), use_container_width=True)
        seleccion = st.selectbox("Eliminar espectro", df_esp_tabla["ID"])
        if st.button("Eliminar espectro"):
            nombre, idx = seleccion.split("__")
            for m in muestras:
                if m["nombre"] == nombre:
                    m["espectros"].pop(int(idx))
                    guardar_muestra(m["nombre"], m.get("observacion", ""), m.get("analisis", []), m.get("espectros", []))
                    st.success("Espectro eliminado.")
                    st.rerun()

        if st.button("Descargar espectros"):
            from tempfile import TemporaryDirectory
            tmp = TemporaryDirectory()
            zip_path = os.path.join(tmp.name, f"espectros_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
            excel_path = os.path.join(tmp.name, "tabla_espectros.xlsx")

            # Guardar Excel con tabla
            with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
                df_esp_tabla.drop(columns=["ID"]).to_excel(writer, index=False, sheet_name="Espectros")

            # Crear carpetas y guardar archivos
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(excel_path, arcname="tabla_espectros.xlsx")
                for m in muestras:
                    for e in m.get("espectros", []):
                        contenido = e.get("contenido")
                        if not contenido:
                            continue
                        carpeta = f"{m['nombre']}"
                        nombre = e.get("nombre_archivo", "espectro")
                        path_temp = os.path.join(tmp.name, carpeta)
                        os.makedirs(path_temp, exist_ok=True)
                        fullpath = os.path.join(path_temp, nombre)
                        with open(fullpath, "wb") as f:
                            if e.get("es_imagen"):
                                f.write(bytes.fromhex(contenido))
                            else:
                                f.write(contenido.encode("latin1"))
                        zipf.write(fullpath, arcname=os.path.join(carpeta, nombre))

            # Leer .zip y ofrecer para descarga
            with open(zip_path, "rb") as f:
                st.download_button("📦 Descargar espectros", data=f.read(),
                                   file_name=os.path.basename(zip_path),
                                   mime="application/zip")

    else:
        st.info("No hay espectros cargados.")
