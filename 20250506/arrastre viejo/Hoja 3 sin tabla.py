# --- HOJA 3 ---
with tab3:
    st.title("Carga de espectros")

    muestras = cargar_muestras()
    nombres_muestras = [m["nombre"] for m in muestras]

    st.subheader("Subir nuevo espectro")
    nombre_sel = st.selectbox("Seleccionar muestra", nombres_muestras)
    tipos_espectro_base = [
        "FTIR-Acetato", "FTIR-Cloroformo", "FTIR-ATR",
        "RMN 1H", "RMN 13C", "RMN-LF 1H"
    ]
    if "tipos_espectro" not in st.session_state:
        st.session_state.tipos_espectro = tipos_espectro_base.copy()
    tipo_espectro = st.selectbox("Tipo de espectro", st.session_state.tipos_espectro)

    # Ingreso manual adicional para RMN 1H
    datos_difusividad = []
    if tipo_espectro == "RMN 1H":
        st.markdown("### Difusividad y Tiempo de relajación – RMN 1H")
        num_registros = st.number_input("Cantidad de registros de difusividad", min_value=0, max_value=20, value=0)
        for i in range(int(num_registros)):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                d_val = st.text_input(f"D [{i+1}] [m²/s]", key=f"d_val_{i}")
            with col2:
                t2_val = st.text_input(f"T2 [{i+1}] [s]", key=f"t2_val_{i}")
            with col3:
                x_min = st.number_input(f"Xmin [{i+1}]", key=f"xmin_{i}")
            with col4:
                x_max = st.number_input(f"Xmax [{i+1}]", key=f"xmax_{i}")
            datos_difusividad.append({"D": d_val, "T2": t2_val, "Xmin": x_min, "Xmax": x_max})

    # Ingreso manual adicional para FTIR-Acetato y FTIR-Cloroformo
    senal_3548 = None
    senal_3611 = None
    peso_muestra = None

    if tipo_espectro == "FTIR-Acetato":
        st.markdown("**Datos manuales opcionales para FTIR-Acetato:**")
        senal_3548 = st.number_input("Señal de Acetato a 3548 cm⁻¹", step=0.0001, format="%.4f")
        peso_muestra = st.number_input("Peso de la muestra [g]", step=0.0001, format="%.4f")
    elif tipo_espectro == "FTIR-Cloroformo":
        st.markdown("**Datos manuales opcionales para FTIR-Cloroformo:**")
        senal_3611 = st.number_input("Señal de Cloroformo a 3611 cm⁻¹", step=0.0001, format="%.4f")
        peso_muestra = st.number_input("Peso de la muestra [g]", step=0.0001, format="%.4f")

    #Agregar nuevo tipo de espectro
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

    if st.button("Guardar espectro") and archivo:
        espectros = next((m for m in muestras if m["nombre"] == nombre_sel), {}).get("espectros", [])
        nuevo = {
            "tipo": tipo_espectro,
            "observaciones": observaciones,
            "nombre_archivo": archivo.name,
            "contenido": base64.b64encode(archivo.getvalue()).decode("utf-8"),
            "es_imagen": archivo.type.startswith("image/"),
            "fecha": str(fecha_espectro),
            "senal_3548": senal_3548,
            "senal_3611": senal_3611,
            "peso_muestra": peso_muestra
        }        
        if tipo_espectro == "RMN 1H" and "datos_difusividad":
            nuevo["difusividad"] = datos_difusividad
        espectros.append(nuevo)

        for m in muestras:
            if m["nombre"] == nombre_sel:
                m["espectros"] = espectros
                guardar_muestra(m["nombre"], m.get("observacion", ""), m.get("analisis", []), espectros)
                st.success("Espectro guardado.")
                st.rerun()

    if st.button("📦 Preparar descarga"):
            with TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, f"espectros_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
                excel_path = os.path.join(tmpdir, "tabla_espectros.xlsx")

                with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
                    # Tabla general
                    filas_general = []
                    for m in muestras:
                        for e in m.get("espectros", []):
                            filas_general.append({
                                "Muestra": m["nombre"],
                                "Tipo": e.get("tipo", ""),
                                "Archivo": e.get("nombre_archivo", ""),
                                "Fecha": e.get("fecha", ""),
                                "Observaciones": e.get("observaciones", "")
                            })
                    pd.DataFrame(filas_general).to_excel(writer, index=False, sheet_name="Espectros")

                    # Difusividad con T2
                    filas_dif = []
                    for m in muestras:
                        for e in m.get("espectros", []):
                            if e.get("tipo") == "RMN 1H" and e.get("difusividad"):
                                for d in e["difusividad"]:
                                    filas_dif.append({
                                        "Muestra": m["nombre"],
                                        "Tipo": e.get("tipo"),
                                        "Fecha": e.get("fecha"),
                                        "Archivo": e.get("nombre_archivo"),
                                        "D [m²/s]": d.get("D"),
                                        "T2 [s]": d.get("T2"),
                                        "Xmin": d.get("Xmin"),
                                        "Xmax": d.get("Xmax"),
                                        "Observaciones": e.get("observaciones")
                                    })
                    if filas_dif:
                        pd.DataFrame(filas_dif).to_excel(writer, index=False, sheet_name="Difusividad")

                # ZIP
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    zipf.write(excel_path, arcname="tabla_espectros.xlsx")
                    for m in muestras:
                        for e in m.get("espectros", []):
                            contenido = e.get("contenido")
                            if not contenido:
                                continue
                            carpeta = f"{m['nombre']}"
                            nombre = e.get("nombre_archivo", "espectro")
                            fullpath = os.path.join(tmpdir, carpeta)
                            os.makedirs(fullpath, exist_ok=True)
                            file_path = os.path.join(fullpath, nombre)
                            with open(file_path, "wb") as file_out:
                                try:
                                    file_out.write(base64.b64decode(contenido))
                                except Exception as error:
                                    st.error(f"Error al decodificar archivo: {nombre} — {error}")
                                    continue
                            zipf.write(file_path, arcname=os.path.join(carpeta, nombre))

                with open(zip_path, "rb") as final_zip:
                    zip_bytes = final_zip.read()
                    st.session_state["zip_bytes"] = zip_bytes
                    st.session_state["zip_name"] = os.path.basename(zip_path)

            if "zip_bytes" in st.session_state:
                st.download_button("📦 Descargar espectros", data=st.session_state["zip_bytes"],
                    file_name=st.session_state["zip_name"], mime="application/zip")
