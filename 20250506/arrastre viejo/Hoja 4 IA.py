# --- HOJA 4 ---
with tab4:
    st.title("Análisis de espectros")

    muestras = cargar_muestras()
    nombres_muestras = [m["nombre"] for m in muestras if m.get("espectros")]

    if not nombres_muestras:
        st.warning("No hay espectros cargados para analizar.")
        st.stop()

    nombre_sel = st.selectbox("Seleccionar muestra", nombres_muestras)
    muestra = next((m for m in muestras if m["nombre"] == nombre_sel), None)

    if not muestra:
        st.error("Muestra no encontrada.")
        st.stop()

    tipos_disponibles = list(set(e.get("tipo", "") for e in muestra.get("espectros", [])))
    tipo_sel = st.selectbox("Seleccionar tipo de espectro", tipos_disponibles)

    espectros_filtrados = [e for e in muestra.get("espectros", []) if e.get("tipo") == tipo_sel]

    if not espectros_filtrados:
        st.warning("No hay espectros para mostrar.")
        st.stop()

    st.markdown("---")
    st.markdown("### Parámetros del gráfico")
    mostrar_sombreado = st.checkbox("Mostrar sombreado para rangos de difusividad (solo RMN 1H)", value=False)

    datos_difusividad_export = []

    for e in espectros_filtrados:
        if e.get("es_imagen"):
            st.image(BytesIO(base64.b64decode(e["contenido"])), caption=e.get("nombre_archivo"), use_container_width=True)
        else:
            try:
                archivo = BytesIO(base64.b64decode(e["contenido"]))
                extension = os.path.splitext(e.get("nombre_archivo", ""))[1].lower()
                if extension == ".xlsx":
                    df = pd.read_excel(archivo)
                else:
                    df = pd.read_csv(archivo, sep=None, engine="python")

                if df.shape[1] >= 2:
                    col_x, col_y = df.columns[:2]
                    fig, ax = plt.subplots()
                    ax.plot(df[col_x], df[col_y], label=e.get("nombre_archivo", "Espectro"))
                    ax.set_xlabel(col_x)
                    ax.set_ylabel(col_y)
                    ax.set_title(f"{muestra['nombre']} - {tipo_sel}")

                    if mostrar_sombreado and tipo_sel == "RMN 1H" and e.get("difusividad"):
                        for d in e["difusividad"]:
                            try:
                                xmin = float(d.get("Xmin", 0))
                                xmax = float(d.get("Xmax", 0))
                                if xmin == xmax:
                                    xmin -= 0.05
                                    xmax += 0.05
                                ax.axvspan(xmin, xmax, color="gray", alpha=0.2)
                                texto = f"D={d.get('D', '')}\nT2={d.get('T2', '')}"
                                x_medio = (xmin + xmax) / 2
                                y_pos = ax.get_ylim()[1] * 0.95
                                ax.text(x_medio, y_pos, texto, fontsize=8, ha="center", va="top", rotation=90)
                                datos_difusividad_export.append({
                                    "Muestra": muestra["nombre"],
                                    "Tipo": tipo_sel,
                                    "Archivo": e.get("nombre_archivo", ""),
                                    "D [m²/s]": d.get("D"),
                                    "T2 [s]": d.get("T2"),
                                    "Xmin": xmin,
                                    "Xmax": xmax,
                                    "Observaciones": e.get("observaciones", "")
                                })
                            except Exception as err:
                                st.warning(f"Error al sombrear difusividad: {err}")

                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("El archivo debe tener al menos dos columnas.")
            except Exception as ex:
                st.error(f"Error al procesar el archivo: {ex}")

    if datos_difusividad_export:
        df_dif = pd.DataFrame(datos_difusividad_export)
        st.markdown("### Exportar datos de difusividad")
        excel_buffer_dif = BytesIO()
        with pd.ExcelWriter(excel_buffer_dif, engine="xlsxwriter") as writer:
            df_dif.to_excel(writer, index=False, sheet_name="Difusividad")
        st.download_button(
            "📥 Descargar tabla de difusividad",
            data=excel_buffer_dif.getvalue(),
            file_name=f"difusividad_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.session_state["df_dif_zip"] = df_dif.copy()

    if st.button("📦 Descargar todo en ZIP"):
        with TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, f"espectros_y_difusividad_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                if "df_dif_zip" in st.session_state:
                    path_excel_dif = os.path.join(tmpdir, "difusividad.xlsx")
                    with pd.ExcelWriter(path_excel_dif, engine="xlsxwriter") as writer:
                        st.session_state["df_dif_zip"].to_excel(writer, index=False, sheet_name="Difusividad")
                    zipf.write(path_excel_dif, arcname="difusividad.xlsx")

                for m in muestras:
                    for e in m.get("espectros", []):
                        if not e.get("contenido"):
                            continue
                        carpeta = m["nombre"]
                        os.makedirs(os.path.join(tmpdir, carpeta), exist_ok=True)
                        nombre = e.get("nombre_archivo", "espectro")
                        path = os.path.join(tmpdir, carpeta, nombre)
                        with open(path, "wb") as f:
                            f.write(base64.b64decode(e["contenido"]))
                        zipf.write(path, arcname=os.path.join(carpeta, nombre))

            with open(zip_path, "rb") as final_zip:
                st.download_button(
                    "📦 Descargar ZIP combinado",
                    data=final_zip.read(),
                    file_name=os.path.basename(zip_path),
                    mime="application/zip"
                )
