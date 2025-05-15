# tabs_tab7_consola.py
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import os
import zipfile
from datetime import datetime
from tempfile import TemporaryDirectory


@st.cache_data(ttl=60)
def obtener_ids_espectros(nombre):
    return [doc.id for doc in firestore.Client().collection("muestras").document(nombre).collection("espectros").list_documents()]

def obtener_espectros_para_muestra(db, nombre):
    ids = obtener_ids_espectros(nombre)
    ref = db.collection("muestras").document(nombre).collection("espectros")
    return [ref.document(doc_id).get().to_dict() for doc_id in ids]

def render_tab7(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Consola")
    st.session_state["current_tab"] = "Consola"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # Bloque con expansores por muestra
    for muestra in muestras:
        with st.expander(f"📁 {muestra['nombre']}"):
            st.markdown(f"📝 **Observación:** {muestra.get('observacion', '—')}")

            # Mostrar y permitir descarga de análisis
            analisis = muestra.get("analisis", [])
            if analisis:
                st.markdown("📊 **Análisis cargados:**")
                for a in analisis:
                    st.markdown(f"- {a['tipo']}: {a['valor']} ({a['fecha']})")
                df_analisis = pd.DataFrame(analisis)
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_analisis.to_excel(writer, index=False, sheet_name="Análisis")
                buffer.seek(0)
                st.download_button("⬇️ Descargar análisis",
                    data=buffer.getvalue(),
                    file_name=f"analisis_{muestra['nombre']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # Mostrar y permitir descarga de espectros
            espectros = muestra.get("espectros", [])
            if espectros:
                st.markdown("🧪 **Espectros cargados:**")
                for e in espectros:
                    etiqueta = f"{e['tipo']} ({e['fecha']})"
                    st.markdown(f"🖼️ {etiqueta}" if e.get("es_imagen", False) else f"📈 {etiqueta}")

                filas_mascaras = []
                for e in espectros:
                    if e.get("mascaras"):
                        for j, mascara in enumerate(e["mascaras"]):
                            filas_mascaras.append({
                                "Archivo": e.get("nombre_archivo", ""),
                                "Máscara N°": j + 1,
                                "D [m2/s]": mascara.get("difusividad"),
                                "T2 [s]": mascara.get("t2"),
                                "Xmin [ppm]": mascara.get("x_min"),
                                "Xmax [ppm]": mascara.get("x_max")
                            })
                df_mascaras = pd.DataFrame(filas_mascaras)
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_mascaras.to_excel(writer, index=False, sheet_name="Mascaras_RMN1H")
                buffer.seek(0)
                if not df_mascaras.empty:
                    st.download_button("📑 Descargar máscaras RMN 1H",
                        data=buffer.getvalue(),
                        file_name=f"mascaras_{muestra['nombre']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_mask_{muestra['nombre']}")
                    
                # Generar ZIP
                buffer_zip = BytesIO()
                with zipfile.ZipFile(buffer_zip, "w") as zipf:
                    for e in espectros:
                        nombre_archivo = e.get("nombre_archivo", "espectro")
                        contenido = e.get("contenido")
                        if not contenido:
                            continue
                        try:
                            binario = base64.b64decode(contenido)
                            zipf.writestr(nombre_archivo, binario)
                        except:
                            continue

                    # Añadir archivo Excel con máscaras si existen
                    if not df_mascaras.empty:
                        zipf.writestr("mascaras_rmn1h.xlsx", buffer.getvalue())
                buffer_zip.seek(0)
                st.download_button("📦 Descargar ZIP de espectros",
                    data=buffer_zip.getvalue(),
                    file_name=f"espectros_{muestra['nombre']}.zip",
                    mime="application/zip",
                    key=f"dl_zip_{muestra['nombre']}")

    st.markdown("---")


    # Las 3 tablas en columnas
    col1, col2, col3 = st.columns(3)

    # Tabla 1: Descargas por muestra
    with col1:
        st.subheader("📋 Descargas por muestra")
        header1 = st.columns([2, 1, 1])
        header1[0].markdown("**Muestra**")
        header1[1].markdown("**📥 Excel**")
        header1[2].markdown("**📦 ZIP**")

        for i, m in enumerate(muestras):
            c1, c2, c3 = st.columns([2, 1, 1])
            nombre = m["nombre"]
            c1.markdown(f"**{nombre}**")

            # Generar Excel de análisis
            df_analisis = pd.DataFrame(m.get("analisis", []))
            df_espectros = pd.DataFrame(obtener_espectros_para_muestra(db, m["nombre"]))

            filas_mascaras = []
            for e in obtener_espectros_para_muestra(db, m["nombre"]):
                if e.get("mascaras"):
                    for j, mascara in enumerate(e["mascaras"]):
                        filas_mascaras.append({
                            "Archivo": e.get("nombre_archivo", ""),
                            "Máscara N°": j + 1,
                            "D [m2/s]": mascara.get("difusividad"),
                            "T2 [s]": mascara.get("t2"),
                            "Xmin [ppm]": mascara.get("x_min"),
                            "Xmax [ppm]": mascara.get("x_max")
                        })
            df_mascaras = pd.DataFrame(filas_mascaras)

            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                df_analisis.to_excel(writer, index=False, sheet_name="Análisis")
                df_espectros.to_excel(writer, index=False, sheet_name="Espectros")
                if not df_mascaras.empty:
                    df_mascaras.to_excel(writer, index=False, sheet_name="Mascaras_RMN1H")                
            buffer_excel.seek(0)

            # Generar ZIP de espectros
            buffer_zip = BytesIO()
            with zipfile.ZipFile(buffer_zip, "w") as zipf:
                for e in obtener_espectros_para_muestra(db, m["nombre"]):
                    nombre_archivo = e.get("nombre_archivo", "espectro")
                    contenido = e.get("contenido")
                    if not contenido:
                        continue
                    try:
                        binario = base64.b64decode(contenido)
                        zipf.writestr(nombre_archivo, binario)
                    except Exception as err:
                        continue
            buffer_zip.seek(0)

            # Botones de descarga
            c2.download_button(f"📥 {len(df_analisis)}", data=buffer_excel.getvalue(), file_name=f"analisis_{nombre}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key=f"excel1_{i}")
            c3.download_button(f"📦 {len(obtener_espectros_para_muestra(db, m['nombre']))}", data=buffer_zip.getvalue(), file_name=f"espectros_{nombre}.zip", mime="application/zip", use_container_width=True, key=f"zip1_{i}")

    # Tabla 2: Descargas por análisis
    with col2:
        st.subheader("🟢 Descargas por análisis")
        conteo_analisis = {}
        for m in muestras:
            for a in m.get("analisis", []):
                tipo = a.get("tipo", "")
                if tipo:
                    conteo_analisis[tipo] = conteo_analisis.get(tipo, 0) + 1
        df2 = pd.DataFrame([{"Tipo de Análisis": k, "Muestras": v} for k, v in conteo_analisis.items()])
        h2 = st.columns([3, 1])
        h2[0].markdown("**Tipo de Análisis**")
        h2[1].markdown("**📥 Excel**")
        for i, row in df2.iterrows():
            tipo = row["Tipo de Análisis"]
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{tipo}** ({row['Muestras']})")

            # Reunir todos los análisis de ese tipo
            filas = []
            for m in muestras:
                for a in m.get("analisis", []):
                    if a.get("tipo") == tipo:
                        fila = a.copy()
                        fila["Muestra"] = m["nombre"]
                        filas.append(fila)

            df_filtrado = pd.DataFrame(filas)

            # Generar Excel
            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                df_filtrado.to_excel(writer, index=False, sheet_name="Análisis")
            buffer_excel.seek(0)

            c2.download_button(
                f"📥 {row['Muestras']}",
                data=buffer_excel.getvalue(),
                file_name=f"analisis_{tipo}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"excel2_{i}"
            )

    # Tabla 3: Descargas por espectros
    with col3:
        st.subheader("🟣 Descargas por espectros")
        conteo_espectros = {}
        for m in muestras:
            for e in obtener_espectros_para_muestra(db, m["nombre"]):
                tipo = e.get("tipo", "")
                if tipo:
                    conteo_espectros[tipo] = conteo_espectros.get(tipo, 0) + 1
        df3 = pd.DataFrame([{"Tipo de Espectro": k, "Muestras": v} for k, v in conteo_espectros.items()])
        h3 = st.columns([3, 1])
        h3[0].markdown("**Tipo de Espectro**")
        h3[1].markdown("**📦 ZIP**")
        for i, row in df3.iterrows():
            tipo = row["Tipo de Espectro"]
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{tipo}** ({row['Muestras']})")

            # Reunir todos los espectros de ese tipo
            buffer_zip = BytesIO()
            with zipfile.ZipFile(buffer_zip, "w") as zipf:
                for m in muestras:
                    for e in obtener_espectros_para_muestra(db, m["nombre"]):
                        if e.get("tipo") == tipo:
                            nombre_archivo = e.get("nombre_archivo", "espectro")
                            contenido = e.get("contenido")
                            if not contenido:
                                continue
                            try:
                                binario = base64.b64decode(contenido)
                                ruta = f"{m['nombre']}_{nombre_archivo}"
                                zipf.writestr(ruta, binario)
                            except Exception:
                                continue
            buffer_zip.seek(0)

            c2.download_button(
                f"📦 {row['Muestras']}",
                data=buffer_zip.getvalue(),
                file_name=f"espectros_{tipo}.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"zip3_{i}"
            )

    st.markdown("---")

    # Botón Descargar T0D0
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        for m in muestras:
            nombre = m["nombre"]
            carpeta = os.path.join(tmpdir, nombre)
            os.makedirs(carpeta, exist_ok=True)

            # Guardar análisis
            df_analisis = pd.DataFrame(m.get("analisis", []))
            path_excel = os.path.join(carpeta, "analisis.xlsx")
            with pd.ExcelWriter(path_excel, engine="xlsxwriter") as writer:
                df_analisis.to_excel(writer, index=False, sheet_name="Análisis")

            # Guardar espectros
            carpeta_espectros = os.path.join(carpeta, "espectros")
            os.makedirs(carpeta_espectros, exist_ok=True)
            for e in obtener_espectros_para_muestra(db, m["nombre"]):
                nombre_archivo = e.get("nombre_archivo", "espectro")
                contenido = e.get("contenido")
                if not contenido:
                    continue
                try:
                    binario = base64.b64decode(contenido)
                    ruta_archivo = os.path.join(carpeta_espectros, nombre_archivo)
                    with open(ruta_archivo, "wb") as f:
                        f.write(binario)
                except Exception:
                    continue

        # Empaquetar todo en ZIP
        buffer_zip = BytesIO()
        with zipfile.ZipFile(buffer_zip, "w") as zipf:
            for root, _, files in os.walk(tmpdir):
                for archivo in files:
                    full_path = os.path.join(root, archivo)
                    rel_path = os.path.relpath(full_path, tmpdir)
                    zipf.write(full_path, arcname=rel_path)
        buffer_zip.seek(0)

        st.download_button(
            "📦 Descargar TODO",
            data=buffer_zip.getvalue(),
            file_name="todo_muestras.zip",
            mime="application/zip"
        )

    st.markdown("---")
    if st.button("Cerrar sesión"):
        st.session_state.pop("token", None)
        st.rerun()

    mostrar_sector_flotante(db, key_suffix="tab7")
