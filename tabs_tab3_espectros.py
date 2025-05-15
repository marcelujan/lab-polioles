
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

    tipos_espectro_base = ["FTIR-Acetato", "FTIR-Cloroformo", "FTIR-ATR", "RMN 1H", "RMN 13C", "RMN-LF 1H"]
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
        observaciones_totales = f"Archivo original: {archivo.name}"
        if observaciones:
            observaciones_totales += f" — {observaciones}"

        nuevo = {
            "tipo": tipo_espectro,
            "observaciones": observaciones_totales,
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


    st.subheader("Espectros cargados")   # Tabla de espectros ya cargados
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

    df_esp_tabla = pd.DataFrame(filas)   # Eliminar espectros (Tabla de seleccion)
    df_mascaras = pd.DataFrame(filas_mascaras)
    if not df_esp_tabla.empty:
        st.dataframe(df_esp_tabla.drop(columns=["ID"]), use_container_width=True)
        seleccion = st.selectbox(
            "Eliminar espectro",
            df_esp_tabla["ID"],
            format_func=lambda i: df_esp_tabla[df_esp_tabla['ID'] == i]['Archivo'].values[0]
            )
        if st.button("Eliminar espectro"):  # Eliminar espectros (Botón)
            nombre, idx = seleccion.split("__")
            for m in muestras:
                if m["nombre"] == nombre:
                    espectros = obtener_espectros_para_muestra(db, nombre)
                    espectro_id = list(db.collection("muestras").document(nombre).collection("espectros").list_documents())[int(idx)].id
                    db.collection("muestras").document(nombre).collection("espectros").document(espectro_id).delete()
                    st.success("Espectro eliminado.")
                    st.rerun()

        if st.button("📦 Preparar descarga"):   #Preparar descarga de espectros (Excel y ZIP)
            with TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, f"espectros_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
                excel_path = os.path.join(tmpdir, "tabla_espectros.xlsx")

                with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
                    df_esp_tabla.drop(columns=["ID"]).to_excel(writer, index=False, sheet_name="Espectros")
                    if not df_mascaras.empty:
                        df_mascaras.to_excel(writer, index=False, sheet_name="Mascaras_RMN1H")
                        
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    zipf.write(excel_path, arcname="tabla_espectros.xlsx")
                    for m in muestras:
                        for e in obtener_espectros_para_muestra(db, m["nombre"]):
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

        if "zip_bytes" in st.session_state:   # Botón de descarga del ZIP preparado
            st.download_button("📦 Descargar espectros", data=st.session_state["zip_bytes"],
                               file_name=st.session_state["zip_name"],
                               mime="application/zip")
    else:
        st.info("No hay espectros cargados.")


    mostrar_sector_flotante(db, key_suffix="tab3")
