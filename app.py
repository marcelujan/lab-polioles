import streamlit as st
import pandas as pd
import toml
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import date, datetime
from io import BytesIO
import os
import base64
import matplotlib.pyplot as plt
import numpy as np

import zipfile
from tempfile import TemporaryDirectory

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

# --- Autenticación ---
config = toml.load("config.toml")
PASSWORD = config["auth"]["password"]
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False
if not st.session_state.autenticado:
    pwd = st.text_input("Contraseña de acceso", type="password")
    if st.button("Ingresar"):
        if pwd == PASSWORD:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")
    st.stop()

# --- Firebase ---
if "firebase_initialized" not in st.session_state:
    cred_dict = json.loads(st.secrets["firebase_key"])
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(cred_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        st.session_state.firebase_initialized = True
db = firestore.client()

# --- Funciones comunes ---
def cargar_muestras():
    try:
        docs = db.collection("muestras").stream()
        return [{**doc.to_dict(), "nombre": doc.id} for doc in docs]
    except:
        return []

def guardar_muestra(nombre, observacion, analisis, espectros=None):
    datos = {
        "observacion": observacion,
        "analisis": analisis
    }
    if espectros is not None:
        datos["espectros"] = espectros
    db.collection("muestras").document(nombre).set(datos)
    backup_name = f"muestras_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(backup_name, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Laboratorio de Polioles",
    "Análisis de datos",
    "Carga de espectros",
    "Análisis de espectros",
    "Índice OH espectroscópico",
    "Consola",
    "Sugerencias"
])


# --- HOJA 1 ---
with tab1:
    st.title("Laboratorio de Polioles")
    muestras = cargar_muestras()
    st.subheader("Añadir muestra")
    nombres = [m["nombre"] for m in muestras]
    opcion = st.selectbox("Seleccionar muestra", ["Nueva muestra"] + nombres)
    if opcion == "Nueva muestra":
        nombre_muestra = st.text_input("Nombre de nueva muestra")
        muestra_existente = None
    else:
        nombre_muestra = opcion
        muestra_existente = next((m for m in muestras if m["nombre"] == opcion), None)

    observacion = st.text_area("Observaciones", value=muestra_existente["observacion"] if muestra_existente else "", height=150)

    st.subheader("Nuevo análisis")
    tipos = [
        "Índice de yodo [% p/p I2 abs]", "Índice OH [mg KHO/g]",
        "Índice de acidez [mg KOH/g]", "Índice de epóxido [mol/100g]",
        "Humedad [%]", "PM [g/mol]", "Funcionalidad [#]",
        "Viscosidad dinámica [cP]", "Densidad [g/mL]", "Otro análisis"
    ]
    df = pd.DataFrame([{"Tipo": "", "Valor": 0.0, "Fecha": date.today(), "Observaciones": ""}])
    nuevos_analisis = st.data_editor(df, num_rows="dynamic", use_container_width=True,
        column_config={"Tipo": st.column_config.SelectboxColumn("Tipo", options=tipos)})

    if st.button("Guardar análisis"):
        previos = muestra_existente["analisis"] if muestra_existente else []
        nuevos = []
        for _, row in nuevos_analisis.iterrows():
            if row["Tipo"] != "":
                nuevos.append({
                    "tipo": row["Tipo"],
                    "valor": row["Valor"],
                    "fecha": str(row["Fecha"]),
                    "observaciones": row["Observaciones"]
                })
        guardar_muestra(nombre_muestra, observacion, previos + nuevos, muestra_existente.get("espectros") if muestra_existente else [])
        st.success("Análisis guardado.")
        st.rerun()

    st.subheader("Análisis cargados")
    muestras = cargar_muestras()
    tabla = []
    for m in muestras:
        for a in m["analisis"]:
            tabla.append({
                "Nombre": m["nombre"],
                "Tipo": a["tipo"],
                "Valor": a["valor"],
                "Fecha": a["fecha"],
                "Observaciones": a["observaciones"]
            })
    df_vista = pd.DataFrame(tabla)
    if not df_vista.empty:
        st.dataframe(df_vista, use_container_width=True)

        st.subheader("Eliminar análisis")
        seleccion = st.selectbox("Seleccionar análisis a eliminar", df_vista.index,
            format_func=lambda i: f"{df_vista.at[i, 'Nombre']} – {df_vista.at[i, 'Tipo']} – {df_vista.at[i, 'Fecha']}")
        if st.button("Eliminar análisis"):
            elegido = df_vista.iloc[seleccion]
            for m in muestras:
                if m["nombre"] == elegido["Nombre"]:
                    m["analisis"] = [a for a in m["analisis"] if not (
                        a["tipo"] == elegido["Tipo"] and str(a["fecha"]) == elegido["Fecha"]
                    )]
                    guardar_muestra(m["nombre"], m["observacion"], m["analisis"], m.get("espectros", []))
                    st.success("Análisis eliminado.")
                    st.rerun()

        st.subheader("Exportar")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_vista.to_excel(writer, index=False, sheet_name="Muestras")
        st.download_button("Descargar Excel",
            data=buffer.getvalue(),
            file_name=f"lab-polioles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No hay análisis cargados.")


# --- HOJA 2 ---
with tab2:
    st.title("Análisis de datos")

    muestras = cargar_muestras()
    tabla = []
    for m in muestras:
        for i, a in enumerate(m.get("analisis", [])):
            tabla.append({
                "Fecha": a.get("fecha", ""),
                "ID": f"{m['nombre']}__{i}",
                "Nombre": m["nombre"],
                "Tipo": a.get("tipo", ""),
                "Valor": a.get("valor", ""),
                "Observaciones": a.get("observaciones", "")
            })

    df = pd.DataFrame(tabla)
    if df.empty:
        st.info("No hay análisis cargados.")
        st.stop()

    st.subheader("Tabla completa de análisis")
    st.dataframe(df.drop(columns=["ID"]), use_container_width=True)

    st.subheader("Seleccionar análisis")
    seleccion = st.multiselect("Seleccione uno o más análisis para graficar", df["ID"].tolist(),
                               format_func=lambda i: f"{df[df['ID'] == i]['Nombre'].values[0]} - {df[df['ID'] == i]['Tipo'].values[0]} - {df[df['ID'] == i]['Fecha'].values[0]}")

    df_sel = df[df["ID"].isin(seleccion)]
    df_avg = df_sel.groupby(["Nombre", "Tipo"], as_index=False)["Valor"].mean()

    st.subheader("Resumen de selección promediada")
    st.dataframe(df_avg, use_container_width=True)

    st.subheader("Gráfico XY")
    tipos_disponibles = sorted(df_avg["Tipo"].unique())

    colx, coly = st.columns(2)
    with colx:
        tipo_x = st.selectbox("Selección de eje X", tipos_disponibles)
    with coly:
        tipo_y = st.selectbox("Selección de eje Y", tipos_disponibles)

    muestras_x = df_avg[df_avg["Tipo"] == tipo_x][["Nombre", "Valor"]].set_index("Nombre")
    muestras_y = df_avg[df_avg["Tipo"] == tipo_y][["Nombre", "Valor"]].set_index("Nombre")
    comunes = muestras_x.index.intersection(muestras_y.index)

    usar_manual_x = st.checkbox("Asignar valores X manualmente")
    if usar_manual_x:
        valores_x_manual = []
        nombres = []
        st.markdown("**Asignar valores X manualmente por muestra:**")
        for nombre in comunes:
            val = st.number_input(f"{nombre}", step=0.1, key=f"manual_x_{nombre}")
            valores_x_manual.append(val)
            nombres.append(nombre)
        x = valores_x_manual
    else:
        x = muestras_x.loc[comunes, "Valor"].tolist()
        nombres = comunes.tolist()

    y = muestras_y.loc[comunes, "Valor"].tolist()

    if x and y and len(x) == len(y):
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        for i, txt in enumerate(nombres):
            ax.annotate(txt, (x[i], y[i]))
        ax.set_xlabel(tipo_x)
        ax.set_ylabel(tipo_y)
        st.pyplot(fig)

        buf_img = BytesIO()
        fig.savefig(buf_img, format="png")
        st.download_button("📷 Descargar gráfico", buf_img.getvalue(),
                           file_name=f"grafico_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
                           mime="image/png")
    else:
        st.warning("Los datos seleccionados no son compatibles para graficar.")

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
            "es_imagen": es_imagen,
            "fecha": str(fecha_espectro),
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
                "Fecha": e.get("fecha", ""),
                "Observaciones": e.get("observaciones", ""),
                "ID": f"{m['nombre']}__{i}"
            })
    df_esp_tabla = pd.DataFrame(filas)
    if not df_esp_tabla.empty:
        st.dataframe(df_esp_tabla.drop(columns=["ID"]), use_container_width=True)
        seleccion = st.selectbox(
            "Eliminar espectro",
            df_esp_tabla["ID"],
            format_func=lambda i: f"{df_esp_tabla[df_esp_tabla['ID'] == i]['Muestra'].values[0]} – {df_esp_tabla[df_esp_tabla['ID'] == i]['Tipo'].values[0]} – {df_esp_tabla[df_esp_tabla['ID'] == i]['Archivo'].values[0]} – {df_esp_tabla[df_esp_tabla['ID'] == i]['Fecha'].values[0]}"
        )
        if st.button("Eliminar espectro"):
            nombre, idx = seleccion.split("__")
            for m in muestras:
                if m["nombre"] == nombre:
                    m["espectros"].pop(int(idx))
                    guardar_muestra(m["nombre"], m.get("observacion", ""), m.get("analisis", []), m.get("espectros", []))
                    st.success("Espectro eliminado.")
                    st.rerun()

        # --- DESCARGA DE ESPECTROS ---
                # Lógica de descarga solo si se hace clic
        if st.button("📦 Preparar descarga"):
            from tempfile import TemporaryDirectory
            import zipfile

            with TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, f"espectros_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
                excel_path = os.path.join(tmpdir, "tabla_espectros.xlsx")

                with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
                    df_esp_tabla.drop(columns=["ID"]).to_excel(writer, index=False, sheet_name="Espectros")

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
                    st.session_state["zip_bytes"] = final_zip.read()
                    st.session_state["zip_name"] = os.path.basename(zip_path)

        # Botón de descarga fuera del evento
        if "zip_bytes" in st.session_state:
            st.download_button("📦 Descargar espectros", data=st.session_state["zip_bytes"],
                               file_name=st.session_state["zip_name"],
                               mime="application/zip")
    else:
        st.info("No hay espectros cargados.")


# --- HOJA 4 ---
with tab4:
    st.title("Análisis de espectros")

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

    # CORREGIDO: usar session_state
    st.multiselect("Muestras", muestras_disp, key="muestras_sel")
    st.multiselect("Tipo de espectro", tipos_disp, key="tipos_sel")

    muestras_sel = st.session_state.get("muestras_sel", [])
    tipos_sel = st.session_state.get("tipos_sel", [])

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
        
        from tempfile import TemporaryDirectory
        import zipfile
        import os

        with TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, f"imagenes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for _, row in seleccionadas.iterrows():
                    carpeta = row["Muestra"]
                    os.makedirs(os.path.join(tmpdir, carpeta), exist_ok=True)

                    nombre_img = row["Nombre archivo"]
                    path_img = os.path.join(tmpdir, carpeta, nombre_img)
                    with open(path_img, "wb") as f:
                        f.write(base64.b64decode(row["Contenido"]))
                    zipf.write(path_img, arcname=os.path.join(carpeta, nombre_img))

                    nombre_txt = os.path.splitext(nombre_img)[0] + ".txt"
                    path_txt = os.path.join(tmpdir, carpeta, nombre_txt)
                    with open(path_txt, "w", encoding="utf-8") as f:
                        f.write(f"Nombre del archivo: {nombre_img}")
                        f.write(f"Tipo de espectro: {row['Tipo']}")
                        f.write(f"Fecha: {row['Fecha']}")
                        f.write(f"Observaciones: {row['Observaciones']}")
                    zipf.write(path_txt, arcname=os.path.join(carpeta, nombre_txt))

            with open(zip_path, "rb") as final_zip:
                zip_bytes = final_zip.read()

        st.download_button("📦 Descargar ZIP de imágenes",
                           data=zip_bytes,
                           file_name=os.path.basename(zip_path),
                           mime="application/zip")

# --- HOJA 5 ---
with tab5:
    st.title("Índice OH espectroscópico")

    try:
        docs = db.collection("muestras").stream()
        muestras = [{**doc.to_dict(), "nombre": doc.id} for doc in docs]
    except Exception as e:
        st.error(f"No se pudieron cargar las muestras: {e}")
        muestras = []

    muestras_sel = st.session_state.get("muestras_sel", [])
    tipos_sel = st.session_state.get("tipos_sel", [])

    if muestras_sel and tipos_sel:
        resultados = []

        for muestra in muestras:
            if muestra["nombre"] in muestras_sel:
                espectros = muestra.get("espectros", [])
                for esp in espectros:
                    tipo = esp.get("tipo", "")
                    if tipo in tipos_sel:
                        if isinstance(esp.get("contenido"), dict) and "datos" in esp["contenido"]:
                            datos = esp["contenido"]["datos"]
                            if isinstance(datos, list) and all(isinstance(x, list) and len(x) == 2 for x in datos):
                                import numpy as np
                                datos_np = np.array(datos)
                                x_valores = datos_np[:, 0]
                                y_valores = datos_np[:, 1]

                                if tipo == "FTIR-Acetato":
                                    objetivo_x = 3548
                                    constante = 52.5253
                                    senal_manual = esp.get("senal_3548", None)
                                elif tipo == "FTIR-Cloroformo":
                                    objetivo_x = 3611
                                    constante = 66.7324
                                    senal_manual = esp.get("senal_3611", None)
                                else:
                                    continue

                                peso_muestra = esp.get("peso_muestra", None)

                                idx_mas_cercano = np.argmin(np.abs(x_valores - objetivo_x))
                                senal_grafica = y_valores[idx_mas_cercano]

                                if senal_manual is not None and peso_muestra is not None and peso_muestra != 0:
                                    indice_oh = ((senal_grafica - senal_manual) * constante) / peso_muestra
                                    indice_oh = round(indice_oh, 4)
                                else:
                                    indice_oh = "No disponible"

                                resultados.append({
                                    "Muestra": muestra["nombre"],
                                    "Tipo": tipo,
                                    "Fecha del espectro": esp.get("fecha", "No disponible"),
                                    "Señal gráfica": round(senal_grafica, 4),
                                    "Señal manual": senal_manual if senal_manual is not None else "No disponible",
                                    "Peso muestra [g]": peso_muestra if peso_muestra is not None else "No disponible",
                                    "Índice OH": indice_oh
                                })

        if resultados:
            import pandas as pd
            import io
            from datetime import datetime

            df_resultados = pd.DataFrame(resultados)
            st.dataframe(df_resultados, use_container_width=True)

            # Botón para descargar Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_resultados.to_excel(writer, index=False, sheet_name="Índice OH")
                writer.save()

            fecha_hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M")
            nombre_archivo = f"indice_oh_resultados_{fecha_hora_actual}.xlsx"

            st.download_button(
                label="📥 Descargar tabla en Excel",
                data=buffer.getvalue(),
                file_name=nombre_archivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("No se encontraron espectros numéricos válidos para calcular Índice OH.")
    else:
        st.warning("Primero debes seleccionar muestras y tipos en la Hoja 4.")


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

# --- HOJA 7 ---
with tab7:
    st.title("Sugerencias")

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