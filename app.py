
import streamlit as st
import pandas as pd
import toml
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import date, datetime
from io import BytesIO
import os
import matplotlib.pyplot as plt
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

tab1, tab2, tab3, tab4 = st.tabs([
    "Laboratorio de Polioles",
    "Análisis de datos",
    "Carga de espectros",
    "Análisis de espectros"
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
                "ID": f"{m['nombre']}__{i}",
                "Nombre": m["nombre"],
                "Tipo": a.get("tipo", ""),
                "Valor": a.get("valor", ""),
                "Fecha": a.get("fecha", ""),
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
        if se_grafico_algo:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            st.pyplot(fig)

            # Descargar imagen del gráfico combinado
            buf_img = BytesIO()
            fig.savefig(buf_img, format="png", bbox_inches="tight")
            st.download_button("🖼️ Descargar gráfico combinado", data=buf_img.getvalue(),
                               file_name="grafico_combinado.png", mime="image/png")

            # Descargar Excel con todos los datos seleccionados
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                for _, row in df_filtrado.iterrows():
                    try:
                        extension = os.path.splitext(row["Nombre archivo"])[1].lower()
                        if extension == ".xlsx":
                            binario = BytesIO(bytes.fromhex(row["Contenido"]))
                            df_data = pd.read_excel(binario)
                        else:
                            contenido = StringIO(bytes.fromhex(row["Contenido"]).decode("latin1"))
                            separadores = [",", "	", ";", " "]
                            for sep in separadores:
                                contenido.seek(0)
                                try:
                                    df_data = pd.read_csv(contenido, sep=sep, engine="python")
                                    if df_data.shape[1] >= 2:
                                        break
                                except:
                                    continue
                            else:
                                continue

                        col_x, col_y = df_data.columns[:2]
                        df_data[col_x] = df_data[col_x].astype(str).str.replace(",", ".", regex=False)
                        df_data[col_y] = df_data[col_y].astype(str).str.replace(",", ".", regex=False)
                        df_data[col_x] = pd.to_numeric(df_data[col_x], errors="coerce")
                        df_data[col_y] = pd.to_numeric(df_data[col_y], errors="coerce")

                        nombre_hoja = f"{row['Muestra']}_{row['Tipo']}".replace(" ", "_")[:31]
                        df_data.to_excel(writer, index=False, sheet_name=nombre_hoja)
                    except:
                        continue
            excel_buffer.seek(0)
            st.download_button("📊 Descargar datos combinados", data=excel_buffer.getvalue(),
                               file_name="datos_combinados.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


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

        if st.button("📦 Descargar espectros"):
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
                                if e.get("es_imagen"):
                                    file_out.write(bytes.fromhex(contenido))
                                else:
                                    file_out.write(contenido.encode("latin1"))
                            zipf.write(file_path, arcname=os.path.join(carpeta, nombre))

                with open(zip_path, "rb") as final_zip:
                    st.download_button("📦 Descargar espectros", data=final_zip.read(),
                                       file_name=os.path.basename(zip_path),
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
                "Observaciones": e.get("observaciones", ""),
                "Contenido": e.get("contenido"),
                "Es imagen": e.get("es_imagen", False)
            })

    df_esp = pd.DataFrame(espectros_info)

    st.subheader("Filtrar espectros")
    muestras_disp = df_esp["Muestra"].unique().tolist()
    tipos_disp = df_esp["Tipo"].unique().tolist()
    muestras_sel = st.multiselect("Muestras", muestras_disp, default=[])
    tipos_sel = st.multiselect("Tipo de espectro", tipos_disp, default=[])
    solo_datos = st.checkbox("Mostrar solo espectros numéricos", value=False)
    solo_imagenes = st.checkbox("Mostrar solo imágenes", value=False)

    df_filtrado = df_esp[df_esp["Muestra"].isin(muestras_sel) & df_esp["Tipo"].isin(tipos_sel)]
    if solo_datos:
        df_filtrado = df_filtrado[~df_filtrado["Es imagen"]]
    if solo_imagenes:
        df_filtrado = df_filtrado[df_filtrado["Es imagen"]]

    st.subheader("Espectros combinados")

    # Definir los rangos globales predeterminados
    rango_x = [float('inf'), float('-inf')]
    rango_y = [float('inf'), float('-inf')]

    for _, row in df_filtrado.iterrows():
        if not row["Es imagen"]:
            try:
                from io import StringIO, BytesIO
                import pandas as pd
                extension = os.path.splitext(row["Nombre archivo"])[1].lower()
                if extension == ".xlsx":
                    binario = BytesIO(bytes.fromhex(row["Contenido"]))
                    df_temp = pd.read_excel(binario)
                else:
                    contenido = StringIO(bytes.fromhex(row["Contenido"]).decode("latin1"))
                    separadores = [",", "\t", ";", " "]
                    for sep in separadores:
                        contenido.seek(0)
                        try:
                            df_temp = pd.read_csv(contenido, sep=sep, engine="python")
                            if df_temp.shape[1] >= 2:
                                break
                        except:
                            continue
                    else:
                        continue

                col_x, col_y = df_temp.columns[:2]
                df_temp[col_x] = df_temp[col_x].astype(str).str.replace(",", ".", regex=False)
                df_temp[col_y] = df_temp[col_y].astype(str).str.replace(",", ".", regex=False)
                df_temp[col_x] = pd.to_numeric(df_temp[col_x], errors="coerce")
                df_temp[col_y] = pd.to_numeric(df_temp[col_y], errors="coerce")

                min_x, max_x = df_temp[col_x].dropna().agg(["min", "max"])
                min_y, max_y = df_temp[col_y].dropna().agg(["min", "max"])

                rango_x[0] = min(rango_x[0], min_x)
                rango_x[1] = max(rango_x[1], max_x)
                rango_y[0] = min(rango_y[0], min_y)
                rango_y[1] = max(rango_y[1], max_y)

            except:
                continue

    if rango_x[0] == float('inf') or rango_y[0] == float('inf'):
        st.warning("No se pudo graficar ningún espectro válido.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min = st.number_input("X mínimo", value=float(rango_x[0]))
        with col2:
            x_max = st.number_input("X máximo", value=float(rango_x[1]))
        with col3:
            y_min = st.number_input("Y mínimo", value=float(rango_y[0]))
        with col4:
            y_max = st.number_input("Y máximo", value=float(rango_y[1]))

        fig, ax = plt.subplots()
        se_grafico_algo = False

        for _, row in df_filtrado.iterrows():
            if not row["Es imagen"]:
                try:
                    extension = os.path.splitext(row["Nombre archivo"])[1].lower()
                    if extension == ".xlsx":
                        binario = BytesIO(bytes.fromhex(row["Contenido"]))
                        df_esp = pd.read_excel(binario)
                    else:
                        contenido = StringIO(bytes.fromhex(row["Contenido"]).decode("latin1"))
                        separadores = [",", "\t", ";", " "]
                        for sep in separadores:
                            contenido.seek(0)
                            try:
                                df_esp = pd.read_csv(contenido, sep=sep, engine="python")
                                if df_esp.shape[1] >= 2:
                                    break
                            except:
                                continue
                        else:
                            continue

                    col_x, col_y = df_esp.columns[:2]
                    df_esp[col_x] = df_esp[col_x].astype(str).str.replace(',', '.', regex=False)
                    df_esp[col_y] = df_esp[col_y].astype(str).str.replace(',', '.', regex=False)
                    df_esp[col_x] = pd.to_numeric(df_esp[col_x], errors='coerce')
                    df_esp[col_y] = pd.to_numeric(df_esp[col_y], errors='coerce')

                    df_fil = df_esp[
                        (df_esp[col_x] >= x_min) & (df_esp[col_x] <= x_max) &
                        (df_esp[col_y] >= y_min) & (df_esp[col_y] <= y_max)
                    ]
                    if df_fil.empty:
                        continue

                    ax.plot(df_fil[col_x], df_fil[col_y], label=f"{row['Muestra']} – {row['Tipo']}")
                    se_grafico_algo = True
                except:
                    continue

        if se_grafico_algo:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No se pudo graficar ningún espectro válido.")
