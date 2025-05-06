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


import requests

FIREBASE_API_KEY = st.secrets["firebase_api_key"]  # clave secreta de Firebase


def registrar_usuario(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        st.success("Usuario registrado correctamente. Ahora puede iniciar sesión.")
    else:
        st.error("No se pudo registrar. El correo puede estar en uso o la contraseña es débil.")


def iniciar_sesion(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["idToken"]
    else:
        st.error("Credenciales incorrectas o cuenta no existente.")
        return None

# --- Autenticación ---
if "token" not in st.session_state:
    st.markdown("### Iniciar sesión")
    email = st.text_input("Correo electrónico")
    password = st.text_input("Contraseña", type="password")

    st.warning("Si usás autocompletar, verificá que los campos estén visibles antes de continuar.")

    if st.button("Iniciar sesión"):
        token = iniciar_sesion(email, password)
        if token:
            st.session_state["token"] = token
            st.success("Inicio de sesión exitoso.")
            st.rerun()

    st.markdown("---")
    st.markdown("### ¿No tenés cuenta? Registrate aquí:")
    with st.form("registro"):
        nuevo_email = st.text_input("Nuevo correo")
        nueva_clave = st.text_input("Nueva contraseña", type="password")
        submit_registro = st.form_submit_button("Registrar")
        if submit_registro:
            registrar_usuario(nuevo_email, nueva_clave)
            token = iniciar_sesion(nuevo_email, nueva_clave)
            if token:
                st.session_state["token"] = token
                st.success("Registro e inicio de sesión exitoso.")
                st.rerun()

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
        nuevos_validos = [a for a in nuevos if a["tipo"] != "" and a["valor"] != 0]

        guardar_muestra(
            nombre_muestra,
            observacion,
            previos + nuevos_validos,
            muestra_existente.get("espectros") if muestra_existente else []
        )
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
            format_func=lambda i: f"{df_vista.at[i, 'Nombre']} – {df_vista.at[i, 'Tipo']} – {df_vista.at[i, 'Fecha']}– {df_vista.at[i, 'Observaciones']}")
        if st.button("Eliminar análisis"):
            elegido = df_vista.iloc[seleccion]
            for m in muestras:
                if m["nombre"] == elegido["Nombre"]:
                    m["analisis"] = [a for a in m["analisis"] if not (
                        a["tipo"] == elegido["Tipo"] and
                        str(a["fecha"]) == elegido["Fecha"] and
                        a["valor"] == elegido["Valor"] and
                        a["observaciones"] == elegido["Observaciones"]
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

                                texto = f"D={d.get('D', '')}
T2={d.get('T2', '')}"
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

# --- HOJA 5 ---
with tab5:
    st.title("Índice OH espectroscópico")

    muestras = cargar_muestras()

    if not muestras:
        st.info("No hay muestras cargadas para analizar.")
        st.stop()
    espectros_info = []

    for m in muestras:
        espectros = m.get("espectros", [])
        for e in espectros:
            tipo = e.get("tipo", "")
            if tipo not in ["FTIR-Acetato", "FTIR-Cloroformo"]:
                continue

            contenido = e.get("contenido")
            es_imagen = e.get("es_imagen", False)

            valor_y_extraido = None

            if contenido and not es_imagen:
                try:
                    extension = e.get("nombre_archivo", "").split(".")[-1].lower()
                    binario = BytesIO(base64.b64decode(contenido))

                    if extension == "xlsx":
                        df = pd.read_excel(binario, header=None)
                    else:
                        sep_try = [",", ";", "\t", " "]
                        for sep in sep_try:
                            binario.seek(0)
                            try:
                                df = pd.read_csv(binario, sep=sep, engine="python", header=None)
                                if df.shape[1] >= 2:
                                    break
                            except:
                                continue
                        else:
                            df = None

                    if df is not None and df.shape[1] >= 2:
                        df = df.dropna()
                        x_valores = pd.to_numeric(df.iloc[:,0], errors='coerce')
                        y_valores = pd.to_numeric(df.iloc[:,1], errors='coerce')
                        df_limpio = pd.DataFrame({"X": x_valores, "Y": y_valores}).dropna()

                        if tipo == "FTIR-Acetato":
                            objetivo_x = 3548
                        else:
                            objetivo_x = 3611

                        idx_cercano = (df_limpio["X"] - objetivo_x).abs().idxmin()
                        valor_y_extraido = df_limpio.loc[idx_cercano, "Y"]
                except Exception as err:
                    valor_y_extraido = None

            espectros_info.append({
                "Muestra": m["nombre"],
                "Tipo espectro": tipo,
                "Fecha espectro": e.get("fecha", ""),
                "Señal": valor_y_extraido,
                "Señal manual 3548": e.get("senal_3548", None),
                "Señal manual 3611": e.get("senal_3611", None),
                "Peso muestra [g]": e.get("peso_muestra", None)
            })

    df_muestras = pd.DataFrame(espectros_info)

    if df_muestras.empty:
        st.warning("No se encontraron espectros válidos para calcular Índice OH.")
        st.stop()

    # Crear columna 'Señal solvente' unificando las manuales
    def obtener_senal_solvente(row):
        if row["Tipo espectro"] == "FTIR-Acetato":
            return row["Señal manual 3548"]
        elif row["Tipo espectro"] == "FTIR-Cloroformo":
            return row["Señal manual 3611"]
        else:
            return None

    df_muestras["Señal solvente"] = df_muestras.apply(obtener_senal_solvente, axis=1)

    # Calcular Índice OH real
    def calcular_indice_oh(row):
        tipo = row["Tipo espectro"]
        peso = row["Peso muestra [g]"]
        senal_grafica = row["Señal"]
        senal_manual = row["Señal solvente"]
        if tipo == "FTIR-Acetato":
            constante = 52.5253
        elif tipo == "FTIR-Cloroformo":
            constante = 66.7324
        else:
            return "No disponible"

        if peso is None or peso == 0 or senal_grafica is None or senal_manual is None:
            return "No disponible"

        return round(((senal_grafica - senal_manual) * constante) / peso, 4)

    df_muestras["Índice OH"] = df_muestras.apply(calcular_indice_oh, axis=1)

    # Mostrar resultados
    columnas_mostrar = ["Muestra", "Tipo espectro", "Fecha espectro", "Señal", "Señal solvente", "Peso muestra [g]", "Índice OH"]
    df_final = df_muestras[columnas_mostrar]
    df_final = df_final.rename(columns={
        "Tipo espectro": "Tipo",
    })
    
    # Aplicar formato de decimales
    df_final["Peso muestra [g]"] = df_final["Peso muestra [g]"].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
    df_final["Índice OH"] = df_final["Índice OH"].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

    # Mostrar tabla final
    st.dataframe(df_final, use_container_width=True)

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
        st.session_state.pop("token", None)
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