# Importación de librerías necesarias para el funcionamiento de la app
import streamlit as st  # - streamlit: interfaz web
import pandas as pd     # - pandas: manejo de datos en tablas
import json             # - json: lectura de archivos de configuración y datos
import firebase_admin   # - firebase_admin: conexión con base de datos Firestore
from firebase_admin import credentials, firestore   # 'credentials': permite inicializar Firebase con las credenciales del proyecto (clave privada) # 'firestore': permite interactuar con la base de datos NoSQL de Firebase (Firestore)
from datetime import date, datetime     # - datetime: manejo de fechas, archivos en memoria y sistema
from io import BytesIO  # - BytesIO: manejo de fechas, archivos en memoria y sistema
import os               # - os: manejo de fechas, archivos en memoria y sistema
import base64           # - base64: codificación/decodificación de archivos binarios
import matplotlib.pyplot as plt         # - matplotlib: generación de gráficos
import numpy as np      # - numpy: generación de gráficos
import zipfile          # - zipfile: creación de archivos comprimidos temporales para descarga
from tempfile import TemporaryDirectory # - TemporaryDirectory: creación de archivos comprimidos temporales para descarga
import requests         # 'requests': permite enviar solicitudes HTTP, utilizado aquí para autenticar usuarios con la API REST de Firebase

# Configuración inicial de la página de Streamlit
st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

# Clave API de Firebase, almacenada de forma segura en los secretos de Streamlit
FIREBASE_API_KEY = st.secrets["firebase_api_key"]  # clave secreta de Firebase

# Función para registrar un nuevo usuario usando Firebase Authentication
# Se envía una solicitud POST a la API REST de Firebase con correo y contraseña
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

# --- Iniciar sesión ---
# Función para iniciar sesión con correo y contraseña utilizando la API REST de Firebase
# Si es exitoso, devuelve el token de autenticación del usuario
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
    st.markdown("### Iniciar sesión")   # Si el usuario aún no está autenticado, se muestra la interfaz de inicio de sesión
    email = st.text_input("Correo electrónico")
    password = st.text_input("Contraseña", type="password") # Se guardará el token en session_state para mantener la sesión activa
    st.warning("Si usás autocompletar, verificá que los campos estén visibles antes de continuar.")

    if st.button("Iniciar sesión"):     # Al hacer clic en "Iniciar sesión", se valida el usuario.
        token = iniciar_sesion(email, password)
        if token:
            st.session_state["token"] = token
            st.success("Inicio de sesión exitoso.")     # Si es correcto, se guarda el token y se reinicia la app para mostrar las pestañas
            st.rerun()

    st.markdown("---")      # Se separa visualmente del login con una línea horizontal ("---")
    st.markdown("### ¿No tenés cuenta? Registrate aquí:")
    with st.form("registro"):   # Formulario para registrar una nueva cuenta
        nuevo_email = st.text_input("Nuevo correo")
        nueva_clave = st.text_input("Nueva contraseña", type="password")
        submit_registro = st.form_submit_button("Registrar")
        if submit_registro:
            registrar_usuario(nuevo_email, nueva_clave) 
            token = iniciar_sesion(nuevo_email, nueva_clave)
            if token:       # Si se registra correctamente, el usuario se autentica automáticamente
                st.session_state["token"] = token
                st.success("Registro e inicio de sesión exitoso.")
                st.rerun()  # Se reinicia la app para mostrar el contenido autenticado
    st.stop()   # Detiene la ejecución de Streamlit si no se ha iniciado sesión, para evitar mostrar el resto de la app

# --- Firebase ---
if "firebase_initialized" not in st.session_state:  # Inicializa Firebase solo una vez por sesión de Streamlit
    cred_dict = json.loads(st.secrets["firebase_key"])  # 1. Se cargan las credenciales del archivo secreto de Firebase desde st.secrets
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")    #se corrige el formato de la clave privada reemplazando '\n' por saltos reales
    cred = credentials.Certificate(cred_dict)   # 2. Se crea el objeto de credenciales para autenticar la conexión
    if not firebase_admin._apps:    # 3. Se inicializa Firebase si no ha sido inicializado aún
        firebase_admin.initialize_app(cred)
        st.session_state.firebase_initialized = True
db = firestore.client()     # 4. Se crea el cliente de Firestore para interactuar con la base de datos

# --- Funciones comunes ---
def cargar_muestras():      # Función para obtener todas las muestras almacenadas en la colección "muestras" de Firestore
    try:
        docs = db.collection("muestras").stream()   # Devuelve una lista de diccionarios, cada uno con los datos de la muestra y su ID como "nombre"

        return [{**doc.to_dict(), "nombre": doc.id} for doc in docs]
    except:
        return []       # Si hay un error (por ejemplo, si Firestore no responde), devuelve una lista vacía

def guardar_muestra(nombre, observacion, analisis, espectros=None): # Función para guardar o actualizar una muestra en la base de datos
    datos = {
        "observacion": observacion, # - 'observacion' y 'analisis' se guardan siempre
        "analisis": analisis        # - 'observacion' y 'analisis' se guardan siempre
    }
    if espectros is not None:       # - 'espectros' se incluye solo si está definido
        datos["espectros"] = espectros
    db.collection("muestras").document(nombre).set(datos)   # - 'nombre' se usa como ID del documento en Firestore

# Definición de las hojas principales de la aplicación
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Laboratorio de Polioles",
    "Análisis de datos",
    "Carga de espectros",
    "Análisis de espectros",
    "Índice OH espectroscópico",
    "Consola",
    "Sugerencias"
])

# --- HOJA 1 --- "Laboratorio de Polioles" ---
with tab1:
    st.title("Laboratorio de Polioles")         # Título principal de la hoja
    muestras = cargar_muestras()                # Carga todas las muestras desde Firestore
    st.subheader("Añadir muestra")              # Sección para crear o editar una muestra existente
    nombres = [m["nombre"] for m in muestras]   # Lista de nombres de muestras ya guardadas

    opcion = st.selectbox("Seleccionar muestra", ["Nueva muestra"] + nombres)    # Selector para elegir si se quiere crear una nueva muestra o editar una existente
    if opcion == "Nueva muestra":
        nombre_muestra = st.text_input("Nombre de nueva muestra") # Campo para ingresar nombre nuevo
        muestra_existente = None
    else:
        nombre_muestra = opcion  # El nombre se toma directamente del selector
        muestra_existente = next((m for m in muestras if m["nombre"] == opcion), None) # Se busca la muestra seleccionada para precargar sus datos

    observacion = st.text_area("Observaciones", value=muestra_existente["observacion"] if muestra_existente else "", height=150)   # Campo para observaciones, precargado si existe

    st.subheader("Nuevo análisis")
    tipos = [           # Lista de tipos de análisis posibles
        "Índice de yodo [% p/p I2 abs]", "Índice OH [mg KHO/g]",
        "Índice de acidez [mg KOH/g]", "Índice de epóxido [mol/100g]",
        "Humedad [%]", "PM [g/mol]", "Funcionalidad [#]",
        "Viscosidad dinámica [cP]", "Densidad [g/mL]", "Otro análisis"
    ]
    df = pd.DataFrame([{"Tipo": "", "Valor": 0.0, "Fecha": date.today(), "Observaciones": ""}])  # Se crea una tabla editable inicial con una fila vacía
    nuevos_analisis = st.data_editor(df, num_rows="dynamic", use_container_width=True,           # Tabla editable para ingresar nuevos análisis
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
        nuevos_validos = [a for a in nuevos if a["tipo"] != "" and a["valor"] != 0]   # Se filtran los análisis con tipo no vacío y valor distinto de cero

        guardar_muestra(    # Guarda los análisis combinando los anteriores y los nuevos válidos
            nombre_muestra,
            observacion,
            previos + nuevos_validos,
            muestra_existente.get("espectros") if muestra_existente else []
        )
        st.success("Análisis guardado.")
        st.rerun()          # Se recarga la página para ver los cambios

    st.subheader("Análisis cargados")
    muestras = cargar_muestras()    # Recarga las muestras después de guardar
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
    df_vista = pd.DataFrame(tabla)  # Tabla completa de todos los análisis
    if not df_vista.empty:
        st.dataframe(df_vista, use_container_width=True)    # Visualiza la tabla con los análisis

        st.subheader("Eliminar análisis")  # Permite elegir un análisis para eliminar mediante descripción compuesta
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

        st.subheader("Exportar")   # Permite descargar todos los análisis en formato Excel
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_vista.to_excel(writer, index=False, sheet_name="Muestras")
        st.download_button("Descargar Excel",
            data=buffer.getvalue(),
            file_name=f"lab-polioles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No hay análisis cargados.")  # Mensaje si no hay datos cargados aún

# --- HOJA 2 --- "Análisis de datos" ---
with tab2:
    st.title("Análisis de datos")  # Título principal de la hoja
    muestras = cargar_muestras()   # Se cargan todas las muestras desde Firestore
    tabla = []
    for m in muestras:
        for i, a in enumerate(m.get("analisis", [])):
            tabla.append({    # Se arma una tabla plana con todos los análisis de todas las muestras
                "Fecha": a.get("fecha", ""),
                "ID": f"{m['nombre']}__{i}",  # Se asigna un ID único combinando nombre e índice
                "Nombre": m["nombre"],
                "Tipo": a.get("tipo", ""),
                "Valor": a.get("valor", ""),
                "Observaciones": a.get("observaciones", "")
            })

    df = pd.DataFrame(tabla)
    if df.empty:  # Si no hay datos, se detiene la ejecución
        st.info("No hay análisis cargados.")
        st.stop()

    st.subheader("Tabla completa de análisis") # Se muestra la tabla sin la columna interna de ID
    st.dataframe(df.drop(columns=["ID"]), use_container_width=True)

    st.subheader("Seleccionar análisis") # Permite al usuario seleccionar uno o más análisis específicos por su descripción completa
    seleccion = st.multiselect("Seleccione uno o más análisis para graficar", df["ID"].tolist(),
                               format_func=lambda i: f"{df[df['ID'] == i]['Nombre'].values[0]} - {df[df['ID'] == i]['Tipo'].values[0]} - {df[df['ID'] == i]['Fecha'].values[0]}")

    df_sel = df[df["ID"].isin(seleccion)]  # Se filtran los análisis seleccionados
    df_avg = df_sel.groupby(["Nombre", "Tipo"], as_index=False)["Valor"].mean()  # Se agrupan por muestra y tipo, calculando el promedio si hay repeticiones

    st.subheader("Resumen de selección promediada")  # Se muestra la tabla resumen con los valores promedio
    st.dataframe(df_avg, use_container_width=True)

    st.subheader("Gráfico XY")  # Lista de tipos únicos disponibles
    tipos_disponibles = sorted(df_avg["Tipo"].unique())

    colx, coly = st.columns(2)  # Permite elegir qué tipo de análisis usar como eje X e Y
    with colx:
        tipo_x = st.selectbox("Selección de eje X", tipos_disponibles)
    with coly:
        tipo_y = st.selectbox("Selección de eje Y", tipos_disponibles)

    # Se obtienen los valores por muestra para X e Y, y se identifican las muestras comunes
    muestras_x = df_avg[df_avg["Tipo"] == tipo_x][["Nombre", "Valor"]].set_index("Nombre")
    muestras_y = df_avg[df_avg["Tipo"] == tipo_y][["Nombre", "Valor"]].set_index("Nombre")
    comunes = muestras_x.index.intersection(muestras_y.index)

    usar_manual_x = st.checkbox("Asignar valores X manualmente") # El usuario ingresa manualmente los valores de X
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
        x = muestras_x.loc[comunes, "Valor"].tolist()   # Sino se toman los valores reales de X
        nombres = comunes.tolist()

    y = muestras_y.loc[comunes, "Valor"].tolist() # Se toman los valores reales de Y

    if x and y and len(x) == len(y):
        fig, ax = plt.subplots()
        ax.scatter(x, y)   # Se grafica un scatter plot de X vs Y
        for i, txt in enumerate(nombres):
            ax.annotate(txt, (x[i], y[i]))  # Se anotan los nombres de las muestras
        ax.set_xlabel(tipo_x)
        ax.set_ylabel(tipo_y)
        st.pyplot(fig)  # Se muestra el gráfico en Streamlit

        buf_img = BytesIO()
        fig.savefig(buf_img, format="png")
        st.download_button("📷 Descargar gráfico", buf_img.getvalue(),   # Permite descargar el gráfico generado como imagen PNG
                           file_name=f"grafico_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
                           mime="image/png")
    else:
        st.warning("Los datos seleccionados no son compatibles para graficar.")# Mensaje de advertencia si no hay suficientes datos coincidentes

# --- HOJA 3 --- "Carga de espectros" ---
with tab3:
    st.title("Carga de espectros")  # Título principal de la hoja

    muestras = cargar_muestras()    # Se cargan todas las muestras desde Firestore
    nombres_muestras = [m["nombre"] for m in muestras]  # Lista de nombres para el selector

    st.subheader("Subir nuevo espectro")
    nombre_sel = st.selectbox("Seleccionar muestra", nombres_muestras)
    tipos_espectro_base = [
        "FTIR-Acetato", "FTIR-Cloroformo", "FTIR-ATR",
        "RMN 1H", "RMN 13C", "RMN-LF 1H"
    ]
    if "tipos_espectro" not in st.session_state: # Si es la primera vez que se abre la app, se inicializa la lista de tipos de espectro
        st.session_state.tipos_espectro = tipos_espectro_base.copy() # Garantiza persistencia dinámica de tipos agregados por el usuario
    tipo_espectro = st.selectbox("Tipo de espectro", st.session_state.tipos_espectro)

    # Variables opcionales de ingreso manual para cálculos específicos en FTIR
    senal_3548 = None
    senal_3611 = None
    peso_muestra = None

   # Campos específicos si el espectro es FTIR-Acetato
    if tipo_espectro == "FTIR-Acetato":
        st.markdown("**Datos manuales opcionales para FTIR-Acetato:**")
        senal_3548 = st.number_input("Señal de Acetato a 3548 cm⁻¹", step=0.0001, format="%.4f")
        peso_muestra = st.number_input("Peso de la muestra [g]", step=0.0001, format="%.4f")

   # Campos específicos si el espectro es FTIR-Cloroformo
    elif tipo_espectro == "FTIR-Cloroformo":
        st.markdown("**Datos manuales opcionales para FTIR-Cloroformo:**")
        senal_3611 = st.number_input("Señal de Cloroformo a 3611 cm⁻¹", step=0.0001, format="%.4f")
        peso_muestra = st.number_input("Peso de la muestra [g]", step=0.0001, format="%.4f")

    # Permite agregar un nuevo tipo de espectro personalizado
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

        st.markdown("### Vista previa") #Se genera una vista previa
        if es_imagen:   # Si es imagen, se muestra directamente
            st.image(archivo, use_container_width=True)
        else:           # Si es tabla, se intenta graficar los primeros dos ejes
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

    if st.button("Guardar espectro") and archivo:  # Guardar espectro
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
        espectros.append(nuevo)

        for m in muestras:
            if m["nombre"] == nombre_sel:
                m["espectros"] = espectros
                guardar_muestra(m["nombre"], m.get("observacion", ""), m.get("analisis", []), espectros)
                st.success("Espectro guardado.")
                st.rerun()

    st.subheader("Espectros cargados")   # Tabla de espectros ya cargados
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

    df_esp_tabla = pd.DataFrame(filas)   # Eliminar espectros (Tabla de seleccion)
    if not df_esp_tabla.empty:
        st.dataframe(df_esp_tabla.drop(columns=["ID"]), use_container_width=True)
        seleccion = st.selectbox(
            "Eliminar espectro",
            df_esp_tabla["ID"],
            format_func=lambda i: f"{df_esp_tabla[df_esp_tabla['ID'] == i]['Muestra'].values[0]} – {df_esp_tabla[df_esp_tabla['ID'] == i]['Tipo'].values[0]} – {df_esp_tabla[df_esp_tabla['ID'] == i]['Archivo'].values[0]} – {df_esp_tabla[df_esp_tabla['ID'] == i]['Fecha'].values[0]}"
        )
        if st.button("Eliminar espectro"):  # Eliminar espectros (Botón)
            nombre, idx = seleccion.split("__")
            for m in muestras:
                if m["nombre"] == nombre:
                    m["espectros"].pop(int(idx))
                    guardar_muestra(m["nombre"], m.get("observacion", ""), m.get("analisis", []), m.get("espectros", []))
                    st.success("Espectro eliminado.")
                    st.rerun()

        if st.button("📦 Preparar descarga"):   #Preparar descarga de espectros (Excel y ZIP)
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
                    st.session_state["zip_bytes"] = zip_bytes
                    st.session_state["zip_name"] = os.path.basename(zip_path)

        if "zip_bytes" in st.session_state:   # Botón de descarga del ZIP preparado
            st.download_button("📦 Descargar espectros", data=st.session_state["zip_bytes"],
                               file_name=st.session_state["zip_name"],
                               mime="application/zip")
    else:
        st.info("No hay espectros cargados.")

# --- HOJA 4 --- "Análisis de espectros" ---
with tab4:
    st.title("Análisis de espectros")  # Título principal de la hoja

    muestras = cargar_muestras()  # Cargar todas las muestras desde Firestore
    if not muestras:
        st.info("No hay muestras cargadas con espectros.")
        st.stop()

    espectros_info = []     # Consolidar la información de todos los espectros en una tabla plana
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

    # Filtros de búsqueda de espectros
    st.subheader("Filtrar espectros")
    muestras_disp = df_esp["Muestra"].unique().tolist()
    tipos_disp = df_esp["Tipo"].unique().tolist()
    muestras_sel = st.multiselect("Muestras", muestras_disp, default=[])
    tipos_sel = st.multiselect("Tipo de espectro", tipos_disp, default=[])
    df_filtrado = df_esp[df_esp["Muestra"].isin(muestras_sel) & df_esp["Tipo"].isin(tipos_sel)]

    # Generar nombres para cada espectro disponible
    espectros_info = []
    for idx, row in df_filtrado.iterrows():
        fecha = row.get("Fecha", "Sin fecha")
        observaciones = row.get("Observaciones", "Sin observaciones")
        if not observaciones:
            observaciones = "Sin observaciones"
        if len(observaciones) > 80:
            observaciones = observaciones[:77] + "..."
        extension = os.path.splitext(row["Nombre archivo"])[1].lower().strip(".")
        nombre_espectro = f"{row['Muestra']} – {row['Tipo']} – {fecha} – {observaciones} ({extension})"
        espectros_info.append({
            "identificador": idx,
            "nombre": nombre_espectro
        })

    # Selección de espectros a visualizar
    if espectros_info:
        espectros_nombres = [e["nombre"] for e in espectros_info]
        seleccionados_nombres = st.multiselect(
            "Seleccionar espectros a visualizar:", 
            espectros_nombres, 
            default=[]
        )
        seleccionados_idx = [e["identificador"] for e in espectros_info if e["nombre"] in seleccionados_nombres]
        df_filtrado = df_filtrado.loc[seleccionados_idx]
    else:
        st.warning("No hay espectros disponibles para seleccionar.")

    # Separar espectros numéricos y espectros en imagen
    df_datos = df_filtrado[~df_filtrado["Es imagen"]]
    df_imagenes = df_filtrado[df_filtrado["Es imagen"]]

    # Gráfico combinado para espectros numéricos
    if not df_datos.empty:
        st.subheader("Gráfico combinado de espectros numéricos")
        fig, ax = plt.subplots()
        rango_x = [float("inf"), float("-inf")]
        rango_y = [float("inf"), float("-inf")]
        data_validos = []

        # Decodificación y limpieza de cada archivo
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

                # Conversión forzada a numérico (corrige errores de coma como decimal)
                for col in [col_x, col_y]:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna()
                if df.empty:
                    continue
                data_validos.append((row["Muestra"], row["Tipo"], df[col_x], df[col_y]))

                # Ajustar rango global automático
                rango_x[0] = min(rango_x[0], df[col_x].min())
                rango_x[1] = max(rango_x[1], df[col_x].max())
                rango_y[0] = min(rango_y[0], df[col_y].min())
                rango_y[1] = max(rango_y[1], df[col_y].max())
            except:
                continue

        # Graficar solo si hay datos válidos
        if not data_validos:
            st.warning("No se pudo graficar ningún espectro válido.")
        else:
            # Selección de rangos de visualización
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x_min = st.number_input("X mínimo", value=rango_x[0])
            with col2:
                x_max = st.number_input("X máximo", value=rango_x[1])
            with col3:
                y_min = st.number_input("Y mínimo", value=rango_y[0])
            with col4:
                y_max = st.number_input("Y máximo", value=rango_y[1])

            # Graficar todos los espectros seleccionados
            for muestra, tipo, x, y in data_validos:
                x_filtrado = x[(x >= x_min) & (x <= x_max)]
                y_filtrado = y[(x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)]
                if not y_filtrado.empty:
                    ax.plot(x_filtrado[:len(y_filtrado)], y_filtrado, label=f"{muestra} – {tipo}")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.legend()            
            st.pyplot(fig)

            # Exportar resumen y hojas individuales en Excel
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

            st.download_button(     # Boton para exportar a excel
                "📥 Exportar resumen a Excel",
                data=excel_buffer.getvalue(),
                file_name=f"espectros_resumen_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

    # Mostrar imágenes cargadas
    if not df_imagenes.empty:
        st.subheader("Imágenes de espectros")
        for _, row in df_imagenes.iterrows():
            try:
                imagen = BytesIO(base64.b64decode(row["Contenido"]))
                st.image(imagen, caption=f"{row['Muestra']} – {row['Tipo']} – {row['Fecha']}", use_container_width=True)
            except:
                st.warning(f"No se pudo mostrar la imagen: {row['Nombre archivo']}")

    # Descarga agrupada de imágenes seleccionadas + info TXT
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
    
                        # Guardar archivo TXT con metadatos
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
    
            st.download_button("📦 Descargar ZIP de imágenes",  # Boton para descargar ZIP
                               data=zip_bytes,
                               file_name=os.path.basename(zip_path),
                               mime="application/zip")

# --- HOJA 5 --- "Índice OH espectroscópico"
with tab5:
    st.title("Índice OH espectroscópico")  # Título principal de la hoja
    muestras = cargar_muestras()  # Cargar todas las muestras con espectros
    if not muestras:
        st.info("No hay muestras cargadas para analizar.")
        st.stop()
    espectros_info = [] # Lista que almacenará los datos procesados por muestra

    # Recorrer todas las muestras para identificar espectros relevantes
    for m in muestras:
        espectros = m.get("espectros", [])
        for e in espectros:
            tipo = e.get("tipo", "")
            if tipo not in ["FTIR-Acetato", "FTIR-Cloroformo"]:  # Solo calcular índice OH para estos tipos de espectro
                continue

            contenido = e.get("contenido")
            es_imagen = e.get("es_imagen", False)
            valor_y_extraido = None

            # Intentar extraer el valor Y cercano a 3548 o 3611 del archivo numérico
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

            # Agregar a la tabla intermedia
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

    # Crear columna 'Señal solvente' unificando los datos manuales ingresados
    def obtener_senal_solvente(row):
        if row["Tipo espectro"] == "FTIR-Acetato":
            return row["Señal manual 3548"]
        elif row["Tipo espectro"] == "FTIR-Cloroformo":
            return row["Señal manual 3611"]
        else:
            return None

    df_muestras["Señal solvente"] = df_muestras.apply(obtener_senal_solvente, axis=1)

    # Calcular Índice OH
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

    # Crear tabla
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

# --- HOJA 6 --- "Consola" ---
with tab6:
    st.title("Consola")  # Título principal de la pestaña

    muestras = cargar_muestras()
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # --- BLOQUE ORIGINAL DE EXPANSORES ---
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
                    st.markdown(f"🖼️ {etiqueta}" if e.get("es_imagen", False) else f"📈 {etiqueta}")
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
                                    f.write(base64.b64decode(contenido))
                                zipf.write(ruta, arcname=nombre)

                        with open(zip_path, "rb") as final_zip:
                            st.download_button("📦 Descargar ZIP de espectros",
                                data=final_zip.read(),
                                file_name=f"espectros_{muestra['nombre']}.zip",
                                mime="application/zip",
                                key=f"dl_zip_{muestra['nombre']}")

    st.markdown("---")

    # --- TABLA 1 ---
    st.subheader("📋 Descargas por muestra")
    header1 = st.columns([2, 1, 1])
    header1[0].markdown("**Muestra**")
    header1[1].markdown("**📥 Excel**")
    header1[2].markdown("**📦 ZIP**")

    for i, m in enumerate(muestras):
        col1, col2, col3 = st.columns([2, 1, 1])
        nombre = m["nombre"]
        analisis = len(m.get("analisis", []))
        espectros = len(m.get("espectros", []))
        col1.markdown(f"**{nombre}**")
        col2.download_button(f"📥 {analisis}", data=b"", file_name=f"analisis_{nombre}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"excel1_{i}", use_container_width=True)
        col3.download_button(f"📦 {espectros}", data=b"", file_name=f"espectros_{nombre}.zip", mime="application/zip", key=f"zip1_{i}", use_container_width=True)

    st.markdown("---")

    # --- TABLAS 2 y 3 LADO A LADO ---
    colA, colB = st.columns(2)

    with colA:
        st.subheader("🟢 Descargas por análisis")
        conteo_analisis = {}
        for m in muestras:
            for a in m.get("analisis", []):
                tipo = a.get("tipo", "")
                if tipo:
                    conteo_analisis[tipo] = conteo_analisis.get(tipo, 0) + 1
        df2 = pd.DataFrame([{"Tipo de Análisis": k, "Muestras": v} for k, v in conteo_analisis.items()])
        header2 = st.columns([3, 1])
        header2[0].markdown("**Tipo de Análisis**")
        header2[1].markdown("**📥 Excel**")
        for i, row in df2.iterrows():
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{row['Tipo de Análisis']}** ({row['Muestras']})")
            col2.download_button("📥", data=b"", file_name=f"analisis_{row['Tipo de Análisis']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"excel2_{i}", use_container_width=True)

    with colB:
        st.subheader("🟣 Descargas por espectros")
        conteo_espectros = {}
        for m in muestras:
            for e in m.get("espectros", []):
                tipo = e.get("tipo", "")
                if tipo:
                    conteo_espectros[tipo] = conteo_espectros.get(tipo, 0) + 1
        df3 = pd.DataFrame([{"Tipo de Espectro": k, "Muestras": v} for k, v in conteo_espectros.items()])
        header3 = st.columns([3, 1])
        header3[0].markdown("**Tipo de Espectro**")
        header3[1].markdown("**📦 ZIP**")
        for i, row in df3.iterrows():
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{row['Tipo de Espectro']}** ({row['Muestras']})")
            col2.download_button("📦", data=b"", file_name=f"espectros_{row['Tipo de Espectro']}.zip", mime="application/zip", key=f"zip3_{i}", use_container_width=True)

    st.markdown("---")
    st.download_button("📦 Descargar TODO", data=b"", file_name="todo_muestras.zip", mime="application/zip")

    st.markdown("---")
    if st.button("Cerrar sesión"):
        st.session_state.pop("token", None)
        st.rerun()


# --- HOJA 7 --- "Sugerencias" ---
with tab7:
    st.title("Sugerencias")   # Título principal de la hoja

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