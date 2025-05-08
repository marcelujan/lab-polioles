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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Laboratorio de Polioles",
    "Análisis de datos",
    "Carga de espectros",
    "Análisis de espectros",
    "Índice OH espectroscópico",
    "Análisis RMN",
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
                tipo = row["Tipo"]
                valor = row["Valor"]
                fecha = str(row["Fecha"])
                obs = row["Observaciones"]
                resumen_obs = obs.replace("\n", " ").strip()[:30].replace(" ", "_")

                id_unico = f"{tipo}-{valor}-{fecha}-{resumen_obs}"

                nuevos.append({
                    "tipo": tipo,
                    "valor": valor,
                    "fecha": fecha,
                    "observaciones": obs,
                    "id": id_unico  # nuevo campo
                })

        nuevos_validos = [a for a in nuevos if a["tipo"] != "" and a["valor"] != 0]

        guardar_muestra(
            nombre_muestra,
            observacion,
            previos + nuevos_validos,
            muestra_existente.get("espectros") if muestra_existente else []
        )
        st.success("Análisis guardado.")
        st.rerun()       # Se recarga la página para ver los cambios

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
    mascaras_rmn1h = []

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

    # Máscara D/T2
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

        # Generar nuevo nombre de archivo basado en muestra, tipo, fecha y resumen de observaciones
        extension = os.path.splitext(archivo.name)[1].lower().strip(".")
        resumen_obs = observaciones.replace("\n", " ").strip()[:30].replace(" ", "_")
        fecha_str = fecha_espectro.strftime("%Y-%m-%d")
        nombre_sin_ext = f"{nombre_sel}_{tipo_espectro}_{fecha_str}-{resumen_obs}"
        nombre_generado = f"{nombre_sin_ext}.{extension}"

        # Mostrar nombre final antes de guardar
        st.markdown(f"**🆔 Nuevo nombre asignado al archivo para su descarga:** `{nombre_generado}`")

    if st.button("Guardar espectro") and archivo:
        espectros = next((m for m in muestras if m["nombre"] == nombre_sel), {}).get("espectros", [])

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

        espectros.append(nuevo)

        for m in muestras:
            if m["nombre"] == nombre_sel:
                m["espectros"] = espectros
                guardar_muestra(m["nombre"], m.get("observacion", ""), m.get("analisis", []), espectros)
                st.success("Espectro guardado.")
                st.rerun()

    st.subheader("Espectros cargados")   # Tabla de espectros ya cargados
    filas = []
    filas_mascaras = []
    for m in muestras:
        for i, e in enumerate(m.get("espectros", [])):
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
                    if not df_mascaras.empty:
                        df_mascaras.to_excel(writer, index=False, sheet_name="Mascaras_RMN1H")
                        
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

# --- HOJA 6 --- "Análisis RMN" ---
with tab6:
    st.title("Análisis RMN")

    muestras = cargar_muestras()
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # --- FILTRAR MUESTRAS Y ESPECTROS ---
    espectros_rmn = []
    for m in muestras:
        for i, e in enumerate(m.get("espectros", [])):
            tipo = e.get("tipo", "").upper()
            if "RMN" in tipo:
                espectros_rmn.append({
                    "muestra": m["nombre"],
                    "tipo": tipo,
                    "es_imagen": e.get("es_imagen", False),
                    "archivo": e.get("nombre_archivo", ""),
                    "contenido": e.get("contenido"),
                    "fecha": e.get("fecha"),
                    "senal_3548": e.get("senal_3548"),
                    "senal_3611": e.get("senal_3611"),
                    "peso_muestra": e.get("peso_muestra"),
                    "mascaras": e.get("mascaras", []),
                    "id": f"{m['nombre']}__{i}"
                })

    df_rmn = pd.DataFrame(espectros_rmn)

    st.subheader("Filtrar espectros")
    muestras_disp = sorted(df_rmn["muestra"].unique())
    muestras_sel = st.multiselect("Muestras", muestras_disp, default=[])

    df_filtrado = df_rmn[df_rmn["muestra"].isin(muestras_sel)]

    espectros_info = []
    for idx, row in df_filtrado.iterrows():
        nombre = f"{row['muestra']} – {row['archivo']}"
        espectros_info.append({"id": row["id"], "nombre": nombre})

    seleccionados = st.multiselect("Seleccionar espectros a visualizar:",
        options=[e["id"] for e in espectros_info],
        format_func=lambda i: next(e["nombre"] for e in espectros_info if e["id"] == i))

    df_sel = df_filtrado[df_filtrado["id"].isin(seleccionados)]

    # --- ZONA RMN 1H ---
    st.subheader("🔬 RMN 1H")
    df_rmn1H = df_sel[(df_sel["tipo"] == "RMN 1H") & (~df_sel["es_imagen"])].copy()
    if df_rmn1H.empty:
        st.info("No hay espectros RMN 1H numéricos seleccionados.")
    else:
        st.markdown("**Máscara D/T2:**")
        usar_mascara = {}
        colores = plt.cm.tab10.colors
        fig, ax = plt.subplots()
        filas_mascaras = []

        for idx, (_, row) in enumerate(df_rmn1H.iterrows()):
            color = colores[idx % len(colores)]
            usar_mascara[row["id"]] = st.checkbox(f"{row['muestra']} – {row['archivo']}", value=False, key=f"chk_mask_{row['id']}")
            try:
                contenido = BytesIO(base64.b64decode(row["contenido"]))
                extension = os.path.splitext(row["archivo"])[1].lower()
                if extension == ".xlsx":
                    df = pd.read_excel(contenido)
                else:
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
                        raise ValueError("No se pudo leer el archivo.")

                col_x, col_y = df.columns[:2]
                df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
                df[col_y] = pd.to_numeric(df[col_y], errors="coerce")
                df = df.dropna()

                # Calcular área de asignación H entre 4.8 y 5.6
                df_h = df[(df[col_x] >= 4.8) & (df[col_x] <= 5.6)]
                asignacion_h = np.trapz(df_h[col_y], df_h[col_x]) if not df_h.empty else np.nan

                ax.plot(df[col_x], df[col_y], label=f"{row['muestra']}", color=color)

                if usar_mascara.get(row["id"], False):
                    for j, mascara in enumerate(row.get("mascaras", [])):
                        x0 = mascara.get("x_min")
                        x1 = mascara.get("x_max")
                        d = mascara.get("difusividad")
                        t2 = mascara.get("t2")
                        if x0 is not None and x1 is not None:
                            sub_df = df[(df[col_x] >= min(x0, x1)) & (df[col_x] <= max(x0, x1))]
                            area = np.trapz(sub_df[col_y], sub_df[col_x]) if not sub_df.empty else 0
                            h = area / asignacion_h if asignacion_h else np.nan
                            ax.axvspan(x0, x1, color=color, alpha=0.3)
                            if d and t2:
                                ax.text((x0+x1)/2, max(df[col_y])*0.9,
                                        f"D={d:.1e}     T2={t2:.3f}", ha="center", va="center", fontsize=6, color="black", rotation=90)
                                filas_mascaras.append({
                                    "Muestra": row["muestra"],
                                    "Archivo": row["archivo"],
                                    "D [m2/s]": f"{d:.1e}",
                                    "T2 [s]": f"{t2:.3f}",
                                    "Xmin [ppm]": f"{x0:.2f}",
                                    "Xmax [ppm]": f"{x1:.2f}",
                                    "Área": f"{area:.2f}",
                                    "H": f"{h:.2f}" if not np.isnan(h) else "—"
                                })
            except:
                st.warning(f"No se pudo graficar espectro: {row['archivo']}")

        ax.set_xlabel("[ppm]")
        ax.set_ylabel("Señal")
        ax.legend()
        st.pyplot(fig)

        if filas_mascaras:
            df_tabla = pd.DataFrame(filas_mascaras)      
            st.dataframe(df_tabla.style.set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]},
                {'selector': 'td', 'props': [('text-align', 'center')]}
            ]), use_container_width=True)
            st.caption("*Asignación: 1 H = integral entre x = 4,8 y x = 5,6")

            # Botón de descarga de tabla de máscaras
            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                df_tabla.to_excel(writer, index=False, sheet_name="Mascaras_RMN1H")
            buffer_excel.seek(0)
            st.download_button("📑 Descargar máscaras D/T2", data=buffer_excel.getvalue(), file_name="mascaras_rmn1h.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Botón para descargar imagen del gráfico RMN 1H
        buffer_img = BytesIO()
        fig.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
        st.download_button("📷 Descargar gráfico RMN 1H", data=buffer_img.getvalue(), file_name="grafico_rmn1h.png", mime="image/png")            

    # --- ZONA RMN 13C ---
    st.subheader("🧪 RMN 13C")
    df_rmn13C = df_sel[(df_sel["tipo"] == "RMN 13C") & (~df_sel["es_imagen"])]
    if df_rmn13C.empty:
        st.info("No hay espectros RMN 13C numéricos seleccionados.")
    else:
        fig13, ax13 = plt.subplots()
        for _, row in df_rmn13C.iterrows():
            try:
                contenido = BytesIO(base64.b64decode(row["contenido"]))
                extension = os.path.splitext(row["archivo"])[1].lower()
                if extension == ".xlsx":
                    df = pd.read_excel(contenido)
                else:
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
                        raise ValueError("No se pudo leer el archivo.")

                col_x, col_y = df.columns[:2]
                df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
                df[col_y] = pd.to_numeric(df[col_y], errors="coerce")
                df = df.dropna()
                ax13.plot(df[col_x], df[col_y], label=f"{row['muestra']}")
            except:
                st.warning(f"No se pudo graficar espectro: {row['archivo']}")

        ax13.set_xlabel("[ppm]")
        ax13.set_ylabel("Señal")
        ax13.legend()
        st.pyplot(fig13)

        # Botón para descargar imagen del gráfico RMN 13C
        buffer_img13 = BytesIO()
        fig13.savefig(buffer_img13, format="png", dpi=300, bbox_inches="tight")
        st.download_button("📷 Descargar gráfico RMN 13C", data=buffer_img13.getvalue(), file_name="grafico_rmn13c.png", mime="image/png")

    # --- ZONA IMÁGENES ---
    st.subheader("🖼️ Espectros imagen")
    df_rmn_img = df_sel[df_sel["es_imagen"]]
    if df_rmn_img.empty:
        st.info("No hay espectros RMN en formato imagen seleccionados.")
    else:
        for _, row in df_rmn_img.iterrows():
            try:
                imagen = BytesIO(base64.b64decode(row["contenido"]))
                st.image(imagen, caption=f"{row['muestra']} – {row['archivo']} ({row['fecha']})", use_container_width=True)
            except:
                st.warning(f"No se pudo mostrar imagen: {row['archivo']}")

        # Botón para descargar ZIP con todas las imágenes mostradas
        with TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, f"imagenes_rmn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for _, row in df_rmn_img.iterrows():
                    nombre = row["archivo"]
                    contenido = row["contenido"]
                    if not contenido:
                        continue
                    try:
                        img_bytes = base64.b64decode(contenido)
                        ruta = os.path.join(tmpdir, nombre)
                        with open(ruta, "wb") as f:
                            f.write(img_bytes)
                        zipf.write(ruta, arcname=nombre)
                    except:
                        continue
            with open(zip_path, "rb") as final_zip:
                st.download_button("📦 Descargar imágenes RMN", data=final_zip.read(), file_name=os.path.basename(zip_path), mime="application/zip")
                
# --- HOJA 7 --- "Consola" ---
with tab7:
    st.title("Consola")  # Título principal de la hoja

    muestras = cargar_muestras()
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # BLOQUE CON EXPANSORES DETALLADOS POR MUESTRA
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
                buffer.seek(0)  # 🔧 Esto garantiza que el archivo se lea desde el principio

                st.download_button(
                    "⬇️ Descargar análisis",
                    data=buffer.getvalue(),
                    file_name=f"analisis_{muestra['nombre']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

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
                    st.download_button("📑 Descargar máscaras RMN 1H", data=buffer.getvalue(), file_name=f"mascaras_{muestra['nombre']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_mask_{muestra['nombre']}")

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
                        except Exception:
                            continue

                    # Añadir archivo Excel con máscaras si existen
                    if not df_mascaras.empty:
                        zipf.writestr("mascaras_rmn1h.xlsx", buffer.getvalue())
                buffer_zip.seek(0)

                st.download_button(
                    "📦 Descargar ZIP de espectros",
                    data=buffer_zip.getvalue(),
                    file_name=f"espectros_{muestra['nombre']}.zip",
                    mime="application/zip",
                    key=f"dl_zip_{muestra['nombre']}"
                )

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
            df_espectros = pd.DataFrame(m.get("espectros", []))

            filas_mascaras = []
            for e in m.get("espectros", []):
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
                for e in m.get("espectros", []):
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
            c3.download_button(f"📦 {len(m.get('espectros', []))}", data=buffer_zip.getvalue(), file_name=f"espectros_{nombre}.zip", mime="application/zip", use_container_width=True, key=f"zip1_{i}")

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
            for e in m.get("espectros", []):
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
                    for e in m.get("espectros", []):
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
            for e in m.get("espectros", []):
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

# --- HOJA 8 --- "Sugerencias" ---
with tab8:
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
