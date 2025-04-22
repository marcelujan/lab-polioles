
import streamlit as st
import pandas as pd
import toml
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import date, datetime
from io import BytesIO, StringIO
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
    cred_dict["private_key"] = cred_dict["private_key"].replace("\n", "\n")
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

    rango_x = [float('inf'), float('-inf')]
    rango_y = [float('inf'), float('-inf')]
    tablas_individuales = []
    x_values = None

    for _, row in df_filtrado.iterrows():
        if not row["Es imagen"]:
            try:
                extension = os.path.splitext(row["Nombre archivo"])[1].lower()
                if extension == ".xlsx":
                    binario = BytesIO(bytes.fromhex(row["Contenido"]))
                    df_temp = pd.read_excel(binario)
                else:
                    contenido = StringIO(bytes.fromhex(row["Contenido"]).decode("latin1"))
                    separadores = [",", "	", ";", " "]
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
                df_temp = df_temp.dropna()

                min_x, max_x = df_temp[col_x].agg(["min", "max"])
                min_y, max_y = df_temp[col_y].agg(["min", "max"])

                rango_x[0] = min(rango_x[0], min_x)
                rango_x[1] = max(rango_x[1], max_x)
                rango_y[0] = min(rango_y[0], min_y)
                rango_y[1] = max(rango_y[1], max_y)

                tablas_individuales.append((row["Muestra"], row["Tipo"], df_temp[[col_x, col_y]]))
                if x_values is None:
                    x_values = df_temp[col_x]
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
        df_resumen = pd.DataFrame({'X': x_values}) if x_values is not None else pd.DataFrame()

        for muestra, tipo, df in tablas_individuales:
            col_x, col_y = df.columns[:2]
            df_fil = df[
                (df[col_x] >= x_min) & (df[col_x] <= x_max) &
                (df[col_y] >= y_min) & (df[col_y] <= y_max)
            ]
            if df_fil.empty:
                continue
            ax.plot(df_fil[col_x], df_fil[col_y], label=f"{muestra} – {tipo}")
            df_resumen[f"{muestra} – {tipo}"] = df_fil[col_y].reset_index(drop=True)
            se_grafico_algo = True

        if se_grafico_algo:
            ax.legend()
            st.pyplot(fig)

            buf_img = BytesIO()
            fig.savefig(buf_img, format="png", bbox_inches="tight")
            st.download_button("📷 Descargar PNG", data=buf_img.getvalue(),
                               file_name="grafico_combinado.png", mime="image/png")

            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df_resumen.to_excel(writer, index=False, sheet_name="Resumen")
                for muestra, tipo, df in tablas_individuales:
                    nombre_hoja = f"{muestra}_{tipo}".replace(" ", "_")[:31]
                    df.to_excel(writer, index=False, sheet_name=nombre_hoja)
            excel_buffer.seek(0)
            st.download_button("📊 Descargar tabla", data=excel_buffer.getvalue(),
                               file_name="espectros_combinados.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("No se pudo graficar ningún espectro válido.")
