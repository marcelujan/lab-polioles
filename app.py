# --- APP COMPLETA: Hoja 1 + Hoja 2 corregida con depuración ---
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

def guardar_muestra(nombre, observacion, analisis):
    datos = {
        "observacion": observacion,
        "analisis": analisis
    }
    db.collection("muestras").document(nombre).set(datos)
    backup_name = f"muestras_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(backup_name, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

tab1, tab2 = st.tabs(["Laboratorio de Polioles", "Análisis de datos"])

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
        guardar_muestra(nombre_muestra, observacion, previos + nuevos)
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
                    guardar_muestra(m["nombre"], m["observacion"], m["analisis"])
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

# --- HOJA 2: Análisis de datos (con depuración) ---
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
    seleccion = st.multiselect("Seleccione uno o más análisis para graficar", df["ID"].tolist(), format_func=lambda i: f"{df[df['ID'] == i]['Nombre'].values[0]} - {df[df['ID'] == i]['Tipo'].values[0]} - {df[df['ID'] == i]['Fecha'].values[0]}", key="multi_e4b96e6c-c8f3-448e-a9b5-72c891d2db71", df["ID"].tolist(),
                               format_func=lambda i: f"{df[df['ID'] == i]['Nombre'].values[0]} - {df[df['ID'] == i]['Tipo'].values[0]} - {df[df['ID'] == i]['Fecha'].values[0]}")

    df_sel = df[df["ID"].isin(seleccion)]
    st.subheader("Resumen de selección")
    st.dataframe(df_sel.drop(columns=["ID"]), use_container_width=True)

    st.subheader("Gráfico XY")
    tipos_disponibles = sorted(df_sel["Tipo"].unique())

    colx, coly = st.columns(2)
    with colx:
        tipo_x = st.selectbox("Selección de eje X", tipos_disponibles)
    with coly:
        tipo_y = st.selectbox("Selección de eje Y", tipos_disponibles)

    muestras_x = df_sel[df_sel["Tipo"] == tipo_x][["Nombre", "Valor"]].set_index("Nombre")
    muestras_y = df_sel[df_sel["Tipo"] == tipo_y][["Nombre", "Valor"]].set_index("Nombre")
    comunes = muestras_x.index.intersection(muestras_y.index)

    usar_manual_x = st.checkbox("Asignar valores X manualmente")

    if usar_manual_x:
        valores_x_manual = []
        nombres = []
        st.markdown("**Asignar valores X manualmente por muestra:**")
        for nombre in comunes:
            val = st.number_input(f"{nombre}", step=0.1, key=f"manual_x_{nombre}_{i}")
            valores_x_manual.append(val)
            nombres.append(nombre)
        x = valores_x_manual
    else:
        x = muestras_x.loc[comunes, "Valor"].tolist()
        nombres = comunes.tolist()

    y = muestras_y.loc[comunes, "Valor"].tolist()

    st.write("🧪 Muestras en común:", nombres)
    st.write("📈 X:", x)
    st.write("📈 Y:", y)

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


# --- HOJA 2: Análisis de datos (con repeticiones y color por muestra) ---
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
    seleccion = st.multiselect("Seleccione uno o más análisis para graficar", df["ID"].tolist(), format_func=lambda i: f"{df[df['ID'] == i]['Nombre'].values[0]} - {df[df['ID'] == i]['Tipo'].values[0]} - {df[df['ID'] == i]['Fecha'].values[0]}", key="multi_e4b96e6c-c8f3-448e-a9b5-72c891d2db71", df["ID"].tolist(),
                               format_func=lambda i: f"{df[df['ID'] == i]['Nombre'].values[0]} - {df[df['ID'] == i]['Tipo'].values[0]} - {df[df['ID'] == i]['Fecha'].values[0]}")

    df_sel = df[df["ID"].isin(seleccion)]
    st.subheader("Resumen de selección")
    st.dataframe(df_sel.drop(columns=["ID"]), use_container_width=True)

    st.subheader("Gráfico XY")
    tipos_disponibles = sorted(df_sel["Tipo"].unique())

    colx, coly = st.columns(2)
    with colx:
        tipo_x = st.selectbox("Selección de eje X", tipos_disponibles)
    with coly:
        tipo_y = st.selectbox("Selección de eje Y", tipos_disponibles)

    df_x = df_sel[df_sel["Tipo"] == tipo_x][["Nombre", "Valor"]].rename(columns={"Valor": "X"})
    df_y = df_sel[df_sel["Tipo"] == tipo_y][["Nombre", "Valor"]].rename(columns={"Valor": "Y"})

    df_xy = pd.merge(df_x, df_y, on="Nombre", how="inner")

    if df_xy.empty:
        st.warning("No hay combinaciones de muestras compatibles para graficar.")
    else:
        st.write("🧪 Datos a graficar:")
        st.dataframe(df_xy)

        fig, ax = plt.subplots()
        muestras_unicas = df_xy["Nombre"].unique()
        colores = cm.get_cmap('tab10', len(muestras_unicas))

        for i, muestra in enumerate(muestras_unicas):
            datos = df_xy[df_xy["Nombre"] == muestra]
            ax.scatter(datos["X"], datos["Y"], label=muestra, color=colores(i))
            for j, row in datos.iterrows():
                ax.annotate(muestra, (row["X"], row["Y"]))

        ax.set_xlabel(tipo_x)
        ax.set_ylabel(tipo_y)
        ax.legend()
        st.pyplot(fig)

        buf_img = BytesIO()
        fig.savefig(buf_img, format="png")
        st.download_button("📷 Descargar gráfico", buf_img.getvalue(),
                           file_name=f"grafico_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
                           mime="image/png")
