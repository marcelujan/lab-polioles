
# --- Fragmento funcional de Hoja 4 modificado ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import os

st.set_page_config(page_title="Laboratorio de Polioles", layout="wide")

tab4 = st.tabs(["Análisis de espectros"])[0]

with tab4:
    st.title("Análisis de espectros")

    def cargar_muestras():
        return []

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

    st.subheader("Espectros visualizados")

    for _, row in df_filtrado.iterrows():
        st.markdown(f"**{row['Muestra']}** – *{row['Tipo']}* – {row['Nombre archivo']}")
        st.markdown(f"`{row['Observaciones']}`")

        if row["Es imagen"]:
            st.image(BytesIO(bytes.fromhex(row["Contenido"])), use_column_width=True)
        else:
            try:
                contenido = StringIO(row["Contenido"])
                df_espectro = pd.read_csv(contenido, sep=None, engine="python")

                if df_espectro.shape[1] >= 2:
                    col_x, col_y = df_espectro.columns[:2]
                    min_x, max_x = float(df_espectro[col_x].min()), float(df_espectro[col_x].max())
                    min_y, max_y = float(df_espectro[col_y].min()), float(df_espectro[col_y].max())

                    col1, col2 = st.columns(2)
                    with col1:
                        x_min = st.number_input("X mínimo", value=min_x, key=f"x_min_{row['Nombre archivo']}")
                        x_max = st.number_input("X máximo", value=max_x, key=f"x_max_{row['Nombre archivo']}")
                    with col2:
                        y_min = st.number_input("Y mínimo", value=min_y, key=f"y_min_{row['Nombre archivo']}")
                        y_max = st.number_input("Y máximo", value=max_y, key=f"y_max_{row['Nombre archivo']}")

                    df_fil = df_espectro[
                        (df_espectro[col_x] >= x_min) & (df_espectro[col_x] <= x_max) &
                        (df_espectro[col_y] >= y_min) & (df_espectro[col_y] <= y_max)
                    ]

                    fig, ax = plt.subplots()
                    ax.plot(df_fil[col_x], df_fil[col_y])
                    ax.set_xlabel(col_x)
                    ax.set_ylabel(col_y)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    st.pyplot(fig)
                else:
                    st.warning("El archivo tiene menos de dos columnas.")
            except Exception as ex:
                st.error(f"No se pudo graficar el archivo: {ex}")
