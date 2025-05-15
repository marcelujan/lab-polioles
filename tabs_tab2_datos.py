# tabs_tab2_datos.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

def render_tab2(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Análisis de datos")
    st.session_state["current_tab"] = "Análisis de datos"

    muestras = cargar_muestras(db)
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

    muestras_seleccionadas = df_sel["Nombre"].unique().tolist()
    st.session_state["muestra_activa"] = muestras_seleccionadas[0] if len(muestras_seleccionadas) == 1 else None

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
        st.download_button("\U0001F4F7 Descargar gráfico", buf_img.getvalue(),
                           file_name=f"grafico_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
                           mime="image/png")
    else:
        st.warning("Los datos seleccionados no son compatibles para graficar.")


    st.subheader("Gráfico de barras por tipo de análisis")
    tipos_disponibles_barras = sorted(df_avg["Tipo"].unique())
    tipo_barras = st.selectbox("Seleccionar tipo de análisis para barras", tipos_disponibles_barras, key="tipo_barras")
    df_barras = df_avg[df_avg["Tipo"] == tipo_barras]
    if not df_barras.empty:
        fig_barras, ax_barras = plt.subplots()
        ax_barras.bar(df_barras["Nombre"], df_barras["Valor"])
        ax_barras.set_ylabel(tipo_barras)
        ax_barras.set_xlabel("Muestra")
        ax_barras.set_title(f"{tipo_barras} por muestra")
        st.pyplot(fig_barras)

        buf_barras = BytesIO()
        fig_barras.savefig(buf_barras, format="png")
        st.download_button("📷 Descargar gráfico de barras", buf_barras.getvalue(),
                           file_name=f"grafico_barras_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
                           mime="image/png")

    mostrar_sector_flotante(db, key_suffix="tab2")
