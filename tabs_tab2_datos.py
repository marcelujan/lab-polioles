import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

def render_tab2(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Análisis de datos")
    st.session_state["current_tab"] = "Análisis de datos"

    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # Consolidar todos los análisis
    filas = []
    for m in muestras:
        for a in m.get("analisis", []):
            fila = a.copy()
            fila["Muestra"] = m["nombre"]
            filas.append(fila)

    df = pd.DataFrame(filas)
    if df.empty:
        st.warning("No hay análisis físico-químicos cargados.")
        st.stop()

    df = df[["Muestra", "Tipo", "Valor", "Fecha", "Observacion"]]

    st.subheader("Todos los análisis")
    st.dataframe(df, use_container_width=True)

    # Selección de análisis a usar
    df["ID"] = df["Muestra"] + " | " + df["Tipo"] + " | " + df["Fecha"]
    seleccion = st.multiselect("Seleccionar análisis a considerar", df["ID"].tolist(), default=df["ID"].tolist())
    df_sel = df[df["ID"].isin(seleccion)]

    # Promedio por muestra y tipo
    df_avg = df_sel.groupby(["Muestra", "Tipo"])["Valor"].mean().reset_index()

    st.subheader("Valores promediados")
    st.dataframe(df_avg, use_container_width=True)

    # Gráfico XY
    st.subheader("Gráfico XY")
    tipos = df_avg["Tipo"].unique().tolist()
    col1, col2 = st.columns(2)
    x_sel = col1.selectbox("Selección de eje X", tipos, key="eje_x")
    y_sel = col2.selectbox("Selección de eje Y", tipos, key="eje_y")

    df_x = df_avg[df_avg["Tipo"] == x_sel][["Muestra", "Valor"]].rename(columns={"Valor": "X"})
    df_y = df_avg[df_avg["Tipo"] == y_sel][["Muestra", "Valor"]].rename(columns={"Valor": "Y"})
    df_merge = pd.merge(df_x, df_y, on="Muestra")

    if st.checkbox("Ingresar manualmente valores de X", key="manual_x"):
        for i, row in df_merge.iterrows():
            df_merge.at[i, "X"] = st.number_input(f"{row['Muestra']} – {x_sel}", value=float(row["X"]), key=f"manual_x_{i}")

    fig, ax = plt.subplots()
    for _, row in df_merge.iterrows():
        ax.scatter(row["X"], row["Y"], label=row["Muestra"])
        ax.annotate(row["Muestra"], (row["X"], row["Y"]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(x_sel)
    ax.set_ylabel(y_sel)
    st.pyplot(fig)

    # 📊 GRÁFICO DE BARRAS por tipo de análisis (SECCIÓN AGREGADA)
    st.subheader("Gráfico de barras por tipo de análisis")
    tipos_analisis = df_avg["Tipo"].unique().tolist()
    if tipos_analisis:
        tipo_barras = st.selectbox("Tipo de análisis para gráfico de barras", tipos_analisis, key="barras_tipo")
        df_barras = df_avg[df_avg["Tipo"] == tipo_barras]
        if not df_barras.empty:
            fig_barras, ax_barras = plt.subplots()
            ax_barras.bar(df_barras["Muestra"], df_barras["Valor"])
            ax_barras.set_ylabel(tipo_barras)
            ax_barras.set_xlabel("Muestra")
            ax_barras.set_title(f"{tipo_barras} por muestra")
            st.pyplot(fig_barras)

            # 📥 DESCARGA del gráfico de barras como PNG
            buffer_png = BytesIO()
            fig_barras.savefig(buffer_png, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                label="📷 Descargar gráfico de barras",
                data=buffer_png.getvalue(),
                file_name=f"grafico_barras_{tipo_barras}.png",
                mime="image/png"
            )

    # Descargar imagen del gráfico XY
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        label="📷 Descargar gráfico XY",
        data=buffer.getvalue(),
        file_name=f"grafico_{x_sel}_vs_{y_sel}.png",
        mime="image/png"
    )

    mostrar_sector_flotante(db, key_suffix="tab2")
