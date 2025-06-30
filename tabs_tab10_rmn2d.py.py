# tabs_tab10_rmn2d.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_tab10(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Comparar mapas 2D RMN")

    st.session_state["current_tab"] = "Comparar mapas 2D RMN"

    # subir dos CSV para comparar
    st.subheader("Subir espectros transformados (CSV)")
    col1, col2 = st.columns(2)
    with col1:
        archivo_1 = st.file_uploader("Espectro 1 (CSV)", type="csv", key="csv1")
    with col2:
        archivo_2 = st.file_uploader("Espectro 2 (CSV)", type="csv", key="csv2")

    if archivo_1 and archivo_2:
        # leer ambos CSV con separador tabulador
        df1 = pd.read_csv(archivo_1, sep="\t")
        df2 = pd.read_csv(archivo_2, sep="\t")

        # preparar ejes
        x1 = df1.columns[1:].astype(float)
        y1 = df1.iloc[:, 0].astype(float)
        z1 = df1.iloc[:, 1:].values

        x2 = df2.columns[1:].astype(float)
        y2 = df2.iloc[:, 0].astype(float)
        z2 = df2.iloc[:, 1:].values

        # graficar con plotly
        fig = go.Figure()

        fig.add_trace(go.Contour(
            x=x1, y=y1, z=z1,
            colorscale="Reds",
            contours=dict(showlabels=True),
            name="Espectro 1",
            opacity=0.7
        ))

        fig.add_trace(go.Contour(
            x=x2, y=y2, z=z2,
            colorscale="Blues",
            contours=dict(showlabels=True),
            name="Espectro 2",
            opacity=0.7
        ))

        fig.update_layout(
            title="Superposición de mapas 2D",
            xaxis_title="F2 (ppm)",
            yaxis_title="F1 (s⁻¹ o m²/s)",
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)

    # muestra flotante
    mostrar_sector_flotante(db, key_suffix="tab10")

