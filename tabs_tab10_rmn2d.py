# tabs_tab10_rmn2d.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def render_tab10(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Comparar mapas 2D RMN")

    st.session_state["current_tab"] = "Comparar mapas 2D RMN"

    st.subheader("Subir espectros transformados (CSV)")
    col1, col2 = st.columns(2)
    with col1:
        archivo_1 = st.file_uploader("Espectro 1 (CSV)", type="csv", key="csv1")
    with col2:
        archivo_2 = st.file_uploader("Espectro 2 (CSV)", type="csv", key="csv2")

    if archivo_1 and archivo_2:
        df1 = pd.read_csv(archivo_1, sep="\t")
        df2 = pd.read_csv(archivo_2, sep="\t")

        # preparar ejes
        x1 = pd.to_numeric(df1.columns[1:], errors="coerce")
        x1 = x1[~pd.isna(x1)]
        raw_y1 = df1.iloc[:, 0].astype(float)
        z1 = df1.iloc[:, 1:len(x1)+1].values

        x2 = pd.to_numeric(df2.columns[1:], errors="coerce")
        x2 = x2[~pd.isna(x2)]
        raw_y2 = df2.iloc[:, 0].astype(float)
        z2 = df2.iloc[:, 1:len(x2)+1].values

        # sliders para nivel de contorno
        level1 = st.slider("Nivel de contorno Espectro 1", 
                           min_value=float(z1.min()), 
                           max_value=float(z1.max()), 
                           value=float(z1.max()/2))

        level2 = st.slider("Nivel de contorno Espectro 2", 
                           min_value=float(z2.min()), 
                           max_value=float(z2.max()), 
                           value=float(z2.max()/2))

        # sliders para rango Y
        y_max = st.number_input(
            "Y máximo (por defecto 1e-9)", 
            value=1e-9, 
            format="%.1e"
        )
        y_min = st.number_input(
            "Y mínimo (por defecto 1e-13)", 
            value=1e-13, 
            format="%.1e"
        )

        # redistribución proporcional logarítmica
        y1 = y_min * (y_max / y_min) ** raw_y1
        y2 = y_min * (y_max / y_min) ** raw_y2

        fig = go.Figure()

        fig.add_trace(go.Contour(
            x=x1, y=y1, z=z1,
            colorscale=[[0, 'red'], [1, 'red']],  
            contours=dict(
                coloring="lines",
                start=level1,
                end=level1,
                size=0.1,
                showlabels=False
            ),
            line=dict(width=1.5),
            showscale=False,
            name="Muestra 1"
        ))

        fig.add_trace(go.Contour(
            x=x2, y=y2, z=z2,
            colorscale=[[0, 'blue'], [1, 'blue']],
            contours=dict(
                coloring="lines",
                start=level2,
                end=level2,
                size=0.1,
                showlabels=False
            ),
            line=dict(width=1.5),
            showscale=False,
            name="Muestra 2"
        ))

        fig.update_layout(
            title="Superposición de mapas 2D",
            xaxis_title="F2 (ppm)",
            yaxis_title="F1 (s⁻¹ o m²/s)",
            height=700,
            xaxis=dict(
                autorange="reversed",
                showgrid=False,
                zeroline=False,
                linecolor="black"
            ),
            yaxis=dict(
                type="log",
                autorange="reversed",
                exponentformat="e",
                showgrid=False,
                zeroline=False,
                linecolor="black"
            ),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="white",
                bordercolor="black"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
