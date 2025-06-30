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

        # 4 campos minimalistas
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            level1 = st.number_input("Nivel contorno 1", value=float(z1.max()/2), format="%.3f")
        with c2:
            level2 = st.number_input("Nivel contorno 2", value=float(z2.max()/2), format="%.3f")
        with c3:
            y_max = st.number_input("Y máximo", value=1e-9, format="%.1e")
        with c4:
            y_min = st.number_input("Y mínimo", value=1e-13, format="%.1e")

        # redistribución proporcional
        y1 = y_min * (y_max / y_min) ** raw_y1
        y2 = y_min * (y_max / y_min) ** raw_y2

        # posiciones intermedias para etiquetas personalizadas
        custom_ticks = np.logspace(np.log10(y_max), np.log10(y_min), num=5)
        custom_texts = [f"{v:.1e}" for v in custom_ticks]

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

        # agregar scatter con etiquetas custom
        fig.add_trace(go.Scatter(
            x=[x1.max() + 0.1] * len(custom_ticks),  # poner fuera del área visible
            y=custom_ticks,
            mode="text",
            text=custom_texts[::-1],  # invertidas
            textposition="middle right",
            showlegend=False,
            textfont=dict(color="black", size=12)
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
                showticklabels=False,  # oculta labels nativos
                showgrid=False,
                zeroline=False,
                linecolor="black",
                range=[np.log10(y_min), np.log10(y_max)]  # ajusta según inputs
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
