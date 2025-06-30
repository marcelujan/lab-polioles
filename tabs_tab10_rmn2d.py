# tabs_tab10_rmn2d.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import base64
import io
import numpy as np

def render_tab10(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Comparar mapas 2D RMN")

    st.session_state["current_tab"] = "Comparar mapas 2D RMN"

    muestras = cargar_muestras(db)
    nombres_muestras = [m["nombre"] for m in muestras]

    # multiselect de muestras
    muestras_sel = st.multiselect("Seleccionar muestras", nombres_muestras,key="multiselect_muestras_tab10")

    espectros_rmn_2d = []
    if muestras_sel:
        for nombre_muestra in muestras_sel:
            ref = db.collection("muestras").document(nombre_muestra).collection("espectros")
            docs = ref.stream()
            for doc in docs:
                data = doc.to_dict()
                if data.get("tipo") == "RMN 2D":
                    espectros_rmn_2d.append({
                        "muestra": nombre_muestra,
                        "nombre": data.get("nombre_archivo", "sin nombre"),
                        "contenido": data.get("contenido"),
                    })

    opciones = [f"{e['muestra']} – {e['nombre']}" for e in espectros_rmn_2d]
    seleccionados = st.multiselect("Elegir espectros 2D a superponer", opciones)

    # campos minimalistas para ajustes de ejes y contornos
    if seleccionados:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            nivel_contorno_1 = st.number_input("Nivel 1", value=0.1, format="%.3f")
        with c2:
            nivel_contorno_2 = st.number_input("Nivel 2", value=0.1, format="%.3f")
        with c3:
            y_max = st.number_input("Y máximo", value=1e-9, format="%.1e")
        with c4:
            y_min = st.number_input("Y mínimo", value=1e-13, format="%.1e")

        fig = go.Figure()

        colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown']  # para varios espectros
        color_idx = 0

        for sel in seleccionados:
            match = next((e for e in espectros_rmn_2d if f"{e['muestra']} – {e['nombre']}" == sel), None)
            if match:
                csv_bytes = base64.b64decode(match["contenido"])
                df = pd.read_csv(io.StringIO(csv_bytes.decode()), sep="\t")

                x = pd.to_numeric(df.columns[1:], errors="coerce")
                x = x[~pd.isna(x)]
                y_raw = df.iloc[:, 0].astype(float)
                z = df.iloc[:, 1:len(x)+1].values

                y_scaled = y_min * (y_max / y_min) ** y_raw

                fig.add_trace(go.Contour(
                    x=x,
                    y=y_scaled,
                    z=z,
                    colorscale=[[0, colores[color_idx % len(colores)]], [1, colores[color_idx % len(colores)]]],
                    contours=dict(
                        coloring="lines",
                        start=nivel_contorno_1 if color_idx == 0 else nivel_contorno_2,
                        end=nivel_contorno_1 if color_idx == 0 else nivel_contorno_2,
                        size=0.1,
                        showlabels=False
                    ),
                    line=dict(width=1.5),
                    showscale=False,
                    name=match["nombre"]
                ))
                color_idx += 1

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
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                linecolor="black",
                range=[np.log10(y_min), np.log10(y_max)]
            ),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="white",
                bordercolor="black"
            )
        )

        # agregar etiquetas de extremos (custom)
        fig.add_trace(go.Scatter(
            x=[x.max()],
            y=[y_min],
            text=[f"{y_min:.1e}"],
            mode="text",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x.max()],
            y=[y_max],
            text=[f"{y_max:.1e}"],
            mode="text",
            showlegend=False
        ))

        st.plotly_chart(fig, use_container_width=True)
