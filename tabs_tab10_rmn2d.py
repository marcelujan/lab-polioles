import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import base64

def render_tab10(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Comparar mapas 2D RMN")

    st.session_state["current_tab"] = "Comparar mapas 2D RMN"

    muestras = cargar_muestras(db)
    nombres_muestras = [m["nombre"] for m in muestras]

    st.subheader("Comparar espectros 2D desde la base de datos")

    nombre_muestra = st.selectbox("Seleccionar muestra", nombres_muestras)
    espectros_rmn_2d = []

    # buscar espectros guardados como RMN 1H D
    ref = db.collection("muestras").document(nombre_muestra).collection("espectros")
    docs = ref.stream()
    for doc in docs:
        data = doc.to_dict()
        if data.get("tipo") == "RMN 1H D":
            espectros_rmn_2d.append({
                "nombre": data.get("nombre_archivo", "sin nombre"),
                "contenido": data.get("contenido"),
            })

    nombres_archivos = [e["nombre"] for e in espectros_rmn_2d]

    seleccionados = st.multiselect("Elegir espectros 2D a superponer", nombres_archivos)

    if seleccionados:
        # 4 campos minimalistas
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            level1 = st.number_input("Nivel 1", value=0.1, format="%.3f")
        with c2:
            level2 = st.number_input("Nivel 2", value=0.1, format="%.3f")
        with c3:
            y_max = st.number_input("Y máximo", value=1e-9, format="%.1e")
        with c4:
            y_min = st.number_input("Y mínimo", value=1e-13, format="%.1e")

        fig = go.Figure()

        colores = ["red", "blue", "green", "orange", "purple", "black"]
        
        for idx, nombre_archivo in enumerate(seleccionados):
            espectro = next(e for e in espectros_rmn_2d if e["nombre"] == nombre_archivo)
            csv_bytes = base64.b64decode(espectro["contenido"])
            csv_str = csv_bytes.decode("utf-8")
            from io import StringIO
            df = pd.read_csv(StringIO(csv_str), sep="\t")

            x = pd.to_numeric(df.columns[1:], errors="coerce")
            x = x[~pd.isna(x)]
            raw_y = df.iloc[:, 0].astype(float)
            z = df.iloc[:, 1:len(x)+1].values

            # remapear Y
            y = y_min * (y_max / y_min) ** raw_y

            nivel = level1 if idx == 0 else level2 if idx == 1 else 0.1  # para más de 2 espectros
            color = colores[idx % len(colores)]

            fig.add_trace(go.Contour(
                x=x, y=y, z=z,
                colorscale=[[0, color], [1, color]],
                contours=dict(
                    coloring="lines",
                    start=nivel,
                    end=nivel,
                    size=0.1,
                    showlabels=False
                ),
                line=dict(width=1.5),
                showscale=False,
                name=nombre_archivo
            ))

        fig.update_layout(
            title="Comparación de mapas 2D",
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

        st.plotly_chart(fig, use_container_width=True)
