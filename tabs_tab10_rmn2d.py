# tabs_tab10_rmn2d.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import base64
import io
import numpy as np
import requests

def render_tab10(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Comparar mapas 2D RMN")

    st.session_state["current_tab"] = "Comparar mapas 2D RMN"

    # Cargar todas las muestras
    muestras = cargar_muestras(db)

    # Filtrar solo las que tienen espectros RMN 1H D
    muestras_filtradas = []
    espectros_dict = {}

    for muestra in muestras:
        espectros_ref = db.collection("muestras").document(muestra["nombre"]).collection("espectros")
        docs = espectros_ref.stream()
        espectros_rmn2d = []
        for doc in docs:
            data = doc.to_dict()
            if data.get("tipo") == "RMN 1H D":
                espectros_rmn2d.append({
                    "muestra": muestra["nombre"],
                    "nombre": data.get("nombre_archivo", "sin nombre"),
                    "url_archivo": data.get("url_archivo"),
                })
        if espectros_rmn2d:
            muestras_filtradas.append(muestra["nombre"])
            espectros_dict[muestra["nombre"]] = espectros_rmn2d

    if not muestras_filtradas:
        st.warning("No hay muestras con espectros RMN 1H D.")
        return

    # Selector múltiple de muestras
    muestras_sel = st.multiselect(
        "Seleccionar muestras con espectros RMN 1H D",
        options=muestras_filtradas
    )

    # Segundo selector de espectros para cada muestra elegida
    espectros_seleccionados = []
    if muestras_sel:
        for m in muestras_sel:
            archivos = espectros_dict[m]
            opciones = [a['nombre'] for a in archivos]
            sel = st.multiselect(
                f"Seleccionar espectros de {m}",
                options=opciones,
                key=f"sel_{m}"
            )
            for s in sel:
                espectros_seleccionados.append({"muestra": m, "nombre_archivo": s})

    # Mostrar resultado parcial
    if espectros_seleccionados:
        st.success(f"Espectros seleccionados: {[e['nombre_archivo'] for e in espectros_seleccionados]}")

    # campos minimalistas para ajustes de ejes y contornos
    if espectros_seleccionados:
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

        for sel in espectros_seleccionados:
            match = next(
                (e for e in espectros_dict[sel['muestra']] if e['nombre'] == sel['nombre_archivo']),
                None
            )
            if match:
                if "url_archivo" in match:
                    response = requests.get(match["url_archivo"])
                    if response.status_code == 200:
                        df = pd.read_csv(io.StringIO(response.text), sep="\t")
                    else:
                        st.error(f"No se pudo leer el archivo en {match['url_archivo']}")
                else:
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
