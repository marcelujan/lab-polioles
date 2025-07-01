import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
import numpy as np
import requests

def render_tab10(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Comparar mapas 2D RMN")

    st.session_state["current_tab"] = "Comparar mapas 2D RMN"

    muestras = cargar_muestras(db)
    espectros_dict = {}

    # construir diccionario con todos los espectros RMN 1H D
    for muestra in muestras:
        espectros_ref = db.collection("muestras").document(muestra["nombre"]).collection("espectros")
        docs = espectros_ref.stream()
        espectros_rmn2d = []
        for doc in docs:
            data = doc.to_dict()
            if data.get("tipo") == "RMN 1H D":
                espectros_rmn2d.append({
                    "nombre": data.get("nombre_archivo", "sin nombre"),
                    "url_archivo": data.get("url_archivo")
                })
        if espectros_rmn2d:
            espectros_dict[muestra["nombre"]] = espectros_rmn2d

    # juntar todos los nombres
    todos_espectros = []
    for espectros in espectros_dict.values():
        for e in espectros:
            todos_espectros.append(e['nombre'])

    if not todos_espectros:
        st.warning("No hay espectros RMN 1H D disponibles.")
        return

    # multiselect unificado
    espectros_seleccionados = st.multiselect(
        "Seleccionar espectros 2D a superponer",
        options=todos_espectros
    )

    if espectros_seleccionados:
        st.success(f"Espectros seleccionados: {espectros_seleccionados}")

        # ajustes globales de Y
        c1, c2 = st.columns(2)
        with c1:
            y_max = st.number_input("Y máximo", value=1e-9, format="%.1e")
        with c2:
            y_min = st.number_input("Y mínimo", value=1e-13, format="%.1e")

        # ajustes de nivel contorno por espectro
        niveles_contorno = {}
        st.subheader("Ajustes de niveles de contorno por espectro")
        for nombre in espectros_seleccionados:
            nivel = st.number_input(
                f"Nivel para {nombre}",
                min_value=0.01,
                max_value=1.0,
                value=0.10,
                format="%.3f",
                key=f"nivel_{nombre}"
            )
            niveles_contorno[nombre] = nivel

        # graficar
        fig = go.Figure()
        colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        color_idx = 0

        for nombre in espectros_seleccionados:
            match = next(
                (e for espectros in espectros_dict.values() for e in espectros if e['nombre'] == nombre),
                None
            )
            if not match:
                st.warning(f"No se encontró el espectro {nombre}")
                continue

            try:
                if "url_archivo" in match:
                    response = requests.get(match["url_archivo"])
                    if response.status_code == 200:
                        df = pd.read_csv(io.StringIO(response.text), sep="\t")
                    else:
                        st.error(f"No se pudo leer el archivo en {match['url_archivo']}")
                        continue
                else:
                    st.warning("No se encontró la URL del espectro para graficar.")
                    continue

                x = pd.to_numeric(df.columns[1:], errors="coerce")
                x = x[~pd.isna(x)]
                y_raw = df.iloc[:, 0].astype(float)
                z = df.iloc[:, 1:len(x)+1].values

                y_scaled = y_min * (y_max / y_min) ** y_raw
                nivel_contorno = niveles_contorno.get(nombre, 0.10)

                fig.add_trace(go.Contour(
                    x=x,
                    y=y_scaled,
                    z=z,
                    colorscale=[[0, colores[color_idx % len(colores)]], [1, colores[color_idx % len(colores)]]],
                    contours=dict(
                        coloring="lines",
                        start=nivel_contorno,
                        end=nivel_contorno,
                        size=0.1,
                        showlabels=False
                    ),
                    line=dict(width=1.5),
                    showscale=False,
                    name=nombre
                ))
                color_idx += 1

            except Exception as e:
                st.warning(f"Error graficando {nombre}: {e}")

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

        st.plotly_chart(fig, use_container_width=True)
