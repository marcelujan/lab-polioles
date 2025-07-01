import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
import numpy as np
import requests

def render_tab10(db, cargar_muestras, mostrar_sector_flotante):
    st.session_state["current_tab"] = "Comparar mapas 2D RMN"
    muestras = cargar_muestras(db)
    espectros_dict = {}

    # recolectar espectros RMN 1H D
    for muestra in muestras:
        espectros_ref = db.collection("muestras").document(muestra["nombre"]).collection("espectros")
        docs = espectros_ref.stream()
        espectros_rmn_d = []
        for doc in docs:
            data = doc.to_dict()
            if data.get("tipo") == "RMN 1H D":
                espectros_rmn_d.append({
                    "nombre": data.get("nombre_archivo", "sin nombre"),
                    "url_archivo": data.get("url_archivo")
                })
        if espectros_rmn_d:
            espectros_dict[muestra["nombre"]] = espectros_rmn_d

    # primer selector de muestras
    muestras_filtradas = list(espectros_dict.keys())
    muestras_sel = st.multiselect(
        "Seleccionar muestras",
        options=muestras_filtradas,
        key="multiselect_muestras_tab10"
    )

    # segundo selector unificado de espectros, SOLO de las muestras seleccionadas
    espectros_opciones = []
    if muestras_sel:
        for m in muestras_sel:
            espectros_opciones.extend([e["nombre"] for e in espectros_dict[m]])

    espectros_seleccionados = st.multiselect(
        "Seleccionar espectros",
        options=espectros_opciones,
        key="multiselect_espectros_tab10"
    )

    if espectros_seleccionados:
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            x_min = st.number_input("X mín", value=0.0, format="%.2f")
        with c2:
            x_max = st.number_input("X máx", value=9.0, format="%.2f")
        with c3:
            y_min_axis = st.number_input("Y mín", value=1e-13, format="%.1e")
        with c4:
            y_max_axis = st.number_input("Y máx", value=1e-9, format="%.1e")
        with c5:
            y_min_scale = st.number_input("Y mín reescalado", value=1e-13, format="%.1e")
        with c6:
            y_max_scale = st.number_input("Y máx reescalado", value=1e-9, format="%.1e")


        # niveles de contorno por espectro
        st.markdown("**Curva de nivel**", unsafe_allow_html=True)
        niveles_contorno = {}
        cols = st.columns(5)

        for idx, nombre in enumerate(espectros_seleccionados):
            col = cols[idx % 5]
            muestra_base = nombre.split("_RMN")[0]  # corta el nombre antes de _RMN
            with col:
                nivel = st.number_input(
                    f"{muestra_base}",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.10,
                    format="%.2f",
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

                y_scaled = y_min_scale * (y_max_scale / y_min_scale) ** y_raw
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
            title="",
            xaxis_title="F2 (ppm)",
            yaxis_title="F1 (s⁻¹ o m²/s)",
            height=700,
            xaxis=dict(
                autorange=False,
                range=[x_max, x_min],  # mayor a menor para sentido RMN
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
                range=[np.log10(y_min_axis), np.log10(y_max_axis)]  # ojo, log10 porque el eje es log
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
