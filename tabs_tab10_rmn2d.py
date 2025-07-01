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

    muestras = cargar_muestras(db)
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

    muestras_sel = st.multiselect(
        "Seleccionar muestras con espectros RMN 1H D",
        options=muestras_filtradas
    )

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

    if espectros_seleccionados:
        st.success(f"Espectros seleccionados: {[e['nombre_archivo'] for e in espectros_seleccionados]}")

        # campos para ajustes globales de Y
        c1, c2 = st.columns(2)
        with c1:
            y_max = st.number_input("Y máximo", value=1e-9, format="%.1e")
        with c2:
            y_min = st.number_input("Y mínimo", value=1e-13, format="%.1e")

        # contornos por espectro
        niveles_contorno = {}

        if espectros_seleccionados:
            st.subheader("Ajustes de niveles de contorno para cada espectro")
            for sel in espectros_seleccionados:
                minimo = 0.01  # podés ajustar si querés
                maximo = 1.0
                nivel = st.number_input(
                    f"Nivel para {sel['nombre_archivo']}",
                    min_value=minimo,
                    max_value=maximo,
                    value=0.10,
                    format="%.3f",
                    key=f"nivel_{sel['nombre_archivo']}"
                )
                niveles_contorno[sel['nombre_archivo']] = nivel


        # graficar
        fig = go.Figure()
        colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        color_idx = 0

        for sel in espectros_seleccionados:
            match = next(
                (e for e in espectros_dict[sel['muestra']] if e['nombre'] == sel['nombre_archivo']),
                None
            )
            if not match:
                st.warning(f"No se encontró el espectro {sel['nombre_archivo']} en {sel['muestra']}")
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

                fig.add_trace(go.Contour(
                    x=x,
                    y=y_scaled,
                    z=z,
                    colorscale=[[0, colores[color_idx % len(colores)]], [1, colores[color_idx % len(colores)]]],
                    contours=dict(
                        coloring="lines",
                        nivel_contorno = niveles_contorno.get(match["nombre_archivo"], 0.10)
                        start=nivel_contorno,
                        end=nivel_contorno,
                        size=0.1,
                        showlabels=False
                    ),
                    line=dict(width=1.5),
                    showscale=False,
                    name=match.get("nombre_archivo", "sin_nombre")
                ))
                color_idx += 1

            except Exception as e:
                st.warning(f"Error graficando {match.get('nombre_archivo', 'desconocido')}: {e}")



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

