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

    # ajustes globales Y
    if espectros_seleccionados:
        c1, c2 = st.columns(2)
        with c1:
            y_min = st.number_input("Y mínimo (global)", value=1e-13, format="%.1e")
        with c2:
            y_max = st.number_input("Y máximo (global)", value=1e-9, format="%.1e")

        # ajustes de nivel para cada espectro
        niveles = {}
        st.markdown("### Configurar niveles de contorno para cada espectro")
        for espectro in espectros_seleccionados:
            try:
                match = next((e for e in espectros_dict[espectro['muestra']] if e['nombre'] == espectro['nombre_archivo']), None)
                if match and "url_archivo" in match:
                    response = requests.get(match["url_archivo"])
                    if response.status_code == 200:
                        df = pd.read_csv(io.StringIO(response.text), sep="\t")
                        valores = df.iloc[:, 1:].values.flatten()
                        min_val, max_val = np.nanmin(valores), np.nanmax(valores)
                        st.write(f"**{espectro['nombre_archivo']}** (min: {min_val:.3f}, max: {max_val:.3f})")
                        nivel = st.number_input(
                            f"Nivel para {espectro['nombre_archivo']}",
                            value=0.1,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            key=f"nivel_{espectro['nombre_archivo']}"
                        )
                        niveles[espectro['nombre_archivo']] = nivel
                    else:
                        st.warning(f"No se pudo leer el archivo {match['nombre_archivo']}")
            except Exception as e:
                st.warning(f"Error leyendo espectro: {e}")

        # graficar
        fig = go.Figure()
        colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        color_idx = 0

        for espectro in espectros_seleccionados:
            match = next((e for e in espectros_dict[espectro['muestra']] if e['nombre'] == espectro['nombre_archivo']), None)
            if match:
                try:
                    response = requests.get(match["url_archivo"])
                    if response.status_code == 200:
                        df = pd.read_csv(io.StringIO(response.text), sep="\t")
                        x = pd.to_numeric(df.columns[1:], errors="coerce")
                        x = x[~pd.isna(x)]
                        y_raw = df.iloc[:, 0].astype(float)
                        z = df.iloc[:, 1:len(x)+1].values
                        y_scaled = y_min * (y_max / y_min) ** y_raw

                        nivel_este = niveles.get(espectro['nombre_archivo'], 0.1)

                        fig.add_trace(go.Contour(
                            x=x,
                            y=y_scaled,
                            z=z,
                            colorscale=[[0, colores[color_idx % len(colores)]], [1, colores[color_idx % len(colores)]]],
                            contours=dict(
                                coloring="lines",
                                start=nivel_este,
                                end=nivel_este,
                                size=0.1,
                                showlabels=False
                            ),
                            line=dict(width=1.5),
                            showscale=False,
                            name=match["nombre_archivo"]
                        ))
                        color_idx += 1
                    else:
                        st.warning(f"No se pudo cargar el archivo {match['nombre_archivo']}")
                except Exception as e:
                    nombre_archivo = match.get("nombre_archivo", "desconocido") if match else "desconocido"
                    st.warning(f"Error graficando {nombre_archivo}: {e}")


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

