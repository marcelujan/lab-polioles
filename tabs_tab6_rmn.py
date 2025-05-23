# --- Hoja 6: Análisis RMN (1H y 13C) con Plotly desde cero ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import base64
import os

def render_tab6(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Análisis RMN – 1H y 13C")

    # --- Cargar muestras y espectros ---
    muestras = cargar_muestras(db)
    if not muestras:
        st.warning("No hay muestras disponibles.")
        st.stop()

    espectros = []
    for m in muestras:
        nombre = m["nombre"]
        docs = db.collection("muestras").document(nombre).collection("espectros").stream()
        for i, doc in enumerate(docs):
            e = doc.to_dict()
            tipo = (e.get("tipo") or "").upper()
            if "RMN" in tipo:
                espectros.append({
                    "muestra": nombre,
                    "tipo": tipo,
                    "archivo": e.get("nombre_archivo", "sin nombre"),
                    "contenido": e.get("contenido"),
                    "mascaras": e.get("mascaras", []),
                    "id": f"{nombre}__{i}"
                })

    df_total = pd.DataFrame(espectros)
    if df_total.empty:
        st.warning("No hay espectros RMN disponibles.")
        st.stop()

    muestras_sel = st.multiselect("Seleccionar muestras", sorted(df_total["muestra"].unique()))
    df_filtrado = df_total[df_total["muestra"].isin(muestras_sel)]

    opciones = [
        f"{row['muestra']} – {row['archivo']}" for _, row in df_filtrado.iterrows()
    ]
    ids_map = dict(zip(opciones, df_filtrado["id"]))
    seleccion = st.multiselect("Seleccionar espectros", opciones)

    df_sel = df_filtrado[df_filtrado["id"].isin([ids_map.get(s) for s in seleccion])]

    st.markdown("## 🧪 RMN 1H")
    df_1h = df_sel[df_sel["tipo"] == "RMN 1H"]
    render_rmn_plot(df_1h, tipo="RMN 1H", key_sufijo="rmn1h")

    st.markdown("## 🧪 RMN 13C")
    df_13c = df_sel[df_sel["tipo"] == "RMN 13C"]
    render_rmn_plot(df_13c, tipo="RMN 13C", key_sufijo="rmn13c")

def render_rmn_plot(df, tipo="RMN 1H", key_sufijo="rmn1h"):
    if df.empty:
        st.info(f"No hay espectros disponibles para {tipo}.")
        return

    # --- Filtros estilo FTIR ---
    col1, col2, col3, col4 = st.columns(4)
    normalizar = col1.checkbox("Normalizar intensidad", key=f"norm_{key_sufijo}")
    mostrar_picos = col2.checkbox("Mostrar picos detectados", key=f"picos_{key_sufijo}")
    restar_espectro = col3.checkbox("Restar espectro de fondo", key=f"resta_{key_sufijo}")
    ajuste_y_manual = col4.checkbox("Ajuste manual eje Y", key=f"ajuste_y_{key_sufijo}")

    ajustes_y = {}
    if ajuste_y_manual:
        st.markdown("#### Ajustes verticales por espectro")
        for _, row in df.iterrows():
            clave = row["archivo"]
            ajustes_y[clave] = st.number_input(f"Ajuste Y para {clave}", value=0.0, step=0.1, key=f"ajuste_val_{clave}")
    else:
        for _, row in df.iterrows():
            ajustes_y[row["archivo"]] = 0.0

    seleccion_resta = None
    if restar_espectro:
        opciones_restar = [f"{row['muestra']} – {row['archivo']}" for _, row in df.iterrows()]
        seleccion_resta = st.selectbox("Seleccionar espectro a restar:", opciones_restar, key=f"sel_resta_{key_sufijo}")

    # --- Rango de visualización ---
    st.markdown("### Rango de visualización")
    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X mínimo", value=0.0, key=f"x_min_{key_sufijo}")
    x_max = colx2.number_input("X máximo", value=10.0 if tipo == "RMN 1H" else 220.0, key=f"x_max_{key_sufijo}")
    if ajuste_y_manual:
        y_min = coly1.number_input("Y mínimo", value=0.0, key=f"y_min_{key_sufijo}")
        y_max = coly2.number_input("Y máximo", value=100.0 if tipo == "RMN 1H" else 2.0, key=f"y_max_{key_sufijo}")
    else:
        y_min = None
        y_max = None

    # --- Decodificar espectro de fondo si aplica ---
    espectro_resta = None
    if restar_espectro and seleccion_resta:
        id_resta = seleccion_resta.split(" – ")[-1].strip()
        fila_resta = df[df["archivo"] == id_resta].iloc[0] if id_resta in set(df["archivo"]) else None
        if fila_resta is not None:
            try:
                espectro_resta = decodificar_csv_o_excel(fila_resta["contenido"], fila_resta["archivo"])
                if espectro_resta is not None:
                    espectro_resta.columns = ["x", "y"]
                    espectro_resta.dropna(inplace=True)
            except:
                espectro_resta = None
                espectro_resta.columns = ["x", "y"]
                espectro_resta.dropna(inplace=True)

    # --- Trazado ---
    fig = go.Figure()
    for _, row in df.iterrows():
        df_esp = decodificar_csv_o_excel(row["contenido"], row["archivo"])
        if df_esp is not None:
            col_x, col_y = df_esp.columns[:2]
            y_data = df_esp[col_y].copy()
            y_data = y_data + ajustes_y.get(row["archivo"], 0.0)
            if espectro_resta is not None:
                df_esp = df_esp.rename(columns={col_x: "x", col_y: "y"}).dropna()
                espectro_resta_interp = np.interp(df_esp["x"], espectro_resta["x"], espectro_resta["y"])
                y_data = df_esp["y"] - espectro_resta_interp
            if normalizar:
                y_data = y_data + ajustes_y.get(row["archivo"], 0.0)
            if normalizar:
                y_data = y_data / y_data.max() if y_data.max() != 0 else y_data
            x_vals = df_esp["x"] if "x" in df_esp.columns else df_esp[col_x]
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_data,
                mode='lines',
                name=row["archivo"]
            ))

    fig.update_layout(
        xaxis_title="[ppm]",
        yaxis_title="Intensidad",
        xaxis=dict(range=[x_max, x_min]),
        yaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None),
        template="simple_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

def decodificar_csv_o_excel(contenido_base64, archivo):
    try:
        contenido = BytesIO(base64.b64decode(contenido_base64))
        ext = os.path.splitext(archivo)[1].lower()
        if ext == ".xlsx":
            return pd.read_excel(contenido)
        else:
            for sep in [",", ";", "\t", " "]:
                contenido.seek(0)
                try:
                    df = pd.read_csv(contenido, sep=sep)
                    if df.shape[1] >= 2:
                        return df
                except:
                    continue
    except Exception as e:
        st.warning(f"Error al decodificar {archivo}: {e}")
    return None
