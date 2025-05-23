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
    y_min = coly1.number_input("Y mínimo", value=0.0, key=f"y_min_{key_sufijo}")
    y_max = coly2.number_input("Y máximo", value=100.0 if tipo == "RMN 1H" else 2.0, key=f"y_max_{key_sufijo}")

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

    # --- Parámetros de picos ---
    if mostrar_picos:
        colp1, colp2 = st.columns(2)
        altura_min = colp1.number_input("Altura mínima", value=0.05, step=0.01, key=f"altura_min_{key_sufijo}")
        distancia_min = colp2.number_input("Distancia mínima entre picos", value=5, step=1, key=f"distancia_min_{key_sufijo}")

# --- Tabla de Cálculo D/T2 ---
    if tipo == "RMN 1H":
        columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                         "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]

        filas_guardadas = []
        for _, row in df.iterrows():
            muestra = row["muestra"]
            archivo = row["archivo"]
            doc = db.collection("muestras").document(muestra).collection("dt2").document("datos")
            data = doc.get().to_dict()
            if data and "filas" in data:
                filas_guardadas.extend([f for f in data["filas"] if f.get("Archivo") == archivo])

        df_dt2 = pd.DataFrame(filas_guardadas)
        for col in columnas_dt2:
            if col not in df_dt2.columns:
                df_dt2[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
        df_dt2 = df_dt2[columnas_dt2]

        st.markdown("### Cálculo D/T2")
        with st.form(f"form_dt2_{key_sufijo}"):
            df_dt2_edit = st.data_editor(
                df_dt2,
                column_config={
                    "Grupo funcional": st.column_config.SelectboxColumn(options=["CH3", "CH2", "OH", "Aromático", "Epóxido", "Ester"]),
                    "δ pico": st.column_config.NumberColumn(format="%.2f"),
                    "X min": st.column_config.NumberColumn(format="%.2f"),
                    "X max": st.column_config.NumberColumn(format="%.2f"),
                    "Área": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "D": st.column_config.NumberColumn(format="%.2e"),
                    "T2": st.column_config.NumberColumn(format="%.3f"),
                    "Xas min": st.column_config.NumberColumn(format="%.2f"),
                    "Xas max": st.column_config.NumberColumn(format="%.2f"),
                    "Has": st.column_config.NumberColumn(format="%.2f"),
                    "Área as": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "H": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "Observaciones": st.column_config.TextColumn(),
                    "Archivo": st.column_config.TextColumn(disabled=True),
                    "Muestra": st.column_config.TextColumn(disabled=True),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",
                key=f"tabla_dt2_{key_sufijo}"
            )
            recalcular = st.form_submit_button("🔴 Recalcular 'Área', 'Área as' y 'H'")

        if recalcular:
            for i, row in df_dt2_edit.iterrows():
                try:
                    muestra = row["Muestra"]
                    archivo = row["Archivo"]
                    x_min = float(row["X min"])
                    x_max = float(row["X max"])
                    xas_min = float(row["Xas min"]) if row["Xas min"] not in [None, ""] else None
                    xas_max = float(row["Xas max"]) if row["Xas max"] not in [None, ""] else None
                    has = float(row["Has"]) if row["Has"] not in [None, ""] else None

                    espectros = db.collection("muestras").document(muestra).collection("espectros").stream()
                    espectro = next((e.to_dict() for e in espectros if e.to_dict().get("nombre_archivo") == archivo), None)
                    if not espectro:
                        continue

                    contenido = BytesIO(base64.b64decode(espectro["contenido"]))
                    extension = os.path.splitext(archivo)[1].lower()
                    if extension == ".xlsx":
                        df_esp = pd.read_excel(contenido)
                    else:
                        for sep in [",", ";", "	", " "]:
                            contenido.seek(0)
                            try:
                                df_esp = pd.read_csv(contenido, sep=sep)
                                if df_esp.shape[1] >= 2:
                                    break
                            except:
                                continue
                        else:
                            continue

                    col_x, col_y = df_esp.columns[:2]
                    df_esp[col_x] = pd.to_numeric(df_esp[col_x], errors="coerce")
                    df_esp[col_y] = pd.to_numeric(df_esp[col_y], errors="coerce")
                    df_esp = df_esp.dropna()

                    df_main = df_esp[(df_esp[col_x] >= min(x_min, x_max)) & (df_esp[col_x] <= max(x_min, x_max))]
                    area = np.trapz(df_main[col_y], df_main[col_x]) if not df_main.empty else None
                    df_dt2_edit.at[i, "Área"] = round(area, 2) if area else None

                    if xas_min is not None and xas_max is not None:
                        df_as = df_esp[(df_esp[col_x] >= min(xas_min, xas_max)) & (df_esp[col_x] <= max(xas_min, xas_max))]
                        area_as = np.trapz(df_as[col_y], df_as[col_x]) if not df_as.empty else None
                        df_dt2_edit.at[i, "Área as"] = round(area_as, 2) if area_as else None

                        if area and area_as and has and area_as != 0:
                            h_calc = (area * has) / area_as
                            df_dt2_edit.at[i, "H"] = round(h_calc, 2)

                except Exception as e:
                    st.warning(f"⚠️ Error en fila {i}: {e}")

            filas_actualizadas = df_dt2_edit.to_dict(orient="records")
            for muestra in df_dt2_edit["Muestra"].unique():
                filas_m = [f for f in filas_actualizadas if f["Muestra"] == muestra]
                doc = db.collection("muestras").document(muestra).collection("dt2").document("datos")
                doc.set({"filas": filas_m})

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

            # Detección de picos
            if mostrar_picos:
                from scipy.signal import find_peaks
                try:
                    peaks, _ = find_peaks(y_data, height=altura_min, distance=distancia_min)
                    for p in peaks:
                        fig.add_trace(go.Scatter(
                            x=[x_vals.iloc[p]],
                            y=[y_data.iloc[p]],
                            mode="markers+text",
                            marker=dict(color="black", size=6),
                            text=[f"{x_vals.iloc[p]:.2f}"],
                            textposition="top center",
                            showlegend=False
                        ))
                except:
                    st.warning(f"⚠️ No se pudieron detectar picos en {row['archivo']}.")

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
