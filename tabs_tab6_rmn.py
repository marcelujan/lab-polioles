# --- Hoja 6: Análisis RMN (1H y 13C) con Plotly desde cero ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import base64
import os

st.title("Análisis RMN – 1H y 13C")

# ================================
# === Funciones auxiliares de carga ===
# ================================

def cargar_muestras(db):
    try:
        docs = db.collection("muestras").stream()
        return [{"nombre": doc.id} for doc in docs]
    except:
        st.warning("⚠️ Error al cargar muestras (simulado sin Firebase).")
        return []

def cargar_espectros_rmn_unificados(muestras, db):
    espectros = []
    for m in muestras:
        nombre = m["nombre"]
        try:
            subcoleccion = db.collection("muestras").document(nombre).collection("espectros").stream()
            for i, doc in enumerate(subcoleccion):
                datos = doc.to_dict()
                tipo = (datos.get("tipo") or "").upper()
                if "RMN" in tipo:
                    espectros.append({
                        "muestra": nombre,
                        "tipo": tipo,
                        "archivo": datos.get("nombre_archivo", "sin nombre"),
                        "contenido": datos.get("contenido"),
                        "fecha": datos.get("fecha"),
                        "mascaras": datos.get("mascaras", []),
                        "es_imagen": datos.get("es_imagen", False),
                        "id": f"{nombre}__{i}"
                    })
        except:
            st.warning(f"⚠️ No se pudo acceder a espectros de la muestra {nombre}.")
    return espectros

# ================================
# === 1. Cargar muestras y espectros ===
# ================================

muestras = cargar_muestras(db)
if not muestras:
    st.warning("No hay muestras disponibles.")
    st.stop()

# Obtener espectros unificados
espectros_rmn = cargar_espectros_rmn_unificados(muestras, db)  # asume implementación existente

# Crear dataframe
df_total = pd.DataFrame(espectros_rmn)
if df_total.empty:
    st.warning("No hay espectros RMN disponibles.")
    st.stop()

# ================================
# === 2. Selector de muestras y espectros ===
# ================================

muestras_disp = sorted(df_total["muestra"].unique())
muestras_sel = st.multiselect("Seleccionar muestras", muestras_disp)

df_filtrado = df_total[df_total["muestra"].isin(muestras_sel)]

ids_info = [
    {"id": row["id"], "nombre": f"{row['muestra']} – {row['archivo']}"}
    for _, row in df_filtrado.iterrows()
]

ids_legibles = {e["id"]: e["nombre"] for e in ids_info}
ids_disponibles = list(ids_legibles.keys())

ids_sel = st.multiselect(
    "Seleccionar espectros a visualizar:",
    options=ids_disponibles,
    format_func=lambda i: ids_legibles.get(i, i)
)

df_sel = df_filtrado[df_filtrado["id"].isin(ids_sel)]

# ================================
# === 3. Sección RMN 1H ===
# ================================

st.markdown("## 🧪 RMN 1H")
df_rmn1h = df_sel[df_sel["tipo"] == "RMN 1H"]
render_rmn_tipo(df_rmn1h, tipo="RMN 1H", key_sufijo="rmn1h")

# ================================
# === 4. Sección RMN 13C ===
# ================================

st.markdown("## 🧪 RMN 13C")
df_rmn13c = df_sel[df_sel["tipo"] == "RMN 13C"]
render_rmn_tipo(df_rmn13c, tipo="RMN 13C", key_sufijo="rmn13c")

# ================================
# === Función de visualización general ===
# ================================

def render_rmn_tipo(df, tipo="RMN 1H", key_sufijo="rmn1h"):
    if df.empty:
        st.info(f"No hay espectros disponibles para {tipo}.")
        return

    with st.expander("⚙️ Filtros y opciones", expanded=True):
        col_f1, col_f2 = st.columns(2)
        normalizar = col_f1.checkbox("Normalizar intensidad", value=False, key=f"chk_norm_{key_sufijo}")
        restar_espectro = col_f2.checkbox("Restar espectro de fondo", value=False, key=f"chk_restar_{key_sufijo}")
        mostrar_picos = col_f1.checkbox("Mostrar picos detectados", value=False, key=f"chk_picos_{key_sufijo}")
        ajuste_y_manual = col_f2.checkbox("Ajuste manual eje Y", value=False, key=f"chk_y_manual_{key_sufijo}")

        colx1, colx2, coly1, coly2 = st.columns(4)
        x_min = colx1.number_input("X mínimo", value=0.0, key=f"x_min_{key_sufijo}")
        x_max = colx2.number_input("X máximo", value=10.0 if tipo == "RMN 1H" else 220.0, key=f"x_max_{key_sufijo}")
        y_min = coly1.number_input("Y mínimo", value=0.0, key=f"y_min_{key_sufijo}") if ajuste_y_manual else None
        y_max = coly2.number_input("Y máximo", value=100.0 if tipo == "RMN 1H" else 2.0, key=f"y_max_{key_sufijo}") if ajuste_y_manual else None

    fig = go.Figure()
    for _, row in df.iterrows():
        df_esp = decodificar_csv_o_excel(row["contenido"], row["archivo"])
        if df_esp is not None:
            col_x, col_y = df_esp.columns[:2]
            y_data = df_esp[col_y] / df_esp[col_y].max() if normalizar else df_esp[col_y]
            fig.add_trace(go.Scatter(
                x=df_esp[col_x],
                y=y_data,
                mode='lines',
                name=f"{row['muestra']} – {row['archivo']}"
            ))

    fig.update_layout(
        xaxis_title="[ppm]",
        yaxis_title="Intensidad",
        xaxis=dict(range=[x_min, x_max], autorange="reversed"),
        yaxis=dict(range=[y_min, y_max] if ajuste_y_manual else None),
        template="simple_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# ================================
# === Función decodificadora ===
# ================================

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
