# tabs_tab6_rmn.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from io import BytesIO
from datetime import datetime
import os
import base64
import zipfile
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt #solo para pruebas

# --- Configuraciones globales ---
GRUPOS_FUNCIONALES = ["Formiato", "Cloroformo", "C=C olefínicos", "Glicerol medio", "Glicerol extremos", "Metil-Éster", "Eter", "Ester", "Ácido carboxílico", "OH", "Epóxido", "C=C", "Alfa-C=O", "Alfa-C=C", "Vecino a alfa-carbonilo", "Alfa-epóxido", "CH2", "CH3"]

def obtener_ids_espectros(nombre):
    return [doc.id for doc in firestore.Client().collection("muestras").document(nombre).collection("espectros").list_documents()]

def render_tab6(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Análisis RMN")
    st.session_state["current_tab"] = "Análisis RMN"

    # --- Selección de muestras ---
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    nombres_muestras = [m["nombre"] for m in muestras]
    muestras_sel = st.multiselect("Seleccionar muestras", nombres_muestras, default=[])
    if not muestras_sel:
        st.warning("Seleccioná al menos una muestra.")
        st.stop()

    # --- Selección de espectros por muestra ---
    espectros_sel = {}
    for muestra in muestras_sel:
        espectros_ref = db.collection("muestras").document(muestra).collection("espectros").stream()
        espectros = [doc.to_dict() for doc in espectros_ref]
        nombres_archivos = [e.get("nombre_archivo", "sin nombre") for e in espectros]
        espectros_sel[muestra] = st.multiselect(f"Espectros para {muestra}", nombres_archivos, default=nombres_archivos[:1])

    st.divider()

    # ==============================
    # === SECCIÓN RMN 1H ===========
    # ==============================
    st.subheader("🔬 RMN 1H")

    # --- Máscara D/T2 ---
    activar_mascara = st.checkbox("Máscara D/T2", value=False)
    usar_mascara = {}
    if activar_mascara:
        st.markdown("Activar sombreado individual por muestra:")
        cols = st.columns(len(muestras_sel))
        for idx, muestra in enumerate(muestras_sel):
            with cols[idx]:
                usar_mascara[muestra] = st.checkbox(muestra, key=f"chk_mask_{muestra}", value=False)

    # --- Cálculo D/T2 ---
    activar_calculo_dt2 = st.checkbox("Cálculo D/T2", value=False)
    if activar_calculo_dt2:
        st.info("Aquí irá la tabla editable de Cálculo D/T2.")

    # --- Señales pico bibliografía ---
    activar_picos = st.checkbox("Señales Pico Bibliografía", value=False)
    if activar_picos:
        editar_tabla_biblio = st.checkbox("Editar Tabla Bibliográfica", value=False)
        if editar_tabla_biblio:
            st.data_editor(
                pd.DataFrame(columns=["Grupo funcional", "X min", "δ pico", "X max", "Tipo de muestra", "Observaciones"]),
                use_container_width=True,
                num_rows="dynamic"
            )

    # --- Control de ejes del gráfico ---
    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X mínimo", value=0.0)
    x_max = colx2.number_input("X máximo", value=10.0)
    y_min = coly1.number_input("Y mínimo", value=0.0)
    y_max = coly2.number_input("Y máximo", value=100.0)

    # --- Gráfico combinado ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Espectros RMN 1H")
    ax.set_xlabel("[ppm]")
    ax.set_ylabel("Señal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color="black", linewidth=0.7)
    st.pyplot(fig)

    # ==============================
    # === Cálculo de señales =======
    # ==============================
    activar_calculo_senales = st.checkbox("Cálculo de señales", value=False)
    if activar_calculo_senales:
        st.info("Aquí irá la tabla editable de Cálculo de señales.")

    # ==============================
    # === RMN 13C ==================
    # ==============================
    st.subheader("🧪 RMN 13C")
    st.info("Aquí se graficarán los espectros RMN 13C seleccionados.")

    # ==============================
    # === Imágenes RMN ============
    # ==============================
    st.subheader("🖼️ Espectros imagen")
    st.info("Aquí se mostrarán las imágenes de espectros RMN.")