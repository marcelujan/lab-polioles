# tabs_tab5_oh.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
import os
from matplotlib import pyplot as plt

def render_tab5(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Índice OH espectroscópico")
    st.session_state["current_tab"] = "Índice OH espectroscópico"  
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas para analizar.")
        st.stop()

    espectros_info = []
     # Recorrer todas las muestras para identificar espectros relevantes   
    for m in muestras:
        espectros = m.get("espectros", [])
        for e in espectros:
            tipo = e.get("tipo", "")
            if tipo not in ["FTIR-Acetato", "FTIR-Cloroformo"]:# Solo calcular índice OH para estos tipos de espectro
                continue

            contenido = e.get("contenido")
            es_imagen = e.get("es_imagen", False)
            valor_y_extraido = None

            # Intentar extraer el valor Y cercano a 3548 o 3611 del archivo numérico
            if contenido and not es_imagen:
                try:
                    extension = e.get("nombre_archivo", "").split(".")[-1].lower()
                    binario = BytesIO(base64.b64decode(contenido))
                    if extension == "xlsx":
                        df = pd.read_excel(binario, header=None)
                    else:
                        sep_try = [",", ";", "\t", " "]
                        for sep in sep_try:
                            binario.seek(0)
                            try:
                                df = pd.read_csv(binario, sep=sep, engine="python", header=None)
                                if df.shape[1] >= 2:
                                    break
                            except:
                                continue
                        else:
                            df = None

                    if df is not None and df.shape[1] >= 2:
                        df = df.dropna()
                        x_valores = pd.to_numeric(df.iloc[:,0], errors='coerce')
                        y_valores = pd.to_numeric(df.iloc[:,1], errors='coerce')
                        df_limpio = pd.DataFrame({"X": x_valores, "Y": y_valores}).dropna()
                        objetivo_x = 3548 if tipo == "FTIR-Acetato" else 3611
                        idx_cercano = (df_limpio["X"] - objetivo_x).abs().idxmin()
                        valor_y_extraido = df_limpio.loc[idx_cercano, "Y"]
                except:
                    valor_y_extraido = None

            # Agregar a la tabla intermedia
            espectros_info.append({
                "Muestra": m["nombre"],
                "Tipo espectro": tipo,
                "Fecha espectro": e.get("fecha", ""),
                "Señal": valor_y_extraido,
                "Señal manual 3548": e.get("senal_3548", None),
                "Señal manual 3611": e.get("senal_3611", None),
                "Peso muestra [g]": e.get("peso_muestra", None)
            })

    df_muestras = pd.DataFrame(espectros_info)
    if df_muestras.empty:
        st.warning("No se encontraron espectros válidos para calcular Índice OH.")
        st.stop()

    # Crear columna 'Señal solvente' unificando los datos manuales ingresados
    def obtener_senal_solvente(row):
        return row["Señal manual 3548"] if row["Tipo espectro"] == "FTIR-Acetato" else row["Señal manual 3611"]

    df_muestras["Señal solvente"] = df_muestras.apply(obtener_senal_solvente, axis=1)

    # Calcular Índice OH
    def calcular_indice_oh(row):
        tipo = row["Tipo espectro"]
        peso = row["Peso muestra [g]"]
        senal_grafica = row["Señal"]
        senal_manual = row["Señal solvente"]
        if tipo == "FTIR-Acetato":
            constante = 52.5253
        elif tipo == "FTIR-Cloroformo":
            constante = 66.7324
        else:
            return "No disponible"
        if peso is None or peso == 0 or senal_grafica is None or senal_manual is None:
            return "No disponible"
        return round(((senal_grafica - senal_manual) * constante) / peso, 4)

    df_muestras["Índice OH"] = df_muestras.apply(calcular_indice_oh, axis=1)

    # Crear tabla
    columnas_mostrar = ["Muestra", "Tipo espectro", "Fecha espectro", "Señal", "Señal solvente", "Peso muestra [g]", "Índice OH"]
    df_final = df_muestras[columnas_mostrar].rename(columns={"Tipo espectro": "Tipo"})
    df_final["Peso muestra [g]"] = df_final["Peso muestra [g]"].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
    df_final["Índice OH"] = df_final["Índice OH"].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

    st.dataframe(df_final, use_container_width=True)

    muestras_unicas = df_final["Muestra"].dropna().unique().tolist()
    st.session_state["muestra_activa"] = muestras_unicas[0] if len(muestras_unicas) == 1 else None

    mostrar_sector_flotante(db)
