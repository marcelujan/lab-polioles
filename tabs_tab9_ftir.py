
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def render_tab9(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Análisis FTIR")
    st.session_state["current_tab"] = "Análisis FTIR"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # --- COMPARACIÓN DE ESPECTROS ---
    st.subheader("Comparación de espectros FTIR")
    espectros = []
    for m in muestras:
        for e in m.get("espectros", []):
            if e.get("tipo", "").startswith("FTIR") and not e.get("es_imagen", False):
                espectros.append({
                    "muestra": m["nombre"],
                    "tipo": e.get("tipo", ""),
                    "archivo": e.get("nombre_archivo", ""),
                    "contenido": e.get("contenido")
                })

    df_espectros = pd.DataFrame(espectros)
    if df_espectros.empty:
        st.warning("No hay espectros FTIR numéricos disponibles.")
        return

    opciones = df_espectros.apply(lambda row: f"{row['muestra']} – {row['tipo']} – {row['archivo']}", axis=1)
    seleccion = st.multiselect("Seleccionar espectros para comparar", opciones, default=[])

    seleccionados = df_espectros[opciones.isin(seleccion)]
    datos_graficar = []
    for _, row in seleccionados.iterrows():
        try:
            contenido = BytesIO(base64.b64decode(row["contenido"]))
            ext = row["archivo"].split(".")[-1].lower()
            if ext == "xlsx":
                df = pd.read_excel(contenido)
            else:
                sep_try = [",", ";", "\t", " "]
                for sep in sep_try:
                    contenido.seek(0)
                    try:
                        df = pd.read_csv(contenido, sep=sep, engine="python")
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
                else:
                    continue
            col_x, col_y = df.columns[:2]
            df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
            df[col_y] = pd.to_numeric(df[col_y], errors="coerce")
            df = df.dropna()
            datos_graficar.append((row["muestra"], row["tipo"], row["archivo"], df))
        except Exception as err:
            st.warning(f"No se pudo procesar: {row['archivo']}")

    if not datos_graficar:
        st.warning("No se pudieron leer espectros válidos.")
        return

    # --- Selección de rango X ---
    st.markdown("### Rango del eje X")
    opcion_rango = st.radio("Seleccionar tipo de rango:", ["Fijo (1400–1800 cm⁻¹)", "Slider interactivo"], horizontal=True)
    if opcion_rango == "Fijo (1400–1800 cm⁻¹)":
        x_min, x_max = 1400, 1800
    else:
        all_x = np.concatenate([df.iloc[:, 0].values for _, _, _, df in datos_graficar])
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        x_min, x_max = st.slider("Seleccionar rango X", min_value=x_min, max_value=x_max, value=(x_min, x_max))

    # --- GRAFICAR ---
    fig, ax = plt.subplots()
    resumen = pd.DataFrame()
    for muestra, tipo, archivo, df in datos_graficar:
        df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
        x = df_filtrado.iloc[:, 0]
        y = df_filtrado.iloc[:, 1]
        ax.plot(x, y, label=f"{muestra} – {tipo}")
        resumen[f"{muestra} – {tipo} (X)"] = x.reset_index(drop=True)
        resumen[f"{muestra} – {tipo} (Y)"] = y.reset_index(drop=True)

    ax.set_xlabel("Número de onda [cm⁻¹]")
    ax.set_ylabel("Absorbancia")
    ax.legend()
    st.pyplot(fig)

    # --- DESCARGAS ---
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        for muestra, tipo, archivo, df in datos_graficar:
            df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
            df_filtrado.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
    excel_buffer.seek(0)

    st.download_button(
        "📥 Descargar Excel",
        data=excel_buffer.getvalue(),
        file_name="comparacion_ftir.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        "📷 Descargar gráfico PNG",
        data=img_buffer.getvalue(),
        file_name="comparacion_ftir.png",
        mime="image/png"
    )

    mostrar_sector_flotante(db)
