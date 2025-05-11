import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks

def render_tab5(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Análisis FTIR")
    st.session_state["current_tab"] = "Análisis FTIR"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    # --- Sección 1: Índice OH espectroscópico ---
    st.subheader("Índice OH espectroscópico")
    espectros_info = []
    for m in muestras:
        for e in m.get("espectros", []):
            tipo = e.get("tipo", "")
            if tipo not in ["FTIR-Acetato", "FTIR-Cloroformo"]:
                continue
            contenido = e.get("contenido")
            es_imagen = e.get("es_imagen", False)
            valor_y_extraido = None
            if contenido and not es_imagen:
                try:
                    extension = e.get("nombre_archivo", "").split(".")[-1].lower()
                    binario = BytesIO(base64.b64decode(contenido))
                    if extension == "xlsx":
                        df = pd.read_excel(binario, header=None)
                    else:
                        for sep in [",", ";", "\t", " "]:
                            binario.seek(0)
                            try:
                                df = pd.read_csv(binario, sep=sep, header=None)
                                if df.shape[1] >= 2:
                                    break
                            except:
                                continue
                        else:
                            df = None
                    if df is not None and df.shape[1] >= 2:
                        df = df.dropna()
                        x_val = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                        y_val = pd.to_numeric(df.iloc[:, 1], errors="coerce")
                        df_limpio = pd.DataFrame({"X": x_val, "Y": y_val}).dropna()
                        objetivo_x = 3548 if tipo == "FTIR-Acetato" else 3611
                        idx = (df_limpio["X"] - objetivo_x).abs().idxmin()
                        valor_y_extraido = df_limpio.loc[idx, "Y"]
                except:
                    valor_y_extraido = None

            espectros_info.append({
                "Muestra": m["nombre"],
                "Tipo": tipo,
                "Fecha": e.get("fecha", ""),
                "Señal": valor_y_extraido,
                "Señal manual 3548": e.get("senal_3548"),
                "Señal manual 3611": e.get("senal_3611"),
                "Peso muestra [g]": e.get("peso_muestra")
            })

    df_oh = pd.DataFrame(espectros_info)
    if df_oh.empty:
        st.warning("No se encontraron espectros válidos.")
    else:
        def get_manual(row):
            if row["Tipo"] == "FTIR-Acetato":
                return row["Señal manual 3548"]
            elif row["Tipo"] == "FTIR-Cloroformo":
                return row["Señal manual 3611"]
            return None

        def calc_oh(row):
            tipo = row["Tipo"]
            peso = row["Peso muestra [g]"]
            senal = row["Señal"]
            ref = row["Señal solvente"]
            if tipo == "FTIR-Acetato":
                const = 52.5253
            elif tipo == "FTIR-Cloroformo":
                const = 66.7324
            else:
                return "—"
            if not all([peso, senal, ref]) or peso == 0:
                return "—"
            return round(((senal - ref) * const) / peso, 4)

        df_oh["Señal solvente"] = df_oh.apply(get_manual, axis=1)
        df_oh["Índice OH"] = df_oh.apply(calc_oh, axis=1)
        df_oh["Peso muestra [g]"] = df_oh["Peso muestra [g]"].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
        df_oh["Índice OH"] = df_oh["Índice OH"].apply(lambda x: round(x, 2) if isinstance(x, float) else x)
        st.dataframe(df_oh[["Muestra", "Tipo", "Fecha", "Señal", "Señal solvente", "Peso muestra [g]", "Índice OH"]],
                     use_container_width=True)

    # --- Sección 2: Comparación de espectros ---
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
        st.stop()

    opciones = df_espectros.apply(lambda row: f"{row['muestra']} – {row['tipo']} – {row['archivo']}", axis=1)
    seleccion = st.multiselect("Seleccionar espectros para comparar", opciones, default=[])
    seleccionados = df_espectros[opciones.isin(seleccion)]

    if seleccionados.empty:
        return

    aplicar_suavizado = st.checkbox("Aplicar suavizado (Savitzky-Golay)", value=False)
    normalizar = st.checkbox("Normalizar intensidad", value=False)
    mostrar_picos = st.checkbox("Mostrar picos detectados automáticamente", value=False)

    if mostrar_picos:
        col1, col2 = st.columns(2)
        altura_min = col1.number_input("Altura mínima", value=0.0, step=0.01)
        distancia_min = col2.number_input("Distancia mínima entre picos", value=70, step=1)

    datos_graficar = []
    for _, row in seleccionados.iterrows():
        try:
            contenido = BytesIO(base64.b64decode(row["contenido"]))
            ext = row["archivo"].split(".")[-1].lower()
            if ext == "xlsx":
                df = pd.read_excel(contenido)
            else:
                for sep in [",", ";", "\t", " "]:
                    contenido.seek(0)
                    try:
                        df = pd.read_csv(contenido, sep=sep)
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
        except:
            continue

    if not datos_graficar:
        return

    all_x = np.concatenate([df.iloc[:, 0].values for _, _, _, df in datos_graficar])
    col1, col2 = st.columns(2)
    x_min = col1.number_input("X min", value=float(np.min(all_x)))
    x_max = col2.number_input("X max", value=float(np.max(all_x)))
    y_min = col1.number_input("Y min", value=float(np.min([df.iloc[:, 1].min() for _, _, _, df in datos_graficar])))
    y_max = col2.number_input("Y max", value=float(np.max([df.iloc[:, 1].max() for _, _, _, df in datos_graficar])))

    fig, ax = plt.subplots()
    resumen = pd.DataFrame()
    for muestra, tipo, archivo, df in datos_graficar:
        df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)].copy()
        if df_filtrado.empty:
            continue
        x = df_filtrado.iloc[:, 0].reset_index(drop=True)
        y = df_filtrado.iloc[:, 1].reset_index(drop=True)

        if aplicar_suavizado and len(y) >= 5:
            window = 7 if len(y) % 2 else 7
            y = pd.Series(savgol_filter(y, window_length=window, polyorder=2)).reset_index(drop=True)
        if normalizar:
            y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y

        label = f"{muestra} – {tipo}"
        ax.plot(x, y, label=label)
        resumen[f"{label} (X)"] = x
        resumen[f"{label} (Y)"] = y

        if mostrar_picos:
            try:
                peaks, _ = find_peaks(y, height=altura_min, distance=distancia_min)
                ax.plot(x.iloc[peaks], y.iloc[peaks], "x", label=f"{label} picos")
                for px, py in zip(x.iloc[peaks], y.iloc[peaks]):
                    ax.annotate(f"   {px:.0f} cm⁻¹ ⇒ {py:.4f}", (px, py), textcoords="offset points", xytext=(0, 8), ha="left", fontsize=6)
            except:
                continue

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Número de onda [cm⁻¹]")
    ax.set_ylabel("Absorbancia")
    ax.legend()
    st.pyplot(fig)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre_base = f"FTIR_{now}"

    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        for muestra, tipo, archivo, df in datos_graficar:
            df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
            df_filtrado.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
    buffer_excel.seek(0)
    st.download_button("📥 Descargar Excel", data=buffer_excel.getvalue(), file_name=f"{nombre_base}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    buffer_img = BytesIO()
    fig.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
    st.download_button("📷 Descargar PNG", data=buffer_img.getvalue(), file_name=f"{nombre_base}.png", mime="image/png")

    mostrar_sector_flotante(db)
