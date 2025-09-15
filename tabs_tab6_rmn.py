# --- Hoja 6: Análisis RMN ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import base64
import os
from functools import lru_cache
import math
from PIL import Image
from scipy.signal import find_peaks
import requests
import io
from theme_plotly_tesis import setup_plotly_tesis, add_tesis_guides
setup_plotly_tesis()  # deja “tesis” como template por defecto


# --- Configuraciones globales ---
GRUPOS_FUNCIONALES = ["Formiato", "Cloroformo", "C=C olefínicos", "Glicerol medio", "Glicerol extremos", "Metil-Éster", "Eter", "Ester", "Ácido carboxílico", "OH", "Epóxido", "C=C", "Alfa-C=O","Alfa-C-OH", "Alfa-C=C", "C=C-Alfa-C=C", "Beta-carbonilo", "Alfa-epóxido", "Epóxido-alfa-epóxido", "CH2", "CH3", "SO3-"]
colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'darkcyan', 'teal', 'crimson', 'goldenrod']

# --- Cacheo de espectros por archivo base64 ---
session_cache = {}

def decodificar_csv_o_excel(contenido_base64, archivo):
    clave_cache = f"{archivo}__{hash(contenido_base64)}"
    if clave_cache in session_cache:
        return session_cache[clave_cache]
    
    # Validar que el contenido no sea None
    if contenido_base64 is None:
        st.warning(f"Error: contenido es None para el archivo {archivo}")
        return None
    
    try:
        contenido = BytesIO(base64.b64decode(contenido_base64))
        ext = os.path.splitext(archivo)[1].lower()
        if ext == ".xlsx":
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
                return None
        df.columns = ["x", "y"]
        df = df.dropna()
        session_cache[clave_cache] = df
        return df
    except Exception as e:
        st.warning(f"Error al decodificar {archivo}: {e}")
    return None

# --- Firebase helpers: precarga completa de espectros RMN por muestra ---
def precargar_espectros_rmn(db, muestras):
    espectros_total = []
    for m in muestras:
        nombre = m["nombre"]
        docs = db.collection("muestras").document(nombre).collection("espectros").stream()
        for i, doc in enumerate(docs):
            e = doc.to_dict()
            tipo = (e.get("tipo") or "").upper()
            if "RMN" in tipo:
                espectros_total.append({
                    "muestra": nombre,
                    "tipo": tipo,
                    "archivo": e.get("nombre_archivo", "sin nombre"),
                    "contenido": e.get("contenido"),
                    "mascaras": e.get("mascaras", []),
                    "fecha": e.get("fecha", "Sin fecha"),
                    "url_archivo": e.get("url_archivo"),
                    "archivos": e.get("archivos", {}),
                    "es_imagen": e.get("es_imagen", False),
                    "id": f"{nombre}__{i}"
                })
    return pd.DataFrame(espectros_total)



def mostrar_correccion_viscosidad_individual(df, tipo="RMN 1H"):
    st.markdown("**Desplazamiento espectral por viscosidad**")
    correcciones = {}

    # Definir rangos y valores por defecto según tipo
    if tipo == "RMN 1H":
        rango_pico1 = (7.20, 7.32)
        valor_pico1_default = 7.26
        rango_pico2 = (0.60, 0.80)
        valor_pico2_default = 0.70
    else:  # RMN 13C u otro
        rango_pico1 = (172.0, 173.0)
        valor_pico1_default = 172.5
        rango_pico2 = (13.0, 14.0)
        valor_pico2_default = 13.5

    for _, row in df.iterrows():
        archivo_actual = row["archivo"]
        df_esp = decodificar_csv_o_excel(row["contenido"], archivo_actual)
        if df_esp is None or df_esp.empty:
            continue

        pico1 = df_esp[(df_esp["x"] >= rango_pico1[0]) & (df_esp["x"] <= rango_pico1[1])]
        p1 = pico1["x"][pico1["y"] == pico1["y"].max()].values[0] if not pico1.empty else valor_pico1_default

        pico2 = df_esp[(df_esp["x"] >= rango_pico2[0]) & (df_esp["x"] <= rango_pico2[1])]
        p2 = pico2["x"][pico2["y"] == pico2["y"].max()].values[0] if not pico2.empty else valor_pico2_default

        col1, col2 = st.columns(2)
        with col1:
            p1_manual = st.number_input(f"Pico 1 ({archivo_actual})", value=float(p1), key=f"pico1_visc_{archivo_actual}")
        with col2:
            p2_manual = st.number_input(f"Pico 2 ({archivo_actual})", value=float(p2), key=f"pico2_visc_{archivo_actual}")

        try:
            a = (valor_pico1_default - valor_pico2_default) / (p1_manual - p2_manual)
            b = valor_pico1_default - a * p1_manual
        except ZeroDivisionError:
            a, b = 1.0, 0.0

        correcciones[archivo_actual] = (a, b)

    return correcciones


def mostrar_ajuste_bibliografia_individual(df):
    st.markdown("### 📘 Ajuste global a bibliografía")
    correcciones = {}
    for _, row in df.iterrows():
        archivo_actual = row["archivo"]

        col1, col2 = st.columns(2)
        with col1:
            p1_bib = st.number_input(f"Pico 1 biblio ({archivo_actual})", value=7.26, key=f"pico1_bib_{archivo_actual}")
        with col2:
            p2_bib = st.number_input(f"Pico 2 biblio ({archivo_actual})", value=0.88, key=f"pico2_bib_{archivo_actual}")

        col3, col4 = st.columns(2)
        with col3:
            p1_medido = st.number_input(f"Pico 1 medido ({archivo_actual})", value=7.26, key=f"pico1_meas_bib_{archivo_actual}")
        with col4:
            p2_medido = st.number_input(f"Pico 2 medido ({archivo_actual})", value=0.70, key=f"pico2_meas_bib_{archivo_actual}")

        try:
            a = (p2_bib - p1_bib) / (p2_medido - p1_medido)
            b = p1_bib - a * p1_medido
        except ZeroDivisionError:
            a, b = 1.0, 0.0

        correcciones[archivo_actual] = (a, b)

    return correcciones


# --- Ajuste global a bibliografía (mismo para todos los espectros) ---
def mostrar_ajuste_bibliografia_global():
    st.markdown("### 📘 Ajuste global a bibliografía")
    col1, col2 = st.columns(2)
    with col1:
        p1_bib = st.number_input("Pico 1 bibliografía", value=7.26, key="pico1_bib_global")
    with col2:
        p2_bib = st.number_input("Pico 2 bibliografía", value=0.88, key="pico2_bib_global")

    col3, col4 = st.columns(2)
    with col3:
        p1_meas = st.number_input("Pico 1 medido", value=7.26, key="pico1_meas_global")
    with col4:
        p2_meas = st.number_input("Pico 2 medido", value=0.70, key="pico2_meas_global")

    try:
        a_bib = (p2_bib - p1_bib) / (p2_meas - p1_meas)
        b_bib = p1_bib - a_bib * p1_meas
    except ZeroDivisionError:
        a_bib, b_bib = 1.0, 0.0

    return a_bib, b_bib

# --- Firebase helpers: precarga de documentos tipo tabla_integral o dt2 ---
def precargar_tabla_global(db, nombre_tabla):
    doc = db.collection("tablas_integrales").document(nombre_tabla).get()
    return doc.to_dict() if doc.exists else {}

def precargar_dt2_muestra(db, muestra, tipo):
    doc = db.collection("muestras").document(muestra).collection("dt2").document(tipo)
    data = doc.get().to_dict()
    return data.get("filas", []) if data else []

# --- Helper para precarga de espectros por muestra (wrapper para obtener_df_esp_precargado) ---
def precargar_espectros_por_muestra(db, muestra):
    # usamos precargar_espectros_rmn pero con lista [{"nombre": muestra}]
    df = precargar_espectros_rmn(db, [{"nombre": muestra}])
    
    # devolvemos un diccionario indexado por archivo
    return {
        row["archivo"]: row for _, row in df.iterrows()
    }

# --- Cálculo de integrales: usar espectros precargados ---
def obtener_df_esp_precargado(db, espectros_dict, muestra, archivo):
    espectro = espectros_dict.get(archivo)
    if espectro is None:
        espectros_dict.update(precargar_espectros_por_muestra(db, muestra))
        espectro = espectros_dict.get(archivo)
    if espectro is None:
        return None
    return decodificar_csv_o_excel(espectro.get("contenido"), archivo)

# Obtiene el espectro combinado visual con las funciones aplicadas
def transformar_espectro(
    row,
    ajustes_y,
    normalizar,
    espectro_resta,
    id_resta,
    correcciones_viscosidad,
    a_bib,
    b_bib
):
    archivo_actual = row["archivo"]
    df_esp = decodificar_csv_o_excel(row.get("contenido"), archivo_actual)
    if df_esp is None or df_esp.empty:
        return None

    df_aux = df_esp.rename(columns={df_esp.columns[0]: "x", df_esp.columns[1]: "y"}).dropna()

    # Corrección por viscosidad
    a_v, b_v = correcciones_viscosidad.get(archivo_actual, (1.0, 0.0))
    x_vals = a_bib * (a_v * df_aux["x"] + b_v) + b_bib

    # Ajuste Y manual
    y_vals = df_aux["y"] + ajustes_y.get(archivo_actual, 0.0)

    # Restar espectro si corresponde
    if espectro_resta is not None:
        # Aplicar corrección de viscosidad al espectro que se resta (si corresponde)
        a_v_resta, b_v_resta = correcciones_viscosidad.get(id_resta, (1.0, 0.0))
        x_resta_corr = a_bib * (a_v_resta * espectro_resta["x"] + b_v_resta) + b_bib

        # Interpolar sobre el eje ya corregido
        y_resta_interp = np.interp(x_vals, x_resta_corr, espectro_resta["y"])
        y_vals = y_vals - (y_resta_interp + ajustes_y.get(id_resta, 0.0))


    # Normalización si corresponde
    if normalizar:
        y_max_val = y_vals.max()
        y_vals = y_vals / y_max_val if y_max_val != 0 else y_vals

    df_transformado = pd.DataFrame({"x": x_vals, "y": y_vals})

    return df_transformado

# --- Optimización aplicada a recálculo de D/T2 y señales ---
def recalcular_areas_y_guardar(
        df_edicion,
        tipo,
        db,
        nombre_tabla,
        tabla_destino="dt2",
        ajustes_y={},
        normalizar=False,
        espectro_resta=None,
        id_resta=None,
        correcciones_viscosidad={},
        a_bib=1.0,
        b_bib=0.0
    ):
    
    espectros_cache = {}
    campo_h = "H" if tipo == "RMN 1H" else "C"
    campo_has = "Has" if tipo == "RMN 1H" else "Cas"

    # Limpieza de nombres de columnas
    df_edicion.columns = [str(col) if not pd.isna(col) else "" for col in df_edicion.columns]

    # Forzar columnas numéricas
    df_edicion["X min"] = pd.to_numeric(df_edicion["X min"], errors="coerce")
    df_edicion["X max"] = pd.to_numeric(df_edicion["X max"], errors="coerce")
    df_edicion["Xas min"] = pd.to_numeric(df_edicion["Xas min"], errors="coerce")
    df_edicion["Xas max"] = pd.to_numeric(df_edicion["Xas max"], errors="coerce")
    df_edicion[campo_has] = pd.to_numeric(df_edicion[campo_has], errors="coerce")

    # Bucle principal
    for i, row in df_edicion.iterrows():
        try:
            row_dict = row.to_dict()

            muestra = row_dict.get("Muestra")
            archivo = row_dict.get("Archivo")

            # Blindaje
            muestra = str(muestra).strip() if pd.notna(muestra) else ""
            archivo = str(archivo).strip() if pd.notna(archivo) else ""

            if muestra == "" or archivo == "":
                continue

            try:
                df_espectros_muestra = precargar_espectros_rmn(db, [{"nombre": muestra}])
                row_espectro = df_espectros_muestra[df_espectros_muestra["archivo"] == archivo].iloc[0]

                df_transformado = transformar_espectro(
                    row=row_espectro,
                    ajustes_y=ajustes_y,
                    normalizar=normalizar,
                    espectro_resta=espectro_resta,
                    id_resta=id_resta,
                    correcciones_viscosidad=correcciones_viscosidad,
                    a_bib=a_bib,
                    b_bib=b_bib
                )
                if df_transformado is None:
                    continue

                tolerancia = 1e-6

                # Calcular Área
                try:
                    x_min = float(row_dict.get("X min"))
                    x_max = float(row_dict.get("X max"))
                    df_main = df_transformado[(df_transformado["x"] >= min(x_min, x_max) - tolerancia) & (df_transformado["x"] <= max(x_min, x_max) + tolerancia)]
                    area = np.trapz(df_main["y"], df_main["x"]) if not df_main.empty else None
                    df_edicion.at[i, "Área"] = round(area, 2) if area is not None else None

                except Exception as e:
                    area = None
                    df_edicion.at[i, "Área"] = None

                # Calcular Área as y H/C
                try:
                    xas_min = float(row_dict.get("Xas min"))
                    xas_max = float(row_dict.get("Xas max"))
                    has_or_cas = float(row_dict.get(campo_has)) if row_dict.get(campo_has) not in [None, ""] else None
                    df_as = df_transformado[(df_transformado["x"] >= min(xas_min, xasMax) - tolerancia) & (df_transformado["x"] <= max(xasMin, xas_max) + tolerancia)]
                    area_as = np.trapz(df_as["y"], df_as["x"]) if not df_as.empty else None
                    df_edicion.at[i, "Área as"] = round(area_as, 2) if area_as is not None else None

                    if area is not None and area_as is not None and has_or_cas is not None and area_as != 0:
                        resultado = (area * has_or_cas) / area_as
                        df_edicion.at[i, campo_h] = round(resultado, 2)
                    else:
                        df_edicion.at[i, campo_h] = None

                except Exception as e:
                    st.warning(f"⚠️ Error en fila {i} al calcular Área as y H/C: {e}")
                    df_edicion.at[i, "Área as"] = None
                    df_edicion.at[i, campo_h] = None

            except Exception as e:
                st.warning(f"⚠️ Error en fila {i}: {e}")


        except Exception as e:
            st.warning(f"⚠️ Error en fila {i}: {e}")

    # Limpiar Muestra para evitar Series ambiguos
    df_edicion["Muestra"] = df_edicion["Muestra"].astype(str).str.strip()

    # Guardar en Firebase (conservar combinaciones no actualizadas)
    filas_actualizadas_raw = df_edicion.to_dict(orient="records")

    # Limpieza extra en filas_actualizadas_raw
    for f in filas_actualizadas_raw:
        f["Muestra"] = str(f.get("Muestra")).strip() if pd.notna(f.get("Muestra")) else ""
        f["Archivo"] = str(f.get("Archivo")).strip() if pd.notna(f.get("Archivo")) else ""

    combinaciones_actualizadas = {
        (f.get("Muestra"), f.get("Archivo"))
        for f in filas_actualizadas_raw
        if (not pd.isna(f.get("Muestra"))) and (not pd.isna(f.get("Archivo"))) and (f.get("Muestra") != "")
    }

    if tabla_destino == "dt2":
        doc_destino = lambda m: db.collection("muestras").document(m).collection("dt2").document(tipo.lower())
    else:
        doc_ref = db.collection("tablas_integrales").document(nombre_tabla)
        doc_data = doc_ref.get().to_dict()
        filas_previas = doc_data.get("filas", []) if doc_data else []

        filas_conservadas = [
            f for f in filas_previas
            if (f.get("Muestra"), f.get("Archivo")) not in combinaciones_actualizadas
        ]

        previas_map = {
            (f.get("Muestra"), f.get("Archivo"), f.get("X min"), f.get("X max")): f
            for f in filas_previas
        }

        campos_a_conservar = ["Grupo funcional", "δ pico", "Observaciones"]

        for f in filas_actualizadas_raw:
            clave = (f.get("Muestra"), f.get("Archivo"), f.get("X min"), f.get("X max"))
            previa = previas_map.get(clave)
            if previa:
                for campo in campos_a_conservar:
                    if campo in f:
                        if f[campo] in [None, ""]:
                            f[campo] = previa.get(campo, "")
        filas_finales = filas_conservadas + filas_actualizadas_raw
        doc_ref.set({"filas": filas_finales})
        return

    
    st.warning(f"VERIFICACIÓN FINAL: df_edicion['Muestra'].unique() = {df_edicion['Muestra'].unique()}")

    for muestra in df_edicion["Muestra"].unique():
        muestra_str = str(muestra)  # forzar string por seguridad
        filas_m = [f for f in filas_actualizadas_raw if str(f.get("Muestra")) == muestra_str]
        doc_out = doc_destino(muestra_str)
        doc_data = doc_out.get().to_dict()
        filas_previas = doc_data.get("filas", []) if doc_data else []
        archivos_actualizados = set(f["Archivo"] for f in filas_m if f.get("Archivo"))
        filas_conservadas = [f for f in filas_previas if f.get("Archivo") not in archivos_actualizados]
        filas_finales = filas_conservadas + filas_m
        doc_out.set({"filas": filas_finales})


    st.success("✅ Datos recalculados y guardados correctamente.")
    st.rerun()



def render_rmn_plot(df, tipo="RMN 1H", key_sufijo="rmn1h", db=None):
    if df.empty:
        st.info(f"No hay espectros disponibles para {tipo}.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    normalizar = col1.checkbox("Normalizar", key=f"norm_{key_sufijo}")
    mostrar_picos = col2.checkbox("Detectar picos", key=f"picos_{key_sufijo}")
    restar_espectro = col3.checkbox("Restar espectro", key=f"resta_{key_sufijo}")
    ajuste_y_manual = col4.checkbox("Ajuste manual Y", key=f"ajuste_y_{key_sufijo}")
    superposicion_vertical = col5.checkbox("Superposición vertical", key=f"offset_{key_sufijo}")

    ajustes_y = {row["archivo"]: st.number_input(f"Y para {row['archivo']}", value=0.0, step=0.1, key=f"ajuste_y_val_{row['archivo']}")
                 for _, row in df.iterrows()} if ajuste_y_manual else {row["archivo"]: 0.0 for _, row in df.iterrows()}

    colv, colb = st.columns(2)
    activar_viscosidad = colv.checkbox("Corregir por viscosidad", key=f"chk_visc_{key_sufijo}")
    activar_biblio = colb.checkbox("Ajustar a bibliografía", key=f"chk_biblio_{key_sufijo}")

    correcciones_viscosidad = mostrar_correccion_viscosidad_individual(df, tipo=tipo) if activar_viscosidad else {}

    # Ajuste global a bibliografía
    if activar_biblio:
        st.markdown("**Desplazamiento espectral por bibliografía**")
        col1, col2 = st.columns(2)
        with col1:
            p1_bib = st.number_input("Pico 1 bibliografía", value=7.26, key=f"pico1_bib_global_{key_sufijo}")
        with col2:
            p2_bib = st.number_input("Pico 2 bibliografía", value=0.88, key=f"pico2_bib_global_{key_sufijo}")

        col3, col4 = st.columns(2)
        with col3:
            p1_medido = st.number_input("Pico 1 medido", value=7.26, key=f"pico1_medido_global_{key_sufijo}")
        with col4:
            p2_medido = st.number_input("Pico 2 medido", value=0.70, key=f"pico2_medido_global_{key_sufijo}")

        try:
            a_bib = (p2_bib - p1_bib) / (p2_medido - p1_medido)
            b_bib = p1_bib - a_bib * p1_medido
        except ZeroDivisionError:
            a_bib, b_bib = 1.0, 0.0
    else:
        a_bib, b_bib = 1.0, 0.0

    seleccion_resta = None
    if restar_espectro:
        opciones_restar = [f"{row['muestra']} – {row['archivo']}" for _, row in df.iterrows()]
        seleccion_resta = st.selectbox("Seleccionar espectro a restar:", opciones_restar, key=f"sel_resta_{key_sufijo}")

    st.markdown("**Selección de rangos de gráficos**")
    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X mínimo", value=0.0, key=f"x_min_{key_sufijo}")
    x_max = colx2.number_input("X máximo", value=9.0 if tipo == "RMN 1H" else 200.0, key=f"x_max_{key_sufijo}")

    if normalizar:
        y_min = coly1.number_input("Y mínimo", value=0.0, key=f"y_min_{key_sufijo}")
        y_max = 1.0
    else:
        y_min = coly1.number_input("Y mínimo", value=0.0, key=f"y_min_{key_sufijo}")
        y_max = coly2.number_input("Y máximo", value=100.0 if tipo == "RMN 1H" else 2, key=f"y_max_{key_sufijo}")

    if mostrar_picos:
        colp1, colp2 = st.columns(2)
        altura_min = colp1.number_input("Altura mínima", value=0.00, step=0.01, key=f"altura_min_{key_sufijo}")
        distancia_min = colp2.number_input("Distancia mínima entre picos", value=400, step=1, key=f"distancia_min_{key_sufijo}")
    else:
        altura_min = distancia_min = None

    col_tabla, col_sombra = st.columns(2)
    mostrar_tabla_dt2_chk = col_tabla.checkbox(f"🧮 Tabla de Cálculos D/T2 (FAMAF) {tipo}", key=f"mostrar_dt2_{key_sufijo}")
    mostrar_tabla_senales_chk = col_tabla.checkbox(f"📈 Tabla de Cálculos {tipo}", key=f"mostrar_senales_{key_sufijo}")
    mostrar_tabla_biblio_chk = col_tabla.checkbox(f"📚 Tabla Bibliográfica {tipo[-3:]}", key=f"mostrar_biblio_{tipo.lower()}_{key_sufijo}")

    aplicar_sombra_dt2 = col_sombra.checkbox(f"Sombrear Tabla de Cálculos D/T2 (FAMAF) {tipo}", key=f"sombra_dt2_{key_sufijo}")
    aplicar_sombra_senales = col_sombra.checkbox(f"Sombrear Tabla de Cálculos {tipo}", key=f"sombra_senales_{key_sufijo}")
    aplicar_sombra_biblio = col_sombra.checkbox(f"Delinear Tabla Bibliográfica {tipo[-3:]}", key=f"sombra_biblio_{key_sufijo}")

    check_d_por_espectro = {}
    check_t2_por_espectro = {}
    if aplicar_sombra_dt2:
        st.markdown("**Espectros para aplicar Sombreados por D/T2**")
        for _, row in df.iterrows():
            archivo = row["archivo"]
            col_d, col_t2 = st.columns([1, 1])
            check_d_por_espectro[archivo] = col_d.checkbox(f"D – {archivo}", key=f"chk_d_{archivo}_{key_sufijo}")
            check_t2_por_espectro[archivo] = col_t2.checkbox(f"T2 – {archivo}", key=f"chk_t2_{archivo}_{key_sufijo}")

    # Preparar espectro_resta e id_resta para pasar a tablas también
    espectro_resta = None
    id_resta = None
    if restar_espectro and seleccion_resta:
        id_resta = seleccion_resta.split(" – ")[-1].strip()
        fila_resta = df[df["archivo"] == id_resta].iloc[0] if id_resta in set(df["archivo"]) else None
        if fila_resta is not None:
            espectro_resta = decodificar_csv_o_excel(fila_resta["contenido"], fila_resta["archivo"])

    if mostrar_tabla_dt2_chk:
        mostrar_tabla_dt2(df, tipo, key_sufijo, db,
                  ajustes_y, normalizar,
                  espectro_resta, id_resta,
                  correcciones_viscosidad,
                  a_bib, b_bib)


    if mostrar_tabla_senales_chk:
        mostrar_tabla_senales(df, tipo, key_sufijo, db,
                          ajustes_y, normalizar,
                          espectro_resta, id_resta,
                          correcciones_viscosidad,
                          a_bib, b_bib)

    if mostrar_tabla_biblio_chk:
        mostrar_tabla_biblio(tipo, key_sufijo, db)

    fig = mostrar_grafico_combinado(
    df=df,
    tipo=tipo,
    key_sufijo=key_sufijo,
    db=db,
    normalizar=normalizar,
    mostrar_picos=mostrar_picos,
    restar_espectro=restar_espectro,
    seleccion_resta=seleccion_resta,
    espectro_resta=espectro_resta, 
    id_resta=id_resta,            
    ajustes_y=ajustes_y,
    superposicion_vertical=superposicion_vertical,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    aplicar_sombra_dt2=aplicar_sombra_dt2,
    aplicar_sombra_senales=aplicar_sombra_senales,
    aplicar_sombra_biblio=aplicar_sombra_biblio,
    check_d_por_espectro=check_d_por_espectro,
    check_t2_por_espectro=check_t2_por_espectro,
    altura_min=altura_min,
    distancia_min=distancia_min,
    correcciones_viscosidad=correcciones_viscosidad,
    a_bib=a_bib,
    b_bib=b_bib,
    )


    if aplicar_sombra_dt2:
        mostrar_sombreados_dt2(fig, df, tipo, y_max, key_sufijo, check_d_por_espectro, check_t2_por_espectro, db)

    st.plotly_chart(fig, use_container_width=True)

    if superposicion_vertical:
        mostrar_grafico_stacked(
            df=df,
            tipo=tipo,
            key_sufijo=key_sufijo,
            normalizar=normalizar,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            ajustes_y=ajustes_y,
            aplicar_sombra_dt2=aplicar_sombra_dt2,
            aplicar_sombra_senales=aplicar_sombra_senales,
            aplicar_sombra_biblio=aplicar_sombra_biblio,
            db=db,
            check_d_por_espectro=check_d_por_espectro,
            check_t2_por_espectro=check_t2_por_espectro,
            correcciones_viscosidad=correcciones_viscosidad,
            a_bib=a_bib,
            b_bib=b_bib,
            mostrar_picos=mostrar_picos,
            restar_espectro=restar_espectro,
            seleccion_resta=seleccion_resta,
            espectro_resta=espectro_resta, 
            id_resta=id_resta,             
            altura_min=altura_min,
            distancia_min=distancia_min
        )

    # --- Mostrar imágenes asociadas al final ---
    imagenes = df[df.get("es_imagen", False)]
    if not imagenes.empty:
        st.markdown("### 📸 Imágenes asociadas")
        for _, row in imagenes.iterrows():
            st.markdown(f"**{row['archivo']}** – {row['muestra']}")
            try:
                image_data = BytesIO(base64.b64decode(row["contenido"]))
                image = Image.open(image_data)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"❌ No se pudo mostrar la imagen: {e}")

    mostrar_indiv = st.checkbox("Gráficos individuales", key=f"chk_indiv_{key_sufijo}")
    if mostrar_indiv:
        mostrar_graficos_individuales(
            df=df,
            tipo=tipo,
            key_sufijo=key_sufijo,
            normalizar=normalizar,
            y_max=y_max,
            y_min=y_min,
            x_max=x_max,
            x_min=x_min,
            ajustes_y=ajustes_y,
            aplicar_sombra_dt2=aplicar_sombra_dt2,
            aplicar_sombra_senales=aplicar_sombra_senales,
            aplicar_sombra_biblio=aplicar_sombra_biblio,
            db=db,
            check_d_por_espectro=check_d_por_espectro,
            check_t2_por_espectro=check_t2_por_espectro,
            correcciones_viscosidad=correcciones_viscosidad,
            a_bib=a_bib,
            b_bib=b_bib,
            mostrar_picos=mostrar_picos,
            restar_espectro=restar_espectro,
            seleccion_resta=seleccion_resta,
            altura_min=altura_min,
            distancia_min=distancia_min
        )


def mostrar_grafico_combinado(
    df,
    tipo,
    key_sufijo,
    db,
    normalizar,
    mostrar_picos,
    restar_espectro,
    seleccion_resta,
    espectro_resta, 
    id_resta, 
    ajustes_y,
    superposicion_vertical,
    x_min,
    x_max,
    y_min,
    y_max,
    aplicar_sombra_dt2,
    aplicar_sombra_senales,
    aplicar_sombra_biblio,
    check_d_por_espectro,
    check_t2_por_espectro,
    altura_min=None,
    distancia_min=None,
    correcciones_viscosidad=None,
    a_bib=1.0, b_bib=0.0,
):

    fig = go.Figure()

    # --- Precargar tabla de señales y bibliografía ---
    tipo_doc_senales = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
    doc_senales = db.collection("tablas_integrales").document(tipo_doc_senales).get()
    filas_senales = doc_senales.to_dict().get("filas", []) if doc_senales.exists else []

    doc_biblio_id = "tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c"
    doc_biblio = db.collection("configuracion_global").document(doc_biblio_id).get()
    filas_biblio = doc_biblio.to_dict().get("filas", []) if doc_biblio.exists else []

    # --- Añadir trazas generadas por función heredada ---
    colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    color_idx = 0

    for _, row in df.iterrows():
        color = colores[color_idx % len(colores)]
        elementos = generar_elementos_rmn(
            row=row,
            ajustes_y=ajustes_y,
            normalizar=normalizar,
            espectro_resta=espectro_resta,
            id_resta=id_resta,
            altura_min=altura_min,
            distancia_min=distancia_min,
            correcciones_viscosidad=correcciones_viscosidad,
            a_bib=a_bib,
            b_bib=b_bib,
            filas_senales=filas_senales,
            filas_biblio=filas_biblio,
            tipo=tipo,
            y_max=y_max,
            aplicar_sombra_senales=aplicar_sombra_senales,
            aplicar_sombra_biblio=aplicar_sombra_biblio,
            mostrar_picos=mostrar_picos,
            color=color,
        )
        color_idx += 1

        for el in elementos:
            if isinstance(el, go.Scatter):
                fig.add_trace(el)
            elif isinstance(el, go.layout.Shape):
                fig.add_shape(el)
            elif isinstance(el, go.layout.Annotation):
                fig.add_annotation(el)

    fig.update_layout(
        xaxis_title="[ppm]",
        yaxis_title="Intensidad",
        xaxis=dict(range=[x_max, x_min]),
        yaxis=dict(range=[y_min, y_max] if y_max > y_min else None),
        template="tesis",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"),
                    linecolor="black", tickcolor="black")
    fig.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"),
                    linecolor="black", tickcolor="black")

    return fig


def mostrar_tabla_dt2(df, tipo, key_sufijo, db,
                      ajustes_y, normalizar,
                      espectro_resta, id_resta,
                      correcciones_viscosidad,
                      a_bib, b_bib):

    if tipo == "RMN 1H":
        columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                        "Xas min", "Xas max", "Has", "Área as", "H", "🔴H", "Observaciones", "Archivo"]
    else:
        columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                        "Xas min", "Xas max", "Cas", "Área as", "C", "🔴C", "Observaciones", "Archivo"]
    filas_guardadas = []
    for _, row in df.iterrows():
        muestra = row["muestra"]
        archivo = row["archivo"]
        df_esp = decodificar_csv_o_excel(row["contenido"], archivo)
        if df_esp is None or df_esp.empty:
            continue

        doc = db.collection("muestras").document(muestra).collection("dt2").document(tipo.lower())
        data = doc.get().to_dict()
        if data and "filas" in data:
            filas_guardadas.extend([f for f in data["filas"] if f.get("Archivo") == archivo])

    df_dt2 = pd.DataFrame(filas_guardadas)
    for col in columnas_dt2:
        if col not in df_dt2.columns:
            df_dt2[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
    df_dt2 = df_dt2[columnas_dt2]
    df_dt2 = df_dt2.sort_values(by=["Archivo", "X max"])

    st.markdown("**🧮 Tabla de Cálculos D/T2 (FAMAF)**")

    # --- Factor H*/C* ---
    if tipo == "RMN 1H":
        factor_hc = st.number_input("Factor H*", value=1.00, format="%.2f", step=0.01, key=f"factor_h_dt2_{key_sufijo}")
        df_dt2["🔴H*"] = df_dt2["H"].apply(lambda h: round(h * factor_hc, 2) if pd.notna(h) else None)
    else:
        factor_hc = st.number_input("Factor C*", value=1.00, format="%.2f", step=0.01, key=f"factor_c_dt2_{key_sufijo}")
        df_dt2["🔴C*"] = df_dt2["C"].apply(lambda c: round(c * factor_hc, 2) if pd.notna(c) else None)

    with st.form(f"form_dt2_{key_sufijo}"):
        col_config = {
            "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
            "δ pico": st.column_config.NumberColumn(format="%.2f"),
            "X min": st.column_config.NumberColumn(format="%.2f"),
            "X max": st.column_config.NumberColumn(format="%.2f"),
            "Área": st.column_config.NumberColumn(format="%.2f", label="🔴Área", disabled=True),
            "D": st.column_config.NumberColumn(format="%.2e"),
            "T2": st.column_config.NumberColumn(format="%.3f"),
            "Xas min": st.column_config.NumberColumn(format="%.2f"),
            "Xas max": st.column_config.NumberColumn(format="%.2f"),
            "Área as": st.column_config.NumberColumn(format="%.2f", label="🔴Área as", disabled=True),
            "Observaciones": st.column_config.TextColumn(),
            "Archivo": st.column_config.TextColumn(disabled=False),
            "Muestra": st.column_config.TextColumn(disabled=False),
        }

        if tipo == "RMN 1H":
            col_config["Has"] = st.column_config.NumberColumn(format="%.2f")
            col_config["H"] = st.column_config.NumberColumn(format="%.2f", label="🔴H", disabled=True)
            col_config["🔴H*"] = st.column_config.NumberColumn(format="%.2f", label="🔴H*", disabled=True)
        else:
            col_config["Cas"] = st.column_config.NumberColumn(format="%.2f")
            col_config["C"] = st.column_config.NumberColumn(format="%.2f", label="🔴C", disabled=True)
            col_config["🔴C*"] = st.column_config.NumberColumn(format="%.2f", label="🔴C*", disabled=True)

        df_dt2_edit = st.data_editor(
            df_dt2,
            column_config=col_config,
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            key=f"tabla_dt2_{key_sufijo}"
        )

        etiqueta_boton_dt2 = "🔴 Recalcular 'Área', 'Área as' y 'H'" if tipo == "RMN 1H" else "🔴 Recalcular 'Área', 'Área as' y 'C'"
        recalcular = st.form_submit_button(etiqueta_boton_dt2)

    if recalcular:
        nombre_tabla = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
        recalcular_areas_y_guardar(
            df_dt2_edit, tipo, db, nombre_tabla, tabla_destino="dt2",
            ajustes_y=ajustes_y,
            normalizar=normalizar,
            espectro_resta=espectro_resta,
            id_resta=id_resta,
            correcciones_viscosidad=correcciones_viscosidad,
            a_bib=a_bib,
            b_bib=b_bib
        )

def mostrar_tabla_senales(df, tipo, key_sufijo, db,
                          ajustes_y, normalizar,
                          espectro_resta, id_resta,
                          correcciones_viscosidad,
                          a_bib, b_bib):
    if tipo == "RMN 1H":
        columnas_senales = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                            "Xas min", "Xas max", "Has", "Área as", "H", "🔴H*", "Observaciones", "Archivo"]
    else:
        columnas_senales = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                            "Xas min", "Xas max", "Cas", "Área as", "C", "🔴C*", "Observaciones", "Archivo"]

    tipo_doc = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
    doc_ref = db.collection("tablas_integrales").document(tipo_doc)
    doc_data = doc_ref.get().to_dict() or {}
    filas_guardadas = doc_data.get("filas", [])

    combinaciones_real = {(row["muestra"], row["archivo"]) for _, row in df.iterrows()}

    agrupadas = {}
    for fila in filas_guardadas:
        clave = (fila.get("Muestra"), fila.get("Archivo"))
        if clave in combinaciones_real:
            agrupadas.setdefault(clave, []).append(fila)

    filas_totales = []
    for (m, a) in combinaciones_real:
        filas = agrupadas.get((m, a), [])
        if not filas:
            fila_vacia = {col: None for col in columnas_senales}
            fila_vacia["Muestra"] = m
            fila_vacia["Archivo"] = a
            filas = [fila_vacia]
        filas_totales.extend(filas)

        hay_fila_vacia = any(
            all(
                f.get(campo) in [None, ""] or (isinstance(f.get(campo), float) and math.isnan(f.get(campo)))
                for campo in ["δ pico", "X min", "X max"]
            )
            for f in filas
        )

        if not hay_fila_vacia:
            nueva = {col: None for col in columnas_senales}
            nueva["Muestra"] = m
            nueva["Archivo"] = a
            filas_totales.append(nueva)

    df_senales = pd.DataFrame(filas_totales)
    for col in columnas_senales:
        if col not in df_senales.columns:
            df_senales[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
    df_senales = df_senales[columnas_senales]
    df_senales = df_senales.sort_values(by=["Archivo", "X max"])

    # --- Factor H*/C* ---
    if tipo == "RMN 1H":
        factor_hc = st.number_input("Factor H*", value=1.00, format="%.2f", step=0.01, key=f"factor_h_senales_{key_sufijo}")
        df_senales["🔴H*"] = df_senales["H"].apply(lambda h: round(h * factor_hc, 2) if pd.notna(h) else None)
    else:
        factor_hc = st.number_input("Factor C*", value=1.00, format="%.2f", step=0.01, key=f"factor_c_senales_{key_sufijo}")
        df_senales["🔴C*"] = df_senales["C"].apply(lambda c: round(c * factor_hc, 2) if pd.notna(c) else None)

    st.markdown("**📈 Tabla de Cálculos**")
    with st.form(f"form_senales_{key_sufijo}"):
        col_config = {
            "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
            "δ pico": st.column_config.NumberColumn(format="%.2f"),
            "X min": st.column_config.NumberColumn(format="%.2f"),
            "X max": st.column_config.NumberColumn(format="%.2f"),
            "Área": st.column_config.NumberColumn(format="%.2f", label="🔴Área", disabled=True),
            "D": st.column_config.NumberColumn(format="%.2e"),
            "T2": st.column_config.NumberColumn(format="%.3f"),
            "Xas min": st.column_config.NumberColumn(format="%.2f"),
            "Xas max": st.column_config.NumberColumn(format="%.2f"),
            "Área as": st.column_config.NumberColumn(format="%.2f", label="🔴Área as", disabled=True),
            "Observaciones": st.column_config.TextColumn(),
            "Archivo": st.column_config.TextColumn(disabled=False),
            "Muestra": st.column_config.TextColumn(disabled=False),
        }
        if tipo == "RMN 1H":
            col_config["Has"] = st.column_config.NumberColumn(format="%.2f")
            col_config["H"] = st.column_config.NumberColumn(format="%.2f", label="🔴H", disabled=True)
            col_config["🔴H*"] = st.column_config.NumberColumn(format="%.2f", label="🔴H*", disabled=True)
        else:
            col_config["Cas"] = st.column_config.NumberColumn(format="%.2f")
            col_config["C"] = st.column_config.NumberColumn(format="%.2f", label="🔴C", disabled=True)
        col_config["🔴C*"] = st.column_config.NumberColumn(format="%.2f", label="🔴C*", disabled=True)

        df_senales_edit = st.data_editor(
            df_senales,
            column_config=col_config,
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            key=f"tabla_senales_{key_sufijo}"
        )

        texto_boton = "🔴 Recalcular 'Área', 'Área as' y 'H'" if tipo == "RMN 1H" else "🔴 Recalcular 'Área', 'Área as' y 'C'"
        recalcular = st.form_submit_button(texto_boton)

    if recalcular:
        nombre_tabla = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
        recalcular_areas_y_guardar(
            df_senales_edit,
            tipo=tipo,
            db=db,
            nombre_tabla=nombre_tabla,
            tabla_destino="senales",
            ajustes_y=ajustes_y,
            normalizar=normalizar,
            espectro_resta=espectro_resta,
            id_resta=id_resta,
            correcciones_viscosidad=correcciones_viscosidad,
            a_bib=a_bib,
            b_bib=b_bib
        )



def mostrar_tabla_biblio(tipo, key_sufijo, db):
    doc_id = "tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c"
    doc_biblio = db.collection("configuracion_global").document(doc_id)

    if not doc_biblio.get().exists:
        doc_biblio.set({"filas": []})

    filas_biblio = doc_biblio.get().to_dict().get("filas", [])
    columnas_biblio = ["Grupo funcional", "X min", "δ pico", "X max", "Tipo de muestra", "Observaciones"]
    df_biblio = pd.DataFrame(filas_biblio)

    for col in columnas_biblio:
        if col not in df_biblio.columns:
            df_biblio[col] = "" if col in ["Grupo funcional", "Tipo de muestra", "Observaciones"] else None
    df_biblio = df_biblio[columnas_biblio]

    st.markdown(f"**📚 Tabla Bibliográfica {tipo[-3:]}**")
    df_biblio_edit = st.data_editor(
        df_biblio,
        column_config={
            "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
            "X min": st.column_config.NumberColumn(format="%.2f"),
            "δ pico": st.column_config.NumberColumn(format="%.2f"),
            "X max": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        key=f"tabla_biblio_{key_sufijo}"
    )

    colb1, colb2 = st.columns([1, 1])
    with colb1:
        if st.button(f"🔴 Actualizar Tabla Bibliográfica {tipo[-3:]}"):
            doc_biblio.set({"filas": df_biblio_edit.to_dict(orient="records")})
            st.success("✅ Datos bibliográficos actualizados.")
    with colb2:
        buffer_excel = BytesIO()
        with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
            df_biblio_edit.to_excel(writer, index=False, sheet_name=f"Bibliografía {tipo[-3:]}")
        buffer_excel.seek(0)
        st.download_button(
            f"📥 Descargar tabla {tipo[-3:]}",
            data=buffer_excel.getvalue(),
            file_name=f"tabla_bibliografica_rmn{tipo[-3:].lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        
def mostrar_grafico_stacked(
    df, tipo, key_sufijo,
    normalizar, x_min, x_max, y_min, y_max,
    ajustes_y,
    aplicar_sombra_dt2,
    aplicar_sombra_senales,
    aplicar_sombra_biblio,
    db,
    check_d_por_espectro=None,
    check_t2_por_espectro=None,
    correcciones_viscosidad={},
    a_bib=1.0,
    b_bib=0.0,
    mostrar_picos=False,
    restar_espectro=False,
    seleccion_resta=None,
    espectro_resta=None, 
    id_resta=None,     
    altura_min=None,
    distancia_min=None
):

    offset_auto = round((y_max - y_min) / (len(df) + 1), 2) if (y_max is not None and y_min is not None and y_max > y_min) else 1.0
    offset_manual = st.slider(
        "Separación entre espectros (offset)",
        min_value=0.1,
        max_value=30.0,
        value=offset_auto,
        step=0.1,
        key=f"offset_val_{key_sufijo}"
    )

    tipo_doc = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
    filas_senales = db.collection("tablas_integrales").document(tipo_doc).get().to_dict().get("filas", [])
    filas_biblio = db.collection("configuracion_global").document(
        "tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c"
    ).get().to_dict().get("filas", [])

    fig_offset = go.Figure()

    for i, (_, row) in enumerate(df.iterrows()):
        archivo_actual = row["archivo"]
        muestra_actual = row["muestra"]

        doc_dt2 = db.collection("muestras").document(muestra_actual).collection("dt2").document(tipo.lower()).get()
        filas_dt2 = doc_dt2.to_dict().get("filas", []) if doc_dt2.exists else []

        elementos = generar_elementos_rmn(
            row=row,
            ajustes_y=ajustes_y,
            normalizar=normalizar,
            espectro_resta=espectro_resta,
            id_resta=id_resta,
            altura_min=altura_min,
            distancia_min=distancia_min,
            correcciones_viscosidad=correcciones_viscosidad,
            a_bib=a_bib,
            b_bib=b_bib,
            filas_senales=filas_senales,
            filas_biblio=filas_biblio,
            tipo=tipo,
            y_max=y_max,
            aplicar_sombra_senales=aplicar_sombra_senales,
            aplicar_sombra_biblio=aplicar_sombra_biblio,
            mostrar_picos=mostrar_picos,
            filas_dt2=filas_dt2,
            check_d_por_espectro=check_d_por_espectro,
            check_t2_por_espectro=check_t2_por_espectro,
        )
        offset = offset_manual * i
        for el in elementos:
            if isinstance(el, go.Scatter):
                el.y = [v + offset for v in el.y]
                fig_offset.add_trace(el)
            elif isinstance(el, go.layout.Shape):
                fig_offset.add_shape(el)
            elif isinstance(el, go.layout.Annotation):
                fig_offset.add_annotation(el)

    altura_base = 500
    height_auto = altura_base + offset_manual * len(df)

    fig_offset.update_layout(
        xaxis_title="[ppm]",
        yaxis_title="Offset + Intensidad",
        xaxis=dict(range=[x_max, x_min]),
        yaxis=dict(range=[y_min, y_max]),
        height=height_auto,
        showlegend=True,
        template="tesis",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    fig_offset.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"),
                            linecolor="black", tickcolor="black")
    fig_offset.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"),
                            linecolor="black", tickcolor="black")

    st.plotly_chart(fig_offset, use_container_width=True)



def generar_elementos_rmn(
    row,
    ajustes_y,
    normalizar,
    espectro_resta,
    id_resta,
    altura_min,
    distancia_min,
    correcciones_viscosidad,
    a_bib,
    b_bib,
    filas_senales,
    filas_biblio,
    tipo,
    y_max,
    aplicar_sombra_senales=False,
    aplicar_sombra_biblio=False,
    mostrar_picos=False,
    filas_dt2=None,
    check_d_por_espectro=None,
    check_t2_por_espectro=None, 
    color=None 
):
    nombre_archivo = row["archivo"]
    elementos = []  # lista de go.Scatter, go.Shape, go.Annotation

    archivo_actual = row["archivo"]
    df_esp = decodificar_csv_o_excel(row.get("contenido"), archivo_actual)
    if df_esp is None or df_esp.empty:
        return elementos

    df_aux = df_esp.rename(columns={df_esp.columns[0]: "x", df_esp.columns[1]: "y"}).dropna()

    a_v, b_v = correcciones_viscosidad.get(archivo_actual, (1.0, 0.0))
    x_vals = a_bib * (a_v * df_aux["x"] + b_v) + b_bib
    y_vals = df_aux["y"] + ajustes_y.get(archivo_actual, 0.0)

    if espectro_resta is not None:
        # Aplicar corrección de viscosidad al espectro que se resta (si corresponde)
        a_v_resta, b_v_resta = correcciones_viscosidad.get(id_resta, (1.0, 0.0))
        x_resta_corr = a_bib * (a_v_resta * espectro_resta["x"] + b_v_resta) + b_bib

        # Interpolar sobre el eje ya corregido
        y_resta_interp = np.interp(x_vals, x_resta_corr, espectro_resta["y"])
        y_vals = y_vals - (y_resta_interp + ajustes_y.get(id_resta, 0.0))

    if normalizar:
        y_vals = y_vals / y_vals.max() if y_vals.max() != 0 else y_vals

    elementos.append(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=archivo_actual,
    line=dict(color=color, width=2)))

    # --- Picos ---
    if mostrar_picos and altura_min is not None and distancia_min is not None:
        try:
            peaks, _ = find_peaks(y_vals, height=altura_min, distance=distancia_min)
            for p in peaks:
                elementos.append(go.Scatter(
                    x=[x_vals.iloc[p]],
                    y=[y_vals.iloc[p]],
                    mode="markers+text",
                    marker=dict(color="black", size=6),
                    text=[f"{x_vals.iloc[p]:.2f}"],
                    textposition="top center",
                    showlegend=False
                ))
        except:
            pass

    # --- Sombra señales ---
    if aplicar_sombra_senales:
        for f in filas_senales:
            if f.get("Archivo") != archivo_actual:
                continue
            x1, x2 = f.get("X min"), f.get("X max")
            grupo, obs = f.get("Grupo funcional"), f.get("Observaciones")
            if x1 is None or x2 is None:
                continue
            elementos.append(go.layout.Shape(
                type="rect",
                x0=min(x1, x2), x1=max(x1, x2),
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(0,255,0,0.3)", line_width=0
            ))
            etiqueta = " | ".join(filter(None, [grupo, obs]))
            if etiqueta:
                elementos.append(go.layout.Annotation(
                    x=(x1 + x2)/2, y=y_max*0.98,
                    text=etiqueta[:20] + ("..." if len(etiqueta) > 20 else ""),
                    showarrow=False,
                    textangle=270,
                    font=dict(size=10),
                    xanchor="center", yanchor="top"
                ))

    # --- Sombra bibliográfica ---
    if aplicar_sombra_biblio:
        for f in filas_biblio:
            delta = f.get("δ pico")
            grupo = f.get("Grupo funcional", "")
            obs = f.get("Observaciones", "")
            if delta is None:
                continue
            etiqueta = " | ".join(filter(None, [grupo, obs]))
            elementos.append(go.layout.Shape(
                type="line", x0=delta, x1=delta,
                y0=0, y1=y_max * 0.8,
                line=dict(color="black", dash="dot", width=1),
                xref="x", yref="y"
            ))
            elementos.append(go.layout.Annotation(
                x=delta,
                y=y_max * 0.8,
                text=etiqueta[:20] + ("..." if len(etiqueta) > 20 else ""),
                showarrow=False,
                textangle=270,
                font=dict(size=10),
                xanchor="center",
                yanchor="bottom"
            ))

    # --- Sombreado D/T2 ---
    if filas_dt2 and (check_d_por_espectro or check_t2_por_espectro):
        for f in filas_dt2:
            if f.get("Archivo") != archivo_actual:
                continue

            x1 = f.get("X min")
            x2 = f.get("X max")
            d_val = f.get("D")
            t2_val = f.get("T2")

            tiene_d = d_val not in [None, ""]
            tiene_t2 = t2_val not in [None, ""]

            mostrar_d = check_d_por_espectro.get(archivo_actual) and tiene_d if check_d_por_espectro else tiene_d
            mostrar_t2 = check_t2_por_espectro.get(archivo_actual) and tiene_t2 if check_t2_por_espectro else tiene_t2

            if not (mostrar_d or mostrar_t2) or x1 is None or x2 is None:
                continue

            partes = []
            if mostrar_d:
                partes.append(f"D = {float(d_val):.2e}")
            if mostrar_t2:
                partes.append(f"T2 = {float(t2_val):.3f}")
            etiqueta = "   ".join(partes)

            color = "rgba(128,128,255,0.3)" if mostrar_d and mostrar_t2 else (
                "rgba(255,0,0,0.3)" if mostrar_d else "rgba(0,0,255,0.3)")

            elementos.append(go.layout.Shape(
                type="rect",
                x0=min(x1, x2),
                x1=max(x1, x2),
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor=color,
                line_width=0
            ))

            elementos.append(go.layout.Annotation(
                x=(x1 + x2) / 2,
                y=y_max * 0.98,
                text=etiqueta,
                showarrow=False,
                font=dict(size=10, color="black"),
                textangle=270,
                xanchor="center",
                yanchor="top"
            ))

    return elementos


def mostrar_graficos_individuales(
    df, tipo, key_sufijo,
    normalizar, y_max, y_min, x_max, x_min,
    ajustes_y,
    aplicar_sombra_dt2,
    aplicar_sombra_senales,
    aplicar_sombra_biblio,
    db,
    check_d_por_espectro=None,
    check_t2_por_espectro=None,
    correcciones_viscosidad={},
    a_bib=1.0,
    b_bib=0.0,
    mostrar_picos=False,
    restar_espectro=False,
    seleccion_resta=None,
    altura_min=None,
    distancia_min=None
):

    tipo_doc = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
    filas_senales = db.collection("tablas_integrales").document(tipo_doc).get().to_dict().get("filas", [])
    filas_biblio = db.collection("configuracion_global").document(
        "tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c"
    ).get().to_dict().get("filas", [])

    espectro_resta = None
    id_resta = None
    if restar_espectro and seleccion_resta:
        id_resta = seleccion_resta.split(" – ")[-1].strip()
        fila_resta = df[df["archivo"] == id_resta].iloc[0] if id_resta in set(df["archivo"]) else None
        if fila_resta is not None:
            espectro_resta = decodificar_csv_o_excel(fila_resta["contenido"], fila_resta["archivo"])

    for _, row in df.iterrows():
        archivo_actual = row["archivo"]
        muestra_actual = row["muestra"]

        doc_dt2 = db.collection("muestras").document(muestra_actual).collection("dt2").document(tipo.lower()).get()
        filas_dt2 = doc_dt2.to_dict().get("filas", []) if doc_dt2.exists else []

        fig = go.Figure()
        elementos = generar_elementos_rmn(
            row=row,
            ajustes_y=ajustes_y,
            normalizar=normalizar,
            espectro_resta=espectro_resta,
            id_resta=id_resta,
            altura_min=altura_min,
            distancia_min=distancia_min,
            correcciones_viscosidad=correcciones_viscosidad,
            a_bib=a_bib,
            b_bib=b_bib,
            filas_senales=filas_senales,
            filas_biblio=filas_biblio,
            tipo=tipo,
            y_max=y_max,
            aplicar_sombra_senales=aplicar_sombra_senales,
            aplicar_sombra_biblio=aplicar_sombra_biblio,
            mostrar_picos=mostrar_picos,
            filas_dt2=filas_dt2,
            check_d_por_espectro=check_d_por_espectro,
            check_t2_por_espectro=check_t2_por_espectro,
        )
        for el in elementos:
            if isinstance(el, go.Scatter):
                fig.add_trace(el)
            elif isinstance(el, go.layout.Shape):
                fig.add_shape(el)
            elif isinstance(el, go.layout.Annotation):
                fig.add_annotation(el)

        fig.update_layout(
            title=archivo_actual,
            xaxis_title="[ppm]",
            yaxis_title="Intensidad",
            xaxis=dict(range=[x_max, x_min]),
            yaxis=dict(range=[y_min, y_max]),
            height=500,
            template="tesis"
        )
        fig.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"),
                 linecolor="black", tickcolor="black")
        fig.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"),
                        linecolor="black", tickcolor="black")

        st.plotly_chart(fig, use_container_width=True)



def mostrar_sombreados_dt2(fig, df, tipo, y_max, key_sufijo, check_d_por_espectro, check_t2_por_espectro, db):
    for _, row in df.iterrows():
        muestra_actual = row["muestra"]
        archivo_actual = row["archivo"]
        doc_dt2 = db.collection("muestras").document(muestra_actual).collection("dt2").document(tipo.lower())
        if doc_dt2.get().exists:
            filas_dt2 = doc_dt2.get().to_dict().get("filas", [])
            for f in filas_dt2:
                if f.get("Archivo") != archivo_actual:
                    continue

                x1 = f.get("X min")
                x2 = f.get("X max")
                d_val = f.get("D")
                t2_val = f.get("T2")

                tiene_d = d_val not in [None, ""]
                tiene_t2 = t2_val not in [None, ""]

                mostrar_d = check_d_por_espectro.get(archivo_actual) and tiene_d
                mostrar_t2 = check_t2_por_espectro.get(archivo_actual) and tiene_t2

                if not (mostrar_d or mostrar_t2) or x1 is None or x2 is None:
                    continue

                partes = []
                if mostrar_d:
                    partes.append(f"D = {float(d_val):.2e}")
                if mostrar_t2:
                    partes.append(f"T2 = {float(t2_val):.3f}")
                etiqueta = "   ".join(partes)

                color = "rgba(128,128,255,0.3)" if mostrar_d and mostrar_t2 else (
                    "rgba(255,0,0,0.3)" if mostrar_d else "rgba(0,0,255,0.3)")

                fig.add_vrect(
                    x0=min(x1, x2),
                    x1=max(x1, x2),
                    fillcolor=color,
                    line_width=0
                )

                fig.add_annotation(
                    x=(x1 + x2) / 2,
                    y=y_max * 0.98,
                    text=etiqueta,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    textangle=270,
                    xanchor="center",
                    yanchor="top"
                )


def render_rmn_1h_d(df_tipo, db):
    if df_tipo.empty:
        st.info("No hay espectros RMN 1H D disponibles.")
        return

    # usar espectros ya seleccionados
    espectros_seleccionados = df_tipo["archivo"].tolist()

    if not espectros_seleccionados:
        st.info("Selecciona al menos un espectro para ver el mapa.")
        return

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

    st.markdown("**Modificar nivel**")
    niveles_contorno = {}
    cols = st.columns(5)
    for idx, nombre in enumerate(espectros_seleccionados):
        col = cols[idx % 5]
        muestra_base = nombre.split("_RMN")[0]
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

    fig = go.Figure()
    color_idx = 0
    zonas_por_archivo = {}
    # --- Cuantificar zonas primero y guardar para cada archivo ---
    for nombre_archivo in espectros_seleccionados:
        zonas_por_archivo[nombre_archivo] = []
        with st.expander(f"🧮 Cuantificar zonas en {nombre_archivo}"):
            # --- Leer valores previos de Firestore ---
            doc_zonas = db.collection("zonas_rmn").document(nombre_archivo).get()
            if doc_zonas.exists:
                data_zonas = doc_zonas.to_dict()
                n_zonas_default = data_zonas.get("n_zonas", 1)
                zonas_default = data_zonas.get("zonas", [])
            else:
                n_zonas_default = 1
                zonas_default = []

            cuantificar = st.checkbox(
                f"Activar cuantificación de zonas para {nombre_archivo}",
                key=f"chk_cuant_{nombre_archivo}"
            )
            if cuantificar:
                n_zonas = st.number_input(
                    "Cantidad de zonas a definir",
                    min_value=1,
                    max_value=10,
                    value=n_zonas_default,
                    step=1,
                    key=f"nzonas_{nombre_archivo}"
                )
                zonas = []
                for i in range(int(n_zonas)):
                    st.markdown(f"**Zona {i+1}**")
                    col1, col2, col3, col4 = st.columns(4)
                    zona_prev = zonas_default[i] if i < len(zonas_default) else {}
                    with col1:
                        x_min_z = st.number_input("X min", value=float(zona_prev.get("x_min", 0.0)), key=f"xmin_{nombre_archivo}_{i}")
                    with col2:
                        x_max_z = st.number_input("X max", value=float(zona_prev.get("x_max", 9.0)), key=f"xmax_{nombre_archivo}_{i}")
                    with col3:
                        y_min_z = st.number_input("Y min", value=float(zona_prev.get("y_min", 1e-13)), format="%.1e", key=f"ymin_{nombre_archivo}_{i}")
                    with col4:
                        y_max_z = st.number_input("Y max", value=float(zona_prev.get("y_max", 1e-9)), format="%.1e", key=f"ymax_{nombre_archivo}_{i}")
                    zonas.append({
                        "x_min": x_min_z,
                        "x_max": x_max_z,
                        "y_min": y_min_z,
                        "y_max": y_max_z
                    })
                zonas_por_archivo[nombre_archivo] = zonas

                # --- Guardado automático si hay cambios ---
                key_guardado = f"zonas_guardadas_{nombre_archivo}"
                zonas_guardadas = st.session_state.get(key_guardado, None)
                datos_actuales = {"n_zonas": int(n_zonas), "zonas": zonas}
                if zonas_guardadas != datos_actuales:
                    db.collection("zonas_rmn").document(nombre_archivo).set(datos_actuales)
                    st.session_state[key_guardado] = datos_actuales
                    st.info("Zonas guardadas automáticamente.", icon="💾")

                # --- Gráficos individuales y tabla de cálculos para cada zona ---
                fila = df_tipo[df_tipo["archivo"] == nombre_archivo].iloc[0]
                url = fila.get("url_archivo")
                if url:
                    try:
                        response = requests.get(url)
                        if response.status_code == 200:
                            df = pd.read_csv(io.StringIO(response.text), sep="\t")
                        else:
                            st.error(f"No se pudo leer el archivo en {url}")
                            df = None
                    except Exception as e:
                        st.warning(f"Error leyendo {nombre_archivo}: {e}")
                        df = None
                    if df is not None:
                        x = pd.to_numeric(df.columns[1:], errors="coerce")
                        x = x[~pd.isna(x)]
                        y_raw = df.iloc[:, 0].astype(float)
                        z = df.iloc[:, 1:len(x)+1].values
                        proy1d_ex = np.sum(z, axis=0)
                        x_ex = x
                        for idx_zona, zona in enumerate(zonas):
                            x_min = zona["x_min"]
                            x_max = zona["x_max"]
                            y_min = zona["y_min"]
                            y_max = zona["y_max"]
                            mask_x = (x >= x_min) & (x <= x_max)
                            y_scaled = y_min_scale * (y_max_scale / y_min_scale) ** y_raw
                            mask_y = (y_scaled >= y_min) & (y_scaled <= y_max)
                            idx_x = np.where(mask_x)[0]
                            idx_y = np.where(mask_y)[0]
                            if len(idx_x) == 0 or len(idx_y) == 0:
                                st.warning(f"⚠️ Zona {idx_zona+1} no tiene datos en rango")
                                continue
                            z_recorte = z[np.ix_(idx_y, idx_x)]
                            proy1d = np.sum(z_recorte, axis=0)
                            # Usar variables locales para cada zona
                            x_zona = np.array(x)[:min(len(x), len(proy1d))]
                            proy1d_zona = np.array(proy1d)[:min(len(x), len(proy1d))]
                            x_ex_zona = np.array(x_ex)[:min(len(x_ex), len(proy1d_ex))]
                            proy1d_ex_zona = np.array(proy1d_ex)[:min(len(x_ex), len(proy1d_ex))]
                            columnas_zona = [
                                "Muestra", "Grupo funcional", "X min", "X max", "Área",
                                "Xas min", "Xas max", "Has", "Área as", "H", "🔴ex2dH", "🔴ex1dH", "Observaciones", "Archivo"
                            ]
                            try:
                                muestra_base = nombre_archivo.split("_RMN")[0]
                                nombre_doc = f"{nombre_archivo}_zona_{idx_zona+1}"
                                doc_ref = db.collection("muestras").document(muestra_base).collection("zonas").document(nombre_doc)
                                datos = doc_ref.get().to_dict()
                                if datos and "filas" in datos:
                                    df_zona = pd.DataFrame(datos["filas"])
                                else:
                                    df_zona = pd.DataFrame([{col: None for col in columnas_zona}])
                            except Exception as e:
                                st.warning(f"No se pudo recuperar tabla previa: {e}")
                                df_zona = pd.DataFrame([{col: None for col in columnas_zona}])
                            for col in columnas_zona:
                                if col not in df_zona.columns:
                                    df_zona[col] = "" if col in ["Grupo funcional", "Observaciones"] else None
                            df_zona = df_zona[columnas_zona]
                            df_zona["Muestra"] = muestra_base
                            df_zona["Archivo"] = nombre_archivo
                            key_tabla = f"tabla_zona_{nombre_archivo}_{idx_zona}"
                            if key_tabla not in st.session_state:
                                st.session_state[key_tabla] = df_zona.copy()
                            if not isinstance(st.session_state[key_tabla], pd.DataFrame):
                                st.session_state[key_tabla] = pd.DataFrame(st.session_state[key_tabla])
                            
                            # Obtener espectro 1D puro para comparación
                            try:
                                df_espectros_muestra = precargar_espectros_rmn(db, [{"nombre": muestra_base}])
                                df_1d_candidates = df_espectros_muestra[
                                    (df_espectros_muestra["tipo"].str.contains("1H")) &
                                    (~df_espectros_muestra["tipo"].str.contains("D|T2"))
                                ]
                                if not df_1d_candidates.empty:
                                    row_espectro_1d = df_1d_candidates.iloc[0]
                                    df_1d_puro = decodificar_csv_o_excel(row_espectro_1d["contenido"], row_espectro_1d["archivo"])
                                    if df_1d_puro is not None:
                                        x_1d = df_1d_puro["x"].values
                                        y_1d = df_1d_puro["y"].values
                                    else:
                                        x_1d = None
                                        y_1d = None
                                else:
                                    x_1d = None
                                    y_1d = None
                            except Exception as e:
                                st.warning(f"No se pudo obtener espectro 1D puro: {e}")
                                x_1d = None
                                y_1d = None
                            
                            def recalcular_tabla_zona(df_zona, x, proy1d, x_ex, proy1d_ex, x_1d=None, y_1d=None):
                                df_calc = df_zona.copy()
                                for i, row in df_calc.iterrows():
                                    try:
                                        def to_float(val):
                                            try:
                                                return float(pd.to_numeric(val, errors="coerce"))
                                            except Exception:
                                                return float('nan')
                                        x_min = to_float(row.get("X min"))
                                        x_max = to_float(row.get("X max"))
                                        xas_min = to_float(row.get("Xas min"))
                                        xas_max = to_float(row.get("Xas max"))
                                        has = to_float(row.get("Has"))
                                        if np.isnan(x_min) or np.isnan(x_max) or np.isnan(xas_min) or np.isnan(xas_max) or np.isnan(has):
                                            df_calc.at[i, "Área"] = None
                                            df_calc.at[i, "Área as"] = None
                                            df_calc.at[i, "H"] = None
                                            df_calc.at[i, "🔴ex2dH"] = None
                                            df_calc.at[i, "🔴ex1dH"] = None
                                            continue
                                        # Calcular denominador: área de referencia usando proyección 1D del 2D completo (no recortado por zona)
                                        mask_xas_ex = (x_ex >= xas_min) & (x_ex <= xas_max)
                                        area_as = np.trapz(proy1d_ex[mask_xas_ex], x_ex[mask_xas_ex]) if np.any(mask_xas_ex) else None
                                        # Cálculo usando proyección 1D del 2D (recortada) para H
                                        mask_x = (x >= x_min) & (x <= x_max)
                                        area = np.trapz(proy1d[mask_x], x[mask_x]) if np.any(mask_x) else None
                                        df_calc.at[i, "Área"] = round(float(area), 2) if area is not None else None
                                        # Cálculo usando proyección 1D del 2D (completa) para ex2dH
                                        mask_x_ex = (x_ex >= x_min) & (x_ex <= x_max)
                                        area_ex = np.trapz(proy1d_ex[mask_x_ex], x_ex[mask_x_ex]) if np.any(mask_x_ex) else None
                                        # Cálculo usando espectro 1D puro para ex1dH
                                        if x_1d is not None and y_1d is not None:
                                            mask_x_1d = (x_1d >= x_min) & (x_1d <= x_max)
                                            area_1d = np.trapz(y_1d[mask_x_1d], x_1d[mask_x_1d]) if np.any(mask_x_1d) else None
                                            mask_xas_1d = (x_1d >= xas_min) & (x_1d <= xas_max)
                                            area_as_1d = np.trapz(y_1d[mask_xas_1d], x_1d[mask_xas_1d]) if np.any(mask_xas_1d) else None
                                        else:
                                            area_1d = None
                                            area_as_1d = None
                                        # Ahora, todos usan el mismo denominador area_as
                                        df_calc.at[i, "Área as"] = round(float(area_as), 2) if area_as is not None else None
                                        if area is not None and area_as is not None and area_as != 0 and not np.isnan(has):
                                            h = (float(area) * has) / float(area_as)
                                            df_calc.at[i, "H"] = round(h, 2)
                                        else:
                                            df_calc.at[i, "H"] = None
                                        if area_1d is not None and area_as_1d is not None and area_as_1d != 0 and not np.isnan(has):
                                            ex1dH = (float(area_1d) * has) / float(area_as_1d)
                                            df_calc.at[i, "🔴ex1dH"] = round(ex1dH, 2)
                                        else:
                                            df_calc.at[i, "🔴ex1dH"] = None
                                        if area_ex is not None and area_as is not None and area_as != 0 and not np.isnan(has):
                                            ex2dH = (float(area_ex) * has) / float(area_as)
                                            df_calc.at[i, "🔴ex2dH"] = round(ex2dH, 2)
                                        else:
                                            df_calc.at[i, "🔴ex2dH"] = None
                                    except Exception as e:
                                        df_calc.at[i, "Área"] = None
                                        df_calc.at[i, "Área as"] = None
                                        df_calc.at[i, "H"] = None
                                        df_calc.at[i, "🔴ex2dH"] = None
                                        df_calc.at[i, "🔴ex1dH"] = None
                                return df_calc
                            df_zona_actualizada = recalcular_tabla_zona(
                                st.session_state[key_tabla],
                                x_zona, proy1d_zona, x_ex_zona, proy1d_ex_zona, x_1d, y_1d
                            )
                            st.session_state[key_tabla] = df_zona_actualizada
                            fig_proy = go.Figure()
                            fig_proy.add_trace(go.Scatter(
                                x=x_ex_zona,
                                y=proy1d_ex_zona,
                                mode="lines",
                                name="Espectro completo",
                                line=dict(color="black", width=1, dash="solid"),
                                opacity=0.5
                            ))
                            # Usar la máscara booleana para graficar la zona
                            if len(idx_x) > 0:
                                fig_proy.add_trace(go.Scatter(
                                    x=x[mask_x],
                                    y=proy1d,
                                    mode="lines",
                                    name=f"Zona {idx_zona+1}",
                                    line=dict(width=2)
                                ))
                            else:
                                st.warning(f"⚠️ Zona {idx_zona+1} no tiene datos en el rango seleccionado para X")
                            fig_proy.update_layout(
                                title=f"Proyección 1D de Zona {idx_zona+1} en {nombre_archivo}",
                                xaxis_title="ppm",
                                yaxis_title="Intensidad integrada",
                                height=300
                            )
                            fig_proy.update_xaxes(autorange="reversed")
                            st.plotly_chart(fig_proy, use_container_width=True)
                            st.markdown(f"**📈 Tabla de integrales para Zona {idx_zona+1}**")
                            for col in columnas_zona:
                                if col not in st.session_state[key_tabla].columns:
                                    st.session_state[key_tabla][col] = "" if col in ["Grupo funcional", "Observaciones"] else None
                            st.session_state[key_tabla] = st.session_state[key_tabla][columnas_zona]
                            st.session_state[key_tabla]["Muestra"] = muestra_base
                            st.session_state[key_tabla]["Archivo"] = nombre_archivo
                            col_config = {
                                "Muestra": st.column_config.TextColumn(disabled=True),
                                "Grupo funcional": st.column_config.SelectboxColumn(options=GRUPOS_FUNCIONALES),
                                "X min": st.column_config.NumberColumn(format="%.2f"),
                                "X max": st.column_config.NumberColumn(format="%.2f"),
                                "Área": st.column_config.NumberColumn(format="%.2f", label="🔴Área", disabled=True),
                                "Xas min": st.column_config.NumberColumn(format="%.2f"),
                                "Xas max": st.column_config.NumberColumn(format="%.2f"),
                                "Has": st.column_config.NumberColumn(format="%.2f"),
                                "Área as": st.column_config.NumberColumn(format="%.2f", label="🔴Área as", disabled=True),
                                "H": st.column_config.NumberColumn(format="%.2f", label="🔴H", disabled=True),
                                "🔴ex2dH": st.column_config.NumberColumn(format="%.2f", label="🔴ex2dH", disabled=True),
                                "🔴ex1dH": st.column_config.NumberColumn(format="%.2f", label="🔴ex1dH", disabled=True),
                                "Observaciones": st.column_config.TextColumn(),
                                "Archivo": st.column_config.TextColumn(disabled=True),
                            }
                            with st.form(f"form_zona_{nombre_archivo}_{idx_zona}"):
                                df_editable = st.data_editor(
                                    st.session_state[key_tabla],
                                    column_config=col_config,
                                    hide_index=True,
                                    use_container_width=True,
                                    num_rows="dynamic",
                                    key=f"tabla_zona_widget_{nombre_archivo}_{idx_zona}"
                                )
                                col1, col2 = st.columns(2)
                                with col1:
                                    recalcular = st.form_submit_button("🔴 Recalcular integrales")
                                with col2:
                                    guardar = st.form_submit_button("💾 Guardar integrales")
                                if recalcular:
                                    df_editable["Muestra"] = muestra_base
                                    df_editable["Archivo"] = nombre_archivo
                                    df_zona_actualizada = recalcular_tabla_zona(df_editable, x, proy1d, x_ex, proy1d_ex, x_1d, y_1d)
                                    df_zona_actualizada = df_zona_actualizada[columnas_zona]
                                    st.session_state[key_tabla] = df_zona_actualizada
                                if guardar:
                                    try:
                                        df_editable["Muestra"] = muestra_base
                                        df_editable["Archivo"] = nombre_archivo
                                        df_para_guardar = df_editable[columnas_zona].copy()
                                        muestra_base = nombre_archivo.split("_RMN")[0]
                                        nombre_doc = f"{nombre_archivo}_zona_{idx_zona+1}"
                                        doc_ref = db.collection("muestras").document(muestra_base).collection("zonas").document(nombre_doc)
                                        doc_ref.set({"filas": df_para_guardar.to_dict(orient="records")})
                                        st.success(f"✅ Guardado correcto para {nombre_doc}")
                                    except Exception as e:
                                        st.error(f"❌ Error al guardar: {e}")

    # --- Graficar espectros y agregar shapes de zonas ---
    for nombre_archivo in espectros_seleccionados:
        color = colores[color_idx % len(colores)]
        fila = df_tipo[df_tipo["archivo"] == nombre_archivo].iloc[0]
        url = fila.get("url_archivo")
        if not url:
            st.warning(f"No se encontró la URL de {nombre_archivo}")
            continue
        try:
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), sep="\t")
            else:
                st.error(f"No se pudo leer el archivo en {url}")
                continue
        except Exception as e:
            st.warning(f"Error leyendo {nombre_archivo}: {e}")
            continue
        x = pd.to_numeric(df.columns[1:], errors="coerce")
        x = x[~pd.isna(x)]
        y_raw = df.iloc[:, 0].astype(float)
        z = df.iloc[:, 1:len(x)+1].values
        y_scaled = y_min_scale * (y_max_scale / y_min_scale) ** y_raw
        nivel_contorno = niveles_contorno.get(nombre_archivo, 0.10)
        fig.add_trace(go.Contour(
            x=x,
            y=y_scaled,
            z=z,
            contours=dict(
                coloring="lines",
                start=nivel_contorno,
                end=nivel_contorno,
                size=0.1,
                showlabels=False
            ),
            line=dict(color=color, width=1.5),
            showscale=False,
            name=f"{muestra_base}",
            hoverinfo="x+y+name",
            colorscale=[[0, color],[1, color]],
        ))
        # Agregar shapes de zonas para este archivo
        for zona in zonas_por_archivo.get(nombre_archivo, []):
            fig.add_shape(
                type="rect",
                x0=zona["x_min"], x1=zona["x_max"],
                y0=zona["y_min"], y1=zona["y_max"],
                xref="x", yref="y",
                fillcolor="rgba(255,0,0,0.2)",
                line_width=0,
                layer="above"
            )
        color_idx += 1

    fig.update_layout(
        title="Mapa 2D RMN 1H D",
        xaxis_title="F2 (ppm)",
        yaxis_title="F1 (s⁻¹ o m²/s)",
        height=700,
        xaxis=dict(
            autorange=False,
            range=[x_max, x_min],
            showgrid=True, gridcolor="#e0e0e0",
            zeroline=False,
            linecolor="black"
        ),
        yaxis=dict(
            type="log",
            showticklabels=False,
            showgrid=True, gridcolor="#e0e0e0",
            zeroline=False,
            linecolor="black",
            range=[np.log10(y_min_axis), np.log10(y_max_axis)]
        ),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="white",
            bordercolor="black"
        ),
        template="tesis"
    )

    fig.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", tickcolor="black")
    fig.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", tickcolor="black")
    st.plotly_chart(fig, use_container_width=True)

    # leyenda en 4 columnas
    referencias = []
    for idx, nombre in enumerate(espectros_seleccionados):
        muestra_base = nombre.split("_RMN")[0]
        color = colores[idx % len(colores)]
        nivel = niveles_contorno[nombre]
        referencias.append(f"<span style='color:{color};font-weight:bold'>—</span> {muestra_base} (nivel {nivel:.2f})")

    cols = st.columns(4)
    for i, ref in enumerate(referencias):
        col = cols[i % 4]
        col.markdown(ref, unsafe_allow_html=True)



def render_rmn_1h_t2(df_tipo):
    if df_tipo.empty:
        st.info("No hay espectros RMN 1H T2 disponibles.")
        return

    # Bloque de niveles de contorno en columnas, como RMN 1H D
    st.markdown("**Modificar nivel**")
    cols = st.columns(5)
    niveles_contorno = {}
    for idx, nombre_archivo in enumerate(df_tipo["archivo"]):
        col = cols[idx % 5]
        muestra_base = nombre_archivo.split("_RMN")[0]
        with col:
            nivel = st.number_input(
                f"{muestra_base}",
                min_value=0.01,
                max_value=1.0,
                value=0.10,
                format="%.2f",
                key=f"nivel_{nombre_archivo}"
            )
            niveles_contorno[nombre_archivo] = nivel

    # Inicializar gráficos combinados
    fig2d = go.Figure()
    fig1d = go.Figure()

    color_idx = 0

    for _, fila in df_tipo.iterrows():
        color = colores[color_idx % len(colores)]
        nombre_archivo = fila["archivo"]
        archivos = fila.get("archivos", {})
        if not archivos:
            st.warning(f"No hay archivos asociados en {nombre_archivo}")
            continue

        try:
            ppm_data = requests.get(archivos["ppmAxis"]).text
            T2axis_data = requests.get(archivos["T2axis"]).text
            T2_proy_data = requests.get(archivos["T2_proy"]).text
            ILT2D_data = requests.get(archivos["ILT2D"]).text

            ppmAxis = np.loadtxt(io.StringIO(ppm_data))
            T2axis = np.loadtxt(io.StringIO(T2axis_data))
            T2_proy = np.loadtxt(io.StringIO(T2_proy_data))
            ILT2D = np.loadtxt(io.StringIO(ILT2D_data))

        except Exception as e:
            st.warning(f"Error descargando archivos para {nombre_archivo}: {e}")
            continue

        # nivel ya capturado con columnas
        nivel = niveles_contorno.get(nombre_archivo, 0.10)
        z = ILT2D.T

        # gráfico 2D combinado
        fig2d.add_trace(go.Contour(
            x=ppmAxis,
            y=T2axis,
            z=z,
            contours=dict(
                coloring="lines",
                start=nivel,
                end=nivel,
                size=0.1,
                showlabels=False
            ),
            line=dict(color=color, width=1.5),
            showscale=False,
            name=nombre_archivo,
            colorscale=[[0, color],[1, color]]
        ))

        # curva de decaimiento combinada
        fig1d.add_trace(go.Scatter(
            x=T2axis,
            y=T2_proy,
            mode="lines",
            name=nombre_archivo,
            line=dict(color=color, width=2)
        ))

        color_idx += 1

    # Mostrar gráficos combinados
    fig2d.update_layout(
        title="ILT2D",
        xaxis=dict(
            autorange=False,
            range=[9, 0],
            title="ppm"
        ),
        yaxis=dict(
            type="log",
            autorange=False,
            range=[
                np.log10(T2axis.min()),
                np.log10(T2axis.max())
            ],
            title="T2 (s)"
        ),
        height=600,
        legend=dict(orientation="h", x=0, y=-0.15),
        template="tesis"
    )

    fig2d.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", tickcolor="black")
    fig2d.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", tickcolor="black")
    st.plotly_chart(fig2d, use_container_width=True)

    fig1d.update_layout(
        title="Decaimiento T2",
        xaxis_title="T2 (s)",
        yaxis_title="Intensidad",
        xaxis_type="log",
        height=400,
        legend=dict(orientation="h", x=0, y=-0.2)
    )
    st.plotly_chart(fig1d, use_container_width=True)



    # Checkbox para gráficos individuales, debajo de los combinados
    mostrar_indiv = st.checkbox("Mostrar gráficos individuales", key="chk_indiv_rmn1h_t2")

    if mostrar_indiv:
        st.markdown("### Gráficos individuales")
        color_idx = 0
        for _, fila in df_tipo.iterrows():
            color = colores[color_idx % len(colores)]
            nombre_archivo = fila["archivo"]
            archivos = fila.get("archivos", {})
            if not archivos:
                continue
            try:
                ppm_data = requests.get(archivos["ppmAxis"]).text
                T2axis_data = requests.get(archivos["T2axis"]).text
                T2_proy_data = requests.get(archivos["T2_proy"]).text
                ILT2D_data = requests.get(archivos["ILT2D"]).text

                ppmAxis = np.loadtxt(io.StringIO(ppm_data))
                T2axis = np.loadtxt(io.StringIO(T2axis_data))
                T2_proy = np.loadtxt(io.StringIO(T2_proy_data))
                ILT2D = np.loadtxt(io.StringIO(ILT2D_data))

            except Exception as e:
                st.warning(f"Error descargando archivos para {nombre_archivo}: {e}")
                continue

            nivel = niveles_contorno.get(nombre_archivo, 0.10)
            z = ILT2D.T

            # gráfico 2D individual
            fig2d_indiv = go.Figure()
            fig2d_indiv.add_trace(go.Contour(
                x=ppmAxis,
                y=T2axis,
                z=z,
                contours=dict(
                    coloring="lines",
                    start=nivel,
                    end=nivel,
                    size=0.1,
                    showlabels=False
                ),
                line=dict(color=color, width=1.5),
                showscale=False,
                colorscale=[[0, color],[1, color]]
            ))
            fig2d_indiv.update_layout(
                title=f"ILT2D de {nombre_archivo}",
                xaxis=dict(
                    autorange=False,
                    range=[9, 0],
                    title="ppm"
                ),
                yaxis=dict(
                    type="log",
                    autorange=False,
                    range=[
                        np.log10(T2axis.min()),
                        np.log10(T2axis.max())
                    ],
                    title="T2 (s)"
                ),
                height=500,
                template="tesis"
            )

            fig2d_indiv.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", tickcolor="black")
            fig2d_indiv.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", tickcolor="black")
            st.plotly_chart(fig2d_indiv, use_container_width=True)

            # curva de decaimiento individual
            fig1d_indiv = go.Figure()
            fig1d_indiv.add_trace(go.Scatter(
                x=T2axis,
                y=T2_proy,
                mode="lines",
                name="Proyección T2",
                line=dict(color=color, width=2)
            ))
            fig1d_indiv.update_layout(
                title=f"Curva de decaimiento T2 de {nombre_archivo}",
                xaxis_title="T2 (s)",
                yaxis_title="Intensidad",
                xaxis_type="log",
                height=400
            )
            st.plotly_chart(fig1d_indiv, use_container_width=True)

            color_idx += 1



def render_tab6(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.session_state["current_tab"] = "Análisis RMN"

    # --- cargar todas las muestras ---
    muestras = cargar_muestras(db)
    if not muestras:
        st.warning("No hay muestras disponibles.")
        st.stop()

    # --- precargar espectros ---
    df_total = precargar_espectros_rmn(db, muestras)
    if df_total.empty:
        st.warning("No hay espectros RMN disponibles.")
        st.stop()

    # --- filtro 1: seleccionar muestras ---
    opciones_muestras = sorted(df_total["muestra"].unique())
    muestras_sel = st.multiselect(
        "Seleccionar muestras",
        opciones_muestras,
        default=opciones_muestras,
        key="muestras_rmn"
    )

    # --- filtro 2: tipos de espectro ---
    tipos_disponibles = sorted(df_total["tipo"].unique())
    tipos_sel = st.multiselect(
        "Tipos de espectro RMN",
        tipos_disponibles,
        default=tipos_disponibles,
        key="tipos_rmn"
    )

    # --- filtro 3: fechas ---
    fechas_disponibles = sorted(df_total["fecha"].unique())
    fechas_sel = st.multiselect(
        "Fechas",
        fechas_disponibles,
        default=fechas_disponibles,
        key="fechas_rmn"
    )

    # --- aplicar filtros ---
    df_filtrado = df_total[
        (df_total["muestra"].isin(muestras_sel)) &
        (df_total["tipo"].isin(tipos_sel)) &
        (df_total["fecha"].isin(fechas_sel))
    ]

    # --- filtro 4: espectros específicos ---
    opciones_archivos = [
        f"{row['muestra']} – {row['tipo']} – {row['archivo']}"
        for _, row in df_filtrado.iterrows()
    ]
    seleccion_archivos = st.multiselect(
        "Seleccionar espectros",
        opciones_archivos,
        key="archivos_rmn"
    )

    # crear mapa id para filtrar
    ids_map = {
        f"{row['muestra']} – {row['tipo']} – {row['archivo']}": row["id"]
        for _, row in df_filtrado.iterrows()
    }
    df_sel = df_filtrado[df_filtrado["id"].isin([ids_map.get(s) for s in seleccion_archivos])]

    # orden de aparición deseado
    orden_prioridad = [
        "RMN 1H",
        "RMN 1H D",
        "RMN 1H T2",
        "RMN-LF 1H",
        "RMN 1H imagen",
        "RMN 13C"
    ]

    for tipo in orden_prioridad:
        if tipo not in tipos_sel:
            continue

        df_tipo = df_sel[df_sel["tipo"] == tipo]
        if df_tipo.empty:
            continue

        # texto antes
        if tipo == "RMN 1H":
            st.markdown("""
            <span style="font-size: 0.85rem; text-align:center; display:block;">
            <br><br><b>RMN 1H: gráfico 1D de intensidad vs desplazamiento químico. Señal de cada grupo de protones en la muestra. Primera identificación química.</b>
            </span>
            """, unsafe_allow_html=True)

        if tipo == "RMN 1H D":
            st.markdown("""
            <span style="font-size: 0.85rem; text-align:center; display:block;">
            <br><br><b>RMN 1H LF: utiliza campos magnéticos más bajos. Los picos se ven más anchos y menos diferenciados pero permite estudiar difusión y relajación T2.</b>
            </span>
            """, unsafe_allow_html=True)

        if tipo == "RMN 1H T2":
            st.markdown("""
            <span style="font-size: 0.85rem; text-align:center; display:block;">
            <br><br><b>T2: tiempo de relajación transversal. Describe qué tan rápido desaparece la señal de un grupo de protones, relacionado con la movilidad molecular. En campo bajo T2 se mide con gran sensibilidad.</b>
            </span>
            """, unsafe_allow_html=True)

        if tipo == "RMN 13C":
            st.markdown("""
            <span style="font-size: 0.85rem; text-align:center; display:block;">
            <br><br><b>RMN 13C: gráfico 1D de intensidad vs desplazamiento químico. Señal de cada grupo de carbonos en la muestra. Complementa la identificación estructural junto al espectro de 1H.</b>
            </span>
            """, unsafe_allow_html=True)

        st.markdown(f"## 🧪 {tipo}")

        # texto después
        if tipo == "RMN 1H D":
            st.markdown("""
            <span style="font-size: 0.85rem;">
            <b>RMN 1H D:</b> gráfico 2D que muestra contornos de difusión vs desplazamiento químico. Las especies químicas se separan por el desplazamiento químico, y en estos gráficos se separan también por difusión. Así se pueden diferenciar compuestos que se solapan en espectros 1D (en campo bajo, aunque el espectro 1D se 'ensucie', la nueva dimensión 'difusión' ayuda a distinguir componentes, ayuda a caracterizar mezclas complejas).<br><br>
            </span>
            """, unsafe_allow_html=True)

        if tipo == "RMN 1H T2":
            st.markdown("""
            <span style="font-size: 0.85rem;">
            <b>RMN 1H T2 (ILT2D):</b> gráfico 2D que muestra contornos de T2 vs. desplazamiento químico. Ideal para separar señales que se solapan en un gráfico 1D. Combina selectividad química (ppm) con la dinámica molecular (T2) para resolver estructuras complejas.<br>
            <b>RMN 1H T2 (Decaimiento):</b> gráfico 1D intensidad vs tiempo de relajación T2. Cada pico indica cuántos protones tienen un cierto T2, reflejando cuan 'rígidos' o 'móviles' son. Mayor T2 indica mayor movilidad molecular (aceites) y un T2 corto indica estructuras más rígidas (polioles).<br><br>
            </span>
            """, unsafe_allow_html=True)

        # render
        if tipo == "RMN 1H":
            render_rmn_plot(df_tipo, tipo="RMN 1H", key_sufijo="rmn1h", db=db)
        elif tipo == "RMN 13C":
            render_rmn_plot(df_tipo, tipo="RMN 13C", key_sufijo="rmn13c", db=db)
        elif tipo == "RMN 1H D":
            render_rmn_1h_d(df_tipo, db)
        elif tipo == "RMN 1H T2":
            render_rmn_1h_t2(df_tipo)
        elif tipo == "RMN-LF 1H":
            render_rmn_plot(df_tipo, tipo="RMN-LF 1H", key_sufijo="rmnlf1h", db=db)
        elif tipo == "RMN 1H imagen":
            render_imagenes(df_tipo)

