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

# --- Configuraciones globales ---
GRUPOS_FUNCIONALES = ["Formiato", "Cloroformo", "C=C olefínicos", "Glicerol medio", "Glicerol extremos", "Metil-Éster", "Eter", "Ester", "Ácido carboxílico", "OH", "Epóxido", "C=C", "Alfa-C=O","Alfa-C-OH", "Alfa-C=C", "Vecino a alfa-carbonilo", "Alfa-epóxido", "CH2", "CH3"]

# --- Cacheo de espectros por archivo base64 ---
session_cache = {}

def decodificar_csv_o_excel(contenido_base64, archivo):
    clave_cache = f"{archivo}__{hash(contenido_base64)}"
    if clave_cache in session_cache:
        return session_cache[clave_cache]
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
                    "id": f"{nombre}__{i}"
                })
    return pd.DataFrame(espectros_total)



# --- Firebase helpers: precarga de documentos tipo tabla_integral o dt2 ---
def precargar_tabla_global(db, nombre_tabla):
    doc = db.collection("tablas_integrales").document(nombre_tabla).get()
    return doc.to_dict() if doc.exists else {}

def precargar_dt2_muestra(db, muestra, tipo):
    doc = db.collection("muestras").document(muestra).collection("dt2").document(tipo)
    data = doc.get().to_dict()
    return data.get("filas", []) if data else []

# --- Cálculo de integrales: usar espectros precargados ---
def obtener_df_esp_precargado(db, espectros_dict, muestra, archivo):
    espectro = espectros_dict.get(archivo)
    if not espectro:
        espectros_dict.update(precargar_espectros_por_muestra(db, muestra))
        espectro = espectros_dict.get(archivo)
    if not espectro:
        return None
    return decodificar_csv_o_excel(espectro.get("contenido"), archivo)

# --- Optimización aplicada a recálculo de D/T2 y señales ---
def recalcular_areas_y_guardar(df_edicion, tipo, db, nombre_tabla, tabla_destino="dt2"):
    espectros_cache = {}
    campo_h = "H" if tipo == "RMN 1H" else "C"
    campo_has = "Has" if tipo == "RMN 1H" else "Cas"

    for i, row in df_edicion.iterrows():
        try:
            muestra = row.get("Muestra")
            archivo = row.get("Archivo")
            if not muestra or not archivo:
                continue

            x_min = float(row.get("X min")) if row.get("X min") not in [None, ""] else None
            x_max = float(row.get("X max")) if row.get("X max") not in [None, ""] else None
            xas_min = float(row.get("Xas min")) if row.get("Xas min") not in [None, ""] else None
            xas_max = float(row.get("Xas max")) if row.get("Xas max") not in [None, ""] else None
            has_or_cas = float(row.get(campo_has)) if row.get(campo_has) not in [None, ""] else None

            df_esp = obtener_df_esp_precargado(db, espectros_cache.setdefault(muestra, {}), muestra, archivo)
            if df_esp is None:
                continue

            df_main = df_esp[(df_esp["x"] >= min(x_min, x_max)) & (df_esp["x"] <= max(x_min, x_max))]
            area = np.trapz(df_main["y"], df_main["x"]) if not df_main.empty else None
            df_edicion.at[i, "Área"] = round(area, 2) if area else None

            if xas_min is not None and xas_max is not None:
                df_as = df_esp[(df_esp["x"] >= min(xas_min, xas_max)) & (df_esp["x"] <= max(xas_min, xas_max))]
                area_as = np.trapz(df_as["y"], df_as["x"]) if not df_as.empty else None
                df_edicion.at[i, "Área as"] = round(area_as, 2) if area_as else None

                if area and area_as and has_or_cas and area_as != 0:
                    resultado = (area * has_or_cas) / area_as
                    df_edicion.at[i, campo_h] = round(resultado, 2)

        except Exception as e:
            st.warning(f"⚠️ Error en fila {i}: {e}")

    # Guardar en Firebase (conservar combinaciones no actualizadas)
    filas_actualizadas_raw = df_edicion.to_dict(orient="records")
    combinaciones_actualizadas = {(f.get("Muestra"), f.get("Archivo")) for f in filas_actualizadas_raw if f.get("Muestra") and f.get("Archivo")}

    if tabla_destino == "dt2":
        doc_destino = lambda m: db.collection("muestras").document(m).collection("dt2").document(tipo.lower())
    else:
        doc_ref = db.collection("tablas_integrales").document(nombre_tabla)
        doc_data = doc_ref.get().to_dict()
        filas_previas = doc_data.get("filas", []) if doc_data else []
        filas_conservadas = [f for f in filas_previas if (f.get("Muestra"), f.get("Archivo")) not in combinaciones_actualizadas]
        filas_finales = filas_conservadas + filas_actualizadas_raw
        doc_ref.set({"filas": filas_finales})
        return

    for muestra in df_edicion["Muestra"].unique():
        filas_m = [f for f in filas_actualizadas_raw if f.get("Muestra") == muestra]
        doc_out = doc_destino(muestra)
        doc_data = doc_out.get().to_dict()
        filas_previas = doc_data.get("filas", []) if doc_data else []
        archivos_actualizados = set(f["Archivo"] for f in filas_m if f.get("Archivo"))
        filas_conservadas = [f for f in filas_previas if f.get("Archivo") not in archivos_actualizados]
        filas_finales = filas_conservadas + filas_m
        doc_out.set({"filas": filas_finales})

    st.success("✅ Datos recalculados y guardados correctamente.")
    st.rerun()
    


def render_tab6(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    # --- Cargar muestras y espectros ---
    muestras = cargar_muestras(db)
    if not muestras:
        st.warning("No hay muestras disponibles.")
        st.stop()

    df_total = precargar_espectros_rmn(db, muestras)
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

    df_rmn1h = df_sel[df_sel["tipo"] == "RMN 1H"]
    if not df_rmn1h.empty:
        st.markdown("## 🧪 RMN 1H")
        render_rmn_plot(df_rmn1h, tipo="RMN 1H", key_sufijo="rmn1h", db=db)

    df_rmn13c = df_sel[df_sel["tipo"] == "RMN 13C"]
    if not df_rmn13c.empty:
        st.markdown("## 🧪 RMN 13C")
        render_rmn_plot(df_rmn13c, tipo="RMN 13C", key_sufijo="rmn13c", db=db)

    imagenes_sel = df_sel[df_sel["archivo"].str.lower().str.endswith((".png", ".jpg", ".jpeg"))]
    if not imagenes_sel.empty:
        st.markdown("## 🧪 RMN Imágenes")
        render_imagenes(imagenes_sel)




def render_rmn_plot(df, tipo="RMN 1H", key_sufijo="rmn1h", db=None):
    if df.empty:
        st.info(f"No hay espectros disponibles para {tipo}.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    normalizar = col1.checkbox("Normalizar", key=f"norm_{key_sufijo}")
    mostrar_picos = col2.checkbox("Detectar picos", key=f"picos_{key_sufijo}")
    restar_espectro = col3.checkbox("Restar espectro", key=f"resta_{key_sufijo}")
    ajuste_y_manual = col4.checkbox("Ajuste manual Y", key=f"ajuste_y_{key_sufijo}")
    superposicion_vertical = col5.checkbox("Superposición vertical de espectros", key=f"offset_{key_sufijo}")

    ajustes_y = {row["archivo"]: st.number_input(f"Y para {row['archivo']}", value=0.0, step=0.1, key=f"ajuste_y_val_{row['archivo']}")
                 for _, row in df.iterrows()} if ajuste_y_manual else {row["archivo"]: 0.0 for _, row in df.iterrows()}

    seleccion_resta = None
    if restar_espectro:
        opciones_restar = [f"{row['muestra']} – {row['archivo']}" for _, row in df.iterrows()]
        seleccion_resta = st.selectbox("Seleccionar espectro a restar:", opciones_restar, key=f"sel_resta_{key_sufijo}")

    colx1, colx2, coly1, coly2 = st.columns(4)
    x_min = colx1.number_input("X mínimo", value=0.0, key=f"x_min_{key_sufijo}")
    x_max = colx2.number_input("X máximo", value=9.0 if tipo == "RMN 1H" else 200.0, key=f"x_max_{key_sufijo}")
    y_min = coly1.number_input("Y mínimo", value=0.0, key=f"y_min_{key_sufijo}")
    y_max = coly2.number_input("Y máximo", value=80.0 if tipo == "RMN 1H" else 1.5, key=f"y_max_{key_sufijo}")

    if mostrar_picos:
        colp1, colp2 = st.columns(2)
        altura_min = colp1.number_input("Altura mínima", value=0.05, step=0.01, key=f"altura_min_{key_sufijo}")
        distancia_min = colp2.number_input("Distancia mínima entre picos", value=5, step=1, key=f"distancia_min_{key_sufijo}")
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

    fig = mostrar_grafico_combinado(
        df=df,
        tipo=tipo,
        key_sufijo=key_sufijo,
        normalizar=normalizar,
        mostrar_picos=mostrar_picos,
        restar_espectro=restar_espectro,
        seleccion_resta=seleccion_resta,
        ajustes_y=ajustes_y,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        altura_min=altura_min,
        distancia_min=distancia_min,
        aplicar_sombra_senales=aplicar_sombra_senales,
        aplicar_sombra_biblio=aplicar_sombra_biblio,
        db=db
    )

    if aplicar_sombra_dt2:
        mostrar_sombreados_dt2(fig, df, tipo, y_max, key_sufijo, check_d_por_espectro, check_t2_por_espectro, db)

    st.plotly_chart(fig, use_container_width=True)

    if mostrar_tabla_dt2_chk:
        mostrar_tabla_dt2(df, tipo, key_sufijo, db)

    if mostrar_tabla_senales_chk:
        mostrar_tabla_senales(df, tipo, key_sufijo, db)

    if mostrar_tabla_biblio_chk:
        mostrar_tabla_biblio(tipo, key_sufijo, db)

    if superposicion_vertical:
        mostrar_grafico_stacked(df, tipo, key_sufijo, normalizar, x_min, x_max, y_min, y_max)

    mostrar_indiv = st.checkbox("Gráficos individuales", key=f"chk_indiv_{key_sufijo}")
    if mostrar_indiv:
        mostrar_graficos_individuales(df, tipo, key_sufijo, normalizar, y_max, y_min, ajustes_y, aplicar_sombra_dt2, aplicar_sombra_senales, aplicar_sombra_biblio, db)



def mostrar_grafico_combinado(
    df,
    tipo,
    key_sufijo,
    db,
    normalizar,
    mostrar_picos,
    restar_espectro,
    seleccion_resta,
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
    distancia_min=None
):

    fig = go.Figure()
    espectro_resta = None

    # --- Decodificar espectro de fondo si aplica ---
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

    # --- Agregar trazas ---
    for _, row in df.iterrows():
        archivo_actual = row["archivo"]
        muestra_actual = row["muestra"]

        df_esp = decodificar_csv_o_excel(row.get("contenido"), archivo_actual)
        if df_esp is None:
            continue

        col_x, col_y = df_esp.columns[:2]
        y_data = df_esp[col_y].copy() + ajustes_y.get(archivo_actual, 0.0)

        if espectro_resta is not None:
            df_esp = df_esp.rename(columns={col_x: "x", col_y: "y"}).dropna()
            espectro_resta_interp = np.interp(df_esp["x"], espectro_resta["x"], espectro_resta["y"])
            y_data = df_esp["y"] - espectro_resta_interp

        if normalizar:
            y_data = y_data / y_data.max() if y_data.max() != 0 else y_data

        x_vals = df_esp[col_x]
        fig.add_trace(go.Scatter(x=x_vals, y=y_data, mode='lines', name=archivo_actual))

    fig.update_layout(
        xaxis_title="[ppm]",
        yaxis_title="Intensidad",
        xaxis=dict(range=[x_max, x_min]),
        yaxis=dict(range=[y_min, y_max] if y_max > y_min else None),
        template="simple_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)


def mostrar_tabla_dt2(df, tipo, key_sufijo, db):

    if tipo == "RMN 1H":
        columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                        "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]
    else:
        columnas_dt2 = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                        "Xas min", "Xas max", "Cas", "Área as", "C", "Observaciones", "Archivo"]
    filas_guardadas = []
    for _, row in df.iterrows():
        muestra = row["muestra"]
        archivo = row["archivo"]
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
        else:
            col_config["Cas"] = st.column_config.NumberColumn(format="%.2f")
            col_config["C"] = st.column_config.NumberColumn(format="%.2f", label="🔴C", disabled=True)

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
        recalcular_areas_y_guardar(df_dt2_edit, tipo, db, nombre_tabla, tabla_destino="dt2")


def mostrar_tabla_senales(df, tipo, key_sufijo, db):
    if tipo == "RMN 1H":
        columnas_senales = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                            "Xas min", "Xas max", "Has", "Área as", "H", "Observaciones", "Archivo"]
    else:
        columnas_senales = ["Muestra", "Grupo funcional", "δ pico", "X min", "X max", "Área", "D", "T2",
                            "Xas min", "Xas max", "Cas", "Área as", "C", "Observaciones", "Archivo"]

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
        else:
            col_config["Cas"] = st.column_config.NumberColumn(format="%.2f")
            col_config["C"] = st.column_config.NumberColumn(format="%.2f", label="🔴C", disabled=True)

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
        recalcular_areas_y_guardar(df_senales_edit, tipo, db, nombre_tabla, tabla_destino="senales")


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

def mostrar_grafico_stacked(df, tipo, key_sufijo, normalizar, x_min, x_max, y_min, y_max):
    offset_auto = round((y_max - y_min) / (len(df) + 1), 2) if (y_max is not None and y_min is not None and y_max > y_min) else 1.0
    offset_manual = st.slider(
        "Separación entre espectros (offset)",
        min_value=0.1,
        max_value=30.0,
        value=offset_auto,
        step=0.1,
        key=f"offset_val_{key_sufijo}"
    )
    fig_offset = go.Figure()
    step_offset = offset_manual

    for i, (_, row) in enumerate(df.iterrows()):
        archivo_actual = row["archivo"]
        df_esp = decodificar_csv_o_excel(row.get("contenido"), archivo_actual)
        if df_esp is None:
            continue

        col_x, col_y = df_esp.columns[:2]
        y_data = df_esp[col_y].copy()
        if normalizar:
            y_data = y_data / y_data.max() if y_data.max() != 0 else y_data

        offset = step_offset * i
        fig_offset.add_trace(go.Scatter(
            x=df_esp[col_x],
            y=y_data + offset,
            mode='lines',
            name=archivo_actual
        ))

    fig_offset.update_layout(
        xaxis_title="[ppm]",
        yaxis_title="Offset + Intensidad",
        xaxis=dict(range=[x_max, x_min]),
        height=500,
        showlegend=True,
        template="simple_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig_offset, use_container_width=True)


def mostrar_graficos_individuales(df, tipo, key_sufijo, normalizar, y_max, y_min, ajustes_y, aplicar_sombra_dt2, aplicar_sombra_senales, aplicar_sombra_biblio, db):
    for _, row in df.iterrows():
        archivo_actual = row["archivo"]
        muestra_actual = row["muestra"]

        df_esp = decodificar_csv_o_excel(row.get("contenido"), archivo_actual)
        if df_esp is None:
            continue

        col_x, col_y = df_esp.columns[:2]
        y_data = df_esp[col_y].copy() + ajustes_y.get(archivo_actual, 0.0)
        if normalizar:
            y_data = y_data / y_data.max() if y_data.max() != 0 else y_data
        x_vals = df_esp[col_x]

        fig_indiv = go.Figure()
        fig_indiv.add_trace(go.Scatter(x=x_vals, y=y_data, mode='lines', name=archivo_actual))

        if aplicar_sombra_dt2:
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

                    if not (tiene_d or tiene_t2) or x1 is None or x2 is None:
                        continue

                    partes = []
                    if tiene_d:
                        partes.append(f"D = {float(d_val):.2e}")
                    if tiene_t2:
                        partes.append(f"T2 = {float(t2_val):.3f}")
                    etiqueta = "   ".join(partes)

                    color = "rgba(128,128,255,0.3)" if tiene_d and tiene_t2 else (
                        "rgba(255,0,0,0.3)" if tiene_d else "rgba(0,0,255,0.3)")

                    fig_indiv.add_vrect(
                        x0=min(x1, x2),
                        x1=max(x1, x2),
                        fillcolor=color,
                        line_width=0
                    )

                    fig_indiv.add_vline(x=x1, line=dict(color="black", width=1))
                    fig_indiv.add_vline(x=x2, line=dict(color="black", width=1))

                    fig_indiv.add_annotation(
                        x=(x1 + x2) / 2,
                        y=y_max * 0.98,
                        text=etiqueta,
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        textangle=270,
                        xanchor="center",
                        yanchor="top"
                    )

        if aplicar_sombra_senales:
            tipo_doc_senales = "rmn1h" if tipo == "RMN 1H" else "rmn13c"
            doc_senales = db.collection("tablas_integrales").document(tipo_doc_senales)
            if doc_senales.get().exists:
                filas_senales = doc_senales.get().to_dict().get("filas", [])
                for f in filas_senales:
                    if f.get("Archivo") != archivo_actual:
                        continue
                    x1 = f.get("X min")
                    x2 = f.get("X max")
                    grupo = f.get("Grupo funcional")
                    valor = f.get("H") if tipo == "RMN 1H" else f.get("C")
                    if x1 is None or x2 is None:
                        continue

                    fig_indiv.add_vrect(x0=min(x1, x2), x1=max(x1, x2), fillcolor="rgba(0,255,0,0.3)", line_width=0)
                    fig_indiv.add_vline(x=x1, line=dict(color="black", width=1))
                    fig_indiv.add_vline(x=x2, line=dict(color="black", width=1))

                    if grupo not in [None, ""] or valor not in [None, ""]:
                        partes = []
                        if grupo:
                            partes.append(f"{grupo}")
                        if valor:
                            partes.append(f"{valor:.2f} {'H' if tipo == 'RMN 1H' else 'C'}")
                        etiqueta = " = ".join(partes) if len(partes) == 2 else " ".join(partes)

                        fig_indiv.add_annotation(
                            x=(x1 + x2) / 2,
                            y=y_max * 0.98,
                            text=etiqueta,
                            showarrow=False,
                            font=dict(size=10, color="black"),
                            textangle=270,
                            xanchor="center",
                            yanchor="top"
                        )

        if aplicar_sombra_biblio:
            doc_biblio = db.collection("configuracion_global").document(
                "tabla_editable_rmn1h" if tipo == "RMN 1H" else "tabla_editable_rmn13c")
            if doc_biblio.get().exists:
                filas_biblio = doc_biblio.get().to_dict().get("filas", [])
                for f in filas_biblio:
                    delta = f.get("δ pico")
                    grupo = f.get("Grupo funcional")
                    if delta is not None:
                        fig_indiv.add_shape(
                            type="line",
                            x0=delta, x1=delta,
                            y0=0, y1=y_max,
                            line=dict(color="black", dash="dot", width=1)
                        )
                        texto = grupo if grupo else f"δ = {delta:.2f}"
                        fig_indiv.add_annotation(
                            x=delta,
                            y=y_max * 0.95,
                            text=texto,
                            showarrow=False,
                            textangle=270,
                            font=dict(size=10, color="black"),
                            xanchor="center",
                            yanchor="top"
                        )

        fig_indiv.update_layout(
            title=f"{archivo_actual}",
            xaxis_title="[ppm]",
            yaxis_title="Intensidad",
            xaxis=dict(range=[x_max, x_min]),
            yaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None),
            height=500,
            template="simple_white"
        )

        st.plotly_chart(fig_indiv, use_container_width=True)


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


def render_imagenes(df):
    #st.markdown("## 🧪 RMN Imágenes")
    imagenes_disponibles = df[df["archivo"].str.lower().str.endswith((".png", ".jpg", ".jpeg"))]

    if imagenes_disponibles.empty:
        st.info("No hay imágenes seleccionadas.")
    else:
        for _, row in imagenes_disponibles.iterrows():
            st.markdown(f"**{row['archivo']}** – {row['muestra']}")
            try:
                image_data = BytesIO(base64.b64decode(row["contenido"]))
                image = Image.open(image_data)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"❌ No se pudo mostrar la imagen: {e}")
