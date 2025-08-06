import streamlit as st
import openai
from datetime import datetime
import base64
import json
from scipy.signal import find_peaks

def generar_resumen_picos_rmn(datos_plotly):
    resumen = []
    picos_dict = {}

    for muestra, tipo, archivo, df in datos_plotly:
        x_vals = df["x"].values
        y_vals = df["y"].values
        peaks, _ = find_peaks(y_vals, height=max(y_vals) * 0.1)
        picos_detectados = [round(x_vals[p], 2) for p in peaks]

        resumen.append(f"""
Muestra: {muestra}
Tipo de espectro: {tipo}
Archivo: {archivo}
Número total de picos detectados: {len(picos_detectados)}
Picos principales (posición en ppm): {picos_detectados}
""")
        picos_dict[f"{muestra} – {archivo}"] = set(picos_detectados)

    resumen_picos_comparativo = ""
    if picos_dict:
        sets_picos = list(picos_dict.values())
        nombres_muestras = list(picos_dict.keys())
        picos_comunes = sorted(set.intersection(*sets_picos)) if len(sets_picos) >= 2 else []

        picos_exclusivos_texto = ""
        for i, nombre in enumerate(nombres_muestras):
            otros_sets = [s for j, s in enumerate(sets_picos) if j != i]
            picos_otros = set.union(*otros_sets) if otros_sets else set()
            picos_exclusivos = sorted(picos_dict[nombre] - picos_otros)
            picos_exclusivos_texto += f"\n{nombre}\nPicos exclusivos: {picos_exclusivos}\n"

        resumen_picos_comparativo = f"""
---
Análisis comparativo automático:

Picos comunes a todos los espectros: {picos_comunes}

{picos_exclusivos_texto}
"""

    return "\n".join(resumen) + "\n" + resumen_picos_comparativo

def consultar_ia_avanzada(muestra_actual=None, comparar_con=None, datos_plotly=None):
    db = st.session_state.get("firebase_db")
    if not db:
        return "⚠️ No se encontró la base de datos."

    def obtener_datos(nombre):
        ref = db.collection("muestras").document(nombre)
        doc = ref.get()
        if not doc.exists:
            return {}
        datos = doc.to_dict()
        espectros_ref = ref.collection("espectros").stream()
        espectros = [e.to_dict() for e in espectros_ref]
        obs_ref = db.collection("observaciones_muestras").document(nombre).get()
        observaciones = obs_ref.to_dict().get("observaciones", []) if obs_ref.exists else []
        conclusiones_ref = db.collection("conclusiones_muestras").document(nombre).get()
        conclusiones = conclusiones_ref.to_dict().get("conclusiones", []) if conclusiones_ref.exists else []
        return {
            "nombre": nombre,
            "observacion": datos.get("observacion", ""),
            "analisis": datos.get("analisis", []),
            "espectros": espectros,
            "observaciones": observaciones,
            "conclusiones": conclusiones
        }

    datos1 = obtener_datos(muestra_actual) if muestra_actual else {}
    datos2 = obtener_datos(comparar_con) if comparar_con else {}

    referencias_globales = db.collection("referencias_globales").stream()
    referencias = [r.to_dict() for r in referencias_globales]

    prompt = """
Sos un asistente experto en análisis de laboratorio (FTIR, RMN, índice OH, viscosidad, etc.).
Tu tarea es analizar las muestras dadas y generar un informe técnico interpretativo.
Incluí observaciones sobre espectros, análisis, comparaciones, y usá bibliografía técnica si aplica.
No repitas disclaimers ni digas que sos una IA.

---
"""

    def resumen_muestra(d):
        if not d:
            return ""
        bloques = [f"Muestra: {d['nombre']}\n"]
        bloques.append(f"Observación general: {d['observacion']}")
        if d["analisis"]:
            bloques.append("Análisis:")
            for a in d["analisis"]:
                bloques.append(f"- {a['tipo']}: {a['valor']} ({a['fecha']})")
        if d["espectros"]:
            bloques.append("Espectros:")
            for e in d["espectros"]:
                bloques.append(f"- {e['tipo']} – {e.get('nombre_archivo', '')} – {e.get('fecha', '')}")
        if d["observaciones"]:
            bloques.append("Observaciones previas:")
            for o in d["observaciones"]:
                bloques.append(f"- {o['fecha']}: {o['texto']}")
        if d["conclusiones"]:
            bloques.append("Conclusiones anteriores:")
            for c in d["conclusiones"]:
                bloques.append(f"- {c['fecha']}: {c['texto']}")
        return "\n".join(bloques)

    prompt += resumen_muestra(datos1)
    if datos2:
        prompt += "\n---\nComparar con:\n"
        prompt += resumen_muestra(datos2)

    if referencias:
        prompt += "\n---\nInformación técnica adicional cargada por el usuario:\n"
        for r in referencias:
            txt = r.get("texto", "")
            etiqueta = r.get("etiqueta", "General")
            prompt += f"\n[{etiqueta}] {txt[:500]}"

    if datos_plotly:
        prompt += "\n---\nResumen automático de espectros cargados:\n"
        prompt += generar_resumen_picos_rmn(datos_plotly)

    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        respuesta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sos un experto en análisis de laboratorio"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1200
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"❌ Error al consultar la IA: {e}"

def mostrar_panel_ia():
    if st.session_state.get("user_email") != "mlujan1863@gmail.com":
        return

    if "mostrar_ia" not in st.session_state:
        st.session_state["mostrar_ia"] = False

    st.markdown("""
    <style>
    .floating-btn {
        position: fixed;
        bottom: 30px;
        right: 220px;
        background-color: #6c63ff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        z-index: 9999;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("🧠", key="btn_flotante_ia"):
        st.session_state["mostrar_ia"] = not st.session_state["mostrar_ia"]

    if st.session_state["mostrar_ia"]:
        with st.sidebar:
            st.markdown("## 🧠 Consultas")
            muestra = st.session_state.get("muestra_activa")
            pregunta = st.text_area(
                "Pregunta IA",
                key="ia_pregunta",
                label_visibility="collapsed"
            )
            datos_plotly = st.session_state.get("datos_plotly", [])
            if datos_plotly:
                st.markdown("**Picos detectados**")
                resumen_previo = generar_resumen_picos_rmn(datos_plotly)
                st.code(resumen_previo, language="markdown")

            if st.button("💬 Consultar"):
                comparar_con = None
                if "comparar con" in pregunta.lower():
                    import re
                    match = re.search(r"muestra\s*(\d+)", pregunta.lower())
                    if match:
                        comparar_con = match.group(1)
                respuesta = consultar_ia_avanzada(muestra_actual=muestra, comparar_con=comparar_con, datos_plotly=datos_plotly)
                st.session_state["respuesta_ia"] = respuesta

            if "respuesta_ia" in st.session_state:
                st.markdown("### Respuesta de IA")
                st.markdown(st.session_state["respuesta_ia"])
                if st.button("💾 Guardar conclusión"):
                    fecha = datetime.now().isoformat()
                    nombre = muestra or "global"
                    db = st.session_state.get("firebase_db")
                    if db:
                        ref = db.collection("conclusiones_muestras").document(nombre)
                        prev = ref.get().to_dict() or {}
                        nuevas = prev.get("conclusiones", []) + [{"fecha": fecha, "texto": st.session_state["respuesta_ia"]}]
                        ref.set({"conclusiones": nuevas})
                        st.success("Conclusión guardada.")

            st.markdown("---")
            st.markdown("### 📚 Cargar info")
            texto = st.text_area(
                    "Referencia IA",
                    key="ia_referencia_texto",
                    label_visibility="collapsed"
                )
            etiqueta = st.text_input("Técnica relacionada (ej: FTIR, RMN, etc.)", key="ia_etiqueta")
            archivo = st.file_uploader("Subir PDF o TXT", key="ia_archivo")

            if st.button("📌 Guardar referencia"):
                db = st.session_state.get("firebase_db")
                if db:
                    ref = db.collection("referencias_globales")
                    contenido = {
                        "fecha": datetime.now().isoformat(),
                        "texto": texto,
                        "etiqueta": etiqueta,
                    }
                    if archivo:
                        try:
                            contenido["archivo_nombre"] = archivo.name
                            contenido["archivo_base64"] = base64.b64encode(archivo.getvalue()).decode("utf-8")
                        except Exception as e:
                            st.error(f"Error al procesar el archivo: {e}")
                            st.info("Por favor, vuelve a subir el archivo.")
                            st.stop()
                    ref.add(contenido)
                    st.success("Referencia guardada.")
