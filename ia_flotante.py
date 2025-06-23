import streamlit as st
import openai
from datetime import datetime
import base64
import json

def consultar_ia_avanzada(muestra_actual=None, comparar_con=None):
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

    # Botón flotante que activa el panel
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
    <form action="#" method="post">
        <button name="abrir_ia" class="floating-btn">🧠</button>
    </form>
    """, unsafe_allow_html=True)

    if st.session_state.get("abrir_ia"):
        st.session_state["mostrar_ia"] = not st.session_state["mostrar_ia"]

    if st.session_state["mostrar_ia"]:
        with st.container():
            st.markdown("""
                <div style='position:fixed; bottom:100px; right:20px; width:400px; background:#fff; padding:20px; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.3); z-index:10000;'>
            """, unsafe_allow_html=True)
            st.markdown("<div style='text-align:right'><button onClick="window.location.reload()">❌</button></div>", unsafe_allow_html=True)
            muestra = st.session_state.get("muestra_activa")
            pregunta = st.text_area("Consulta a la IA", key="ia_pregunta")
            if st.button("💬 Consultar IA"):
                comparar_con = None
                if "comparar con" in pregunta.lower():
                    import re
                    match = re.search(r"muestra\s*(\d+)", pregunta.lower())
                    if match:
                        comparar_con = match.group(1)
                respuesta = consultar_ia_avanzada(muestra_actual=muestra, comparar_con=comparar_con)
                st.session_state["respuesta_ia"] = respuesta

            if "respuesta_ia" in st.session_state:
                st.markdown("### Respuesta de IA")
                st.markdown(st.session_state["respuesta_ia"])
                if st.button("💾 Guardar como conclusión"):
                    fecha = datetime.now().isoformat()
                    nombre = muestra or "global"
                    db = st.session_state.get("firebase_db")
                    if db:
                        ref = db.collection("conclusiones_muestras").document(nombre)
                        prev = ref.get().to_dict() or {}
                        nuevas = prev.get("conclusiones", []) + [{"fecha": fecha, "texto": st.session_state["respuesta_ia"]}]
                        ref.set({"conclusiones": nuevas})
                        st.success("Conclusión guardada.")
            st.markdown("</div>", unsafe_allow_html=True)
