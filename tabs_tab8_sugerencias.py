
# tabs_tab8_sugerencias.py
import streamlit as st
from datetime import datetime
import pandas as pd

def render_tab8(db, mostrar_sector_flotante):
    st.title("Sugerencias")
    st.session_state["current_tab"] = "Sugerencias"
    sugerencias_ref = db.collection("sugerencias")

    st.subheader("Dejar una sugerencia")
    comentario = st.text_area("Escribí tu sugerencia o comentario aquí:")
    if st.button("Enviar sugerencia"):
        if comentario.strip():
            sugerencias_ref.add({
                "comentario": comentario.strip(),
                "fecha": datetime.now().isoformat()
            })
            st.success("Gracias por tu comentario.")
            st.rerun()
        else:
            st.warning("El comentario no puede estar vacío.")

    st.subheader("Comentarios recibidos")
    docs = sugerencias_ref.order_by("fecha", direction=st.session_state.db.Query.DESCENDING).stream()
    sugerencias = [{"id": doc.id, **doc.to_dict()} for doc in docs]

    for s in sugerencias:
        st.markdown(f"**{s['fecha'][:19].replace('T',' ')}**")
        st.markdown(s["comentario"])
        if st.button("Eliminar", key=f"del_{s['id']}"):
            sugerencias_ref.document(s["id"]).delete()
            st.success("Comentario eliminado.")
            st.rerun()

    # 🔐 Sección mlujan1863
    if st.session_state.get("user_email") == "mlujan1863@gmail.com":
        st.markdown("---")
        st.subheader("🧠 Sección mlujan1863@gmail.com")

        muestras_disponibles = [doc.id for doc in db.collection("muestras").stream()]
        muestra_actual = st.selectbox("Seleccionar muestra para observación", muestras_disponibles, key="obs_muestra_sel")

        obs_ref = db.collection("observaciones_muestras").document(muestra_actual)
        obs_doc = obs_ref.get()
        observaciones = obs_doc.to_dict().get("observaciones", []) if obs_doc.exists else []

        if observaciones:
            st.markdown("### Observaciones anteriores")
            for obs in sorted(observaciones, key=lambda x: x["fecha"], reverse=True):
                st.markdown(f"- **{obs['fecha'].strftime('%Y-%m-%d %H:%M')}** — {obs['texto']}")

        nueva_obs = st.text_area("Agregar nueva observación", key="nueva_obs_texto")
        if st.button("💾 Guardar observación"):
            nueva_entrada = {
                "texto": nueva_obs,
                "fecha": datetime.now()
            }
            observaciones.append(nueva_entrada)
            obs_ref.set({"observaciones": observaciones})
            st.success("Observación guardada correctamente.")

    mostrar_sector_flotante(db)
