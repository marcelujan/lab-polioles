import streamlit as st
from datetime import datetime
import firebase_admin
from firebase_admin import firestore

# Asegurarse de que la instancia de Firestore esté disponible
if not firebase_admin._apps:
    firebase_admin.initialize_app()
db = firestore.client()

def mostrar_sector_flotante():
    if st.session_state.get("user_email") != "mlujan1863@gmail.com":
        return  # Solo se muestra para ese usuario

    st.markdown("---")
    st.markdown("🧠 **Observación rápida (sector flotante)**")

    muestra_activa = st.session_state.get("muestra_activa", None)

    if muestra_activa:
        st.info(f"Observación vinculada automáticamente a: **{muestra_activa}**")
    else:
        st.warning("No se detectó ninguna muestra activa. La observación no podrá guardarse.")

    nueva_obs = st.text_area("Escribí tu observación", key=f"obs_flotante_{st.session_state.get('current_tab', '')}")
    if st.button("💾 Guardar observación rápida"):
        if not muestra_activa:
            st.error("No se puede guardar la observación sin muestra activa.")
        else:
            obs_ref = db.collection("observaciones_muestras").document(muestra_activa)
            obs_doc = obs_ref.get()
            observaciones = obs_doc.to_dict().get("observaciones", []) if obs_doc.exists else []
            nueva_entrada = {
                "texto": nueva_obs,
                "fecha": datetime.now(),
                "origen": f"observación rápida desde {st.session_state.get('current_tab', 'desconocido')}"
            }
            observaciones.append(nueva_entrada)
            obs_ref.set({"observaciones": observaciones})
            st.success("Observación guardada correctamente.")
