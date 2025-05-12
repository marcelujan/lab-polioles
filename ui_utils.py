import streamlit as st
from datetime import datetime
import firebase_admin
from firebase_admin import firestore

def mostrar_sector_flotante(db, key_suffix=""):
    """Muestra el cuadro de observación flotante solo si el usuario es Marcelo."""
    if st.session_state.get("user_email") != "mlujan1863@gmail.com":
        return  # No mostrar para otros usuarios

    st.markdown("---")
    st.markdown("🧠 **Observación rápida (sector flotante)**")

    muestra_activa = st.session_state.get("muestra_activa", None)
    current_tab = st.session_state.get("current_tab", "desconocido")
    unique_suffix = f"{muestra_activa}_{current_tab}_{datetime.now().strftime('%H%M%S%f')}"

    if muestra_activa:
        st.info(f"Observación vinculada automáticamente a: **{muestra_activa}**")
    else:
        st.warning("No se detectó ninguna muestra activa. La observación no podrá guardarse.")

    text_key = f"obs_flotante_{unique_suffix}"
    button_key = f"btn_guardar_obs_rapida_{unique_suffix}"

    nueva_obs = st.text_area("Escribí tu observación", key=text_key)
    if st.button("💾 Guardar observación rápida", key=button_key):
        if not muestra_activa:
            st.error("No se puede guardar la observación sin muestra activa.")
        else:
            obs_ref = db.collection("observaciones_muestras").document(muestra_activa)
            obs_doc = obs_ref.get()
            observaciones = obs_doc.to_dict().get("observaciones", []) if obs_doc.exists else []
            nueva_entrada = {
                "texto": nueva_obs,
                "fecha": datetime.now(),
                "origen": f"observación rápida desde {current_tab}"
            }
            observaciones.append(nueva_entrada)
            obs_ref.set({"observaciones": observaciones})
            st.success("Observación guardada correctamente.")
