# tabs_tab8_sugerencias.py
import streamlit as st
from datetime import datetime
from firebase_admin import firestore


def _format_fecha(valor):
    if valor is None:
        return ""
    if hasattr(valor, "strftime"):
        try:
            return valor.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    if isinstance(valor, str):
        # intenta ISO primero
        try:
            return datetime.fromisoformat(valor.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return valor[:19].replace("T", " ")
    return str(valor)


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

    docs = sugerencias_ref.order_by("fecha", direction=firestore.Query.DESCENDING).stream()
    st.subheader("Comentarios recibidos")
    sugerencias = [{"id": doc.id, **doc.to_dict()} for doc in docs]

    for s in sugerencias:
        st.markdown(f"**{_format_fecha(s.get('fecha'))}**")
        st.markdown(s.get("comentario") or s.get("texto") or "")
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
            def _obs_sort_key(item):
                valor = item.get("fecha")
                if hasattr(valor, "timestamp"):
                    return valor.timestamp()
                if isinstance(valor, str):
                    try:
                        return datetime.fromisoformat(valor.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        return valor
                return str(valor)

            for obs in sorted(observaciones, key=_obs_sort_key, reverse=True):
                st.markdown(f"- **{_format_fecha(obs.get('fecha'))}** — {obs.get('texto', '')}")

        nueva_obs = st.text_area("Agregar nueva observación", key="nueva_obs_texto")
        if st.button("💾 Guardar observación"):
            nueva_entrada = {
                "texto": nueva_obs,
                "fecha": datetime.now()
            }
            observaciones.append(nueva_entrada)
            obs_ref.set({"observaciones": observaciones})
            st.success("Observación guardada correctamente.")
