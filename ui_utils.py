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

    if muestra_activa:
        st.info(f"Observación vinculada automáticamente a: **{muestra_activa}**")
    else:
        st.warning("No se detectó ninguna muestra activa. La observación no podrá guardarse.")

    # Claves únicas por tab y muestra
    text_key = f"obs_flotante_{key_suffix}"
    button_key = f"btn_guardar_obs_rapida_{key_suffix}"

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

def get_caracteristicas():
    return [
        "Índice de yodo [% p/p I2 abs]", "Índice OH [mg KHO/g]",
        "Índice de acidez [mg KOH/g]", "Índice de epóxido [mol/100g]",
        "Humedad [%]", "PM [g/mol]", "Funcionalidad [#]",
        "Viscosidad dinámica [cP]", "Densidad [g/mL]",
        "PCV [%]", "Otro análisis"
    ]

# Aclaraciones por característica
ACLARACIONES_CARACTERISTICAS = {
    "Índice de yodo [% p/p I2 abs]": "Estimar IY y colocar Pm (o blanco) con 5 mL de ciclohexano en erlenmeyer de 250 mL. Añadir 10-25 mL Wijs, tapar con algodon embebido en KI 15% p/v y dejar en oscuridad 1-2 h. Añadir 10-20 mL KI 15% p/v y 50-100 mL agua destilada. Titular con tiosulfato de sodio 0,1 N<br>"
    "Pm en función del índice esperado (problema en resultados). *Pm = 1g para IY = 20. Pm = 0,2 g para IY = 130.* Problemas con dobles enlaces conjugados.",
    "Índice OH [mg KHO/g]": "En 'Observaciones' indicar si se utiliza método espectroscopía FTIR o acetilación.",
    "Índice de acidez [mg KOH/g]": "1 g muestra + 50 mL solvente (etanol/tolueno) + fenolftaleína 1%. Titular con KOH 0,1 N etanólico<br>"
    "*Valoración de KOH 0,1 N etanólico con biftalato ácido de potasio (KHP, patrón primario): secar KHP por 2 h a 120 °C y enfriar en desecador. Disolver 0,40 g KHP con 50 mL de agua y añadir fenolftaleína. Titular con aproximadamente 20 mL de KOH 0,1 N etanólico.*",
    "Índice de epóxido [mol/100g]": "sin aclaraciones",
    "Humedad [%]": "sin aclaraciones",
    "PM [g/mol]": "sin aclaraciones",
    "Funcionalidad [#]": "sin aclaraciones",
    "Viscosidad dinámica [cP]": "sin aclaraciones",
    "Densidad [g/mL]": "sin aclaraciones",
    "PCV [%]": "sin aclaraciones",
    "Otro análisis": "sin aclaraciones",
}

def get_aclaracion(caracteristica: str) -> str:
    return ACLARACIONES_CARACTERISTICAS.get(caracteristica, "sin aclaraciones")
