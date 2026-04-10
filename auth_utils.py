import requests
import streamlit as st
from firebase_admin import auth as admin_auth

FIREBASE_API_KEY = st.secrets["firebase_api_key"]


def iniciar_sesion(email, password):
    """Autentica con Firebase Auth y devuelve el contexto autorizado del usuario."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }
    response = requests.post(url, json=payload, timeout=30)

    if response.status_code != 200:
        st.error("Credenciales incorrectas o cuenta no habilitada.")
        return None

    data = response.json()
    id_token = data["idToken"]

    try:
        decoded = admin_auth.verify_id_token(id_token)
        user = admin_auth.get_user(decoded["uid"])
    except Exception as exc:
        st.error(f"No se pudo validar la sesión en el servidor: {exc}")
        return None

    claims = user.custom_claims or {}

    return {
        "id_token": id_token,
        "uid": user.uid,
        "email": user.email or email,
        "can_use_app": bool(claims.get("can_use_app", False)),
        "role": claims.get("role", "auditor"),
    }
