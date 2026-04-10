import requests
import streamlit as st
from firebase_admin import auth as admin_auth

FIREBASE_API_KEY = st.secrets["firebase_api_key"]


def _post_auth(endpoint: str, payload: dict):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}?key={FIREBASE_API_KEY}"
    return requests.post(url, json=payload, timeout=20)


def registrar_usuario(email, password):
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }
    response = _post_auth("signUp", payload)
    return response.status_code == 200


def iniciar_sesion(email, password):
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }

    response = _post_auth("signInWithPassword", payload)

    if response.status_code != 200:
        # Alta silenciosa: si no existe o no puede iniciar, intenta crearla sin exponer el flujo.
        signup = _post_auth("signUp", payload)
        if signup.status_code != 200:
            return None
        response = _post_auth("signInWithPassword", payload)
        if response.status_code != 200:
            return None

    data = response.json()
    id_token = data.get("idToken")
    if not id_token:
        return None

    decoded = admin_auth.verify_id_token(id_token)
    user = admin_auth.get_user(decoded["uid"])
    claims = user.custom_claims or {}

    return {
        "id_token": id_token,
        "uid": user.uid,
        "email": user.email,
        "can_use_app": bool(claims.get("can_use_app", False)),
        "role": claims.get("role", "auditor"),
    }
