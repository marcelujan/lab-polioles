import requests
import streamlit as st
from firebase_admin import auth as admin_auth

FIREBASE_API_KEY = st.secrets["firebase_api_key"]


def _post(endpoint: str, payload: dict):
    url = f"https://identitytoolkit.googleapis.com/v1/{endpoint}?key={FIREBASE_API_KEY}"
    return requests.post(url, json=payload, timeout=20)


def _build_auth_context(id_token: str):
    decoded = admin_auth.verify_id_token(id_token)
    user = admin_auth.get_user(decoded["uid"])
    claims = user.custom_claims or {}
    return {
        "id_token": id_token,
        "uid": user.uid,
        "email": (user.email or "").strip().lower(),
        "can_use_app": bool(claims.get("can_use_app", False)),
        "role": claims.get("role", "audit"),
    }


def iniciar_sesion(email: str, password: str):
    email = (email or "").strip().lower()
    password = password or ""
    if not email or not password:
        return None

    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }

    # 1) Intentar iniciar sesión con cuenta existente.
    response = _post("accounts:signInWithPassword", payload)
    if response.status_code == 200:
        return _build_auth_context(response.json()["idToken"])

    # 2) Si no entra, intentar alta silenciosa. Si el mail ya existe, Firebase devolverá EMAIL_EXISTS.
    signup = _post("accounts:signUp", payload)
    if signup.status_code == 200:
        return _build_auth_context(signup.json()["idToken"])

    return None
