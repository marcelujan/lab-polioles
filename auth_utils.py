import requests
import streamlit as st

FIREBASE_API_KEY = st.secrets["firebase_api_key"]

def registrar_usuario(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        st.success("Usuario registrado correctamente. Ahora puede iniciar sesión.")
    else:
        st.error("No se pudo registrar. El correo puede estar en uso o la contraseña es débil.")

def iniciar_sesion(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["idToken"]
    else:
        st.error("Credenciales incorrectas o cuenta no existente.")
        return None
