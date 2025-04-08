import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import json

def iniciar_firebase():
    # Si ya hay una app inicializada, simplemente devolvemos el cliente de Firestore.
    if not firebase_admin._apps:
        try:
            # Obtén el string desde los secrets
            secret_str = st.secrets["firebase_key"]
            # Parsear el JSON
            cred_dict = json.loads(secret_str)
            # Crear las credenciales con el diccionario obtenido
            cred = credentials.Certificate(cred_dict)
            # Inicializar la app con las credenciales
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error("Error al inicializar Firebase: " + str(e))
            raise e
    return firestore.client()
