import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import json

def iniciar_firebase():
    if not firebase_admin._apps:
        cred_dict = json.loads(st.secrets["firebase_key"])
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    return firestore.client()
