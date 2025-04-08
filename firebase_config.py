import firebase_admin
from firebase_admin import credentials, firestore

def iniciar_firebase():
    cred = credentials.Certificate("firebase_key.json")
    try:
        firebase_admin.initialize_app(cred)
    except ValueError:
        pass  # Ya fue inicializado

    return firestore.client()
