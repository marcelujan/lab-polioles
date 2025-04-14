import firebase_admin
from firebase_admin import credentials, firestore
import json
import streamlit as st

cred_dict = json.loads(st.secrets["firebase_key"])
cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
cred = credentials.Certificate(cred_dict)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

def borrar_coleccion(nombre):
    coleccion = db.collection(nombre).stream()
    count = 0
    for doc in coleccion:
        doc.reference.delete()
        count += 1
    print(f"{count} documentos eliminados de la colección '{nombre}'.")

if __name__ == "__main__":
    borrar_coleccion("muestras")
    borrar_coleccion("espectros")
    print("🔥 Limpieza completada.")
