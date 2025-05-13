# firestore_utils.py
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Inicialización global de Firebase (solo una vez)
def iniciar_firebase(cred_json_str):
    cred_dict = json.loads(cred_json_str)
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(cred_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()

# Función para cargar muestras
def cargar_muestras(db):
    try:
        docs = db.collection("muestras").stream()
        return [{**doc.to_dict(), "nombre": doc.id} for doc in docs]
    except:
        return []

# Función para guardar una muestra
def guardar_muestra(db, nombre, observacion, analisis, espectros=None):
    datos = {
        "observacion": observacion,
        "analisis": analisis
    }
    if espectros is not None:
        datos["espectros"] = espectros
    db.collection("muestras").document(nombre).set(datos)

# Eliminar muestra
#def eliminar_muestra(db, nombre):
#    try:
 #       db.collection("muestras").document(nombre).delete()
  #  except Exception as e:
   #     print(f"Error al eliminar muestra '{nombre}': {e}")
def eliminar_muestra(db, nombre):
    try:
        ref = db.collection("muestras").document(nombre)
        if ref.get().exists:
            ref.delete()
            print(f"Muestra '{nombre}' eliminada correctamente.")
        else:
            print(f"⚠ La muestra '{nombre}' no existe en Firestore.")
    except Exception as e:
        print(f"❌ Error al eliminar muestra '{nombre}': {e}")
