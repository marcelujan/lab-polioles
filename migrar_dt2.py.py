import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict

# Inicializar Firebase con tu archivo de credenciales
cred = credentials.Certificate("ruta/a/credencial.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Leer documento centralizado
doc_central = db.collection("tablas_dt2").document("cuantificable").get()
if not doc_central.exists:
    print("❌ No se encontró 'tablas_dt2/cuantificable'")
    exit()

filas = doc_central.to_dict().get("filas", [])
if not filas:
    print("⚠️ No hay filas para migrar.")
    exit()

# Agrupar por muestra
agrupado = defaultdict(list)
for fila in filas:
    muestra = fila.get("Muestra")
    if muestra:
        agrupado[muestra].append(fila)

# Migrar cada grupo a su ruta correspondiente
for muestra, filas_muestra in agrupado.items():
    ruta = f"muestras/{muestra}/dt2/rmn 1h"
    print(f"🔁 Migrando {len(filas_muestra)} filas a → {ruta}")
    doc_destino = db.collection("muestras").document(muestra).collection("dt2").document("rmn 1h")
    doc_destino.set({"filas": filas_muestra})

print("✅ Migración completada.")
