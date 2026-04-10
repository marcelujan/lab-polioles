import base64
import copy
import io
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class _Node:
    data: Dict[str, Any] = field(default_factory=dict)
    subcollections: Dict[str, Dict[str, "_Node"]] = field(default_factory=dict)


class FakeDocSnapshot:
    def __init__(self, doc_id: str, node: Optional[_Node]):
        self.id = doc_id
        self._node = node
        self.exists = node is not None

    def to_dict(self):
        return copy.deepcopy(self._node.data) if self._node else {}


class FakeDocumentRef:
    def __init__(self, db: "FakeFirestoreClient", collection_name: str, doc_id: str, collection_store: Dict[str, _Node]):
        self._db = db
        self._collection_name = collection_name
        self.id = doc_id
        self._collection_store = collection_store

    def _get_node(self, create: bool = False) -> Optional[_Node]:
        node = self._collection_store.get(self.id)
        if node is None and create:
            node = _Node()
            self._collection_store[self.id] = node
        return node

    def get(self):
        return FakeDocSnapshot(self.id, self._collection_store.get(self.id))

    def set(self, data: Dict[str, Any]):
        node = self._get_node(create=True)
        node.data = copy.deepcopy(data)

    def update(self, data: Dict[str, Any]):
        node = self._get_node(create=True)
        node.data.update(copy.deepcopy(data))

    def delete(self):
        self._collection_store.pop(self.id, None)

    def collection(self, name: str):
        node = self._get_node(create=True)
        sub = node.subcollections.setdefault(name, {})
        return FakeCollectionRef(self._db, name, sub)


class FakeQuery:
    def __init__(self, docs: List[FakeDocSnapshot]):
        self._docs = docs

    def stream(self):
        return list(self._docs)


class FakeCollectionRef:
    def __init__(self, db: "FakeFirestoreClient", name: str, store: Dict[str, _Node]):
        self._db = db
        self._name = name
        self._store = store

    def document(self, doc_id: Optional[str] = None):
        if doc_id is None:
            doc_id = uuid.uuid4().hex[:12]
        return FakeDocumentRef(self._db, self._name, doc_id, self._store)

    def stream(self):
        return [FakeDocSnapshot(doc_id, node) for doc_id, node in self._sorted_items()]

    def list_documents(self):
        return [FakeDocumentRef(self._db, self._name, doc_id, self._store) for doc_id, _ in self._sorted_items()]

    def order_by(self, field: str, direction: Any = None):
        reverse = str(direction).upper().endswith("DESCENDING")

        def sort_key(item):
            _, node = item
            value = node.data.get(field)
            return (value is None, value)

        items = sorted(self._store.items(), key=sort_key, reverse=reverse)
        return FakeQuery([FakeDocSnapshot(doc_id, node) for doc_id, node in items])

    def _sorted_items(self):
        return sorted(self._store.items(), key=lambda item: item[0])


class FakeFirestoreClient:
    def __init__(self, seed_data: Optional[Dict[str, Dict[str, _Node]]] = None):
        self._root: Dict[str, Dict[str, _Node]] = seed_data or {}

    def collection(self, name: str):
        store = self._root.setdefault(name, {})
        return FakeCollectionRef(self, name, store)

    def document(self, path: str):
        parts = [p for p in path.split("/") if p]
        if len(parts) % 2 != 0:
            raise ValueError(f"Ruta inválida: {path}")

        current_collection = self._root
        doc_ref = None
        for i in range(0, len(parts), 2):
            collection_name = parts[i]
            doc_id = parts[i + 1]
            collection_store = current_collection.setdefault(collection_name, {})
            doc_ref = FakeDocumentRef(self, collection_name, doc_id, collection_store)
            node = doc_ref._get_node(create=True)
            current_collection = node.subcollections
        return doc_ref


class FakeBlob:
    def __init__(self, path: str, bucket_store: Dict[str, bytes]):
        self.path = path
        self._bucket_store = bucket_store

    def upload_from_string(self, data, content_type: Optional[str] = None):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._bucket_store[self.path] = data

    def make_public(self):
        return None

    @property
    def public_url(self):
        return f"audit://{self.path}"


class FakeBucket:
    def __init__(self):
        self._objects: Dict[str, bytes] = {}

    def blob(self, path: str):
        return FakeBlob(path, self._objects)


_AUDIT_BUCKET = FakeBucket()


def patch_storage_for_audit():
    from firebase_admin import storage

    storage.bucket = lambda *args, **kwargs: _AUDIT_BUCKET


# ---------- Datos demo ----------

def _df_to_base64_csv(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return base64.b64encode(csv_bytes).decode("utf-8")


def _ftir_curve(center: float, width: float, amp: float, x_start=4000, x_end=650, n=900):
    x = np.linspace(x_start, x_end, n)
    y = 100 - amp * np.exp(-0.5 * ((x - center) / width) ** 2)
    return pd.DataFrame({"x": x, "y": y})


def _rmn_curve(peaks: List[tuple], x_start: float, x_end: float, n: int = 1200):
    x = np.linspace(x_start, x_end, n)
    y = np.zeros_like(x)
    for center, width, amp in peaks:
        y += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
    return pd.DataFrame({"x": x, "y": y})


def _plot_png_base64(df: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df.iloc[:, 0], df.iloc[:, 1])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _seed_client() -> FakeFirestoreClient:
    db = FakeFirestoreClient()

    # Muestras base
    muestras = {
        "POL-001": {
            "observacion": "Poliol epoxidado piloto. Ejemplo visible solo para auditoría.",
            "analisis": [
                {"tipo": "Índice OH [mg KHO/g]", "valor": 182.4, "fecha": "2025-03-05", "observaciones": "Método FTIR", "id": "oh-001"},
                {"tipo": "Viscosidad dinámica [cP]", "valor": 1450, "fecha": "2025-03-06", "observaciones": "25 °C", "id": "visc-001"},
            ],
        },
        "POL-002": {
            "observacion": "Poliol formulado. Datos demostrativos para inspección de interfaz.",
            "analisis": [
                {"tipo": "Índice de acidez [mg KOH/g]", "valor": 0.82, "fecha": "2025-03-07", "observaciones": "Titulación", "id": "ac-002"},
                {"tipo": "Humedad [%]", "valor": 0.12, "fecha": "2025-03-08", "observaciones": "Karl Fischer", "id": "hum-002"},
            ],
        },
    }

    for muestra, data in muestras.items():
        db.collection("muestras").document(muestra).set(data)

    # Espectros demo
    ftir_1 = _ftir_curve(1740, 45, 32)
    ftir_2 = _ftir_curve(3611, 70, 22)
    rmn1h = _rmn_curve([(0.70, 0.05, 60), (3.65, 0.08, 35), (5.35, 0.06, 18), (7.26, 0.03, 12)], 0, 9)
    rmn13c = _rmn_curve([(14.1, 0.7, 18), (62.5, 1.2, 22), (128.4, 0.9, 12), (172.5, 0.8, 26)], 0, 180)
    imagen_demo = _plot_png_base64(ftir_1, "FTIR demo")

    demo_spectra = {
        "POL-001": [
            {
                "doc_id": "esp_ftir_acetato",
                "tipo": "FTIR-Acetato",
                "observaciones": "Acetilación demo",
                "archivo_original": "POL-001_FTIR_Acetato_demo.csv",
                "nombre_archivo": "POL-001_FTIR_Acetato_demo.csv",
                "contenido": _df_to_base64_csv(ftir_1),
                "url_archivo": None,
                "es_imagen": False,
                "fecha": "2025-03-05",
                "peso_muestra": 0.4521,
            },
            {
                "doc_id": "esp_rmn_1h",
                "tipo": "RMN 1H",
                "observaciones": "RMN 1H demo",
                "archivo_original": "POL-001_RMN1H_demo.csv",
                "nombre_archivo": "POL-001_RMN1H_demo.csv",
                "contenido": _df_to_base64_csv(rmn1h),
                "url_archivo": None,
                "es_imagen": False,
                "fecha": "2025-03-05",
                "mascaras": [
                    {"difusividad": 1.2e-10, "t2": 0.18, "x_min": 3.2, "x_max": 4.1},
                    {"difusividad": 8.5e-11, "t2": 0.09, "x_min": 5.0, "x_max": 5.6},
                ],
            },
            {
                "doc_id": "esp_img_ftir",
                "tipo": "FTIR-ATR",
                "observaciones": "Imagen demostrativa",
                "archivo_original": "POL-001_FTIR_ATR_demo.png",
                "nombre_archivo": "POL-001_FTIR_ATR_demo.png",
                "contenido": imagen_demo,
                "url_archivo": None,
                "es_imagen": True,
                "fecha": "2025-03-05",
            },
        ],
        "POL-002": [
            {
                "doc_id": "esp_ftir_cloroformo",
                "tipo": "FTIR-Cloroformo",
                "observaciones": "Cloroformo demo",
                "archivo_original": "POL-002_FTIR_Cloroformo_demo.csv",
                "nombre_archivo": "POL-002_FTIR_Cloroformo_demo.csv",
                "contenido": _df_to_base64_csv(ftir_2),
                "url_archivo": None,
                "es_imagen": False,
                "fecha": "2025-03-08",
                "peso_muestra": 0.3985,
            },
            {
                "doc_id": "esp_rmn_13c",
                "tipo": "RMN 13C",
                "observaciones": "RMN 13C demo",
                "archivo_original": "POL-002_RMN13C_demo.csv",
                "nombre_archivo": "POL-002_RMN13C_demo.csv",
                "contenido": _df_to_base64_csv(rmn13c),
                "url_archivo": None,
                "es_imagen": False,
                "fecha": "2025-03-08",
            },
        ],
    }

    for muestra, espectros in demo_spectra.items():
        col = db.collection("muestras").document(muestra).collection("espectros")
        for e in espectros:
            doc_id = e.pop("doc_id")
            col.document(doc_id).set(e)

    # Observaciones y conclusiones demo
    db.collection("observaciones_muestras").document("POL-001").set({
        "observaciones": [
            {"fecha": "2025-03-06T10:15:00", "texto": "Se observa banda intensa en 1740 cm⁻¹.", "origen": "auditoría demo"},
            {"fecha": "2025-03-07T09:00:00", "texto": "Comparar con muestra formulada POL-002.", "origen": "auditoría demo"},
        ]
    })
    db.collection("observaciones_muestras").document("POL-002").set({
        "observaciones": [
            {"fecha": "2025-03-08T11:30:00", "texto": "Disminución de acidez respecto a lote base.", "origen": "auditoría demo"}
        ]
    })
    db.collection("conclusiones_muestras").document("POL-001").set({
        "conclusiones": [
            {"fecha": "2025-03-07T18:00:00", "texto": "Perfil compatible con apertura parcial de anillo epóxido."}
        ]
    })

    # FTIR
    db.collection("tablas_ftir").document("POL-001__POL-001_FTIR_Acetato_demo.csv").set({
        "filas": [
            {"Grupo funcional": "Ester", "D pico": 1740, "X min": 1710, "X max": 1760, "Área": 980.2, "Observaciones": "Carbonilo"},
            {"Grupo funcional": "OH", "D pico": 3548, "X min": 3500, "X max": 3600, "Área": 420.5, "Observaciones": "OH acetilado"},
        ]
    })
    db.collection("tablas_ftir").document("POL-002__POL-002_FTIR_Cloroformo_demo.csv").set({
        "filas": [
            {"Grupo funcional": "OH", "D pico": 3611, "X min": 3570, "X max": 3650, "Área": 510.7, "Observaciones": "OH libre"},
        ]
    })
    db.document("tablas_ftir_bibliografia/default").set({
        "filas": [
            {"Grupo funcional": "OH", "X min": 3200, "δ pico": 3550, "X max": 3650, "Tipo de muestra": "Poliol", "Observaciones": "Banda ancha OH"},
            {"Grupo funcional": "Ester", "X min": 1715, "δ pico": 1740, "X max": 1765, "Tipo de muestra": "Éster / acetato", "Observaciones": "Carbonilo"},
        ]
    })
    db.document("tablas_indice_oh/manual").set({
        "filas": [
            {"Muestra": "POL-001", "Archivo": "POL-001_FTIR_Acetato_demo.csv", "Señal": 3548, "Peso [g]": 0.4521, "Indice OH": 182.4, "Observaciones": "Demo"},
            {"Muestra": "POL-002", "Archivo": "POL-002_FTIR_Cloroformo_demo.csv", "Señal": 3611, "Peso [g]": 0.3985, "Indice OH": 160.8, "Observaciones": "Demo"},
        ]
    })

    # RMN
    db.collection("tablas_integrales").document("rmn1h").set({
        "filas": [
            {"Muestra": "POL-001", "Archivo": "POL-001_RMN1H_demo.csv", "Grupo funcional": "Glicerol medio", "δ pico": 3.65, "X min": 3.45, "X max": 3.85, "Área": 22.3, "Xas min": 7.20, "Xas max": 7.32, "Área as": 6.1, "Has": 1.0, "H": 3.66, "Observaciones": "Referencia demo"},
            {"Muestra": "POL-001", "Archivo": "POL-001_RMN1H_demo.csv", "Grupo funcional": "C=C olefínicos", "δ pico": 5.35, "X min": 5.15, "X max": 5.50, "Área": 8.4, "Xas min": 7.20, "Xas max": 7.32, "Área as": 6.1, "Has": 1.0, "H": 1.38, "Observaciones": "Referencia demo"},
        ]
    })
    db.collection("tablas_integrales").document("rmn13c").set({
        "filas": [
            {"Muestra": "POL-002", "Archivo": "POL-002_RMN13C_demo.csv", "Grupo funcional": "CH3", "δ pico": 14.1, "X min": 13.2, "X max": 14.8, "Área": 11.8, "Xas min": 171.8, "Xas max": 173.2, "Área as": 7.0, "Cas": 1.0, "C": 1.69, "Observaciones": "Referencia demo"},
        ]
    })
    db.collection("configuracion_global").document("tabla_editable_rmn1h").set({
        "filas": [
            {"Grupo funcional": "Glicerol medio", "δ pico": 3.65, "X min": 3.45, "X max": 3.85, "Observaciones": "Señal principal"},
            {"Grupo funcional": "C=C olefínicos", "δ pico": 5.35, "X min": 5.15, "X max": 5.50, "Observaciones": "Insaturación"},
        ]
    })
    db.collection("configuracion_global").document("tabla_editable_rmn13c").set({
        "filas": [
            {"Grupo funcional": "CH3", "δ pico": 14.1, "X min": 13.2, "X max": 14.8, "Observaciones": "Metilo terminal"},
            {"Grupo funcional": "Ester", "δ pico": 172.5, "X min": 171.8, "X max": 173.2, "Observaciones": "Carbonilo"},
        ]
    })

    db.collection("muestras").document("POL-001").collection("dt2").document("rmn 1h").set({
        "filas": [
            {"Muestra": "POL-001", "Archivo": "POL-001_RMN1H_demo.csv", "Grupo funcional": "Glicerol medio", "δ pico": 3.65, "X min": 3.45, "X max": 3.85, "Área": 22.3, "Xas min": 7.20, "Xas max": 7.32, "Área as": 6.1, "Has": 1.0, "H": 3.66, "Observaciones": "DT2 demo"},
        ]
    })
    db.collection("muestras").document("POL-002").collection("dt2").document("rmn 13c").set({
        "filas": [
            {"Muestra": "POL-002", "Archivo": "POL-002_RMN13C_demo.csv", "Grupo funcional": "CH3", "δ pico": 14.1, "X min": 13.2, "X max": 14.8, "Área": 11.8, "Xas min": 171.8, "Xas max": 173.2, "Área as": 7.0, "Cas": 1.0, "C": 1.69, "Observaciones": "DT2 demo"},
        ]
    })

    db.collection("zonas_rmn").document("POL-001_RMN1H_demo.csv").set({
        "zonas": [
            {"nombre": "Zona 1", "x_min": 3.2, "x_max": 4.0},
            {"nombre": "Zona 2", "x_min": 5.0, "x_max": 5.6},
        ]
    })
    db.collection("muestras").document("POL-001").collection("zonas").document("POL-001_RMN1H_demo.csv__zona_1").set({
        "filas": [
            {"Muestra": "POL-001", "Archivo": "POL-001_RMN1H_demo.csv", "Grupo funcional": "Glicerol medio", "δ pico": 3.65, "X min": 3.45, "X max": 3.85, "Área": 22.3, "Observaciones": "Zona demo"}
        ]
    })

    # Sugerencias y referencias
    db.collection("sugerencias").document("sug_001").set({
        "fecha": "2025-03-09T10:00:00",
        "comentario": "Agregar validación de consistencia entre índice OH y FTIR.",
        "texto": "Agregar validación de consistencia entre índice OH y FTIR.",
        "usuario": "auditor-demo"
    })
    db.collection("referencias_globales").document("ref_001").set({
        "etiqueta": "FTIR",
        "texto": "Asignación orientativa: OH 3200-3650 cm⁻¹; carbonilo 1715-1765 cm⁻¹. Datos de demostración."
    })

    return db


def get_audit_db() -> FakeFirestoreClient:
    return _seed_client()
