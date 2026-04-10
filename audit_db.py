from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class AuditDocumentSnapshot:
    path: str = ""
    _data: dict[str, Any] | None = None
    exists: bool = False

    @property
    def id(self) -> str:
        return self.path.rstrip("/").split("/")[-1] if self.path else ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data or {})


class AuditDocumentReference:
    def __init__(self, path: str = "") -> None:
        self.path = path.strip("/")

    @property
    def id(self) -> str:
        return self.path.split("/")[-1] if self.path else ""

    def collection(self, name: str) -> "AuditCollectionReference":
        next_path = f"{self.path}/{name}" if self.path else name
        return AuditCollectionReference(next_path)

    def get(self) -> AuditDocumentSnapshot:
        return AuditDocumentSnapshot(path=self.path, _data={}, exists=False)

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def delete(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class AuditCollectionReference:
    def __init__(self, path: str = "") -> None:
        self.path = path.strip("/")

    def document(self, doc_id: str | None = None) -> AuditDocumentReference:
        doc_id = doc_id or "audit-doc"
        next_path = f"{self.path}/{doc_id}" if self.path else doc_id
        return AuditDocumentReference(next_path)

    def stream(self, *_args: Any, **_kwargs: Any) -> list[AuditDocumentSnapshot]:
        return []

    def list_documents(self, *_args: Any, **_kwargs: Any) -> list[AuditDocumentReference]:
        return []

    def add(self, _data: dict[str, Any] | None = None, *_args: Any, **_kwargs: Any):
        doc_ref = self.document()
        return None, doc_ref

    def order_by(self, *_args: Any, **_kwargs: Any) -> "AuditCollectionReference":
        return self

    def where(self, *_args: Any, **_kwargs: Any) -> "AuditCollectionReference":
        return self

    def limit(self, *_args: Any, **_kwargs: Any) -> "AuditCollectionReference":
        return self


class AuditFirestoreClient:
    def collection(self, name: str) -> AuditCollectionReference:
        return AuditCollectionReference(name)

    def document(self, path: str) -> AuditDocumentReference:
        return AuditDocumentReference(path)


class AuditBlob:
    def __init__(self, path: str) -> None:
        self.path = path
        self.public_url = ""

    def upload_from_string(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def make_public(self) -> None:
        self.public_url = ""


class AuditBucket:
    def blob(self, path: str) -> AuditBlob:
        return AuditBlob(path)


class _AuditClientFactory:
    def __call__(self, *_args: Any, **_kwargs: Any) -> AuditFirestoreClient:
        return AuditFirestoreClient()


def apply_audit_mode() -> None:
    from firebase_admin import firestore as firestore_module
    from firebase_admin import storage as storage_module

    firestore_module.client = _AuditClientFactory()
    firestore_module.Client = _AuditClientFactory()
    storage_module.bucket = lambda *_args, **_kwargs: AuditBucket()
