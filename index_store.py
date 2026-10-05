"""Bounded, immutable, reusable retrieval indexes for document sets."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from document_service import DocumentRecord, DocumentService

if TYPE_CHECKING:
    from retrieval import RetrievalIndex


INDEX_VERSION = "v1"


@dataclass(frozen=True)
class IndexEntry:
    key: str
    owner: str
    document_ids: frozenset[str]
    index: RetrievalIndex
    expires_at: float


class IndexStore:
    def __init__(self, documents: DocumentService, shared_models: RetrievalIndex | None = None,
                 index_factory: Callable[[], RetrievalIndex] | None = None,
                 max_entries: int = 8, ttl_seconds: int = 3600):
        if max_entries <= 0 or ttl_seconds <= 0:
            raise ValueError("Index limits must be positive")
        self.documents = documents
        self.shared_models = shared_models
        self.index_factory = index_factory or self._new_index
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, IndexEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._build_count: dict[str, int] = {}

    def _new_index(self) -> RetrievalIndex:
        if self.shared_models is None:
            raise RuntimeError("Shared embedding models are unavailable")
        from retrieval import RetrievalIndex
        return RetrievalIndex(self.shared_models.bge_model, self.shared_models.all_mini_model)

    @staticmethod
    def _key(owner: str, records: list[DocumentRecord]) -> str:
        payload = [INDEX_VERSION, owner, [(record.document_id, record.content_hash) for record in records]]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def _cached(self, key: str) -> IndexEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.time():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry

    def get_or_build(self, owner: str, sources: list[str]) -> tuple[RetrievalIndex, bool]:
        records = self.documents.get_records(owner, sources)
        records = sorted({record.document_id: record for record in records}.values(),
                         key=lambda record: record.document_id)
        key = self._key(owner, records)
        cached = self._cached(key)
        if cached:
            return cached.index, True
        with self._build_lock:
            cached = self._cached(key)
            if cached:
                return cached.index, True
            index = self.index_factory()
            chunks = []
            for record in records:
                extraction = self.documents.extract(record)
                if not extraction.success or not extraction.text.strip():
                    raise ValueError(f"Document {record.document_id} could not be extracted")
                chunks.extend(index._create_chunks(extraction.text, f"upload:{record.document_id}"))
            if not chunks:
                raise ValueError("No searchable text was extracted")
            index.build_indices(chunks)
            # A concurrent deletion or expiry must prevent publishing this index.
            self.documents.get_records(owner, [f"upload:{record.document_id}" for record in records])
            expires_at = min(time.time() + self.ttl_seconds,
                             *(record.expires_at for record in records))
            entry = IndexEntry(key=key, owner=owner,
                               document_ids=frozenset(record.document_id for record in records),
                               index=index, expires_at=expires_at)
            with self._lock:
                self._entries[key] = entry
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
                self._build_count[owner] = self._build_count.get(owner, 0) + 1
            return index, False

    def invalidate(self, document_ids: list[str]):
        ids = set(document_ids)
        if not ids:
            return
        with self._build_lock, self._lock:
            for key, entry in list(self._entries.items()):
                if entry.document_ids & ids:
                    del self._entries[key]

    def stats(self, owner: str) -> dict:
        with self._lock:
            return {"cached_indexes": sum(entry.owner == owner for entry in self._entries.values()),
                    "builds": self._build_count.get(owner, 0),
                    "max_cached_indexes": self.max_entries}

    def clear(self, owner: str):
        with self._build_lock, self._lock:
            for key, entry in list(self._entries.items()):
                if entry.owner == owner:
                    del self._entries[key]
