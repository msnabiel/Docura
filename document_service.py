"""Persistent document records, ownership, expiry, and extracted text cache."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from extraction_models import ExtractionResult
from source_security import MAX_SOURCE_BYTES, UPLOAD_ID_PATTERN

if TYPE_CHECKING:
    from text_extractor import EnhancedTextExtractionService


ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff",
    ".bmp", ".txt", ".eml", ".html", ".csv", ".json", ".xlsx",
    ".xls", ".zip",
}
EXTRACTION_VERSION = "v1"


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    owner: str
    filename: str
    storage_name: str
    content_hash: str
    size: int
    created_at: float
    expires_at: float


class DocumentService:
    def __init__(self, data_dir: str, extractor: EnhancedTextExtractionService | None = None,
                 ttl_seconds: int = 86400):
        if ttl_seconds <= 0:
            raise ValueError("Document TTL must be positive")
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.upload_dir.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self.data_dir, 0o700)
        os.chmod(self.upload_dir, 0o700)
        self.db_path = self.data_dir / "documents.sqlite3"
        if extractor is None:
            from text_extractor import EnhancedTextExtractionService
            extractor = EnhancedTextExtractionService()
        self.extractor = extractor
        self.ttl_seconds = ttl_seconds
        self._write_lock = threading.RLock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._write_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    storage_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS documents_owner_expiry ON documents(owner, expires_at)")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS extracted_text (
                    cache_key TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
        os.chmod(self.db_path, 0o600)

    @staticmethod
    def _record(row) -> DocumentRecord:
        return DocumentRecord(**dict(row))

    def store_bytes(self, owner: str, filename: str, data: bytes) -> DocumentRecord:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported file type")
        if not data or len(data) > MAX_SOURCE_BYTES:
            raise ValueError("Document is empty or exceeds the size limit")
        document_id = uuid.uuid4().hex
        storage_name = f"{document_id}{suffix}"
        content_hash = hashlib.sha256(data).hexdigest()
        created_at = time.time()
        record = DocumentRecord(
            document_id=document_id,
            owner=owner,
            filename=Path(filename).name,
            storage_name=storage_name,
            content_hash=content_hash,
            size=len(data),
            created_at=created_at,
            expires_at=created_at + self.ttl_seconds,
        )
        path = self.upload_dir / storage_name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
            with self._write_lock, self._connect() as connection:
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    tuple(record.__dict__.values()),
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return record

    def get_records(self, owner: str, sources: list[str]) -> list[DocumentRecord]:
        if not sources or len(sources) > 20:
            raise ValueError("Provide between 1 and 20 document IDs")
        records = []
        now = time.time()
        with self._connect() as connection:
            for source in sources:
                match = UPLOAD_ID_PATTERN.fullmatch(source)
                if not match:
                    raise ValueError("Expected an upload ID")
                row = connection.execute(
                    "SELECT * FROM documents WHERE document_id = ? AND owner = ? AND expires_at > ?",
                    (match.group(1), owner, now),
                ).fetchone()
                if row is None:
                    raise LookupError("Document not found or expired")
                record = self._record(row)
                if not (self.upload_dir / record.storage_name).is_file():
                    raise LookupError("Document content is unavailable")
                records.append(record)
        return records

    def delete(self, owner: str, source: str) -> bool:
        match = UPLOAD_ID_PATTERN.fullmatch(source)
        if not match:
            raise ValueError("Expected an upload ID")
        with self._write_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ? AND owner = ?",
                (match.group(1), owner),
            ).fetchone()
            if row is None:
                return False
            record = self._record(row)
            connection.execute("DELETE FROM documents WHERE document_id = ?", (record.document_id,))
            self._prune_extraction(connection, record)
        (self.upload_dir / record.storage_name).unlink(missing_ok=True)
        return True

    def cleanup_expired(self) -> list[str]:
        now = time.time()
        with self._write_lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents WHERE expires_at <= ?", (now,)).fetchall()
            records = [self._record(row) for row in rows]
            for record in records:
                connection.execute("DELETE FROM documents WHERE document_id = ?", (record.document_id,))
                self._prune_extraction(connection, record)
        for record in records:
            (self.upload_dir / record.storage_name).unlink(missing_ok=True)
        return [record.document_id for record in records]

    @staticmethod
    def _cache_key(record: DocumentRecord) -> str:
        return f"{EXTRACTION_VERSION}:{record.owner}:{Path(record.storage_name).suffix}:{record.content_hash}"

    def _prune_extraction(self, connection, record: DocumentRecord):
        still_used = any(
            Path(row[0]).suffix == Path(record.storage_name).suffix
            for row in connection.execute(
                "SELECT storage_name FROM documents WHERE owner = ? AND content_hash = ?",
                (record.owner, record.content_hash),
            )
        )
        if not still_used:
            connection.execute(
                "DELETE FROM extracted_text WHERE cache_key = ?",
                (self._cache_key(record),),
            )

    def extract(self, record: DocumentRecord) -> ExtractionResult:
        cache_key = self._cache_key(record)
        with self._connect() as connection:
            cached = connection.execute(
                "SELECT text, metadata FROM extracted_text WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if cached:
            metadata = json.loads(cached["metadata"])
            metadata["filename"] = record.storage_name
            metadata["file_size"] = record.size
            return ExtractionResult(text=cached["text"], metadata=metadata)
        path = self.upload_dir / record.storage_name
        with path.open("rb") as document:
            data = document.read(MAX_SOURCE_BYTES + 1)
        if len(data) > MAX_SOURCE_BYTES or hashlib.sha256(data).hexdigest() != record.content_hash:
            raise ValueError("Stored document failed integrity validation")
        result = self.extractor.extract_text_from_bytes(data, record.storage_name)
        if result.success and result.text.strip():
            with self._write_lock, self._connect() as connection:
                still_owned = connection.execute(
                    "SELECT 1 FROM documents WHERE document_id = ? AND owner = ? AND expires_at > ?",
                    (record.document_id, record.owner, time.time()),
                ).fetchone()
                if still_owned:
                    connection.execute(
                        "INSERT OR IGNORE INTO extracted_text VALUES (?, ?, ?, ?, ?)",
                        (cache_key, record.owner, result.text, json.dumps(result.metadata), time.time()),
                    )
        return result

    def stats(self, owner: str) -> dict:
        with self._connect() as connection:
            document_count = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE owner = ? AND expires_at > ?",
                (owner, time.time()),
            ).fetchone()[0]
            extraction_count = connection.execute(
                "SELECT COUNT(*) FROM extracted_text WHERE owner = ?", (owner,)
            ).fetchone()[0]
        return {"documents": document_count, "cached_extractions": extraction_count}

    def clear_extractions(self, owner: str):
        with self._write_lock, self._connect() as connection:
            connection.execute("DELETE FROM extracted_text WHERE owner = ?", (owner,))
