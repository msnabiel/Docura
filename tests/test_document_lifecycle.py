import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from document_service import DocumentService
from extraction_models import ExtractionResult
from index_store import IndexStore


class FakeExtractor:
    def __init__(self):
        self.calls = 0

    def extract_text_from_bytes(self, data, filename):
        self.calls += 1
        return ExtractionResult(text=data.decode(), metadata={"filename": filename})


class FakeIndex:
    def __init__(self):
        self.chunks = []

    def _create_chunks(self, text, source):
        return [(text, source)]

    def build_indices(self, chunks):
        self.chunks = chunks


class DocumentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.extractor = FakeExtractor()
        self.documents = DocumentService(self.temp.name, extractor=self.extractor, ttl_seconds=60)
        self.indexes = IndexStore(self.documents, index_factory=FakeIndex, max_entries=2, ttl_seconds=60)

    def tearDown(self):
        self.temp.cleanup()

    def test_persistent_records_owner_and_extraction_reuse(self):
        first = self.documents.store_bytes("alice", "one.txt", b"hello")
        second = self.documents.store_bytes("alice", "two.txt", b"hello")
        source = f"upload:{first.document_id}"
        self.assertEqual(self.documents.get_records("alice", [source])[0], first)
        with self.assertRaises(LookupError):
            self.documents.get_records("bob", [source])
        self.documents.extract(first)
        self.documents.extract(second)
        self.assertEqual(self.extractor.calls, 1)
        reopened = DocumentService(self.temp.name, extractor=self.extractor)
        self.assertEqual(reopened.get_records("alice", [source])[0], first)
        self.assertEqual(reopened.stats("alice")["cached_extractions"], 1)
        self.assertEqual(reopened.stats("bob")["cached_extractions"], 0)

    def test_index_reuse_order_and_delete(self):
        first = self.documents.store_bytes("alice", "one.txt", b"one")
        second = self.documents.store_bytes("alice", "two.txt", b"two")
        sources = [f"upload:{first.document_id}", f"upload:{second.document_id}"]
        index, reused = self.indexes.get_or_build("alice", sources)
        self.assertFalse(reused)
        again, reused = self.indexes.get_or_build("alice", sources[::-1])
        self.assertTrue(reused)
        self.assertIs(index, again)
        self.assertEqual(self.indexes.stats("alice")["builds"], 1)
        self.assertFalse(self.documents.delete("bob", sources[0]))
        self.assertTrue(self.documents.delete("alice", sources[0]))
        self.indexes.invalidate([first.document_id])
        self.assertEqual(self.indexes.stats("alice")["cached_indexes"], 0)
        self.assertFalse((Path(self.temp.name) / "uploads" / first.storage_name).exists())
        with self.assertRaises(LookupError):
            self.indexes.get_or_build("alice", sources)

    def test_expiry_cleanup_and_index_ttl(self):
        record = self.documents.store_bytes("alice", "one.txt", b"one")
        source = f"upload:{record.document_id}"
        self.indexes.ttl_seconds = 0.01
        self.indexes.get_or_build("alice", [source])
        time.sleep(0.02)
        _, reused = self.indexes.get_or_build("alice", [source])
        self.assertFalse(reused)
        with self.documents._connect() as connection:
            connection.execute("UPDATE documents SET expires_at = 0 WHERE document_id = ?", (record.document_id,))
        removed = self.documents.cleanup_expired()
        self.indexes.invalidate(removed)
        self.assertEqual(removed, [record.document_id])
        self.assertEqual(self.documents.stats("alice")["documents"], 0)
        self.assertEqual(self.documents.stats("alice")["cached_extractions"], 0)

    def test_concurrent_requests_build_once(self):
        record = self.documents.store_bytes("alice", "one.txt", b"one")
        source = f"upload:{record.document_id}"
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.indexes.get_or_build("alice", [source]), range(4)))
        self.assertEqual(len({id(index) for index, _ in results}), 1)
        self.assertEqual(self.indexes.stats("alice")["builds"], 1)


if __name__ == "__main__":
    unittest.main()
