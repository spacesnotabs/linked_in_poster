from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from src.services.rag import ChunkJobManager, ChunkingService


class ChunkingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.workspace = Path(self._tmpdir.name)
        self.source_root = self.workspace / "repo"
        (self.source_root / "docs").mkdir(parents=True, exist_ok=True)
        (self.source_root / "code").mkdir(parents=True, exist_ok=True)
        (self.source_root / "docs" / "guide.md").write_text(
            "# Title\n\nThis is a sample guide used for chunking.\n\n"
            "It contains multiple paragraphs so the document chunker produces slices.\n",
            encoding="utf-8",
        )
        (self.source_root / "code" / "example.py").write_text(
            '"""Example module for chunking tests."""\n\n'
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n",
            encoding="utf-8",
        )
        self.database_path = self.workspace / "chunks.sqlite3"
        self.service = ChunkingService(database_path=self.database_path)

    def test_run_generates_chunks_and_summary(self) -> None:
        summary = self.service.run([self.source_root])
        self.assertEqual(summary.file_count, 2)
        self.assertEqual(summary.files_with_chunks, 2)
        self.assertGreater(summary.chunks_written, 0)

        db_summary = self.service.database.summarize()
        self.assertEqual(db_summary["total_files"], 2)
        self.assertEqual(db_summary["total_chunks"], summary.chunks_written)

        files_table = self.service.database.list_files()
        self.assertEqual(files_table["count"], 2)
        first_file = files_table["items"][0]["file_path"]
        chunks = self.service.database.list_chunks(file_path=first_file)
        self.assertGreater(chunks["count"], 0)

    def test_job_manager_tracks_completion(self) -> None:
        manager = ChunkJobManager(self.service)
        job = manager.start_job([str(self.source_root)])

        deadline = time.time() + 5
        snapshot = None
        while time.time() < deadline:
            snapshot = manager.get(job.job_id)
            if snapshot and snapshot["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["status"], "completed")
        self.assertGreater(snapshot["chunk_count"], 0)
        self.assertGreater(snapshot["files_with_chunks"], 0)
        self.assertTrue(snapshot["logs"])


if __name__ == "__main__":
    unittest.main()
