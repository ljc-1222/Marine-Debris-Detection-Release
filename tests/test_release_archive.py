"""Check archive integrity and path boundaries with tiny standalone fixtures."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.release_archive import build, safe_name, verify


class SourceArchiveTest(unittest.TestCase):
    def test_build_uses_a_readable_filename_and_only_committed_sources(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=source, check=True)
            (source / "example.py").write_text("print('source fixture')\n")
            subprocess.run(["git", "add", "example.py"], cwd=source, check=True)
            subprocess.run(["git", "-c", "user.name=Archive Fixture", "-c",
                            "user.email=archive-fixture@example.invalid", "commit", "-qm", "Source fixture"],
                           cwd=source, check=True)
            (source / "notes.txt").write_text("Untracked fixture; excluded from the release.\n")
            with patch("scripts.release_archive.SOURCE_REVISIONS", {}):
                manifest = build(root / "release", root=source)
            self.assertEqual(manifest["archive"]["filename"], "marine-debris-source.tar.gz")
            self.assertEqual(set(manifest["files"]), {"marine-debris/example.py"})
            self.assertEqual(verify(root / "release"), manifest)

    def test_archive_and_extracted_file_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            archive = root / "source.tar.gz"
            value = b"source fixture\n"
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("marine-debris/example.py")
                member.size, member.mode = len(value), 0o644
                stream.addfile(member, io.BytesIO(value))
            manifest = {"schema_version": 1, "project_commit": "a" * 40,
                        "archive": {"filename": archive.name, "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
                        "files": {"marine-debris/example.py": {"kind": "file", "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest(), "mode": 0o644}}}
            (root / "source-manifest.json").write_text(json.dumps(manifest))
            extracted = root / "source"
            extracted.mkdir()
            (extracted / "example.py").write_bytes(value)
            verify(root, extracted)
            (extracted / "example.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Extracted source"):
                verify(root, extracted)
            archive.write_bytes(archive.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256/byte"):
                verify(root)

    def test_source_names_cannot_escape(self):
        for value in ("../secret", "/absolute", "source/../secret", "source\\secret", "bad\nname"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_name(value)


if __name__ == "__main__":
    unittest.main()
