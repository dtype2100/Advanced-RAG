"""Blob / object storage helper for raw source files.

Placeholder for future S3-compatible or local-disk storage integration.
"""

from __future__ import annotations

from pathlib import Path


class LocalBlobStore:
    """Simple local-filesystem blob store for raw uploaded files.

    Suitable for development; replace with S3/GCS adapter for production.
    """

    def __init__(self, base_dir: str = "data/raw") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, content: bytes) -> Path:
        """Write ``content`` to ``base_dir/filename`` and return the path."""
        dest = self.base_dir / filename
        dest.write_bytes(content)
        return dest

    def load(self, filename: str) -> bytes:
        """Read and return the raw bytes of a stored file."""
        return (self.base_dir / filename).read_bytes()

    def exists(self, filename: str) -> bool:
        """Return True if ``filename`` exists in the blob store."""
        return (self.base_dir / filename).exists()

    def list_files(self) -> list[str]:
        """Return a sorted list of all stored filenames."""
        return sorted(p.name for p in self.base_dir.iterdir() if p.is_file())
