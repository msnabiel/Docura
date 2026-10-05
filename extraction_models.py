"""Lightweight extraction result shared by storage and format extractors."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractionResult:
    text: str
    metadata: dict[str, Any]
    success: bool = True
    error: str | None = None
    processing_time: float = 0.0
    file_hash: str | None = None

    def __post_init__(self):
        if not self.success and not self.error:
            self.error = "Unknown error occurred"
