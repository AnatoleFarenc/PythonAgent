"""Data models shared by the chunking, indexing and evaluation code."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class BaseModel:
    """Base dataclass providing dict serialisation."""

    def to_dict(self) -> dict[str, Any]:
        """Return the dataclass (recursively) as a plain dict."""
        return asdict(self)


@dataclass(slots=True)
class ChunkModel(BaseModel):
    """A contiguous piece of a source file.

    Attributes:
        id: Position of the chunk in the index.
        file: Path of the source file.
        start_byte: Offset of the first byte of the chunk.
        end_byte: Offset just after the last byte of the chunk.
        start_line: First line of the chunk (1-based).
        end_line: Last line of the chunk (1-based, inclusive).
        paths: Enclosing class/function names, e.g. ``"Foo > bar"``.
    """

    id: int
    file: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    paths: str


@dataclass(slots=True)
class ChunkListModel(BaseModel):
    """Ordered list of chunks; chunk ``i`` matches vector row ``i``."""

    chunks: list[ChunkModel]
