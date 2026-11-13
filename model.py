from dataclasses import asdict, dataclass


@dataclass(slots=True)
class BaseModel:
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ChunkModel(BaseModel):
    id: int
    file: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    paths: str


@dataclass(slots=True)
class ChunkListModel(BaseModel):
    chunks: list[ChunkModel]
