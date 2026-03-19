from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    
@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float