from typing import List
from src.models import RetrievedChunk

def build_context(retrieved: List[RetrievedChunk]) -> str:
    parts = []
    for item in retrieved:
        parts.append(
            f"Fuente: {item.chunk.source}\n"
            f"Score: {item.score:.4f}\n"
            f"Contenido: {item.chunk.text}"
        )
    return "\n\n".join(parts)