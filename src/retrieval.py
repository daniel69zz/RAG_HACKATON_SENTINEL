import numpy as np
from typing import List
from src.models import Chunk, RetrievedChunk

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcular similitud coseno entre dos vectores"""
    return float(np.dot(a, b))

def retrieve_top_k(
    question: str,
    chunks: List[Chunk],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int = 3
) -> List[RetrievedChunk]:
    """Recuperar top K chunks más similares"""
    scored = []
    
    for chunk, emb in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scored.append(RetrievedChunk(chunk=chunk, score=score))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]

def build_context(retrieved: List[RetrievedChunk]) -> str:
    """Construir contexto a partir de chunks recuperados"""
    parts = []
    for item in retrieved:
        parts.append(
            f"Fuente: {item.chunk.source}\n"
            f"Score: {item.score:.4f}\n"
            f"Contenido: {item.chunk.text}"
        )
    return "\n\n".join(parts)