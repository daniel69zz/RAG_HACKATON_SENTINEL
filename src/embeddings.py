import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generar embeddings para lista de textos"""
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generar embedding para una query"""
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]