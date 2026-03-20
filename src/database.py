from typing import List, Optional
import numpy as np
import psycopg
from psycopg.rows import dict_row
from src.models import Chunk, RetrievedChunk
from src.config import PG_DSN, EMBEDDING_DIM


class RAGDatabase:
    def __init__(self):
        self.conn = psycopg.connect(PG_DSN, row_factory=dict_row)
        self.init_db()

    def init_db(self):
        with self.conn.cursor() as cur:
            # habilitar pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({EMBEDDING_DIM}) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        self.conn.commit()
        print("✓ Base de datos PostgreSQL inicializada con pgvector")

    @staticmethod
    def _to_pgvector_str(embedding: np.ndarray) -> str:
        # formato esperado por pgvector: [0.1,0.2,...]
        return "[" + ",".join(map(str, embedding.astype(np.float32).tolist())) + "]"

    def add_chunks_batch(self, chunks: List[Chunk], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks y embeddings deben tener la misma longitud")

        with self.conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                emb_vec = self._to_pgvector_str(emb)
                cur.execute(
                    """
                    INSERT INTO chunks (chunk_id, source, text, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        source = EXCLUDED.source,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding;
                    """,
                    (chunk.chunk_id, chunk.source, chunk.text, emb_vec)
                )
        self.conn.commit()

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 3) -> List[RetrievedChunk]:
        qvec = self._to_pgvector_str(query_embedding)

        with self.conn.cursor() as cur:
            # cosine distance: <=> (menor es mejor)
            cur.execute(
                """
                SELECT
                    chunk_id,
                    source,
                    text,
                    (1 - (embedding <=> %s::vector)) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (qvec, qvec, top_k)
            )
            rows = cur.fetchall()

        results: List[RetrievedChunk] = []
        for r in rows:
            c = Chunk(chunk_id=r["chunk_id"], source=r["source"], text=r["text"])
            results.append(RetrievedChunk(chunk=c, score=float(r["score"])))
        return results

    def get_chunk_count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM chunks;")
            return int(cur.fetchone()["total"])

    def clear_chunks(self):
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks;")
        self.conn.commit()
        print("✓ Chunks eliminados de la BD")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, source, text FROM chunks WHERE chunk_id = %s;",
                (chunk_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return Chunk(chunk_id=row["chunk_id"], source=row["source"], text=row["text"])

    def close(self):
        if self.conn:
            self.conn.close()