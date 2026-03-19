import sqlite3
import numpy as np
from typing import List, Optional
import json
from src.models import Chunk

class RAGDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.init_db()
    
    def init_db(self):
        """Inicializar base de datos con tabla de chunks"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        print(f"✓ Base de datos inicializada: {self.db_path}")
    
    def add_chunk(self, chunk: Chunk, embedding: np.ndarray):
        """Guardar chunk con embedding en BD"""
        cursor = self.conn.cursor()
        
        # Convertir embedding a bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        cursor.execute('''
            INSERT OR REPLACE INTO chunks (chunk_id, source, text, embedding)
            VALUES (?, ?, ?, ?)
        ''', (chunk.chunk_id, chunk.source, chunk.text, embedding_bytes))
        
        self.conn.commit()
    
    def add_chunks_batch(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Guardar múltiples chunks con embeddings"""
        cursor = self.conn.cursor()
        
        for chunk, embedding in zip(chunks, embeddings):
            embedding_bytes = embedding.astype(np.float32).tobytes()
            cursor.execute('''
                INSERT OR REPLACE INTO chunks (chunk_id, source, text, embedding)
                VALUES (?, ?, ?, ?)
            ''', (chunk.chunk_id, chunk.source, chunk.text, embedding_bytes))
        
        self.conn.commit()
    
    def get_all_chunks(self) -> tuple[List[Chunk], np.ndarray]:
        """Obtener todos los chunks y sus embeddings"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT chunk_id, source, text, embedding FROM chunks ORDER BY chunk_id')
        rows = cursor.fetchall()
        
        chunks = []
        embeddings = []
        
        for row in rows:
            chunk_id, source, text, embedding_bytes = row
            chunks.append(Chunk(chunk_id=chunk_id, source=source, text=text))
            embeddings.append(np.frombuffer(embedding_bytes, dtype=np.float32))
        
        return chunks, np.array(embeddings) if embeddings else np.array([])
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Obtener un chunk específico"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT chunk_id, source, text FROM chunks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        
        if row:
            return Chunk(chunk_id=row[0], source=row[1], text=row[2])
        return None
    
    def clear_chunks(self):
        """Limpiar todos los chunks de la BD"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM chunks')
        self.conn.commit()
        print("✓ Chunks eliminados de la BD")
    
    def get_chunk_count(self) -> int:
        """Contar chunks en BD"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM chunks')
        return cursor.fetchone()[0]
    
    def close(self):
        """Cerrar conexión"""
        if self.conn:
            self.conn.close()