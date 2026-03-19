from typing import Dict
from pathlib import Path
from src.database import RAGDatabase
from src.parser import load_markdown_files, chunk_text
from src.embeddings import EmbeddingGenerator
from src.retrieval import retrieve_top_k, build_context
from src.generation_v2 import AnswerGenerator
from src.models import Chunk
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MIN_SIMILARITY, DB_PATH


class RAGPipeline:
    def __init__(self, markdown_dir: str = None):
        self.db = RAGDatabase(DB_PATH)
        self.embedding_gen = EmbeddingGenerator()
        self.answer_gen = AnswerGenerator(max_messages_per_conversation=7)

        if markdown_dir is None:
            project_root = Path(__file__).parent.parent
            markdown_dir = project_root / "data"
        else:
            markdown_dir = Path(markdown_dir)

        self.markdown_dir = markdown_dir

        if not self.markdown_dir.exists():
            print(f"⚠️  Carpeta no encontrada: {self.markdown_dir}")
            print(f"📍 Buscando en: {self.markdown_dir.absolute()}")

    def ingest_from_markdown(self):
        """Ingesta: cargar .md, chunkear, embedear y guardar en BD"""
        print("\n📚 Iniciando ingesta de archivos Markdown...\n")
        print(f"📍 Buscando archivos en: {self.markdown_dir.absolute()}\n")

        documents = load_markdown_files(str(self.markdown_dir))

        if not documents:
            print("⚠️  No se encontraron archivos .md")
            print(f"   Verifica que existan archivos .md en: {self.markdown_dir.absolute()}")
            return

        print(f"✓ Documentos cargados: {len(documents)}\n")

        count_before = self.db.get_chunk_count()
        if count_before > 0:
            self.db.clear_chunks()
            print(f"✓ Base de datos limpiada ({count_before} chunks eliminados)\n")

        all_chunks = []
        all_embeddings = []

        for doc in documents:
            source = doc["source"]
            content = doc["content"]
            pieces = chunk_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

            print(f"  📄 {source}: {len(pieces)} chunks")

            for i, piece in enumerate(pieces):
                chunk = Chunk(
                    chunk_id=f"{source}::chunk_{i}",
                    source=source,
                    text=piece
                )
                all_chunks.append(chunk)

            embeddings = self.embedding_gen.embed_texts(pieces)
            all_embeddings.extend(embeddings)

        import numpy as np
        all_embeddings = np.array(all_embeddings)
        self.db.add_chunks_batch(all_chunks, all_embeddings)

        print(f"\n✓ Ingesta completada: {len(all_chunks)} chunks guardados en BD\n")

    def ingest(self, raw_text: str):
        """Ingesta desde texto plano (mantener para compatibilidad)"""
        from src.parser import parse_documents

        print("\n📚 Iniciando ingesta de texto plano...\n")

        documents = parse_documents(raw_text)
        print(f"✓ Documentos parseados: {len(documents)}")

        count_before = self.db.get_chunk_count()
        if count_before > 0:
            self.db.clear_chunks()

        all_chunks = []
        all_embeddings = []

        for doc in documents:
            source = doc["source"]
            content = doc["content"]
            pieces = chunk_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

            print(f"  • {source}: {len(pieces)} chunks")

            for i, piece in enumerate(pieces):
                chunk = Chunk(
                    chunk_id=f"{source}::chunk_{i}",
                    source=source,
                    text=piece
                )
                all_chunks.append(chunk)

            embeddings = self.embedding_gen.embed_texts(pieces)
            all_embeddings.extend(embeddings)

        import numpy as np
        all_embeddings = np.array(all_embeddings)
        self.db.add_chunks_batch(all_chunks, all_embeddings)

        print(f"\n✓ Ingesta completada: {len(all_chunks)} chunks guardados en BD\n")

    def query(self, conversation_id: str, question: str) -> Dict:
        """Realizar una query RAG con memoria por conversación y reescritura de consulta."""

        chunks, chunk_embeddings = self.db.get_all_chunks()

        if len(chunks) == 0:
            return {
                "answer": "⚠️ Base de conocimiento vacía. Realiza una ingesta primero.",
                "sources": [],
                "scores": [],
                "retrieved_chunks": [],
                "original_question": question,
                "rewritten_question": question
            }

        # Reescribir usando historial reciente de la conversación
        rewritten_question = self.answer_gen.rewrite_question(conversation_id, question)

        # Embedding de la consulta reescrita
        query_embedding = self.embedding_gen.embed_query(rewritten_question)

        retrieved = retrieve_top_k(
            question=rewritten_question,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            query_embedding=query_embedding,
            top_k=TOP_K
        )

        best_score = retrieved[0].score if retrieved else 0.0

        # Si luego quieres reactivar el filtro mínimo:
        # if not retrieved or best_score < MIN_SIMILARITY:
        #     return {
        #         "answer": "No tengo suficiente información en la base para responder con seguridad.",
        #         "sources": [],
        #         "scores": [],
        #         "retrieved_chunks": [],
        #         "original_question": question,
        #         "rewritten_question": rewritten_question
        #     }

        context = build_context(retrieved)

        answer = self.answer_gen.generate_answer(
            conversation_id=conversation_id,
            question=question,
            context=context
        )

        return {
            "answer": answer,
            "sources": [item.chunk.source for item in retrieved],
            "scores": [round(item.score, 4) for item in retrieved],
            "retrieved_chunks": [
                {
                    "source": item.chunk.source,
                    "score": round(item.score, 4),
                    "text": item.chunk.text
                }
                for item in retrieved
            ],
            "original_question": question,
            "rewritten_question": rewritten_question,
            "best_score": round(best_score, 4) if retrieved else 0.0
        }

    def clear_conversation(self, conversation_id: str) -> None:
        self.answer_gen.clear_conversation(conversation_id)

    def get_conversation_history(self, conversation_id: str):
        return self.answer_gen.get_history(conversation_id)

    def close(self):
        """Cerrar conexión a BD"""
        self.db.close()