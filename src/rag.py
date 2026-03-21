from typing import Dict
from pathlib import Path
from src.database import RAGDatabase
from src.embeddings import EmbeddingGenerator
from src.retrieval import build_context
from src.generation_v2 import AnswerGenerator
from src.config import TOP_K, MIN_SIMILARITY


class RAGPipeline:
    def __init__(self, markdown_dir: str = None):
        self.db = RAGDatabase()
        self.embedding_gen = EmbeddingGenerator()
        self.answer_gen = AnswerGenerator(max_messages_per_conversation=7)

        if markdown_dir is None:
            project_root = Path(__file__).parent.parent
            markdown_dir = project_root / "data"
        else:
            markdown_dir = Path(markdown_dir)

        self.markdown_dir = markdown_dir

    def query(self, conversation_id: str, question: str) -> Dict:
        rewritten_question = self.answer_gen.rewrite_question(conversation_id, question)
        query_embedding = self.embedding_gen.embed_query(rewritten_question)

        retrieved = self.db.search_similar(query_embedding=query_embedding, top_k=TOP_K)
        best_score = retrieved[0].score if retrieved else 0.0

        # ✅ Regla de confianza:
        # si no hay resultados o score bajo -> responder por criterio LLM (sin contexto RAG)
        use_rag_context = bool(retrieved) and (best_score >= MIN_SIMILARITY)

        context = build_context(retrieved) if use_rag_context else ""

        answer = self.answer_gen.generate_answer(
            conversation_id=conversation_id,
            question=question,
            context=context
        )

        return {
            "answer": answer,
            "sources": [item.chunk.source for item in retrieved] if use_rag_context else [],
            "scores": [round(item.score, 4) for item in retrieved] if use_rag_context else [],
            "retrieved_chunks": [
                {
                    "source": item.chunk.source,
                    "score": round(item.score, 4),
                    "text": item.chunk.text
                }
                for item in retrieved
            ] if use_rag_context else [],
            "original_question": question,
            "rewritten_question": rewritten_question,
            "best_score": round(best_score, 4),
            "used_rag_context": use_rag_context,
            "confidence": "high" if use_rag_context else "low"
        }

    def get_law_protections_for_evidence(self, evidence: str, max_laws: int = 3) -> Dict:
        safe_max_laws = max(1, min(int(max_laws), 3))

        query_embedding = self.embedding_gen.embed_query(evidence)
        retrieval_top_k = max(TOP_K, 12)
        retrieved = self.db.search_similar(query_embedding=query_embedding, top_k=retrieval_top_k)
        best_score = retrieved[0].score if retrieved else 0.0

        # Umbral un poco mas flexible que el chat general para no perder normas potencialmente utiles.
        use_rag_context = bool(retrieved) and (best_score >= 0.15)
        context = build_context(retrieved) if use_rag_context else ""

        legal_result = self.answer_gen.extract_laws_for_evidence(
            evidence=evidence,
            context=context,
            max_laws=safe_max_laws,
        )

        return {
            "evidencia": evidence,
            "best_score": round(best_score, 4),
            "used_rag_context": use_rag_context,
            "leyes": legal_result.get("leyes", []),
            "retrieved_chunks": [
                {
                    "source": item.chunk.source,
                    "score": round(item.score, 4),
                    "text": item.chunk.text,
                }
                for item in retrieved
            ] if use_rag_context else [],
        }

    def clear_conversation(self, conversation_id: str) -> None:
        self.answer_gen.clear_conversation(conversation_id)

    def get_conversation_history(self, conversation_id: str):
        return self.answer_gen.get_history(conversation_id)

    def close(self):
        self.db.close()