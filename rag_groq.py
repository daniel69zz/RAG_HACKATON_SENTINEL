import os
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# =========================================================
# MODELOS
# =========================================================

@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


# =========================================================
# CONFIG
# =========================================================

GROQ_MODEL = "llama-3.3-70b-versatile"   # puedes cambiarlo
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # local, no usa API externa

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MIN_SIMILARITY = 0.30

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("No se encontró GROQ_API_KEY en el entorno.")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedder = SentenceTransformer(EMBEDDING_MODEL)

# =========================================================
# TEXTO PLANO DE EJEMPLO
# =========================================================

def answer_question(
    question: str,
    chunks: List[Chunk],
    chunk_embeddings: np.ndarray
) -> Dict:
    retrieved = retrieve_top_k(
        question=question,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=TOP_K
    )

    best_score = retrieved[0].score if retrieved else 0.0

    if not retrieved or best_score < MIN_SIMILARITY:
        return {
            "answer": "No tengo suficiente información en la base para responder con seguridad.",
            "sources": [],
            "scores": [],
            "retrieved_chunks": []
        }

    context = build_context(retrieved)
    answer = generate_answer_with_groq(question, context)

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
        ]
    }

RAW_TEXT = """
[DOC: consentimiento_basico.txt]
El consentimiento significa que una persona acepta de manera libre, clara e informada participar en una situación.
El consentimiento no debe ser forzado, manipulado ni asumido.
Una persona puede cambiar de opinión en cualquier momento.
El silencio o la falta de resistencia no significan consentimiento.

[DOC: relaciones_sanas.txt]
Una relación sana se basa en respeto, comunicación, confianza y límites claros.
Es importante que ambas personas puedan expresar incomodidad sin miedo.
Los celos, el control excesivo y la presión constante no son señales de una relación sana.

[DOC: prevencion_its.txt]
Las infecciones de transmisión sexual pueden prevenirse con educación, medidas de protección y atención médica oportuna.
El uso correcto del preservativo reduce el riesgo de transmisión de varias ITS.
Las pruebas médicas y el acceso a información confiable son importantes para la prevención.

[DOC: anticonceptivos_general.txt]
Los métodos anticonceptivos ayudan a prevenir embarazos.
Existen métodos de barrera, hormonales, intrauterinos y permanentes.
Cada método tiene ventajas, limitaciones y recomendaciones de uso distintas.
La elección del método ideal depende de la situación de cada persona y debe basarse en información confiable.

[DOC: apoyo_y_orientacion.txt]
Si una persona siente presión, miedo o confusión en una relación, buscar apoyo de un adulto de confianza o de un profesional puede ser útil.
En situaciones de riesgo o violencia, es importante acudir a servicios de ayuda, líneas de atención o centros de salud.
La seguridad personal y el acceso a apoyo confiable son prioritarios.
""".strip()


# =========================================================
# PARSER Y CHUNKING
# =========================================================

def parse_documents(raw_text: str) -> List[Dict[str, str]]:
    docs = []
    current_source = None
    current_lines = []

    for line in raw_text.splitlines():
        line = line.strip()

        if line.startswith("[DOC:") and line.endswith("]"):
            if current_source and current_lines:
                docs.append({
                    "source": current_source,
                    "content": "\n".join(current_lines).strip()
                })
                current_lines = []

            current_source = line.replace("[DOC:", "").replace("]", "").strip()
        else:
            if line:
                current_lines.append(line)

    if current_source and current_lines:
        docs.append({
            "source": current_source,
            "content": "\n".join(current_lines).strip()
        })

    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size debe ser mayor que overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks

# =========================================================
# EMBEDDINGS Y SIMILITUD
# =========================================================

def embed_texts(texts: List[str]) -> np.ndarray:
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def embed_query(text: str) -> np.ndarray:
    return embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# =========================================================
# INDEXACIÓN
# =========================================================

def build_knowledge_base(raw_text: str):
    documents = parse_documents(raw_text)
    all_chunks: List[Chunk] = []

    for doc in documents:
        source = doc["source"]
        content = doc["content"]
        pieces = chunk_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{source}::chunk_{i}",
                    source=source,
                    text=piece
                )
            )

    chunk_texts = [c.text for c in all_chunks]
    chunk_embeddings = embed_texts(chunk_texts)

    return all_chunks, chunk_embeddings

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_top_k(
    question: str,
    chunks: List[Chunk],
    chunk_embeddings: np.ndarray,
    top_k: int = 3
) -> List[RetrievedChunk]:
    query_embedding = embed_query(question)

    scored = []
    for chunk, emb in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scored.append(RetrievedChunk(chunk=chunk, score=score))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]

def build_context(retrieved: List[RetrievedChunk]) -> str:
    parts = []
    for item in retrieved:
        parts.append(
            f"Fuente: {item.chunk.source}\n"
            f"Score: {item.score:.4f}\n"
            f"Contenido: {item.chunk.text}"
        )
    return "\n\n".join(parts)

# =========================================================
# GENERACIÓN CON GROQ
# =========================================================

def generate_answer_with_groq(question: str, context: str) -> str:
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente educativo sobre sexualidad. "
                    "Responde SOLO con la información del contexto dado. "
                    "Si el contexto no alcanza, dilo claramente. "
                    "No inventes datos. "
                    "Usa lenguaje claro, neutral y educativo. "
                    "Al final agrega una línea breve con las fuentes usadas."
                )
            },
            {
                "role": "user",
                "content": f"""PREGUNTA:
{question}

CONTEXTO:
{context}"""
            }
        ]
    )

    return completion.choices[0].message.content

# =========================================================
# MAIN
# =========================================================

def main():
    print("Indexando conocimiento...")
    chunks, chunk_embeddings = build_knowledge_base(RAW_TEXT)
    print(f"Chunks creados: {len(chunks)}")
    print("Escribe una pregunta. Usa 'salir' para terminar.\n")

    while True:
        question = input("Tú: ").strip()

        if question.lower() in {"salir", "exit", "quit"}:
            print("Fin.")
            break

        result = answer_question(question, chunks, chunk_embeddings)

        print("\nBot:")
        print(result["answer"])

        print("\nTop chunks recuperados:")
        for i, chunk in enumerate(result["retrieved_chunks"], start=1):
            print(f"\n--- Chunk {i} ---")
            print(f"Fuente: {chunk['source']}")
            print(f"Score: {chunk['score']}")
            print(f"Texto: {chunk['text']}")

        print("\nFuentes:", result["sources"])
        print("Scores:", result["scores"])
        print("-" * 60)
        
if __name__ == "__main__":
    main()