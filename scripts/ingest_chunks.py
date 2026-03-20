from pathlib import Path
import numpy as np

from src.database import RAGDatabase
from src.parser import load_markdown_files, chunk_text
from src.embeddings import EmbeddingGenerator
from src.models import Chunk
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def main():
    project_root = Path(__file__).resolve().parent.parent
    markdown_dir = project_root / "data"

    print("\n📚 Ingesta única de Markdown -> chunks -> embeddings -> PostgreSQL (pgvector)\n")
    print(f"📍 Directorio: {markdown_dir}")

    docs = load_markdown_files(str(markdown_dir))
    if not docs:
        print("⚠️ No hay archivos .md para ingerir.")
        return

    db = RAGDatabase()
    embedder = EmbeddingGenerator()

    print(f"✓ Documentos cargados: {len(docs)}")

    # si quieres ingesta limpia total (una vez)
    db.clear_chunks()

    all_chunks = []
    all_embeddings = []

    for doc in docs:
        source = doc["source"]
        content = doc["content"]
        pieces = chunk_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        print(f"  📄 {source}: {len(pieces)} chunks")

        if not pieces:
            continue

        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{source}::chunk_{i}",
                    source=source,
                    text=piece
                )
            )

        emb = embedder.embed_texts(pieces)
        all_embeddings.extend(emb)

    if not all_chunks:
        print("⚠️ No se generaron chunks.")
        db.close()
        return

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    db.add_chunks_batch(all_chunks, all_embeddings)

    print(f"\n✅ Ingesta completada: {len(all_chunks)} chunks guardados.")
    print(f"📦 Total en BD: {db.get_chunk_count()}")
    db.close()


if __name__ == "__main__":
    main()