from src.rag import RAGPipeline

def main():
    rag = RAGPipeline()

    print("\nEscribe una pregunta (o 'salir' para terminar):\n")

    while True:
        print("\n" + "=" * 60)
        question = input("Tú: ").strip()

        if question.lower() in {"salir", "exit", "quit"}:
            print("\nFin. ¡Hasta luego!")
            break

        if not question:
            continue

        conversation_id = "chat_principal"
        result = rag.query(conversation_id, question)

        print("\n" + "=" * 60)
        print("RESPUESTA:")
        print(result["answer"])
        print("=" * 60)

        print(f"used_rag_context: {result['used_rag_context']}")
        print(f"best_score: {result['best_score']}")
        print(f"confidence: {result['confidence']}")
        print(f"rewritten_question: {result['rewritten_question']}")

        print("\nCHUNKS RECUPERADOS:")
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"source: {chunk['source']}")
            print(f"score: {chunk['score']}")
            print(chunk["text"])

        print("\n" + "=" * 60 + "\n")

    rag.close()

if __name__ == "__main__":
    main()