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
        print(result["answer"])
        print("=" * 60 + "\n")

    rag.close()

if __name__ == "__main__":
    main()