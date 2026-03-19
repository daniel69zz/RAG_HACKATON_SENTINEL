from src.rag import RAGPipeline

def main():
    rag = RAGPipeline() 
    
    print("=" * 60)
    print("🤖 RAG EDUCATIVO - SEXUALIDAD BASADA EN EVIDENCIA")
    print("=" * 60)
    
    print("\n¿Deseas hacer ingesta de archivos Markdown? (s/n): ", end="")
    if input().lower() in {"s", "si", "sí"}:
        rag.ingest_from_markdown()
    
    # Loop de preguntas
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
        # print("Bot:")
        print(result["answer"])
        
        # print("\n📊 Top chunks recuperados:")
        # for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        #     print(f"\n--- Chunk {i} ---")
        #     print(f"Fuente: {chunk['source']}")
        #     print(f"Score: {chunk['score']}")
        #     print(f"Texto: {chunk['text'][:200]}...")
        
        # print(f"\n📌 Fuentes: {result['sources']}")
        # print(f"📈 Scores: {result['scores']}")
        print("=" * 60 + "\n")
    
    rag.close()


if __name__ == "__main__":
    main()