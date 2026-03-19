from groq import Groq
from typing import List
from src.models import RetrievedChunk
from src.config import GROQ_API_KEY, GROQ_MODEL

class AnswerGenerator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generar respuesta usando Groq"""
        completion = self.client.chat.completions.create(
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