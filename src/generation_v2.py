from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL
from src.ChatMemory import ConversationMemory
from datetime import datetime


class AnswerGenerator:
    def __init__(self, max_messages_per_conversation: int = 7):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.memory = ConversationMemory(max_messages=max_messages_per_conversation)

    def _get_system_prompt(self) -> str:
        return """
Eres un chatbot educativo y de orientación dirigido principalmente a adolescentes y jóvenes. Brindas información clara, segura, respetuosa y fácil de entender sobre sexualidad, consentimiento, salud sexual y reproductiva, prevención de violencia y derechos sexuales y reproductivos.

Regla principal:
- Si el CONTEXTO contiene información útil y relevante, úsalo como base principal.
- Si el CONTEXTO es parcial, úsalo como base y complétalo con conocimiento general confiable.
- Si el CONTEXTO está vacío, no es relevante o no responde la pregunta, responde con normalidad usando conocimiento general confiable y seguro.
- Nunca finjas que una respuesta viene del CONTEXTO si no está allí.

Muy importante (cumple el objetivo del proyecto):
- La ausencia de CONTEXTO útil NO es una razón para negarte a responder preguntas generales.
- Si puedes ayudar con conocimiento general confiable, hazlo.

Tiempo real (fecha/hora):
- Para preguntas como “qué día es hoy”, “qué fecha es hoy”, “qué hora es”, usa los METADATOS DEL SISTEMA (fecha/hora) que te entrega el mensaje del usuario.
- No inventes fecha/hora si ya tienes esos metadatos disponibles.

Seguridad:
- No inventes datos específicos no verificados (teléfonos, direcciones, instituciones, leyes locales exactas, estadísticas con números, fechas “de hoy” o afirmaciones factuales muy concretas si no estás seguro).
- No des diagnósticos médicos, psicológicos o legales.
- No des instrucciones peligrosas o ilegales.
- En temas sensibles (abuso/violencia/coerción), responde con empatía y pasos seguros; menciona recursos específicos solo si el sistema los proporcionó o están en el CONTEXTO.

Estilo:
- Cercano, empático, calmado y claro.
- Responde primero de forma directa; luego amplía si hace falta.
- No uses un tono moralista ni juzgador.
"""

    def _build_user_prompt(self, question: str, context: str) -> str:
        # Contexto puede venir vacío si el retrieval no encontró nada confiable.
        safe_context = context.strip() if context and context.strip() else ""

        # ✅ Metadatos reales del sistema (Python sí conoce la fecha/hora actuales)
        now = datetime.now()
        fecha_actual = now.strftime("%Y-%m-%d")
        hora_actual = now.strftime("%H:%M:%S")

        # ✅ Día de la semana en español (sin depender del locale del sistema)
        dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        dia_semana = dias[now.weekday()]

        return f"""PREGUNTA ACTUAL:
{question}

METADATOS DEL SISTEMA (confiables):
- Fecha actual: {fecha_actual}
- Día de la semana: {dia_semana}
- Hora actual: {hora_actual}

CONTEXTO DISPONIBLE (puede estar vacío):
{safe_context if safe_context else "[vacío]"}

Instrucciones:
- Si el contexto responde la pregunta, úsalo como base principal.
- Si el contexto ayuda parcialmente, complétalo con conocimiento general confiable.
- Si el contexto está vacío o no es útil, responde igualmente usando tu criterio y conocimiento general confiable (no te detengas por falta de contexto).
- No digas “no puedo responder solo porque no hay contexto”.
- Usa el historial reciente para entender referencias o continuidad.
- Para preguntas de fecha/hora “de hoy/ahora”, usa los METADATOS DEL SISTEMA (no inventes).
- No inventes datos específicos no verificados.
- Responde claro, natural y fácil de entender para una persona joven.
- No menciones el contexto si no aporta nada.
"""

    def rewrite_question(self, conversation_id: str, question: str) -> str:
        history_text = self.memory.get_formatted_history(conversation_id)

        messages = [
            {
                "role": "system",
                "content": """
Tu tarea es reescribir la pregunta del usuario para que sea autosuficiente y clara para un sistema de búsqueda RAG.

Reglas:
- Usa el historial reciente solo para resolver referencias ambiguas.
- Mantén el significado original.
- No respondas la pregunta.
- No agregues información inventada.
- Si la pregunta ya es clara por sí sola, devuélvela casi igual.
- Devuelve solo la pregunta reescrita.
"""
            },
            {
                "role": "user",
                "content": f"""HISTORIAL RECIENTE:
{history_text}

PREGUNTA ACTUAL:
{question}
"""
            }
        ]

        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=messages
        )

        rewritten = completion.choices[0].message.content.strip()
        return rewritten or question

    def generate_answer(self, conversation_id: str, question: str, context: str) -> str:
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_user_prompt(question, context)
        history = self.memory.get_messages(conversation_id)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            messages=messages
        )

        answer = completion.choices[0].message.content.strip()

        # Guardar historial: conserva la pregunta original del usuario (no la reescrita)
        self.memory.add_message(conversation_id, "user", question)
        self.memory.add_message(conversation_id, "assistant", answer)

        return answer

    def clear_conversation(self, conversation_id: str) -> None:
        self.memory.clear_conversation(conversation_id)

    def get_history(self, conversation_id: str):
        return self.memory.get_messages(conversation_id)