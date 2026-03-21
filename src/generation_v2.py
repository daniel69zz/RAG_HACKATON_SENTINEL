import json
import re
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL
from src.ChatMemory import ConversationMemory
from datetime import datetime


_SYSTEM_PROMPT = """
Eres un chatbot educativo y de orientación dirigido principalmente a adolescentes y jóvenes. Brindas información clara, segura, respetuosa y fácil de entender sobre sexualidad, consentimiento, salud sexual y reproductiva, prevención de violencia y derechos sexuales y reproductivos.

Crisis:
- Si la persona indica que está en peligro inmediato o describe una situación de emergencia activa, prioriza decirle que busque ayuda de emergencia antes de dar cualquier información educativa.

Tiempo real (fecha/hora):
- Para preguntas como que dia es hoy, que fecha es hoy o que hora es, usa los METADATOS DEL SISTEMA (fecha/hora) que te entrega el mensaje del usuario.
- No inventes fecha/hora si ya tienes esos metadatos disponibles.

Seguridad:
- No inventes datos específicos no verificados (teléfonos, direcciones, instituciones, leyes locales exactas, estadísticas con números o afirmaciones factuales muy concretas si no estás seguro).
- No des diagnósticos médicos, psicológicos o legales.
- No des instrucciones peligrosas o ilegales.
- En temas sensibles (abuso, violencia, coerción), responde con empatía y pasos seguros; menciona recursos específicos solo si el sistema los proporcionó o están en el CONTEXTO.

Estilo:
- Cercano, empático, calmado y claro.
- Responde primero de forma directa; luego amplía si hace falta.
- No uses un tono moralista ni juzgador.
- Usa lenguaje sencillo y evita tecnicismos innecesarios; si usas un término técnico, explícalo brevemente.
- Usa ejemplos concretos y cercanos a la realidad de una persona joven cuando ayude a entender mejor.
""".strip()


class AnswerGenerator:
    def __init__(self, max_messages_per_conversation: int = 7):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.memory = ConversationMemory(max_messages=max_messages_per_conversation)

    def _get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

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
- Nunca digas frases como “según el contexto...”, “basándome en la información proporcionada...” o similares; integra el conocimiento de forma natural.
- Usa el historial reciente para entender referencias o continuidad.
- Para preguntas de fecha/hora “de hoy/ahora”, usa los METADATOS DEL SISTEMA (no inventes).
- No inventes datos específicos no verificados.
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

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict:
        text = (raw_text or "").strip()
        if not text:
            return {}

        # Primer intento: parsear todo el texto como JSON.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: tomar el primer bloque JSON delimitado por llaves.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    def extract_laws_for_evidence(self, evidence: str, context: str, max_laws: int = 3):
        safe_max = max(1, min(int(max_laws), 3))
        safe_context = context.strip() if context and context.strip() else "[vacío]"

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un analista legal educativo especializado en violencia sexual en Bolivia. "
                    "Tu tarea es extraer leyes/normas protectoras aplicables a una evidencia. "
                    "No inventes leyes, articulos, ni hechos. Usa solo el CONTEXTO RECUPERADO."
                ),
            },
            {
                "role": "user",
                "content": f"""
EVIDENCIA DEL CASO:
{evidence}

CONTEXTO RECUPERADO:
{safe_context}

Devuelve SOLO JSON valido con este formato exacto:
{{
  "leyes": [
    {{
      "ley": "Nombre de la ley/norma",
      "articulos": ["Art. X", "Art. Y"],
      "descripcion_breve": "Explicacion en 1 o 2 frases",
      "por_que_aplica": "Relacion directa con la evidencia"
    }}
  ]
}}

Reglas:
- Maximo {safe_max} leyes (pueden ser menos).
- Si no hay articulos explicitos, devuelve [] en articulos.
- Prioriza normas de proteccion de la victima.
- No incluyas texto fuera del JSON.
""",
            },
        ]

        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=messages,
        )

        raw = completion.choices[0].message.content.strip() if completion.choices else ""
        parsed = self._extract_json_object(raw)

        laws = parsed.get("leyes", []) if isinstance(parsed, dict) else []
        cleaned = []

        for item in laws:
            if not isinstance(item, dict):
                continue

            law_name = str(item.get("ley", "")).strip()
            if not law_name:
                continue

            articles = item.get("articulos", [])
            if not isinstance(articles, list):
                articles = []
            articles = [str(a).strip() for a in articles if str(a).strip()]

            description = str(item.get("descripcion_breve", "")).strip()
            reason = str(item.get("por_que_aplica", "")).strip()

            cleaned.append(
                {
                    "ley": law_name,
                    "articulos": articles,
                    "descripcion_breve": description,
                    "por_que_aplica": reason,
                }
            )

            if len(cleaned) >= safe_max:
                break

        return {
            "leyes": cleaned,
            "raw_model_output": raw,
        }