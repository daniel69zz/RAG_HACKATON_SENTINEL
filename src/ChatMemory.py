from collections import defaultdict, deque
from threading import Lock
from typing import Dict, List


class ConversationMemory:
    def __init__(self, max_messages: int = 7):
        self.max_messages = max_messages
        self._conversations = defaultdict(lambda: deque(maxlen=self.max_messages))
        self._lock = Lock()

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conversations[conversation_id].append({
                "role": role,
                "content": content
            })

    def get_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._conversations[conversation_id])

    def clear_conversation(self, conversation_id: str) -> None:
        with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]

    def get_formatted_history(self, conversation_id: str) -> str:
        history = self.get_messages(conversation_id)

        if not history:
            return "No hay historial reciente."

        lines = []
        for msg in history:
            role = "Usuario" if msg["role"] == "user" else "Asistente"
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)