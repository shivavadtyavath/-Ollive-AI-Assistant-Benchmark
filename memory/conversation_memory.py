"""
Conversation Memory Module
--------------------------
Implements short-term (sliding window) and long-term (vector store) memory
for both assistants. This is the same memory interface used by both OSS and
Frontier assistants to ensure a fair comparison.
"""

from __future__ import annotations
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import os


@dataclass
class Message:
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    model: str = ""

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    Sliding-window short-term memory with optional summarisation hook.
    Keeps the last `max_turns` user+assistant pairs in context.
    """

    SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI personal assistant.
You have access to conversation history and can reference earlier parts of the conversation.
You are knowledgeable, concise, and always prioritise user safety.
If you are unsure about something, say so rather than making things up.
Never produce harmful, biased, or discriminatory content."""

    def __init__(self, max_turns: int = 10, model_name: str = ""):
        self.max_turns = max_turns
        self.model_name = model_name
        self._history: deque[Message] = deque()
        self.session_id: str = f"session_{int(time.time())}"

    def add_user_message(self, content: str) -> None:
        self._history.append(Message(role="user", content=content, model=self.model_name))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self._history.append(Message(role="assistant", content=content, model=self.model_name))
        self._trim()

    def _trim(self) -> None:
        """Keep only the last max_turns * 2 messages (user+assistant pairs)."""
        max_messages = self.max_turns * 2
        while len(self._history) > max_messages:
            self._history.popleft()

    def get_messages_for_api(self) -> List[dict]:
        """Return messages in OpenAI-compatible format with system prompt prepended."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend([m.to_dict() for m in self._history])
        return messages

    def get_history(self) -> List[Message]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self.session_id = f"session_{int(time.time())}"

    def to_json(self) -> str:
        return json.dumps([asdict(m) for m in self._history], indent=2)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._history if m.role == "user")

    def __len__(self) -> int:
        return len(self._history)
