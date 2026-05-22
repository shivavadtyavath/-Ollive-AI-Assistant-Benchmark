"""
Frontier Model Assistant - Llama-3.3-70B via Groq API
------------------------------------------------------
Uses Groq's FREE inference API for ultra-fast LLM responses.
Model: llama-3.3-70b-versatile (state-of-the-art, free on Groq)

Groq provides the fastest LLM inference available (up to 750 tokens/sec)
making it ideal for real-time assistant applications.
"""

from __future__ import annotations
import os
import time
from typing import Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential

from memory.conversation_memory import ConversationMemory
from guardrails.safety_layer import SafetyLayer
from tools.tool_registry import ToolRegistry
from observability.tracer import Tracer, LLMTrace

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class FrontierAssistant:
    """
    Personal assistant powered by Llama-3.3-70B via Groq (frontier-class model).
    Groq provides free API access with generous rate limits.
    """

    MODEL_ID = os.getenv("FRONTIER_MODEL_ID", "llama-3.3-70b-versatile")

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.memory = ConversationMemory(
            max_turns=int(os.getenv("MAX_HISTORY_TURNS", "10")),
            model_name=self.MODEL_ID,
        )
        self.safety = SafetyLayer()
        self.tools = ToolRegistry()
        self.tracer = Tracer()
        self._client = None
        if GROQ_AVAILABLE and self.api_key and self.api_key != "your_groq_api_key_here":
            self._client = Groq(api_key=self.api_key)

    @property
    def model_name(self) -> str:
        return self.MODEL_ID

    def chat(self, user_message: str) -> Tuple[str, dict]:
        """
        Process a user message and return (response_text, metadata).
        """
        start_time = time.time()
        trace = self.tracer.start_trace(
            model=self.MODEL_ID,
            model_type="frontier",
            session_id=self.memory.session_id,
        )

        # --- Input Safety Check ---
        input_check = self.safety.check_input(user_message)
        if input_check.blocked:
            refusal = self.safety.get_refusal_message(input_check.flags)
            self.tracer.finish_trace(
                trace, start_time, refusal, user_message,
                safety_flags=input_check.flags,
            )
            return refusal, self._build_meta(trace, input_check.flags, None)

        sanitized_input = input_check.sanitized_text

        # --- Tool Detection ---
        tool_name = self.tools.detect_tool(sanitized_input)
        tool_context = ""
        if tool_name:
            tool_result = self.tools.run(tool_name, sanitized_input)
            if tool_result.success:
                tool_context = (
                    f"\n\n[Real-time data from {tool_name}]:\n{tool_result.output}\n"
                    "Please use this information to answer the user's question accurately."
                )

        # --- Build messages with memory ---
        self.memory.add_user_message(sanitized_input)
        messages = self.memory.get_messages_for_api()

        # Inject tool context into last user message
        if tool_context:
            messages[-1]["content"] += tool_context

        # --- Call Groq API ---
        response_text = self._call_groq(messages)

        # --- Output Safety Check ---
        output_check = self.safety.check_output(response_text)
        safe_response = output_check.sanitized_text

        # --- Update memory ---
        self.memory.add_assistant_message(safe_response)

        # --- Finish trace ---
        all_flags = input_check.flags + output_check.flags
        self.tracer.finish_trace(
            trace, start_time, safe_response,
            str(messages),
            safety_flags=all_flags,
            tool_used=tool_name,
        )

        return safe_response, self._build_meta(trace, all_flags, tool_name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def _call_groq(self, messages: list) -> str:
        """Call Groq API with retry logic."""
        if self._client is None:
            return self._fallback_response(messages)

        try:
            completion = self._client.chat.completions.create(
                model=self.MODEL_ID,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stream=False,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                return "Rate limit reached. Please wait a moment and try again."
            if "invalid_api_key" in error_msg.lower():
                return "Invalid API key. Please check your GROQ_API_KEY in .env"
            return f"I encountered an error: {error_msg}"

    def _fallback_response(self, messages: list) -> str:
        """Demo response when no API key is configured."""
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return (
            f"[Frontier Demo Mode - No Groq Key] I received: '{last_user[:100]}'. "
            "Please configure GROQ_API_KEY in your .env file to enable real responses."
        )

    def _build_meta(self, trace: LLMTrace, flags: list, tool: Optional[str]) -> dict:
        return {
            "model": self.MODEL_ID,
            "model_type": "Frontier",
            "latency_ms": trace.latency_ms,
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "safety_flags": flags,
            "tool_used": tool,
            "turn_count": self.memory.turn_count,
        }

    def reset(self) -> None:
        self.memory.clear()
