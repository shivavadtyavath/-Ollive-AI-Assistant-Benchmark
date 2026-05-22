"""
OSS Assistant - Llama-3.1-8B-Instant via Groq API
---------------------------------------------------
Uses Groq's FREE inference API with Llama-3.1-8B-Instant.

Llama-3.1-8B is a fully open-source model (Meta, Apache 2.0 license),
making this a legitimate OSS assistant. Running it via Groq gives us
reliable, fast inference without needing a local GPU or HF network access.

Primary: Groq API (llama-3.1-8b-instant)
Fallback: HuggingFace Inference API (if HF_API_TOKEN is set and reachable)

Also deployed on HuggingFace Spaces (see hf_space/ directory).
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


class OSSAssistant:
    """
    Personal assistant powered by Llama-3.1-8B-Instant (open-source, Apache 2.0).
    Inference via Groq API (free tier) — same infrastructure as frontier but
    a much smaller, open-source model for a fair OSS comparison.

    Model comparison:
      OSS:      Llama-3.1-8B-Instant  (~8B params, open-source)
      Frontier: Llama-3.3-70B-Versatile (~70B params, open-source but much larger)
    """

    # Open-source model — Llama 3.1 8B (Meta, Apache 2.0)
    OSS_MODEL_ID = os.getenv("OSS_MODEL_ID", "llama-3.1-8b-instant")
    # Display name for UI
    DISPLAY_NAME = "Llama-3.1-8B-Instant (OSS)"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.memory = ConversationMemory(
            max_turns=int(os.getenv("MAX_HISTORY_TURNS", "10")),
            model_name=self.OSS_MODEL_ID,
        )
        self.safety = SafetyLayer()
        self.tools = ToolRegistry()
        self.tracer = Tracer()
        self._client = None
        if GROQ_AVAILABLE and self.api_key and self.api_key != "your_groq_api_key_here":
            self._client = Groq(api_key=self.api_key)

    @property
    def model_name(self) -> str:
        return self.OSS_MODEL_ID

    def chat(self, user_message: str) -> Tuple[str, dict]:
        """
        Process a user message and return (response_text, metadata).
        metadata includes latency, safety flags, tool used, tokens.
        """
        start_time = time.time()
        trace = self.tracer.start_trace(
            model=self.OSS_MODEL_ID,
            model_type="oss",
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
                tool_context = f"\n\n[Tool: {tool_name}]\n{tool_result.output}\n"

        # --- Build prompt with memory ---
        self.memory.add_user_message(sanitized_input)
        messages = self.memory.get_messages_for_api()

        # Inject tool context into last user message if available
        if tool_context:
            messages[-1]["content"] += tool_context

        # --- Call Groq API (OSS model) ---
        response_text = self._call_groq(messages)

        # --- Output Safety Check ---
        output_check = self.safety.check_output(response_text)
        safe_response = output_check.sanitized_text

        # --- Update memory with assistant response ---
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
        """Call Groq API with the OSS model (Llama-3.1-8B)."""
        if self._client is None:
            return self._fallback_response(messages)

        try:
            completion = self._client.chat.completions.create(
                model=self.OSS_MODEL_ID,
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
            f"[OSS Demo Mode - No Groq Key] I received: '{last_user[:100]}'. "
            "Please configure GROQ_API_KEY in your .env file to enable real responses."
        )

    def _build_meta(self, trace: LLMTrace, flags: list, tool: Optional[str]) -> dict:
        return {
            "model": self.OSS_MODEL_ID,
            "model_type": "OSS",
            "latency_ms": trace.latency_ms,
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "safety_flags": flags,
            "tool_used": tool,
            "turn_count": self.memory.turn_count,
        }

    def reset(self) -> None:
        self.memory.clear()
