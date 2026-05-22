"""
Observability & Tracing Module
-------------------------------
Lightweight tracing that works with or without LangSmith.
Every LLM call is logged with:
  - model name
  - latency (ms)
  - input/output token estimates
  - safety flags
  - tool usage

Falls back to local JSON logging if LangSmith is not configured.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class LLMTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = ""
    model: str = ""
    model_type: str = ""          # "oss" | "frontier"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    safety_flags: List[str] = field(default_factory=list)
    tool_used: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Tracer:
    """
    Dual-sink tracer: LangSmith (if configured) + local JSONL file.
    """

    LOG_PATH = Path(__file__).parent.parent / "observability" / "traces.jsonl"

    def __init__(self):
        self._langsmith_enabled = self._init_langsmith()
        self.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _init_langsmith(self) -> bool:
        api_key = os.getenv("LANGCHAIN_API_KEY", "")
        if not api_key or api_key == "your_langsmith_api_key_here":
            return False
        try:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ollive-ai-assistants")
            os.environ["LANGCHAIN_API_KEY"] = api_key
            # Quick connectivity check
            import requests as _req
            r = _req.get(
                "https://api.smith.langchain.com/api/v1/workspaces",
                headers={"x-api-key": api_key},
                timeout=5,
            )
            return r.status_code == 200
        except Exception:
            return False

    def log(self, trace: LLMTrace) -> None:
        """Write trace to local JSONL log and optionally to LangSmith."""
        # Always write locally
        try:
            with open(self.LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(trace)) + "\n")
        except Exception:
            pass

        # Send to LangSmith if connected
        if self._langsmith_enabled:
            try:
                import requests as _req
                payload = {
                    "name": f"{trace.model_type.upper()} — {trace.model}",
                    "run_type": "llm",
                    "inputs": {"session_id": trace.session_id},
                    "outputs": {
                        "latency_ms": trace.latency_ms,
                        "prompt_tokens": trace.prompt_tokens,
                        "completion_tokens": trace.completion_tokens,
                        "safety_flags": trace.safety_flags,
                        "tool_used": trace.tool_used,
                        "success": trace.success,
                    },
                    "extra": {"model": trace.model, "model_type": trace.model_type},
                }
                _req.post(
                    "https://api.smith.langchain.com/api/v1/runs",
                    headers={"x-api-key": os.getenv("LANGCHAIN_API_KEY", "")},
                    json=payload,
                    timeout=3,
                )
            except Exception:
                pass  # Never crash on observability failure

    def start_trace(self, model: str, model_type: str, session_id: str) -> LLMTrace:
        return LLMTrace(model=model, model_type=model_type, session_id=session_id)

    def finish_trace(
        self,
        trace: LLMTrace,
        start_time: float,
        response_text: str,
        input_text: str,
        safety_flags: Optional[List[str]] = None,
        tool_used: Optional[str] = None,
        error: Optional[str] = None,
    ) -> LLMTrace:
        trace.latency_ms = round((time.time() - start_time) * 1000, 2)
        trace.prompt_tokens = self._estimate_tokens(input_text)
        trace.completion_tokens = self._estimate_tokens(response_text)
        trace.safety_flags = safety_flags or []
        trace.tool_used = tool_used
        trace.success = error is None
        trace.error = error
        self.log(trace)
        return trace

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    def get_all_traces(self) -> List[Dict[str, Any]]:
        """Load all traces from local log."""
        if not self.LOG_PATH.exists():
            return []
        traces = []
        with open(self.LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return traces

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate stats per model type."""
        traces = self.get_all_traces()
        stats: Dict[str, Any] = {"oss": {}, "frontier": {}}
        for model_type in ["oss", "frontier"]:
            subset = [t for t in traces if t.get("model_type") == model_type]
            if not subset:
                continue
            latencies = [t["latency_ms"] for t in subset if t.get("success")]
            stats[model_type] = {
                "total_calls": len(subset),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
                "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 2),
                "total_tokens": sum(t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in subset),
                "error_rate": round(sum(1 for t in subset if not t.get("success")) / len(subset), 3),
                "safety_triggered": sum(1 for t in subset if t.get("safety_flags")),
            }
        return stats
