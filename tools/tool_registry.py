"""
Tool Registry
-------------
Lightweight tool-use system. Both assistants can call these tools
when they need real-time information. Tools are invoked via keyword
detection in the user prompt (no function-calling API required,
making it model-agnostic and free).

Available tools:
  - web_search    : DuckDuckGo search (free, no API key)
  - wikipedia     : Wikipedia summary lookup
  - calculator    : Safe math expression evaluator
  - datetime_tool : Current date/time
"""

from __future__ import annotations
import math
import re
import datetime
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    import wikipedia
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None


class ToolRegistry:
    """Registry of callable tools available to both assistants."""

    # Keywords that trigger each tool
    TOOL_TRIGGERS: Dict[str, list] = {
        "web_search":    ["search for", "look up", "find information about", "what is the latest", "current news", "latest news"],
        "wikipedia":     ["wikipedia", "who is", "history of", "biography of"],
        "calculator":    ["calculate", "compute", "solve", "math:", "what is \\d", "\\d+ [+\\-*/] \\d+"],
        "datetime_tool": ["what time", "what date", "today's date", "current date", "current time", "what day"],
    }

    def detect_tool(self, user_message: str) -> Optional[str]:
        """Heuristically detect if a tool should be invoked."""
        msg_lower = user_message.lower()
        # Calculator: explicit keywords or math expressions
        if re.search(r"(calculate|compute|solve)\s", msg_lower):
            return "calculator"
        if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", msg_lower):
            return "calculator"
        # DateTime: explicit date/time questions
        if re.search(r"(what (time|date|day)|today'?s? date|current (date|time|day))", msg_lower):
            return "datetime_tool"
        # Web search: explicit search requests
        if any(t in msg_lower for t in ["search for", "look up", "find information about", "latest news", "current news"]):
            return "web_search"
        # Wikipedia: explicit biography/history requests
        if any(t in msg_lower for t in ["wikipedia", "who is", "history of", "biography of"]):
            return "wikipedia"
        return None

    def run(self, tool_name: str, query: str) -> ToolResult:
        dispatch: Dict[str, Callable] = {
            "web_search":    self._web_search,
            "wikipedia":     self._wikipedia,
            "calculator":    self._calculator,
            "datetime_tool": self._datetime_tool,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return ToolResult(tool_name=tool_name, success=False, output="", error="Unknown tool")
        return fn(query)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _web_search(self, query: str) -> ToolResult:
        if not DDGS_AVAILABLE:
            return ToolResult("web_search", False, "", "duckduckgo-search not installed")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            if not results:
                return ToolResult("web_search", True, "No results found.")
            output = "\n\n".join(
                f"**{r['title']}**\n{r['body']}\nSource: {r['href']}"
                for r in results
            )
            return ToolResult("web_search", True, output)
        except Exception as e:
            return ToolResult("web_search", False, "", str(e))

    def _wikipedia(self, query: str) -> ToolResult:
        if not WIKI_AVAILABLE:
            return ToolResult("wikipedia", False, "", "wikipedia not installed")
        try:
            # Extract the main subject from the query
            clean = re.sub(r"(who is|what is|tell me about|explain|wikipedia|history of)", "", query, flags=re.I).strip()
            summary = wikipedia.summary(clean, sentences=4, auto_suggest=True)
            return ToolResult("wikipedia", True, summary)
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                summary = wikipedia.summary(e.options[0], sentences=4)
                return ToolResult("wikipedia", True, summary)
            except Exception:
                return ToolResult("wikipedia", False, "", "Disambiguation error")
        except Exception as e:
            return ToolResult("wikipedia", False, "", str(e))

    def _calculator(self, query: str) -> ToolResult:
        # Extract math expression from query
        expr = re.sub(r"(calculate|compute|what is|solve|math:)", "", query, flags=re.I).strip()
        expr = expr.rstrip("?").strip()
        # Safe eval: only allow math operations
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "pow": pow})
        try:
            result = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
            return ToolResult("calculator", True, f"{expr} = {result}")
        except Exception as e:
            return ToolResult("calculator", False, "", f"Could not evaluate: {e}")

    def _datetime_tool(self, query: str) -> ToolResult:
        now = datetime.datetime.now()
        output = (
            f"Current date: {now.strftime('%A, %B %d, %Y')}\n"
            f"Current time: {now.strftime('%I:%M %p')}\n"
            f"Timezone: Local system time"
        )
        return ToolResult("datetime_tool", True, output)
