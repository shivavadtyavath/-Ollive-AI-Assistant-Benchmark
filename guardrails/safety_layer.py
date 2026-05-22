"""
Safety & Guardrails Layer
--------------------------
Multi-layer safety system applied to BOTH input (user prompt) and
output (model response). Inspired by production AI safety practices.

Layers:
  1. Keyword / pattern blocklist  (fast, zero-cost)
  2. Profanity filter             (better-profanity)
  3. PII detection & redaction    (regex-based, no external API)
  4. Jailbreak pattern detection  (heuristic)
  5. Output toxicity check        (keyword-based fallback)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Tuple, List
from better_profanity import profanity

profanity.load_censor_words()


# ---------------------------------------------------------------------------
# Jailbreak / prompt-injection patterns
# ---------------------------------------------------------------------------
JAILBREAK_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore (all |previous |prior )?(instructions?|prompts?|rules?)", re.I),
    re.compile(r"you are now (dan|jailbroken|unrestricted|evil|without restrictions)", re.I),
    re.compile(r"pretend (you are|to be) (an? )?(evil|unrestricted|unfiltered|jailbroken)", re.I),
    re.compile(r"do anything now", re.I),
    re.compile(r"(bypass|override|disable) (safety|filter|guardrail|restriction)", re.I),
    re.compile(r"act as (if you have no|without any) (restriction|limit|filter|rule)", re.I),
    re.compile(r"(forget|disregard|ignore) (your|all) (training|guidelines|rules|ethics)", re.I),
    re.compile(r"developer mode", re.I),
    re.compile(r"sudo (mode|override)", re.I),
    re.compile(r"you must (comply|obey|follow) (without|regardless)", re.I),
]

# ---------------------------------------------------------------------------
# Harmful topic blocklist (input)
# ---------------------------------------------------------------------------
HARMFUL_TOPICS: List[re.Pattern] = [
    re.compile(r"how to (make|build|create|synthesize) (a |an )?(bomb|explosive|weapon|poison|drug)", re.I),
    re.compile(r"make (a |an )?(bomb|explosive|weapon|poison|drug)", re.I),
    re.compile(r"child (porn|abuse|exploitation|grooming)", re.I),
    re.compile(r"suicide (method|how to|instructions?)", re.I),
    re.compile(r"hack (into|a|the) (bank|government|hospital|system)", re.I),
    re.compile(r"(ransomware|malware) (code|script|source)", re.I),
    re.compile(r"(synthesize|manufacture) (meth|heroin|fentanyl|cocaine)", re.I),
]

# ---------------------------------------------------------------------------
# PII patterns (for output redaction)
# ---------------------------------------------------------------------------
PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL",   re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")),
    ("PHONE",   re.compile(r"\b(\+?\d[\d\s\-().]{7,}\d)\b")),
    ("SSN",     re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT",  re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("IP",      re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
]


@dataclass
class SafetyResult:
    is_safe: bool
    flags: List[str]
    sanitized_text: str
    risk_level: str   # "none" | "low" | "medium" | "high"

    @property
    def blocked(self) -> bool:
        return not self.is_safe


class SafetyLayer:
    """
    Stateless safety checker. Call check_input() before sending to model,
    and check_output() before displaying to user.
    """

    def check_input(self, text: str) -> SafetyResult:
        flags: List[str] = []
        sanitized = text

        # Layer 1: Harmful topics
        for pattern in HARMFUL_TOPICS:
            if pattern.search(text):
                flags.append("HARMFUL_TOPIC")
                break

        # Layer 2: Jailbreak detection
        for pattern in JAILBREAK_PATTERNS:
            if pattern.search(text):
                flags.append("JAILBREAK_ATTEMPT")
                break

        # Layer 3: Profanity
        if profanity.contains_profanity(text):
            flags.append("PROFANITY")
            sanitized = profanity.censor(text)

        is_safe = "HARMFUL_TOPIC" not in flags and "JAILBREAK_ATTEMPT" not in flags
        risk_level = self._compute_risk(flags)

        return SafetyResult(
            is_safe=is_safe,
            flags=flags,
            sanitized_text=sanitized,
            risk_level=risk_level,
        )

    def check_output(self, text: str) -> SafetyResult:
        flags: List[str] = []
        sanitized = text

        # Redact PII from model output
        for label, pattern in PII_PATTERNS:
            if pattern.search(text):
                flags.append(f"PII_{label}")
                sanitized = pattern.sub(f"[{label} REDACTED]", sanitized)

        # Profanity in output
        if profanity.contains_profanity(text):
            flags.append("OUTPUT_PROFANITY")
            sanitized = profanity.censor(sanitized)

        is_safe = True  # Output warnings are non-blocking (we redact instead)
        risk_level = self._compute_risk(flags)

        return SafetyResult(
            is_safe=is_safe,
            flags=flags,
            sanitized_text=sanitized,
            risk_level=risk_level,
        )

    @staticmethod
    def _compute_risk(flags: List[str]) -> str:
        if not flags:
            return "none"
        if any(f in flags for f in ["HARMFUL_TOPIC", "JAILBREAK_ATTEMPT"]):
            return "high"
        if any(f in flags for f in ["PROFANITY", "OUTPUT_PROFANITY"]):
            return "medium"
        return "low"

    @staticmethod
    def get_refusal_message(flags: List[str]) -> str:
        if "JAILBREAK_ATTEMPT" in flags:
            return (
                "I noticed this message contains patterns that attempt to override my "
                "safety guidelines. I'm designed to be helpful, harmless, and honest — "
                "I can't bypass those principles. How can I help you with something else?"
            )
        if "HARMFUL_TOPIC" in flags:
            return (
                "I'm not able to assist with that request as it involves potentially "
                "harmful content. If you're in distress, please reach out to appropriate "
                "support services. I'm happy to help with other topics."
            )
        return "I'm unable to process that request. Please try rephrasing."
