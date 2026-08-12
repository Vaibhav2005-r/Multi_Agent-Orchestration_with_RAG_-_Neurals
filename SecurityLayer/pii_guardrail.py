"""
PII Guardrail — Fast Regex-Based Detector
==========================================
Replaces the deprecated nvidia/gliner-pii NeMo Guardrails model (EOL: 2026-07-27)
with a local regex-based PII scanner. This is synchronous, zero-latency, and
requires no external API calls.

Detects: emails, phone numbers, SSNs, credit card numbers, Aadhaar numbers,
         PAN cards, dates of birth, IP addresses, and passport numbers.
"""

import re
from typing import Optional

# ── PII Patterns ───────────────────────────────────────────────────────────────
_PII_PATTERNS = [
    # Email addresses
    (r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "EMAIL"),
    # Phone numbers (various formats: +91, 0, or bare 10-digit)
    (r"(?:\+?91[\-\s]?)?[6-9]\d{9}", "PHONE"),
    # US SSN
    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "SSN"),
    # Credit/Debit card (13–19 digits, optionally grouped)
    (r"\b(?:\d{4}[\s\-]?){3}\d{1,4}\b", "CREDIT_CARD"),
    # Aadhaar (India) — 12 digits
    (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "AADHAAR"),
    # PAN card (India) — e.g. ABCDE1234F
    (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "PAN"),
    # IPv4 address
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "IP_ADDRESS"),
    # Date of birth patterns (dd/mm/yyyy, mm-dd-yyyy etc.)
    (r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", "DATE_OF_BIRTH"),
    # Passport (generic: letter + 7 digits)
    (r"\b[A-Z]\d{7}\b", "PASSPORT"),
]

_COMPILED = [(re.compile(pat, re.IGNORECASE), label) for pat, label in _PII_PATTERNS]


class PIIGuardrail:
    """
    Fast, local regex-based PII detector.
    Replaces the deprecated nvidia/gliner-pii NeMo Guardrails integration.
    """

    def __init__(self, config_path: Optional[str] = None):
        # config_path kept for API compatibility — ignored in regex mode
        self._detected_types: list = []

    def detect(self, text: str) -> list[str]:
        """
        Returns a list of PII type labels found in the text.
        e.g. ['EMAIL', 'PHONE']  or  [] if clean.
        """
        found = []
        for pattern, label in _COMPILED:
            if pattern.search(text):
                found.append(label)
        return found

    async def check_query(self, query: str) -> str:
        """
        Async-compatible interface (kept for drop-in compatibility with SecurityOrchestrator).
        Returns the original query — detection result is stored in self._detected_types.
        The SecurityOrchestrator compares the returned value to the original to detect PII.
        """
        self._detected_types = self.detect(query)
        if self._detected_types:
            # Return a lightly masked placeholder so SecurityOrchestrator knows PII was found
            return f"[PII_DETECTED:{','.join(self._detected_types)}] {query}"
        return query
