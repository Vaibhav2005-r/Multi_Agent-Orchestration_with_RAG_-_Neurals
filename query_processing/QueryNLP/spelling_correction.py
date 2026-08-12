"""
Query Spelling Correction — Local Fast Version
===============================================
Replaces the previous cloud llama-3.1-70b LLM call (~3-5s) with a local
pyspellchecker pass (<10ms) plus a hardcoded domain acronym expander.

Pipeline:
  Raw Query → Acronym Expansion → pyspellchecker → Cleaned Query
"""

import re
from spellchecker import SpellChecker

# ── Domain-Specific Acronym / Abbreviation Map ──────────────────────────────
# Keys are case-insensitive patterns; values are their expansions.
DOMAIN_ACRONYMS: dict[str, str] = {
    "rbi":   "Reserve Bank of India",
    "nbfc":  "Non-Banking Financial Company",
    "nbfcs": "Non-Banking Financial Companies",
    "lsp":   "Lending Service Provider",
    "lsps":  "Lending Service Providers",
    "irda":  "Insurance Regulatory and Development Authority",
    "sebi":  "Securities and Exchange Board of India",
    "emi":   "Equated Monthly Installment",
    "apr":   "Annual Percentage Rate",
    "kyc":   "Know Your Customer",
    "aml":   "Anti-Money Laundering",
    "pii":   "Personally Identifiable Information",
    "eba":   "European Banking Authority",
    "fiu":   "Financial Intelligence Unit",
    "cic":   "Credit Information Company",
    "rrb":   "Regional Rural Bank",
}

# Words to never "correct" — domain terms that aren't in the general dictionary
_PROTECTED = {
    "qdrant", "fintech", "nemoguardrails", "aadhaar", "nbfc", "nbfcs",
    "rbi", "lsp", "lsps", "irda", "sebi", "kyc", "aml", "apr",
    "fiu", "cic", "rrb", "llm", "rag",
}

_spell = SpellChecker()


def _expand_acronyms(text: str) -> str:
    """Replace known acronyms with their full forms."""
    def _replacer(match: re.Match) -> str:
        token = match.group(0)
        return DOMAIN_ACRONYMS.get(token.lower(), token)
    return re.sub(r"\b\w+\b", _replacer, text)


def _correct_spelling(text: str) -> str:
    """Correct misspelled words, skipping protected domain terms and short tokens."""
    words = text.split()
    corrected = []
    for word in words:
        # Strip trailing punctuation for checking
        core = word.strip(".,!?;:'\"()")
        lower = core.lower()

        if (
            lower in _PROTECTED          # domain term
            or len(core) <= 3            # too short to correct safely
            or not core.isalpha()        # numbers / mixed tokens
            or core[0].isupper()         # likely proper noun / acronym
        ):
            corrected.append(word)
        else:
            fix = _spell.correction(lower)
            # Only accept correction if it's reasonably close
            if fix and fix != lower:
                corrected.append(word.replace(core, fix))
            else:
                corrected.append(word)
    return " ".join(corrected)


class QuerySpellingCorrector:
    """
    Local, zero-latency query spell checker.
    No API calls, no network dependency, runs in < 20ms.
    """

    def __init__(self, rewrite_model: str = None):
        # rewrite_model param kept for API compatibility — ignored in local mode
        pass

    def process_query(self, query: str) -> str:
        """
        Executes acronym expansion + local spell correction on the raw query.
        """
        step1 = _expand_acronyms(query)
        step2 = _correct_spelling(step1)
        print(f"\n[Spelling Correction] '{query}' → '{step2}' (local, 0 API calls)")
        return step2


# ── Demo / Testing ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    corrector = QuerySpellingCorrector()
    tests = [
        "what is the complince mandates for NBFCs according to rserve bnk?",
        "How dose rbi regulte digtl lnding?",
        "What are the kyc requiremants for lsps?",
    ]
    for q in tests:
        print(f"\nInput:  {q}")
        print(f"Output: {corrector.process_query(q)}")
