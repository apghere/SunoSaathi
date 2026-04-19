"""
NLP simplification module — re-exports simplifier backends and adds
keyword extraction for sign-language glossary lookup.
"""

from __future__ import annotations

# Re-export the existing simplifier backends
from simplifier import (  # noqa: F401
    BaseSimplifier,
    PassthroughSimplifier,
    RuleBasedSimplifier,
    BARTSimplifier,
    get_simplifier,
)


# ---------------------------------------------------------------------------
# Keyword extraction (used for sign-glossary lookup)
# ---------------------------------------------------------------------------

def extract_keywords(text: str, max_keywords: int = 6) -> list[str]:
    """Extract content words (nouns, verbs, adjectives) from *text*.

    Used to look up matching sign animations in the glossary.

    Falls back to simple stop-word filtering when spaCy is not available.

    Parameters
    ----------
    text         : input string (typically a simplified caption chunk)
    max_keywords : maximum number of keywords to return

    Returns
    -------
    list of lowercase lemma strings
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        doc = nlp(text.lower())
        keywords = [
            token.lemma_
            for token in doc
            if token.pos_ in ("NOUN", "VERB", "ADJ", "PROPN")
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 2
        ]
        return keywords[:max_keywords]
    except Exception:
        return _fallback_keywords(text, max_keywords)


def extract_keywords_from_captions(captions: list[str], max_per_chunk: int = 3) -> list[str]:
    """Extract keywords from all caption chunks, deduplicated, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for cap in captions:
        for kw in extract_keywords(cap, max_per_chunk):
            if kw not in seen:
                seen.add(kw)
                result.append(kw)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their", "its",
    "and", "or", "but", "so", "yet", "for", "nor", "not", "very",
    "just", "also", "even", "still", "than", "then", "when", "where",
    "how", "what", "who", "which", "with", "from", "into", "onto",
    "upon", "about", "over", "under", "after", "before", "since",
    "until", "while", "because", "although", "though", "if",
})


def _fallback_keywords(text: str, max_keywords: int) -> list[str]:
    import re
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2][:max_keywords]
