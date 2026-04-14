"""
Caption simplification — three swappable backends.

  none   → PassthroughSimplifier   — returns raw transcript unchanged
  rules  → RuleBasedSimplifier     — spaCy dependency splitting + vocab lookup
  bart   → BARTSimplifier          — facebook/bart-large-cnn summarization (~1.6 GB)

Factory:
    from simplifier import get_simplifier
    s = get_simplifier("rules")
    chunks = s.simplify("The patient demonstrated significant improvement after...")
    # → ["The patient showed big improvement", "after the new treatment"]
"""

import re
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_WORDS        = 10   # hard cap per output caption chunk
LONG_SENT_TOKENS = 40   # trigger dependency splitting above this token count

# ---------------------------------------------------------------------------
# Vocabulary lookup table  (formal/complex → plain)
# Multi-word phrases must be listed before their component words so the
# sorted-by-length pattern matching hits the phrase before the sub-word.
# ---------------------------------------------------------------------------

_VOCAB_RAW: dict[str, str] = {
    # phrases (length-sensitive — keep these first)
    "due to the fact that":  "because",
    "at this point in time": "now",
    "in the event that":     "if",
    "a large number of":     "many",
    "with regard to":        "about",
    "in addition to":        "besides",
    "in order to":           "to",
    "prior to":              "before",
    # single words
    "utilize":      "use",
    "utilise":      "use",
    "commence":     "start",
    "terminate":    "end",
    "approximately":"about",
    "subsequently": "then",
    "however":      "but",
    "therefore":    "so",
    "furthermore":  "also",
    "nevertheless": "still",
    "consequently": "so",
    "demonstrate":  "show",
    "demonstrated": "showed",
    "indicate":     "show",
    "indicated":    "showed",
    "facilitate":   "help",
    "sufficient":   "enough",
    "obtain":       "get",
    "require":      "need",
    "required":     "needed",
    "assist":       "help",
    "attempt":      "try",
    "attempted":    "tried",
    "purchase":     "buy",
    "purchased":    "bought",
    "inquire":      "ask",
    "respond":      "reply",
    "observe":      "see",
    "construct":    "build",
    "determine":    "find",
    "additional":   "more",
    "numerous":     "many",
    "significant":  "big",
    "primary":      "main",
    "initial":      "first",
    "currently":    "now",
    "previously":   "before",
    "implement":    "use",
    "implemented":  "used",
}

# Pre-compile patterns sorted longest-first so phrases match before sub-words
_VOCAB_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b' + re.escape(src) + r'\b', re.IGNORECASE), dst)
    for src, dst in sorted(_VOCAB_RAW.items(), key=lambda kv: len(kv[0]), reverse=True)
]

# ---------------------------------------------------------------------------
# Shared text utilities
# ---------------------------------------------------------------------------

def _remove_parentheticals(text: str) -> str:
    """Strip (…) and […] blocks, then tidy surrounding whitespace/punctuation."""
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\s+([,;:])', r'\1', text)   # fix " , " → ","
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _apply_vocab(text: str) -> str:
    """Replace complex words/phrases with simpler equivalents."""
    for pattern, replacement in _VOCAB_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _cap_words(text: str, limit: int = MAX_WORDS) -> str:
    """Truncate to at most `limit` whitespace-separated tokens."""
    words = text.split()
    return ' '.join(words[:limit])


def _clean_chunk(text: str) -> str:
    """Strip leading/trailing punctuation artifacts and capitalise."""
    text = text.strip().lstrip(',;:-').rstrip(',;:')
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSimplifier(ABC):
    @abstractmethod
    def simplify(self, text: str) -> list[str]:
        """Return a list of simplified caption chunks for one transcript."""
        ...


# ---------------------------------------------------------------------------
# Backend: passthrough
# ---------------------------------------------------------------------------

class PassthroughSimplifier(BaseSimplifier):
    """Returns the raw transcript as a single-element list. No processing."""

    def simplify(self, text: str) -> list[str]:
        return [text] if text.strip() else []


# ---------------------------------------------------------------------------
# Backend: rule-based (spaCy)
# ---------------------------------------------------------------------------

class RuleBasedSimplifier(BaseSimplifier):
    """
    Pipeline per sentence:
      1. Remove parentheticals
      2. If > LONG_SENT_TOKENS tokens, split at dependency clause boundaries
         (dep_ in: advcl, relcl, conj, ccomp)
      3. Apply vocabulary lookup table
      4. Cap each chunk at MAX_WORDS words
    """

    _SPLIT_DEPS = frozenset(('advcl', 'relcl', 'conj', 'ccomp'))

    def __init__(self):
        try:
            import spacy
            # ner not needed and slows things down
            self._nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError:
            raise RuntimeError(
                "spaCy English model missing. Install it with:\n"
                "  python -m spacy download en_core_web_sm"
            )

    # ------------------------------------------------------------------

    def _clause_boundaries(self, sent) -> list[int]:
        """
        Return sorted doc-absolute token indices where a new clause begins.
        We look for tokens whose dependency role marks them as clause heads
        (advcl / relcl / conj / ccomp) and whose head is within the same
        sentence, then take the leftmost token of that clause's subtree.
        """
        bounds: set[int] = set()
        for tok in sent:
            if (
                tok.dep_ in self._SPLIT_DEPS
                and sent.start <= tok.head.i < sent.end
            ):
                subtree_start = min(t.i for t in tok.subtree)
                if subtree_start > sent.start:
                    bounds.add(subtree_start)
        return sorted(bounds)

    def _split_sentence(self, sent) -> list[str]:
        """
        Return one or more plain-text chunks for a single spaCy sentence Span.
        Splits only when the sentence exceeds LONG_SENT_TOKENS real tokens.
        """
        real_tok_count = sum(1 for t in sent if not t.is_space)
        if real_tok_count <= LONG_SENT_TOKENS:
            return [sent.text.strip()]

        boundaries = self._clause_boundaries(sent)
        if not boundaries:
            # No dependency boundary found — return the sentence whole;
            # _cap_words will truncate it later.
            return [sent.text.strip()]

        doc    = sent.doc
        chunks = []
        prev   = sent.start
        for bi in boundaries:
            span_text = doc[prev:bi].text.strip().rstrip(',;')
            if span_text:
                chunks.append(span_text)
            prev = bi
        tail = doc[prev:sent.end].text.strip()
        if tail:
            chunks.append(tail)

        return chunks if chunks else [sent.text.strip()]

    # ------------------------------------------------------------------

    def simplify(self, text: str) -> list[str]:
        text = _remove_parentheticals(text)
        if not text:
            return []

        doc     = self._nlp(text)
        results = []

        for sent in doc.sents:
            for chunk in self._split_sentence(sent):
                chunk = _apply_vocab(chunk)
                chunk = _cap_words(chunk)
                chunk = _clean_chunk(chunk)
                if chunk:
                    results.append(chunk)

        return results


# ---------------------------------------------------------------------------
# Backend: BART (optional – requires transformers)
# ---------------------------------------------------------------------------

class BARTSimplifier(BaseSimplifier):
    """
    Summarises each transcript chunk with facebook/bart-large-cnn.
    Downloads ~1.6 GB on first use (cached in ~/.cache/huggingface).

    Parentheticals are stripped before passing text to the model.
    """

    MODEL_ID = "facebook/bart-large-cnn"

    def __init__(self, max_length: int = 30, min_length: int = 8):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise RuntimeError(
                "transformers not installed. Run:\n"
                "  pip install transformers sentencepiece"
            )
        self._max_length = max_length
        self._min_length = min_length
        self._pipe = hf_pipeline(
            "summarization",
            model=self.MODEL_ID,
            tokenizer=self.MODEL_ID,
        )

    def simplify(self, text: str) -> list[str]:
        text = _remove_parentheticals(text)
        if not text:
            return []

        outputs = self._pipe(
            text,
            max_length=self._max_length,
            min_length=self._min_length,
            do_sample=False,
            truncation=True,
        )
        summary = outputs[0]["summary_text"].strip()
        return [summary] if summary else []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_simplifier(backend: str = "rules") -> BaseSimplifier:
    """
    Return the appropriate simplifier.

    backend: "none" | "rules" | "bart"
    """
    backend = backend.lower()
    if backend == "none":
        return PassthroughSimplifier()
    if backend == "rules":
        return RuleBasedSimplifier()
    if backend == "bart":
        return BARTSimplifier()
    raise ValueError(
        f"Unknown simplifier backend {backend!r}. Choose: none, rules, bart"
    )
