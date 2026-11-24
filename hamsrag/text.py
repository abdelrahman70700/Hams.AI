"""Arabic / English text processing shared by indexing, querying and training.

Two levels of normalization are deliberately kept apart:

* ``clean_text``      - light, meaning-preserving cleanup used for text that is shown to users
                        and fed to neural models (embedder, reranker, LLM): Unicode NFKC
                        (folds Arabic presentation forms produced by PDF extraction),
                        tashkeel / tatweel removal, Eastern-Arabic -> ASCII digits,
                        whitespace cleanup.  Orthography (ة, ى, hamza seats) is kept because
                        the neural models were pre-trained on real orthography.
* ``analyze``         - aggressive lexical analysis for BM25: everything above plus alef/ya/
                        ta-marbuta folding, Arabic/Latin script-boundary splitting
                        (``الـpricing`` -> ``ال`` + ``pricing``), Arabic light stemming
                        (Light10-style), English Snowball stemming, stopword removal and
                        number canonicalisation (``١٬٥٠٠`` == ``1,500`` == ``1500``).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import Stemmer  # PyStemmer (Snowball, C implementation)

# --------------------------------------------------------------------------- character tables
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_PUA = re.compile(r"[\uE000-\uF8FF]")
_TATWEEL = "ـ"
_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",  # Arabic-Indic + Extended (Persian/Urdu) digits
    "01234567890123456789",
)
# Applied in clean_text: separators that belong to numbers, and invisible formatting marks.
# Digits are folded to ASCII everywhere (also in text shown to the LLM): small LLMs copy ASCII
# numbers more reliably, and it makes "١٤" and "14" identical for every component.
_PUNCT_MAP = str.maketrans({
    "٫": ".",  # Arabic decimal separator
    "٬": ",",  # Arabic thousands separator
    "٪": "%",  # Arabic percent sign
    "‌": " ", "‍": "", "‎": "", "‏": "",  # ZWNJ/ZWJ/LRM/RLM
    " ": " ",
})
# Applied only in lexical analysis: Arabic sentence punctuation.
_AR_PUNCT = str.maketrans({"،": ",", "؛": ";", "؟": "?"})
_ALEF = re.compile(r"[أإآٱ]")

_AR_LETTER = r"ء-غف-يٱ-ۓ"
_AR_RUN = re.compile(f"[{_AR_LETTER}]+")
_LATIN_RUN = re.compile(r"[A-Za-z]+")
# A token is an Arabic run, a Latin run (with digits allowed after letters, e.g. "B2B", "G15")
# or a number (with optional decimal part).  Script changes always start a new token.
_TOKEN = re.compile(
    rf"[{_AR_LETTER}]+|[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?"
)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")

# --------------------------------------------------------------------------- stopwords
# Small on purpose: BM25's IDF already discounts frequent words; we only drop pure
# function words. Lists are in *analyzed* (normalized) form.
_AR_STOP = {
    "في", "من", "على", "علي", "الى", "الي", "عن", "مع", "او", "ثم", "هل", "ما", "ماذا", "هو", "هي",
    "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "الذين", "ان", "إن", "كان", "كانت", "لا", "لم", "لن",
    "قد", "كل", "اي", "عند", "بين", "حتي", "حتى", "اذا", "و", "ف", "ب", "ل", "ك", "ال", "يا", "به",
    "بها", "له", "لها", "فيه", "فيها", "منه", "منها", "عليه", "عليها",
}
_EN_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by", "from", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "as", "do",
    "does", "did", "what", "which", "who", "whom", "how", "i", "me", "my", "we", "our", "you",
    "your", "can", "could", "would", "should", "will", "about", "there", "if", "so", "any",
}

# Light10 affixes, written in normalized form (ة->ه, ى->ي, أإآ->ا).
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_AR_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")


# --------------------------------------------------------------------------- public API
def clean_text(text: str) -> str:
    """Light normalization for display and for neural models (keeps orthography)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_PUNCT_MAP).translate(_DIGITS)
    t = _TASHKEEL.sub("", t).replace(_TATWEEL, "")
    t = _PUA.sub("", t)  # private-use glyphs (PDF bullets, icon fonts)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def normalize_ar(text: str) -> str:
    """Aggressive Arabic orthographic folding used only for lexical matching."""
    t = _ALEF.sub("ا", text)
    return t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")


def stem_ar(tok: str) -> str:
    """Light10-style light stemmer (Larkey et al.): strip one conjunction/article prefix and
    common suffixes while keeping at least 2-3 letters. Input must already be normalized."""
    if len(tok) >= 4 and tok.startswith("و"):
        tok = tok[1:]
    for p in _AR_PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 2:
            tok = tok[len(p):]
            break
    for s in _AR_SUFFIXES:
        if tok.endswith(s) and len(tok) - len(s) >= 3:  # >=3 avoids ضمان -> ضم
            tok = tok[: -len(s)]
    return tok


@lru_cache(maxsize=1)
def _en_stemmer() -> Stemmer.Stemmer:
    return Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    """Script-aware tokenization on normalized text (no stemming)."""
    t = normalize_ar(clean_text(text)).translate(_AR_PUNCT).lower()
    t = _THOUSANDS.sub("", t)
    return _TOKEN.findall(t)


def analyze(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Full lexical analysis for BM25 (index and query side must use the same function)."""
    out: list[str] = []
    en = _en_stemmer()
    for tok in tokenize(text):
        if tok[0].isdigit():
            out.append(tok.rstrip("0").rstrip(".") if "." in tok else tok)
        elif _AR_RUN.fullmatch(tok):
            if not keep_stopwords and tok in _AR_STOP:
                continue
            out.append(stem_ar(tok))
        else:
            if not keep_stopwords and tok in _EN_STOP:
                continue
            out.append(en.stemWord(tok))
    return out


# --------------------------------------------------------------------------- language
_EXPLICIT_EN = re.compile(r"\b(answer|reply|respond)\s+in\s+english\b|بالانجليزي|بالإنجليزي|باللغة الانجليزية|باللغة الإنجليزية", re.I)
_EXPLICIT_AR = re.compile(r"\b(answer|reply|respond)\s+in\s+arabic\b|بالعربي|باللغة العربية", re.I)


def script_stats(text: str) -> tuple[int, int]:
    """Number of Arabic-script words and Latin-script words."""
    t = clean_text(text)
    return len(_AR_RUN.findall(t)), len(_LATIN_RUN.findall(t))


def detect_language(text: str) -> dict:
    """Return {'lang': 'ar'|'en', 'mixed': bool, 'answer_lang': 'ar'|'en'}.

    The answer language follows the dominant script of the question, unless the user explicitly
    asks for a language ("answer in English", "بالعربي").  Ties go to Arabic: a question with an
    Arabic frame and English product terms (``أبغى أعرف الـpricing``) is an Arabic question.
    """
    ar, en = script_stats(text)
    lang = "ar" if ar >= en and ar > 0 else "en"
    mixed = ar > 0 and en > 0
    answer = lang
    if _EXPLICIT_EN.search(text):
        answer = "en"
    elif _EXPLICIT_AR.search(text):
        answer = "ar"
    return {"lang": lang, "mixed": mixed, "answer_lang": answer}
