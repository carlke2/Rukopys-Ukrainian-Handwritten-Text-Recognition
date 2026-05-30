"""
src/postprocessing/normalize_text.py  —  Clean raw OCR output.

WHY TEXT NORMALIZATION:
  Raw model output often contains:
    - Extra whitespace around words
    - Repeated spaces between words
    - Unicode artifacts (different dash variants, zero-width chars)
    - Mixed half-width / full-width characters
    - Trailing punctuation inconsistencies

  Normalizing before scoring improves CER because these are not
  meaningful differences — just formatting noise.

  NOTE: Apply normalization to BOTH prediction AND ground truth
  before computing CER, otherwise you penalize the model for things
  that are not actual errors.

USAGE:
  from src.postprocessing.normalize_text import normalize

  clean = normalize("  Доброго   ранку  ")  # → "Доброго ранку"
"""

import re
import unicodedata


# Unicode replacements: map look-alike characters to their canonical form.
UNICODE_MAP = {
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u00ab": '"',   # left guillemet
    "\u00bb": '"',   # right guillemet
    "\u2026": "...", # ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero-width space
    "\u200c": "",    # zero-width non-joiner
    "\u200d": "",    # zero-width joiner
    "\ufeff": "",    # BOM
}

# Latin to Cyrillic lookalikes (crucial for OCR)
# Latin c, o, p, x, a, e, i, y, k, m, n, t, h → Cyrillic equivalents
LOOKALIKES = {
    "c": "с", "o": "о", "p": "р", "x": "х", "a": "а", "e": "е",
    "i": "і", "y": "у", "k": "к", "m": "м", "n": "п", "t": "т", "h": "н",
    "C": "С", "O": "О", "P": "Р", "X": "Х", "A": "А", "E": "Е",
    "I": "І", "Y": "У", "K": "К", "M": "М", "N": "П", "T": "Т", "H": "Н",
}

# LaTeX symbol mappings (per competition rules)
LATEX_SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ",
    r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ",
    r"\nu": "ν", r"\xi": "ξ", r"\omicron": "ο", r"\pi": "π",
    r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ",
    r"\phi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\cdot": "·", r"\rightarrow": "→", r"\leftarrow": "←",
    r"\infty": "∞", r"\approx": "≈", r"\neq": "≠", r"\leq": "≤", r"\geq": "≥",
}

# Superscripts and Subscripts
SUPER_SUB_MAP = {
    "²": "^2", "³": "^3", "¹": "^1", "⁰": "^0",
    "⁴": "^4", "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
    "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
}


def normalize(text: str, lowercase: bool = False) -> str:
    """
    Normalize a text string for consistent comparison.
    Aligned with 'Handwritten to Data' challenge rules.
    """
    if not isinstance(text, str):
        return ""

    # 1. Strikethrough: ~~old~~{new} -> new; ~~text~~ -> text
    # First, handle the replacement case
    text = re.sub(r"~~.*?~~\{(.*?)\}", r"\1", text)
    # Then handle the simple strikethrough
    text = re.sub(r"~~(.*?)~~", r"\1", text)

    # 2. LaTeX symbols
    for sym, val in LATEX_SYMBOLS.items():
        text = text.replace(sym, val)

    # 3. LaTeX braces: x_{3} -> x_3
    text = re.sub(r"([_^])\{(.*?)\}", r"\1\2", text)

    # 4. Superscripts/Subscripts
    for src, dst in SUPER_SUB_MAP.items():
        text = text.replace(src, dst)

    # 5. Latin/Cyrillic lookalikes
    # Note: only replace if the surrounding context is Cyrillic or if it's a standalone char
    # For simplicity in metric calculation, we often replace all.
    for src, dst in LOOKALIKES.items():
        text = text.replace(src, dst)

    # 6. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 7. Map common Unicode variants (dashes, quotes)
    for src, dst in UNICODE_MAP.items():
        text = text.replace(src, dst)

    # 8. Collapse whitespace and strip
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if lowercase:
        text = text.lower()

    return text


def normalize_batch(
    texts: list[str],
    lowercase: bool = False,
) -> list[str]:
    """Normalize a list of texts."""
    return [normalize(t, lowercase=lowercase) for t in texts]


if __name__ == "__main__":
    tests = [
        "  Доброго   ранку  ",
        "E\u2013mc\u00b2",
        "текст\u200bз\u200bнульовими\u200bпробілами",
        "«Привіт»",
    ]
    for t in tests:
        print(f"  IN : {repr(t)}")
        print(f"  OUT: {repr(normalize(t))}")
        print()
    print("✅ Normalization test complete.")
