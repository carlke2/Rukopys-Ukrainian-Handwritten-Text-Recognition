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
# This handles copy-paste artifacts from different keyboard layouts.
UNICODE_MAP = {
    "\u2013": "-",   # en dash → hyphen
    "\u2014": "-",   # em dash → hyphen
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u00ab": '"',   # left guillemet
    "\u00bb": '"',   # right guillemet
    "\u2026": "...", # ellipsis → three dots
    "\u00a0": " ",   # non-breaking space → regular space
    "\u200b": "",    # zero-width space → remove
    "\u200c": "",    # zero-width non-joiner → remove
    "\u200d": "",    # zero-width joiner → remove
    "\ufeff": "",    # BOM → remove
}


def normalize(text: str, lowercase: bool = False) -> str:
    """
    Normalize a text string for consistent comparison.

    Steps applied:
      1. Unicode NFC normalization (canonical form)
      2. Replace common Unicode look-alikes (dashes, quotes, etc.)
      3. Collapse multiple whitespace into single space
      4. Strip leading/trailing whitespace
      5. Optionally lowercase

    Args:
        text:      raw text string
        lowercase: if True, convert to lowercase (default False —
                   Ukrainian proper nouns are case-sensitive)

    Returns:
        cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Step 1: NFC normalization (combine accented characters)
    text = unicodedata.normalize("NFC", text)

    # Step 2: replace look-alike characters
    for src, dst in UNICODE_MAP.items():
        text = text.replace(src, dst)

    # Step 3: collapse whitespace (spaces, tabs, newlines → single space)
    text = re.sub(r"\s+", " ", text)

    # Step 4: strip
    text = text.strip()

    # Step 5: optional lowercase
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
