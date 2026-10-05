"""Text cleanup shared by answer generation and OCR."""

import re
from typing import Optional


def clean_string_post(text: str) -> str:
    """
    Clean a string by:
    - Replacing escaped quotes (\" -> ")
    - Removing unescaped backslashes (\)
    - Replacing newlines with space
    - Replacing em dashes (\u2014) with --
    - Collapsing multiple spaces
    """
    import re

    # Replace escaped double quotes
    cleaned = text.replace('\\"', '"')

    # Replace em dash Unicode with "--"
    cleaned = cleaned.replace('\u2014', '--')

    # Remove unescaped backslashes (not part of escape sequences)
    # Use regex to avoid removing backslashes that are part of Unicode escapes
    cleaned = re.sub(r'\\(?!u[0-9a-fA-F]{4})', '', cleaned)

    # Replace one or more newline characters with a space
    cleaned = re.sub(r'\n+', ' ', cleaned)

    # Collapse multiple spaces into one
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def clean_ocr_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    return text.strip()

