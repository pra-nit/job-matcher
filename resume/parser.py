"""
Resume text extraction (PDF or plain text).

Tries PyMuPDF first (fast, accurate), falls back to pypdf, and accepts
``.txt``/``.md`` files directly.  Imports are lazy so the module works in
environments without PDF libraries installed.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_CHARS = 200_000


def extract_resume_text(path: Path) -> str:
    """Extract resume text from PDF/TXT/MD. Raises ValueError on hard failure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".text"}:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        text = _extract_pdf(path)
    else:
        raise ValueError(f"Unsupported resume format '{suffix}' (expected .pdf or .txt)")

    text = _clean(text)
    if len(text.strip()) < 40:
        raise ValueError(
            f"Extracted only {len(text.strip())} characters from {path.name}; "
            "the PDF may be scanned/image-only. Try a text-based PDF."
        )
    logger.info("Extracted %d characters of resume text from %s", len(text), path.name)
    return text


def _extract_pdf(path: Path) -> str:
    errors: list[str] = []
    try:
        try:
            import pymupdf as pdf_lib  # modern PyMuPDF
        except ImportError:  # older releases expose only `fitz`
            import fitz as pdf_lib

        with pdf_lib.open(path) as doc:
            pages = [page.get_text("text") for page in doc]
        return "\n".join(pages)
    except ImportError:
        errors.append("PyMuPDF not installed")
    except Exception as exc:
        errors.append(f"PyMuPDF failed: {exc}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # pragma: no cover - damaged page
                continue
        return "\n".join(pages)
    except ImportError:
        errors.append("pypdf not installed")
    except Exception as exc:
        errors.append(f"pypdf failed: {exc}")

    raise ValueError(
        "Could not extract PDF text (" + "; ".join(errors) + "). "
        "Install PyMuPDF or pypdf (see requirements.txt)."
    )


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:_MAX_CHARS]


def guess_name_from_text(text: str) -> Optional[str]:
    """Best-effort: the first non-empty, short line without digits/@."""
    for line in text.splitlines()[:10]:
        line = line.strip()
        if not line or len(line) > 60:
            continue
        if re.search(r"\d|@|http|www\.|\.com", line, re.I):
            continue
        if line.lower().strip(": ") in {"resume", "curriculum vitae", "cv"}:
            continue
        if len(line.split()) <= 5:
            return line
    return None
