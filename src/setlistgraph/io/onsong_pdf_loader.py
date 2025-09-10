from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

# soft deps; keep imports inside functions to avoid hard runtime reqs
from setlistgraph.io.onsong_loader import import_onsong_to_catalog  # reuse existing pipeline

HEADER_KV = re.compile(r"^\s*([A-Za-zÄÖÅäöå]+)\s*:\s*(.+)$")

def _normalize_text(s: str) -> str:
    # de-hyphenate common line wraps: "ord-\net" -> "ordet"
    s = re.sub(r"-\s*\n", "", s)
    # collapse over-wrapped lines
    s = re.sub(r"[ \t]+\n", "\n", s)
    # collapse 3+ blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _pdf_text(pdf_path: str, ocr_fallback: bool = True, dpi: int = 300, max_pages: Optional[int] = None) -> str:
    """
    Try embedded text first (pdfplumber). If too little text, fallback to OCR (pytesseract + pdf2image).
    """
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            parts = []
            for p in pages:
                parts.append(p.extract_text() or "")
            text = "\n".join(parts)
    except Exception:
        text = ""

    if (not text or len(text.strip()) < 80) and ocr_fallback:
        try:
            from pdf2image import convert_from_path
            import pytesseract
            pages = convert_from_path(pdf_path, dpi=dpi)
            ocr_parts = []
            for img in (pages[:max_pages] if max_pages else pages):
                ocr_parts.append(pytesseract.image_to_string(img))
            text = "\n".join(ocr_parts)
        except Exception as e:
            # leave text possibly empty; caller will handle
            pass

    return _normalize_text(text or "")

def _infer_title_artist(raw_lines: List[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Heuristic: if no explicit "Title:" or "Artist:" lines exist,
    use the first non-empty line as title, second as artist if it smells like a name.
    """
    lines = [ln.strip() for ln in raw_lines if ln.strip()]
    if not lines:
        return None, None
    title = lines[0]
    artist = None
    if len(lines) > 1 and len(lines[1].split()) <= 5:
        artist = lines[1]
    return title, artist

def _as_onsong_text(extracted: str, title: Optional[str], artist: Optional[str], key: Optional[str], tempo: Optional[float]) -> str:
    """
    Wrap the PDF text in a light OnSong-like header so the existing importer can parse it.
    """
    header = []
    if title:  header.append(f"Title: {title}")
    if artist: header.append(f"Artist: {artist}")
    if key:    header.append(f"Key: {key}")
    if tempo:  header.append(f"Tempo: {int(tempo)}")
    hdr = "\n".join(header)
    body = extracted.strip()
    return f"{hdr}\n\n{body}" if hdr else body

def import_onsong_pdf_to_catalog(
    pdf_path: str,
    catalog_path: str,
    lyrics_dir: str = "lyrics_private",
    default_bpm: float = 120.0,
    key: Optional[str] = None,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    ocr_fallback: bool = True,
    dpi: int = 300,
    max_pages: Optional[int] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract text (OCR if needed) from a PDF chord/lyric sheet, convert to OnSong-ish text,
    and reuse the normal OnSong importer (saves plain lyrics, appends to catalog).
    """
    text = _pdf_text(pdf_path, ocr_fallback=ocr_fallback, dpi=dpi, max_pages=max_pages)
    if not text.strip():
        raise ValueError(f"Could not extract text from PDF: {pdf_path}")

    # If not explicitly provided, infer title/artist from top lines
    if not (title and artist):
        lines = text.splitlines()
        t_guess, a_guess = _infer_title_artist(lines[:8])
        title = title or t_guess
        artist = artist or a_guess

    onsong_text = _as_onsong_text(text, title=title, artist=artist, key=key, tempo=default_bpm)
    row = import_onsong_to_catalog(
        onsong_text=onsong_text,
        catalog_path=catalog_path,
        lyrics_dir=lyrics_dir,
        default_bpm=default_bpm,
        **(extra_meta or {})
    )
    # Stash original file path in metadata-like field if you want
    row["source_pdf"] = str(pdf_path)
    return row
