from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

import pandas as pd

from .onsong_loader import import_onsong_to_catalog

# Optional: PDF support (text/ocr -> OnSong -> import)
try:
    from setlistgraph.io.onsong_pdf_loader import import_onsong_pdf_to_catalog
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# Optional: live vector DB upsert (Chroma)
try:
    from setlistgraph.retrievers.vdb import ChromaSongRetriever
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False

# Optional: legacy batch embedding rebuild (.rag_cache)
try:
    from setlistgraph.pipelines.embed_catalog import build_embeddings
    HAS_EMBED = True
except Exception:
    HAS_EMBED = False


@dataclass
class BatchResult:
    added_rows: pd.DataFrame
    successes: int
    failures: int
    errors: list[str]


def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(errors="ignore")


def _load_catalog(catalog_path: Path) -> pd.DataFrame:
    if catalog_path.exists():
        if catalog_path.suffix.lower() == ".csv":
            return pd.read_csv(catalog_path)
        return pd.read_parquet(catalog_path)
    return pd.DataFrame(
        columns=[
            "song_id", "title", "artist", "bpm", "key", "mode", "energy",
            "scripture_refs", "spotify_uri", "copyright_status",
            "lyrics_path", "theme_summary"
        ]
    )


def _save_catalog(df: pd.DataFrame, catalog_path: Path) -> None:
    if catalog_path.suffix.lower() == ".csv":
        df.to_csv(catalog_path, index=False)
    else:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(catalog_path, index=False)


def _remove_row_from_catalog(catalog_path: Path, song_id: str) -> None:
    """Remove a single row by song_id (used to rollback duplicate PDF imports)."""
    df = _load_catalog(catalog_path)
    if "song_id" in df.columns:
        df = df[df["song_id"].astype(str) != str(song_id)]
        _save_catalog(df, catalog_path)


def batch_import_from_dir(
    input_dir: str | Path,
    catalog_path: str | Path,
    lyrics_dir: str | Path = "lyrics_private",
    glob_patterns: Tuple[str, ...] = ("*.txt", "*.onsong", "*.pdf"),
    default_bpm: float = 120.0,
    skip_if_exists: bool = True,
    rebuild_embeddings: bool = False,
    embeddings_out_dir: str | Path = ".rag_cache",
    # VDB live upsert
    vdb_upsert: bool = False,
    vdb_dir: str | Path = ".chroma",
) -> BatchResult:
    """
    Import TXT / OnSong / PDF files from a folder into the catalog.
    - If PDF: uses onsong_pdf_loader (embedded text or OCR fallback).
    - Duplicate detection by (title, artist); for PDFs, we rollback if detected post-import.
    - Optional: rebuild legacy .rag_cache embeddings or upsert live to Chroma.
    """
    input_dir = Path(input_dir)
    catalog_path = Path(catalog_path)
    lyrics_dir = Path(lyrics_dir)
    vdb_dir = Path(vdb_dir)
    lyrics_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot existing (title, artist) pairs to detect duplicates
    df_cat = _load_catalog(catalog_path)
    existing_pairs = set()
    if skip_if_exists and not df_cat.empty:
        for _, r in df_cat.iterrows():
            existing_pairs.add(
                (
                    str(r.get("title", "")).strip().lower(),
                    str(r.get("artist", "")).strip().lower(),
                )
            )

    # Collect files per pattern
    files: List[Path] = []
    for pat in glob_patterns:
        files += list(input_dir.rglob(pat))
    files = sorted(set(files))

    added: List[Dict[str, Any]] = []
    errors: List[str] = []
    ok = 0
    fail = 0

    for fp in files:
        try:
            ext = fp.suffix.lower()

            # ---------- PDF path ----------
            if ext == ".pdf":
                if not HAS_PDF:
                    raise RuntimeError(
                        "PDF support not available. Install pdfplumber/pdf2image/pytesseract "
                        "and add setlistgraph.io.onsong_pdf_loader."
                    )

                # Import (this writes lyrics file + appends to catalog)
                row = import_onsong_pdf_to_catalog(
                    pdf_path=str(fp),
                    catalog_path=str(catalog_path),
                    lyrics_dir=str(lyrics_dir),
                    default_bpm=default_bpm,
                )
                # Duplicate rollback check
                title = str(row.get("title", "")).strip().lower()
                artist = str(row.get("artist", "")).strip().lower()
                pair = (title, artist)

                if skip_if_exists and pair in existing_pairs:
                    # rollback: remove lyrics file + remove catalog row we just added
                    lp = row.get("lyrics_path")
                    if lp:
                        try:
                            Path(lp).unlink(missing_ok=True)  # py3.8+: missing_ok
                        except Exception:
                            pass
                    _remove_row_from_catalog(catalog_path, str(row.get("song_id")))
                    # do not count as failure (silently skipped)
                    continue

                # Not duplicate: proceed
                added.append(row)
                ok += 1

                # Optional: live upsert into Chroma
                if vdb_upsert and HAS_CHROMA:
                    try:
                        retr = ChromaSongRetriever(chroma_dir=str(vdb_dir))
                        retr.upsert_rows([row])
                    except Exception as e:
                        errors.append(f"{fp.name}: VDB upsert failed: {e}")

                # Add to seen pairs so subsequent files can detect duplicates against this one
                existing_pairs.add(pair)
                continue

            # ---------- TXT / .onsong path ----------
            txt = _read_text_file(fp)
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            guess_title = lines[0] if lines else ""
            guess_artist = lines[1] if len(lines) > 1 else ""
            pair = (guess_title.lower(), guess_artist.lower())

            if skip_if_exists and pair in existing_pairs:
                continue

            row = import_onsong_to_catalog(
                onsong_text=txt,
                catalog_path=str(catalog_path),
                lyrics_dir=str(lyrics_dir),
                default_bpm=default_bpm,
            )
            added.append(row)
            ok += 1

            if vdb_upsert and HAS_CHROMA:
                try:
                    retr = ChromaSongRetriever(chroma_dir=str(vdb_dir))
                    retr.upsert_rows([row])
                except Exception as e:
                    errors.append(f"{fp.name}: VDB upsert failed: {e}")

            existing_pairs.add(pair)

        except Exception as e:
            errors.append(f"{fp.name}: {e}")
            fail += 1

    added_df = pd.DataFrame(added)

    # Optional: rebuild legacy embeddings (.rag_cache)
    if rebuild_embeddings:
        if not HAS_EMBED:
            errors.append(
                "Embeddings pipeline not available. Install sentence-transformers and ensure setlistgraph.pipelines.embed_catalog exists."
            )
        else:
            try:
                build_embeddings(str(catalog_path), out_dir=str(embeddings_out_dir))
            except Exception as e:
                errors.append(f"Embedding rebuild failed: {e}")

    return BatchResult(added_rows=added_df, successes=ok, failures=fail, errors=errors)


def batch_import_from_uploads(
    uploads: Iterable[Tuple[str, str]],
    catalog_path: str | Path,
    lyrics_dir: str | Path = "lyrics_private",
    default_bpm: float = 120.0,
    rebuild_embeddings: bool = False,
    embeddings_out_dir: str | Path = ".rag_cache",
    # VDB live upsert
    vdb_upsert: bool = False,
    vdb_dir: str | Path = ".chroma",
) -> BatchResult:
    """
    Import a set of in-memory uploads (filename, text) into the catalog.
    (Uploads are assumed TXT / OnSong. For PDFs, use the directory importer or write a separate upload path.)
    """
    catalog_path = Path(catalog_path)
    lyrics_dir = Path(lyrics_dir)
    vdb_dir = Path(vdb_dir)
    lyrics_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot existing (title, artist) to dedupe
    df_cat = _load_catalog(catalog_path)
    existing_pairs = set()
    for _, r in df_cat.iterrows():
        existing_pairs.add(
            (
                str(r.get("title", "")).strip().lower(),
                str(r.get("artist", "")).strip().lower(),
            )
        )

    added: List[Dict[str, Any]] = []
    errors: List[str] = []
    ok = 0
    fail = 0

    for name, text in uploads:
        try:
            # Lightweight duplicate detection from first two non-empty lines
            lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
            guess_title = lines[0] if lines else ""
            guess_artist = lines[1] if len(lines) > 1 else ""
            pair = (guess_title.lower(), guess_artist.lower())

            if pair in existing_pairs:
                continue

            row = import_onsong_to_catalog(
                onsong_text=text,
                catalog_path=str(catalog_path),
                lyrics_dir=str(lyrics_dir),
                default_bpm=default_bpm,
            )
            added.append(row)
            ok += 1

            if vdb_upsert and HAS_CHROMA:
                try:
                    retr = ChromaSongRetriever(chroma_dir=str(vdb_dir))
                    retr.upsert_rows([row])
                except Exception as e:
                    errors.append(f"{name}: VDB upsert failed: {e}")

            existing_pairs.add(pair)

        except Exception as e:
            errors.append(f"{name}: {e}")
            fail += 1

    added_df = pd.DataFrame(added)

    if rebuild_embeddings:
        if not HAS_EMBED:
            errors.append(
                "Embeddings pipeline not available. Install sentence-transformers and ensure setlistgraph.pipelines.embed_catalog exists."
            )
        else:
            try:
                build_embeddings(str(catalog_path), out_dir=str(embeddings_out_dir))
            except Exception as e:
                errors.append(f"Embedding rebuild failed: {e}")

    return BatchResult(added_rows=added_df, successes=ok, failures=fail, errors=errors)
