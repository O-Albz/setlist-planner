
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import pandas as pd

from .onsong_loader import import_onsong_to_catalog
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

def batch_import_from_dir(
    input_dir: str | Path,
    catalog_path: str | Path,
    lyrics_dir: str | Path = "lyrics_private",
    glob_patterns: Tuple[str, ...] = ("*.txt", "*.onsong"),
    default_bpm: float = 120.0,
    skip_if_exists: bool = True,
    rebuild_embeddings: bool = False,
    embeddings_out_dir: str | Path = ".rag_cache",
) -> BatchResult:
    """Import all OnSong files from a folder into the catalog, optionally rebuild embeddings."""
    input_dir = Path(input_dir)
    catalog_path = Path(catalog_path)
    lyrics_dir = Path(lyrics_dir)
    lyrics_dir.mkdir(parents=True, exist_ok=True)

    # Load existing catalog if present (for duplicate checks)
    if catalog_path.exists():
        if catalog_path.suffix.lower() == ".csv":
            df_cat = pd.read_csv(catalog_path)
        else:
            df_cat = pd.read_parquet(catalog_path)
    else:
        df_cat = pd.DataFrame(columns=[
            "song_id","title","artist","bpm","key","mode","energy","scripture_refs",
            "spotify_uri","copyright_status","lyrics_path","theme_summary"
        ])

    existing_pairs = set()
    if skip_if_exists and not df_cat.empty:
        for _, r in df_cat.iterrows():
            existing_pairs.add((str(r.get("title","")).strip().lower(), str(r.get("artist","")).strip().lower()))

    files: List[Path] = []
    for pat in glob_patterns:
        files += list(input_dir.rglob(pat))
    files = sorted(set(files))

    added = []
    errors: List[str] = []
    ok = 0
    fail = 0

    for fp in files:
        try:
            txt = _read_text_file(fp)
            # quick first two lines as title/artist heuristic for duplicate check
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            guess_title = lines[0] if lines else ""
            guess_artist = lines[1] if len(lines) > 1 else ""
            pair = (guess_title.lower(), guess_artist.lower())

            if skip_if_exists and pair in existing_pairs:
                continue

            row = import_onsong_to_catalog(
                onsong_text=txt,
                catalog_path=catalog_path,
                lyrics_dir=lyrics_dir,
                default_bpm=default_bpm,
            )
            added.append(row)
            ok += 1
        except Exception as e:
            errors.append(f"{fp.name}: {e}")
            fail += 1

    added_df = pd.DataFrame(added)

    if rebuild_embeddings:
        if not HAS_EMBED:
            errors.append("Embeddings pipeline not available. Install sentence-transformers and ensure setlistgraph.pipelines.embed_catalog exists.")
        else:
            try:
                build_embeddings(catalog_path, out_dir=embeddings_out_dir)
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
) -> BatchResult:
    """Import a set of in-memory uploads (filename, text) into the catalog."""
    catalog_path = Path(catalog_path)
    lyrics_dir = Path(lyrics_dir)
    lyrics_dir.mkdir(parents=True, exist_ok=True)

    added = []
    errors: List[str] = []
    ok = 0
    fail = 0

    for name, text in uploads:
        try:
            row = import_onsong_to_catalog(
                onsong_text=text,
                catalog_path=catalog_path,
                lyrics_dir=lyrics_dir,
                default_bpm=default_bpm,
            )
            added.append(row)
            ok += 1
        except Exception as e:
            errors.append(f"{name}: {e}")
            fail += 1

    added_df = pd.DataFrame(added)

    if rebuild_embeddings:
        if not HAS_EMBED:
            errors.append("Embeddings pipeline not available. Install sentence-transformers and ensure setlistgraph.pipelines.embed_catalog exists.")
        else:
            try:
                build_embeddings(catalog_path, out_dir=embeddings_out_dir)
            except Exception as e:
                errors.append(f"Embedding rebuild failed: {e}")

    return BatchResult(added_rows=added_df, successes=ok, failures=fail, errors=errors)
