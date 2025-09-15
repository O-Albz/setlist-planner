from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import uuid, re
import pandas as pd

# We keep a broad schema for internal compatibility, but your CSV can be minimal:
# title, artist, bpm, lyrics   (others are backfilled below)

REQUIRED_COLUMNS = [
    "song_id","title","artist","bpm","key","mode","energy","scripture_refs",
    "spotify_uri","copyright_status","lyrics_path","theme_summary",
    # extended (kept for compat; we backfill with safe defaults)
    "themes","lyrics_blurb","default_key","vocal_range","meter","segments","youth_score","keys_available",
]

DEFAULTS: Dict[str, Any] = {
    "key": "", "mode": "", "energy": "",  # <- leave energy blank; we do NOT derive it from BPM
    "scripture_refs": "",
    "spotify_uri": "",
    "copyright_status": "unknown",
    "lyrics_path": "",
    "theme_summary": "",
    "themes": "",
    "lyrics_blurb": "",
    "default_key": "",
    "vocal_range": "",
    "meter": "4/4",
    "segments": "",
    "youth_score": 0.0,
    "keys_available": "",
}

def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-]+", "-", str(s).strip())
    s = re.sub(r"-{2,}", "-", s).strip("-").lower()
    return s or "untitled"

def _ensure_schema(df: pd.DataFrame, lyrics_dir: Path) -> pd.DataFrame:
    # Minimal columns sanity
    if "title" not in df.columns or "artist" not in df.columns:
        raise ValueError("Catalog needs at least 'title' and 'artist'.")
    if "bpm" not in df.columns:
        df["bpm"] = 0.0

    # song_id
    if "song_id" not in df.columns:
        df["song_id"] = [uuid.uuid4().hex[:8] for _ in range(len(df))]
    df["song_id"] = df["song_id"].astype(str)

    # normalize bpm to float
    def _to_float(x):
        try: return float(x)
        except Exception: return 0.0
    df["bpm"] = df["bpm"].apply(_to_float)

    # If lyrics text is provided inline, persist to a file and record lyrics_path
    if "lyrics_path" not in df.columns:
        df["lyrics_path"] = ""

    if "lyrics" in df.columns:
        lyrics_dir.mkdir(parents=True, exist_ok=True)
        new_paths = []
        for _, row in df.iterrows():
            lp = str(row.get("lyrics_path") or "").strip()
            if lp:
                new_paths.append(lp)
                continue
            text = str(row.get("lyrics") or "").strip()
            if not text:
                new_paths.append("")
                continue
            name = f"{_slug(row.get('title'))}-{_slug(row.get('artist'))}-{row['song_id']}.txt"
            path = lyrics_dir / name
            try:
                path.write_text(text, encoding="utf-8")
            except Exception:
                path.write_text(text)
            new_paths.append(str(path))
        df["lyrics_path"] = new_paths
        # You may drop 'lyrics' to keep memory small; tests usually don't need it.
        # df = df.drop(columns=["lyrics"])

    # Derive a compact theme_summary from lyrics (or blank)
    if "theme_summary" not in df.columns:
        def _summ(r):
            lp = str(r.get("lyrics_path") or "").strip()
            txt = ""
            if lp and Path(lp).exists():
                try:
                    txt = Path(lp).read_text(encoding="utf-8")
                except Exception:
                    txt = Path(lp).read_text(errors="ignore")
            if not txt:
                txt = str(r.get("lyrics") or "")
            txt = txt.strip().replace("\n", " ")
            return (txt[:240] + "…") if len(txt) > 240 else txt
        df["theme_summary"] = df.apply(_summ, axis=1)

    # Backfill all required columns with safe defaults
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, "")

    # default_key mirrors key if empty
    df["default_key"] = df["default_key"].where(df["default_key"].astype(str).str.len() > 0, df["key"])

    # Return in canonical order
    return df[REQUIRED_COLUMNS].copy()

def load_catalog(csv_path: str | Path, lyrics_dir: str | Path = "lyrics_private") -> pd.DataFrame:
    """
    Load a minimal or full catalog CSV and guarantee REQUIRED_COLUMNS exist.
    If a 'lyrics' column is present, it writes files under `lyrics_dir/` and sets 'lyrics_path'.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Catalog not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return _ensure_schema(df, Path(lyrics_dir))

# --- Catalog validation for tests & UI ---

def validate_catalog(df: pd.DataFrame) -> None:
    """
    Validate a catalog DataFrame. Raises ValueError on problems, otherwise returns None.
    - All REQUIRED_COLUMNS present
    - Not empty
    - bpm numeric (at least some values)
    - song_id unique
    - title/artist not blank
    """
    # columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # not empty
    if df.empty:
        raise ValueError("Catalog is empty")

    # bpm numeric
    try:
        df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    except Exception:
        pass
    if df["bpm"].isna().all():
        raise ValueError("Column 'bpm' has no numeric values")

    # song_id unique
    if df["song_id"].duplicated().any():
        dups = df.loc[df["song_id"].duplicated(), "song_id"].unique().tolist()[:5]
        raise ValueError(f"Duplicate song_id values: {dups}")

    # required text fields non-blank
    if (df["title"].astype(str).str.strip() == "").any():
        raise ValueError("Blank titles present")
    if (df["artist"].astype(str).str.strip() == "").any():
        raise ValueError("Blank artists present")

    # default_key present (we already backfill from key in _ensure_schema)
    # nothing else to do; success == no exception
    return None

