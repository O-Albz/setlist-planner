
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import pandas as pd

# Required schema for the song catalog CSV
REQUIRED_COLUMNS: List[str] = [
    "song_id","title","artist","themes","scripture_refs","lyrics_blurb","default_key",
    "vocal_range","energy","bpm","meter","segments","youth_score","spotify_uri","keys_available"
]

def load_catalog(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the song catalog CSV and validate required columns.
    Returns a pandas DataFrame with the same columns.
    Raises ValueError if required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Catalog not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def validate_catalog(df: pd.DataFrame) -> None:
    """
    Perform basic sanity checks on the catalog.
    Raises ValueError with a helpful message if something is off.
    """
    # Check emptiness
    if df.empty:
        raise ValueError("Catalog is empty — add at least one song row.")
    # Check a few critical fields
    for col in ["song_id", "title", "artist", "default_key", "bpm"]:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise ValueError(f"Column '{col}' has {n} missing value(s).")
    # Ensure numeric columns are numeric-ish
    for col in ["song_id", "energy", "bpm", "youth_score"]:
        try:
            pd.to_numeric(df[col])
        except Exception as e:
            raise ValueError(f"Column '{col}' must be numeric. Error: {e}")
    # Optional: keys_available should be semicolon-separated list
    if "keys_available" in df.columns:
        bad = df["keys_available"].fillna("").apply(lambda s: isinstance(s, str) and ";" not in s and len(s) > 0)
        if bad.any():
            idxs = bad[bad].index.tolist()[:3]
            raise ValueError(f"'keys_available' should be a ';' separated list of keys (e.g., 'C;D;E'). Problem rows: {idxs}")
