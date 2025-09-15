# src/setlistgraph/agents/tools.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import re

import pandas as pd
from setlistgraph.io.loaders import load_catalog

def _simple_score(row: Dict, terms: List[str]) -> float:
    hay = " ".join([
        str(row.get("title","")),
        str(row.get("artist","")),
        str(row.get("theme_summary","")),
    ]).lower()
    # quick-and-dirty term hits
    return sum(1.0 for t in terms if t and t in hay)

def plan_setlist(theme: str, scripture: str, gathering: str, n_songs: int = 4, catalog_path: str = "src/setlistgraph/data/song_catalog.sample.csv") -> Dict:
    """
    Legacy simple planner used by tests/test_graph.py.
    - loads the CSV (minimal schema supported)
    - does a naive relevance score on title/artist/theme_summary
    - returns top N as 'songs' (no LLM or vector DB required)
    """
    df = load_catalog(catalog_path)
    if df.empty:
        return {"songs": [], "notes": "Catalog is empty.", "catalog_path": catalog_path}

    terms = re.split(r"\W+", f"{theme} {scripture}".lower())
    terms = [t for t in terms if t]

    # naive scoring
    scored = df.copy()
    scored["score_simple"] = scored.apply(lambda r: _simple_score(r, terms), axis=1)
    scored = scored.sort_values(["score_simple"], ascending=[False])

    # fallback: if all zeros, just take the first rows
    if (scored["score_simple"] == 0).all():
        scored = df.copy()

    pick = scored.head(max(1, n_songs))  # ensure at least 1
    out_cols = ["song_id","title","artist","bpm","key","default_key","scripture_refs"]
    for c in out_cols:
        if c not in pick.columns: pick[c] = None

    return {
        "songs": pick[out_cols].to_dict(orient="records"),
        "notes": "Simple fallback planner (no embeddings).",
        "catalog_path": catalog_path,
    }
