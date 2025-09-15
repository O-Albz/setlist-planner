import numpy as np
import pandas as pd
from setlistgraph.retrievers.vdb import ChromaSongRetriever

def test_upsert_rows_handles_nan_lyrics_path(tmp_path, chroma_dir, stub_sentence_transformer):
    # Simulate rows without lyrics_path and with NaN values
    rows = [
        {"song_id": "a1", "title": "A", "artist": "X", "bpm": 90, "lyrics_path": float("nan"), "lyrics": "", "theme_summary": "hope"},
        {"song_id": "b2", "title": "B", "artist": "Y", "bpm": 80, "lyrics_path": "", "lyrics": "grace mercy", "theme_summary": ""},
    ]
    retr = ChromaSongRetriever(chroma_dir=chroma_dir)
    n = retr.upsert_rows(rows)
    assert n == 2
    res = retr.search("grace", top_k=5)
    assert not res.empty
