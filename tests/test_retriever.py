
from pathlib import Path
import os
import pandas as pd
from setlistgraph.io.loaders import load_catalog
from setlistgraph.retrievers.songs import SimpleSongRetriever

SAMPLE_PATHS = [
    Path('src/setlistgraph/data/song_catalog.sample.csv'),
    Path('setlistgraph/src/setlistgraph/data/song_catalog.sample.csv'),
]

def _find_sample():
    for p in SAMPLE_PATHS:
        if p.exists():
            return p
    env_path = os.getenv('SONG_CATALOG_PATH')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    raise FileNotFoundError('sample catalog not found')

def test_tfidf_search_runs():
    path = _find_sample()
    df = load_catalog(path)
    r = SimpleSongRetriever(df)
    hits = r.search("hope resurrection 1 Peter 1:3", top_k=5)
    assert len(hits) >= 1
    assert "score" in hits.columns
