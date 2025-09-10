
import os
from pathlib import Path
import pandas as pd
import pytest

# Import from the installed package path (pythonpath should include 'src')
from setlistgraph.io.loaders import load_catalog, validate_catalog, REQUIRED_COLUMNS

SAMPLE_PATHS = [
    Path('src/setlistgraph/data/song_catalog.sample.csv'),
    Path('setlistgraph/src/setlistgraph/data/song_catalog.sample.csv'),  # fallback if running from repo root's parent
]

def find_sample_csv() -> Path:
    for p in SAMPLE_PATHS:
        if p.exists():
            return p
    # Allow override via env var
    env_path = os.getenv('SONG_CATALOG_PATH')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    raise FileNotFoundError('Could not find song_catalog.sample.csv. Set SONG_CATALOG_PATH env var to the CSV.')

def test_catalog_has_required_columns():
    path = find_sample_csv()
    df = load_catalog(path)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"

def test_catalog_not_empty_and_valid():
    path = find_sample_csv()
    df = load_catalog(path)
    validate_catalog(df)  # should not raise
    assert len(df) >= 1
