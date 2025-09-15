from pathlib import Path
from setlistgraph.io.loaders import load_catalog, REQUIRED_COLUMNS

def test_minimal_csv_is_normalized(tmp_path, tiny_catalog_csv):
    lyrics_dir = tmp_path / "lyrics_private"
    df = load_catalog(tiny_catalog_csv, lyrics_dir=lyrics_dir)

    # Has all required columns
    for col in REQUIRED_COLUMNS:
        assert col in df.columns

    # song_id generated
    assert df["song_id"].notna().all()

    # bpm cast to float
    assert df["bpm"].dtype.kind in ("i", "f")

    # lyrics persisted to files
    assert df["lyrics_path"].notna().all()
    for p in df["lyrics_path"]:
        if p:
            assert Path(p).exists()

    # theme_summary backfilled
    assert df["theme_summary"].fillna("").astype(str).str.len().mean() > 0
