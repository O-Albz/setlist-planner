import pandas as pd
from setlistgraph.io.loaders import load_catalog
from setlistgraph.retrievers.vdb import ChromaSongRetriever

def test_chroma_upsert_and_search(tmp_path, tiny_catalog_csv, chroma_dir, stub_sentence_transformer):
    df = load_catalog(tiny_catalog_csv, lyrics_dir=tmp_path / "lyrics_private")
    retr = ChromaSongRetriever(chroma_dir=chroma_dir)
    n = retr.upsert_rows(df.to_dict(orient="records"))
    assert n == len(df)

    res = retr.search("hope 1 Peter 1:3", top_k=5)
    assert not res.empty
    assert set(["title", "artist", "score"]).issubset(res.columns)
    assert (res["score"] >= 0).all()
