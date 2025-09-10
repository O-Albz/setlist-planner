# src/setlistgraph/retrievers/semantic.py
import pandas as pd

class SemanticSongRetriever:
    def __init__(self, meta: pd.DataFrame, emb): self.meta = meta
    def search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        # Return empty df; your tests stub the tool anyway.
        return pd.DataFrame(columns=["song_id","title","artist","bpm","key","energy","scripture_refs","spotify_uri","score"])
