from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# We won't store lyrics; documents can be empty or theme_summary.
# We store vectors + rich metadata.

COLL_NAME = "songs"

def _st_model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

class ChromaSongRetriever:
    def __init__(self, chroma_dir: str = ".chroma", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.model = _st_model(model_name)
        # cosine space
        self.coll = self.client.get_or_create_collection(name=COLL_NAME, metadata={"hnsw:space": "cosine"})

    def upsert_batch(self, meta_df: pd.DataFrame, emb: np.ndarray, doc_fallback: Optional[List[str]] = None):
        ids = meta_df["song_id"].astype(str).tolist()
        metadatas = meta_df.to_dict(orient="records")
        documents = doc_fallback or [""] * len(ids)  # keep lyrics out; can pass theme_summary strings
        # upsert with precomputed embeddings
        self.coll.upsert(ids=ids, metadatas=metadatas, documents=documents, embeddings=emb.tolist())

    def search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        q_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]
        res = self.coll.query(query_embeddings=[q_vec.tolist()], n_results=top_k)
        # Normalize output
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] or res.get("embeddings", [[]])[0]  # chroma returns distances for cosine
        # Convert cosine distance → similarity
        sims = [1.0 - float(d) for d in dists] if dists else [0.0] * len(ids)
        for m, s in zip(metas, sims):
            m["score"] = s
        df = pd.DataFrame(metas)
        # keep common columns only
        keep = ["song_id","title","artist","bpm","key","energy","scripture_refs","spotify_uri","score"]
        for k in keep:
            if k not in df.columns:
                df[k] = None
        return df[keep].copy()

# Migration from your existing .rag_cache to Chroma
def migrate_rag_cache_to_chroma(rag_dir: str = ".rag_cache", chroma_dir: str = ".chroma"):
    rag = Path(rag_dir)
    meta = pd.read_parquet(rag / "songs_meta.parquet")
    emb = np.load(rag / "songs_emb.npy")
    retr = ChromaSongRetriever(chroma_dir=chroma_dir)
    # Use theme_summary as a harmless document if present; otherwise empty strings
    docs = (meta.get("theme_summary").fillna("").astype(str).tolist()) if "theme_summary" in meta.columns else None
    retr.upsert_batch(meta_df=meta, emb=emb, doc_fallback=docs)
    return len(meta)
