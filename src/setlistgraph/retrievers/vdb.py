from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd

import chromadb
from sentence_transformers import SentenceTransformer

# Persistent local vector DB for songs
COLL_NAME = "songs"

def _st_model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

class ChromaSongRetriever:
    """
    Chroma-backed retriever. Stores vectors + metadata (no lyrics text).
    - upsert_rows: compute embeddings from lyrics_path OR lyrics OR theme_summary
    - upsert_batch: accepts precomputed embeddings (for migrations)
    - search: cosine similarity (1 - distance)
    """
    def __init__(self, chroma_dir: str = ".chroma", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.model = _st_model(model_name)
        self.coll = self.client.get_or_create_collection(name=COLL_NAME, metadata={"hnsw:space": "cosine"})

    # ---------- Helpers ----------
    def _row_to_text(self, r: Dict[str, Any], text_field: str = "theme_summary") -> str:
        # 1) lyrics_path on disk
        lp = r.get("lyrics_path")
        if isinstance(lp, str):
            s = lp.strip()
            if s:
                p = Path(s)
                if p.exists():
                    try:
                        return p.read_text(encoding="utf-8")
                    except Exception:
                        return p.read_text(errors="ignore")
        # 2) inline lyrics field (may be NaN/float)
        txt = r.get("lyrics")
        if isinstance(txt, float) or txt is None:
            txt = ""
        txt = str(txt)
        if txt.strip():
            return txt
        # 3) fallback summary
        return str(r.get(text_field) or "")


    # ---------- Upserts ----------
    def upsert_rows(self, rows: List[Dict[str, Any]], text_field: str = "theme_summary") -> int:
        if not rows:
            return 0
        ids = [str(r["song_id"]) for r in rows]
        metadatas = [{k: v for k, v in r.items() if k != "lyrics_path"} for r in rows]
        texts = [self._row_to_text(r, text_field=text_field) for r in rows]
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        self.coll.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=[""] * len(ids),  # privacy: do NOT store lyrics text
            embeddings=embs.tolist(),
        )
        return len(ids)

    def upsert_batch(self, meta_df: pd.DataFrame, emb: np.ndarray, doc_fallback: Optional[List[str]] = None):
        ids = meta_df["song_id"].astype(str).tolist()
        metadatas = meta_df.to_dict(orient="records")
        documents = doc_fallback or [""] * len(ids)  # keep lyrics out; can pass theme_summary strings
        self.coll.upsert(ids=ids, metadatas=metadatas, documents=documents, embeddings=emb.tolist())

    # ---------- Query ----------
    def search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        q_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]
        res = self.coll.query(query_embeddings=[q_vec.tolist()], n_results=top_k)
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]  # cosine distance
        sims = [1.0 - float(d) for d in (dists or [])]
        for m, s in zip(metas, sims):
            m["score"] = s
        df = pd.DataFrame(metas)
        keep = ["song_id","title","artist","bpm","key","energy","scripture_refs","spotify_uri","score","theme_summary"]
        for k in keep:
            if k not in df.columns:
                df[k] = None
        return df[keep].copy()

# ---------- One-shot migration from .rag_cache ----------
def migrate_rag_cache_to_chroma(rag_dir: str = ".rag_cache", chroma_dir: str = ".chroma") -> int:
    rag = Path(rag_dir)
    meta = pd.read_parquet(rag / "songs_meta.parquet")
    emb = np.load(rag / "songs_emb.npy")
    client = chromadb.PersistentClient(path=chroma_dir)
    coll = client.get_or_create_collection(name=COLL_NAME, metadata={"hnsw:space":"cosine"})
    ids = meta["song_id"].astype(str).tolist()
    metas = meta.to_dict(orient="records")
    coll.upsert(ids=ids, metadatas=metas, documents=[""] * len(ids), embeddings=emb.tolist())
    return len(ids)
