# src/setlistgraph/pipelines/embed_catalog.py
import numpy as np
import pandas as pd

def load_index(index_dir: str):
    # Minimal stub so imports work in tests; replace with your real implementation.
    meta = pd.DataFrame()
    emb = np.zeros((0, 384), dtype="float32")
    return meta, emb, {"model": "stub"}

def build_embeddings(catalog_path: str, out_dir: str = ".rag_cache"):
    return {"status": "stub"}
