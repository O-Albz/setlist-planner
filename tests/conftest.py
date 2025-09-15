import os
import json
import re
import numpy as np
import pandas as pd
import pytest

# --------- Fast, offline vector stub (replaces SentenceTransformer) ----------
class _DummyST:
    def __init__(self, *a, **k): pass
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            v = np.zeros(16, dtype="float32")
            for i, ch in enumerate(str(t)[:256]):
                v[i % 16] += (ord(ch) % 31) / 31.0
            if normalize_embeddings:
                n = np.linalg.norm(v)
                if n > 0:
                    v /= n
            vecs.append(v)
        return np.stack(vecs, axis=0)

@pytest.fixture(autouse=True)
def no_ollama_env(monkeypatch):
    # Ensure code never attempts to call a real LLM during tests
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    # Optional: if you add a USE_STUB_LLM flag later, keep this true:
    monkeypatch.setenv("USE_STUB_LLM", "1")

@pytest.fixture
def stub_sentence_transformer(monkeypatch):
    """Patch vdb to use the tiny in-memory ST model (no downloads)."""
    import setlistgraph.retrievers.vdb as vdb
    monkeypatch.setattr(vdb, "SentenceTransformer", _DummyST)
    return _DummyST

# --------- LLM stub for agentic_graph ---------------------------------------
class _StubResp:
    def __init__(self, content): self.content = content

class _StubLLM:
    """
    Handles both:
      1) Supervisor loop (list of messages with CONTEXT=..., TOOLS=...)
      2) Plan prompt (single string containing CANDIDATES:[...])
    """
    def __init__(self, *a, **k): pass

    def invoke(self, msg):
        # Supervisor loop uses a list of messages
        if isinstance(msg, list):
            ctx = {}
            for m in msg:
                cont = m.get("content", "")
                if isinstance(cont, str) and cont.startswith("CONTEXT="):
                    ctx = json.loads(cont.split("CONTEXT=")[-1])
            if ctx.get("have_candidates", 0) == 0:
                return _StubResp(json.dumps({"action": "retrieve_semantic", "action_input": {"top_k": 10}}))
            if ctx.get("have_plan", 0) == 0:
                n = ctx.get("n_songs", 3)
                return _StubResp(json.dumps({"action": "plan_llm", "action_input": {"n_songs": n}}))
            if not ctx.get("history") or len(ctx.get("history", [])) < 3:
                return _StubResp(json.dumps({"action": "audit_metrics", "action_input": {}}))
            return _StubResp(json.dumps({"action": "final", "action_input": {"summary": "stubbed OK"}}))

        # Plan step uses a big string with CANDIDATES: [...]
        if isinstance(msg, str):
            m = re.search(r"CANDIDATES:\s*(\[[\s\S]*\])\s*$", msg)
            if not m:
                return _StubResp('{"order":[],"perceived_energy":{},"reasoning":"no candidates"}')
            mini = json.loads(m.group(1))
            order = [r["song_id"] for r in mini[:3]]
            pe = {sid: 3 for sid in order}
            return _StubResp(json.dumps({"order": order, "perceived_energy": pe, "reasoning": "stub order"}))

        return _StubResp('{"action":"final","action_input":{"summary":"unexpected call"}}')

@pytest.fixture
def stub_llm(monkeypatch):
    import setlistgraph.agents.agentic_graph as ag
    # Make sure the module exposes ChatOllama and allows monkeypatch
    monkeypatch.setattr(ag, "_HAS_LLM", True)
    monkeypatch.setattr(ag, "ChatOllama", _StubLLM)
    return _StubLLM

# --------- Tiny catalog fixture ---------------------------------------------
@pytest.fixture
def tiny_catalog_csv(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "title,artist,bpm,lyrics\n"
        "Hope In You,O Albz,100,You lift my eyes my heart awakens your hope is near I trust in You\n"
        "Grace Unending,O Albz,74,Grace upon grace you carry me mercy that sings through every season\n"
        "Lift My Voice,O Albz,92,I will lift my voice sing until the night is done you are faithful still\n",
        encoding="utf-8",
    )
    return csv

@pytest.fixture
def chroma_dir(tmp_path):
    d = tmp_path / ".chroma"
    d.mkdir(exist_ok=True)
    return str(d)
