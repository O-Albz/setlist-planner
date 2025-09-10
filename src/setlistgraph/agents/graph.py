# src/setlistgraph/agents/graph.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from setlistgraph.scoring.compatibility import is_transition_ok
from setlistgraph.retrievers.songs import SimpleSongRetriever, filter_by_gathering
from setlistgraph.io.loaders import load_catalog

# Optional semantic RAG (embeddings over lyrics/theme_summary)
try:
    from setlistgraph.pipelines.embed_catalog import load_index
    from setlistgraph.retrievers.semantic import SemanticSongRetriever

    _HAS_SEMANTIC = True
except Exception:
    _HAS_SEMANTIC = False

# add near the top
try:
    from setlistgraph.retrievers.vdb import ChromaSongRetriever
    _HAS_CHROMA = True
except Exception:
    _HAS_CHROMA = False

def retrieve_candidates(state: S) -> S:
    s = state["service"]
    query = f'{s.get("theme","")} {s.get("scripture","")}'.strip() or "*"

    # 1) Chroma (if present)
    if _HAS_CHROMA:
        try:
            retr = ChromaSongRetriever(chroma_dir=state.get("index_dir") or ".chroma")
            hits = retr.search(query, top_k=50)
            cands = hits.to_dict(orient="records")
            cands = filter_by_gathering_list(cands, s.get("gathering", "family"))
            return {"candidates": cands}
        except Exception:
            pass

    # 2) fall back to your .rag_cache semantic retriever
    # ... existing SemanticSongRetriever path ...


# ---------------------------
# State definition
# ---------------------------
class S(TypedDict, total=False):
    # Inputs
    service: Dict[str, Any]  # {theme, scripture, gathering, n_songs, start_key}
    catalog_path: str        # used by TF-IDF fallback
    index_dir: str           # directory with embeddings cache (default: ".rag_cache")

    # Working/outputs
    candidates: List[Dict[str, Any]]
    planned: List[Dict[str, Any]]
    transitions: List[Dict[str, Any]]
    violations: List[Dict[str, Any]]
    songs: List[Dict[str, Any]]
    notes: str



# ---------------------------
# Helpers
# ---------------------------
def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _energy(x: Any, default: int = 3) -> int:
    try:
        v = int(x)
        return v if 1 <= v <= 5 else default
    except Exception:
        return default


def _bpm(x: Any) -> float:
    return _to_float(x, 0.0)


def _pick_seed(cands: List[Dict[str, Any]]) -> int:
    """
    Seed = highest semantic score, slight preference for higher-BPM openers.
    (No hard BPM filtering — BPM is just metadata here.)
    """
    if not cands:
        return -1
    best_i, best = 0, -1e9
    for i, r in enumerate(cands):
        base = float(r.get("score", 0) or 0)
        bonus = min(_bpm(r.get("bpm", 0)) / 200.0, 1.0) * 0.08  # tiny tilt to faster openers
        val = base + bonus
        if val > best:
            best, best_i = val, i
    return best_i


def _phase_targets(pos: int, total: int) -> Dict[str, Any]:
    """
    Heuristic arc:
      1 -> Opener: prefer faster, energy 4–5
      middle-low -> Reflective: prefer slower, energy 2–3
      middle-high -> Build: mid tempo, energy 3–4
      last -> Sending: faster, energy 4–5
    Returns soft target hints (no hard constraints).
    """
    if pos == 1:
        return {"phase": "opener", "target_energy": 4, "tempo_pref": "fast"}
    if pos == total:
        return {"phase": "sending", "target_energy": 4, "tempo_pref": "fast"}
    mid_break = max(2, total // 2)
    if pos <= mid_break:
        return {"phase": "reflective", "target_energy": 2, "tempo_pref": "slow"}
    return {"phase": "build", "target_energy": 3, "tempo_pref": "mid"}


def _tempo_fit(bpm: float, tempo_pref: str) -> float:
    """
    Soft preference score for tempo buckets:
      fast ~ >= 96, slow ~ <= 80, mid ~ 80–100
    Returns [0..1] (bounded).
    """
    if bpm <= 0:
        return 0.0
    if tempo_pref == "fast":
        # reward higher tempos (cap at 120+)
        return min(max((bpm - 96) / 24.0, 0.0), 1.0)
    if tempo_pref == "slow":
        # reward slower tempos (≤80 best)
        return min(max((80 - bpm) / 20.0, 0.0), 1.0)
    # mid: best around 88–96, fade out to 80/104
    center = 92.0
    spread = 12.0
    return max(0.0, 1.0 - abs(bpm - center) / spread)


def _transition_ok(prev: Dict[str, Any], nxt: Dict[str, Any]) -> bool:
    k1 = (prev.get("key") or prev.get("default_key") or "").strip()
    k2 = (nxt.get("key") or nxt.get("default_key") or "").strip()
    b1 = _bpm(prev.get("bpm"))
    b2 = _bpm(nxt.get("bpm"))
    chk = is_transition_ok(k1, b1, k2, b2, semitone_limit=2, drift=0.15)
    return chk.keys_ok and chk.tempo_ok


def filter_by_gathering_list(rows: List[Dict[str, Any]], gathering: str) -> List[Dict[str, Any]]:
    """
    Similar to filter_by_gathering(DataFrame, ...), but operates on list of dicts.
    """
    g = (gathering or "").strip().lower()
    if g == "youth":
        return sorted(rows, key=lambda r: (_energy(r.get("energy", 3)), r.get("score", 0.0)), reverse=True)
    if g == "traditional":
        # prefer lower energy, but still respect score
        return sorted(rows, key=lambda r: (_energy(r.get("energy", 3)), r.get("score", 0.0)))
    # family/general: high score first, moderate energy next
    return sorted(rows, key=lambda r: (r.get("score", 0.0), -abs(_energy(r.get("energy", 3)) - 3)), reverse=True)


# ---------------------------
# Graph nodes
# ---------------------------
def intake(state: S) -> S:
    """Normalize incoming service params."""
    s = state.get("service", {}) or {}
    s["theme"] = (s.get("theme") or "").strip()
    s["scripture"] = (s.get("scripture") or "").strip()
    s["gathering"] = (s.get("gathering") or "family").strip().lower()
    s["n_songs"] = _to_int(s.get("n_songs", 4), 4)
    s["start_key"] = (s.get("start_key") or "").strip()
    return {"service": s}


def retrieve_candidates(state: S) -> S:
    """
    Retrieve top candidate songs using Semantic RAG (embeddings) if available,
    falling back to the TF-IDF retriever over the raw catalog.
    NOTE: BPM is NOT used here — it's just metadata; similarity is purely textual.
    """
    s = state["service"]
    query = f'{s.get("theme","")} {s.get("scripture","")}'.strip()

    # Semantic path (no BPM filters/penalties)
    if _HAS_SEMANTIC:
        try:
            index_dir = state.get("index_dir") or ".rag_cache"
            meta, emb, _ = load_index(index_dir)
            retriever = SemanticSongRetriever(meta, emb)
            hits = retriever.search(query or "*", top_k=50)  # no SearchFilters
            keep = ["song_id", "title", "artist", "bpm", "key", "energy", "scripture_refs", "spotify_uri", "score"]
            cands = hits[keep].to_dict(orient="records")
            cands = filter_by_gathering_list(cands, s.get("gathering", "family"))
            return {"candidates": cands}
        except Exception:
            pass  # fall back silently

    # TF-IDF fallback
    catalog_path = state.get("catalog_path") or "src/setlistgraph/data/song_catalog.sample.csv"
    df = load_catalog(catalog_path)
    tfidf = SimpleSongRetriever(df)
    hits = tfidf.search(query or "*", top_k=50)
    hits = filter_by_gathering(hits, s.get("gathering", "family"))
    # Harmonize column names with semantic path
    if "default_key" in hits.columns and "key" not in hits.columns:
        hits = hits.rename(columns={"default_key": "key"})
    keep = ["song_id", "title", "artist", "bpm", "key", "energy", "scripture_refs", "spotify_uri", "score"]
    cands = hits[keep].to_dict(orient="records")
    return {"candidates": cands}


def flow_plan(state: S) -> S:
    """
    Greedy planner:
      - Seed by semantic score (tiny preference to faster openers)
      - For each next slot, choose a candidate that:
          * passes transition feasibility (≤2 semitones, ≤15% tempo drift)
          * matches the phase's soft preferences (tempo & energy)
          * keeps overall similarity score high
    """
    s = state["service"]
    n = max(1, _to_int(s.get("n_songs", 4), 4))
    cand: List[Dict[str, Any]] = state.get("candidates", []) or []
    if not cand:
        return {"planned": []}

    # Seed
    idx = _pick_seed(cand)
    if idx < 0:
        idx = 0
    planned = [cand[idx]]
    remaining = cand[:idx] + cand[idx + 1 :]

    # Build the rest
    while len(planned) < n and remaining:
        pos = len(planned) + 1
        targets = _phase_targets(pos, n)
        t_pref = targets["tempo_pref"]
        e_target = targets["target_energy"]

        best, best_row = -1e9, None
        for r in remaining:
            if not _transition_ok(planned[-1], r):
                continue
            base = float(r.get("score", 0.0) or 0.0)
            tempo_score = _tempo_fit(_bpm(r.get("bpm")), t_pref)  # [0..1]
            energy_score = max(0.0, 1.0 - abs(_energy(r.get("energy", 3)) - e_target) / 2.0)  # [0..1]
            val = base + 0.10 * tempo_score + 0.08 * energy_score
            if val > best:
                best, best_row = val, r

        # Fallbacks if no feasible transition found
        if best_row is None:
            # try ignoring arc and just keep feasible transition
            for r in remaining:
                if _transition_ok(planned[-1], r):
                    best_row = r
                    break
        if best_row is None:
            # as last resort, ignore transition rule (pads/keyswitch assumed)
            best_row = remaining[0]

        planned.append(best_row)
        remaining = [x for x in remaining if x is not best_row]

    return {"planned": planned[:n]}


def judge_transitions(state: S) -> S:
    """
    Audit the planned sequence and compute per-edge feasibility.
    Adds:
      - transitions: list of checks between adjacent songs
      - violations: subset where keys_ok or tempo_ok is False
    """
    planned = state.get("planned", []) or []
    rows: List[Dict[str, Any]] = []
    for i in range(len(planned) - 1):
        a, b = planned[i], planned[i + 1]
        k1 = (a.get("key") or a.get("default_key") or "").strip()
        k2 = (b.get("key") or b.get("default_key") or "").strip()
        b1 = _bpm(a.get("bpm"))
        b2 = _bpm(b.get("bpm"))
        chk = is_transition_ok(k1, b1, k2, b2, semitone_limit=2, drift=0.15)
        rows.append(
            {
                "i": i,
                "from_title": a.get("title"),
                "to_title": b.get("title"),
                "from_key": k1,
                "to_key": k2,
                "from_bpm": b1,
                "to_bpm": b2,
                "semitones": chk.semitones,
                "bpm_delta": chk.bpm_delta,
                "keys_ok": chk.keys_ok,
                "tempo_ok": chk.tempo_ok,
            }
        )
    violations = [r for r in rows if not (r["keys_ok"] and r["tempo_ok"])]
    return {"transitions": rows, "violations": violations}


def assemble(state: S) -> S:
    out: S = {
        "songs": state.get("planned", []),
        "notes": "Planner uses BPM only for placement; transitions constrained to ≤2 semitones & ≤15% tempo drift.",
    }
    if "transitions" in state:
        out["transitions"] = state["transitions"]
    if "violations" in state:
        out["violations"] = state["violations"]
    return out


# ---------------------------
# Build graph
# ---------------------------

def build_app():
    g = StateGraph(S)
    g.add_node("intake", intake)
    g.add_node("retrieve_candidates", retrieve_candidates)
    g.add_node("flow_plan", flow_plan)

    g.add_node("judge_transitions", judge_transitions)  # ← new audit node

    g.add_node("assemble", assemble)

    g.set_entry_point("intake")
    g.add_edge("intake", "retrieve_candidates")
    g.add_edge("retrieve_candidates", "flow_plan")
    g.add_edge("flow_plan", "judge_transitions")        # ← audit before assemble
    g.add_edge("judge_transitions", "assemble")         # ← pass audit to output
    g.add_edge("assemble", END)
    return g.compile()


# ---------------------------
# Convenience runner
# ---------------------------
def plan_setlist(
    theme: str = "",
    scripture: str = "",
    gathering: str = "family",
    n_songs: int = 4,
    start_key: Optional[str] = None,
    catalog_path: str = "src/setlistgraph/data/song_catalog.sample.csv",
    index_dir: str = ".rag_cache",
) -> Dict[str, Any]:
    """
    Wrapper for quick calls. NOTE: BPM is *not* used to filter retrieval.
    It’s only considered during planning/placement.
    """
    app = build_app()
    state: S = {
        "service": {
            "theme": theme,
            "scripture": scripture,
            "gathering": gathering,
            "n_songs": n_songs,
            "start_key": start_key or "",
        },
        "catalog_path": catalog_path,
        "index_dir": index_dir,
    }
    out = app.invoke(state)
    if "songs" not in out and "planned" in out:
        out["songs"] = out["planned"]
    if "notes" not in out:
        out["notes"] = "Planner uses BPM for placement only; retrieval is purely semantic."
    return out


__all__ = ["build_app", "plan_setlist"]

