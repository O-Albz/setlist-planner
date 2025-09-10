# src/setlistgraph/agents/graph.py
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from setlistgraph.io.loaders import load_catalog
from setlistgraph.retrievers.songs import SimpleSongRetriever, filter_by_gathering
from setlistgraph.scoring.compatibility import is_transition_ok


class S(TypedDict, total=False):
    # inputs
    service: Dict[str, Any]        # {theme, scripture, gathering, n_songs}
    catalog_path: str

    # working state
    candidates: List[Dict[str, Any]]
    planned: List[Dict[str, Any]]

    # outputs
    songs: List[Dict[str, Any]]
    notes: str


def intake(state: S) -> S:
    s = state["service"]
    s["theme"] = (s.get("theme") or "").strip()
    s["scripture"] = (s.get("scripture") or "").strip()
    return {"service": s}


def retrieve_candidates(state: S) -> S:
    df = load_catalog(state["catalog_path"])
    query = f'{state["service"]["theme"]} {state["service"]["scripture"]}'.strip()
    retriever = SimpleSongRetriever(df)  # swap to RagSongRetriever when ready
    hits = retriever.search(query, top_k=50)
    hits = filter_by_gathering(hits, state["service"]["gathering"])
    keep = ["song_id", "title", "artist", "default_key", "bpm", "energy", "scripture_refs", "spotify_uri"]
    return {"candidates": hits[keep].to_dict(orient="records")}


def flow_plan(state: S) -> S:
    n = int(state["service"].get("n_songs", 4))
    cand = state.get("candidates", [])
    if not cand:
        return {"planned": []}

    planned = [cand[0]]
    remaining = cand[1:]

    def ok(prev, nxt):
        chk = is_transition_ok(prev["default_key"], prev["bpm"], nxt["default_key"], nxt["bpm"])
        return chk.keys_ok and chk.tempo_ok

    while len(planned) < n and remaining:
        prev = planned[-1]
        # gentle arc toward mid energy in the middle, then back up
        target_energy = 3 if 1 < len(planned) < n - 1 else prev.get("energy", 3)
        pick = None

        for r in remaining:
            if ok(prev, r) and abs((r.get("energy", 3)) - target_energy) <= 1:
                pick = r
                break

        if pick is None:
            for r in remaining:
                if ok(prev, r):
                    pick = r
                    break

        if pick is None:
            pick = remaining[0]

        planned.append(pick)
        remaining = [x for x in remaining if x is not pick]

    return {"planned": planned[:n]}


def assemble(state: S) -> S:
    # IMPORTANT: update the state (don’t replace it). Put the final plan in 'songs'.
    return {
        "songs": state.get("planned", []),
        "notes": "Keys ≤ 2 semitones between songs; tempo drift ≤ 15%.",
    }


def build_app():
    g = StateGraph(S)
    g.add_node("intake", intake)
    g.add_node("retrieve_candidates", retrieve_candidates)
    g.add_node("flow_plan", flow_plan)
    g.add_node("assemble", assemble)

    g.set_entry_point("intake")
    g.add_edge("intake", "retrieve_candidates")
    g.add_edge("retrieve_candidates", "flow_plan")
    g.add_edge("flow_plan", "assemble")
    g.add_edge("assemble", END)
    return g.compile()
