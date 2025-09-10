from __future__ import annotations

import json, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

# ---- Local LLM via Ollama (agentic supervisor + tools that need generation) ----
try:
    from langchain_ollama import ChatOllama
    _HAS_LLM = True
except Exception:
    _HAS_LLM = False

# ---- Retrieval & catalog plumbing (your existing modules) ----
from setlistgraph.pipelines.embed_catalog import load_index
from setlistgraph.retrievers.semantic import SemanticSongRetriever
from setlistgraph.io.onsong_loader import import_onsong_to_catalog

# ---- Optional hard metrics (for reporting only; NOT enforced) ----
from setlistgraph.scoring.compatibility import is_transition_ok


# =========================
# Utilities
# =========================
def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(0))

def _to_float(x: Any, default: float = 0.0) -> float:
    try: return float(x)
    except Exception: return default

def _energy(x: Any, default: int = 3) -> int:
    try:
        v = int(x)
        return v if 1 <= v <= 5 else default
    except Exception:
        return default


# =========================
# Agent State
# =========================
@dataclass
class AgentState:
    goal: str
    theme: str = ""
    scripture: str = ""
    n_songs: int = 4
    index_dir: str = ".rag_cache"
    catalog_path: str = "src/setlistgraph/data/catalog.csv"
    llm_model: str = "llama3.2"
    temperature: float = 0.4
    top_k: int = 50

    # runtime
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    critique: str = ""
    transitions_report: List[Dict[str, Any]] = field(default_factory=list)  # metrics only
    notes: str = ""
    history: List[Dict[str, str]] = field(default_factory=list)             # tool call log


# =========================
# Tools (side effects + state updates)
# =========================
ToolFn = Callable[[AgentState, Dict[str, Any]], str]

def tool_retrieve_semantic(state: AgentState, args: Dict[str, Any]) -> str:
    """args: {"query": str, "top_k": int}  — pure text similarity; BPM/Key not filtered here."""
    query = (args.get("query") or f"{state.theme} {state.scripture}").strip()
    top_k = int(args.get("top_k", state.top_k))
    meta, emb, _ = load_index(state.index_dir)
    ret = SemanticSongRetriever(meta, emb)
    hits = ret.search(query or "*", top_k=top_k)
    keep = ["song_id","title","artist","bpm","key","energy","scripture_refs","spotify_uri","score"]
    state.candidates = hits[keep].to_dict(orient="records")
    return f"Retrieved {len(state.candidates)} candidates."

def tool_plan_llm(state: AgentState, args: Dict[str, Any]) -> str:
    """
    args: {"n_songs": int}
    LLM proposes an order from top candidates (metadata only; no lyrics).
    """
    if not _HAS_LLM:
        raise RuntimeError("langchain-ollama not installed.")

    n = int(args.get("n_songs", state.n_songs))
    slate = state.candidates[: max(12, n)]  # keep prompt small
    if not slate:
        return "No candidates to plan."

    # Only metadata to the LLM (privacy)
    mini = [
        {
            "song_id": c.get("song_id"),
            "title": c.get("title"),
            "artist": c.get("artist"),
            "bpm": _to_float(c.get("bpm")),
            "key": (c.get("key") or c.get("default_key") or ""),
            "energy": _energy(c.get("energy", 3)),
            "scripture_refs": c.get("scripture_refs"),
            "score": float(c.get("score") or 0.0),
        } for c in slate if c.get("song_id")
    ]
    id_to_row = {r["song_id"]: r for r in mini}

    llm = ChatOllama(model=state.llm_model, temperature=state.temperature)
    prompt = f"""
You are a worship set planner. Choose an order of exactly {n} songs (IDs) from CANDIDATES.
Targets:
- Musical arc: opener (higher energy/tempo), reflective middle (lower), build, sending (higher)
- Prefer smoother transitions (small key/tempo jumps), but you MAY break this with a reason
- Keep semantic score high

Return ONLY JSON:
{{
  "order": ["<song_id>", ... exactly {n} ids ...],
  "reasoning": "short notes about the flow and any tradeoffs"
}}

CANDIDATES:
{json.dumps(mini, ensure_ascii=False)}
""".strip()

    resp = llm.invoke(prompt)
    data = _extract_json(resp.content)
    order_ids = [i for i in data.get("order", []) if i in id_to_row][:n]
    state.plan = [id_to_row[i] for i in order_ids]
    state.critique = data.get("reasoning", "")
    return f"LLM planned {len(state.plan)} songs."

def tool_reflect_and_repair(state: AgentState, args: Dict[str, Any]) -> str:
    """
    LLM critiques the current plan (e.g., rough transitions, arc issues) and returns a revised order.
    This gives you a fully agentic, non-deterministic planner.
    """
    if not _HAS_LLM:
        raise RuntimeError("langchain-ollama not installed.")
    if not state.plan:
        return "No plan to repair."

    # Compact view for the model
    plan_view = [
        {"song_id": s.get("song_id"), "title": s.get("title"), "bpm": _to_float(s.get("bpm")),
         "key": (s.get("key") or s.get("default_key") or ""), "energy": _energy(s.get("energy", 3))}
        for s in state.plan
    ]
    llm = ChatOllama(model=state.llm_model, temperature=state.temperature)
    prompt = f"""
You are a strict but creative music director.
Given PLAN, identify transition risks (key jumps, tempo clashes, energy whiplash).
Then produce a better order using ONLY the same song IDs (no new songs, no removals), or say it's already optimal.

Return ONLY JSON:
{{
  "reasons": "what you changed and why (short)",
  "order": ["<song_id>", ... same count as PLAN ...]
}}

PLAN:
{json.dumps(plan_view, ensure_ascii=False)}
""".strip()
    resp = llm.invoke(prompt)
    data = _extract_json(resp.content)
    order_ids = [i for i in data.get("order", [])]
    # Validate: if malformed, keep old plan
    if len(order_ids) != len(state.plan) or set(order_ids) != {s["song_id"] for s in state.plan}:
        return "Repair skipped (invalid order)."
    gid = {s["song_id"]: s for s in state.plan}
    state.plan = [gid[i] for i in order_ids]
    state.critique = (state.critique + "\n" if state.critique else "") + data.get("reasons", "")
    return "Plan repaired."

def tool_audit_metrics(state: AgentState, args: Dict[str, Any]) -> str:
    """
    Compute *metrics only* (NOT enforcing): semitone distance and tempo deltas between adjacent songs.
    This keeps the system agentic: the LLM may break rules with justification; you still get visibility.
    """
    trs: List[Dict[str, Any]] = []
    p = state.plan
    for i in range(len(p) - 1):
        a, b = p[i], p[i+1]
        k1 = (a.get("key") or a.get("default_key") or "").strip()
        k2 = (b.get("key") or b.get("default_key") or "").strip()
        b1 = _to_float(a.get("bpm")); b2 = _to_float(b.get("bpm"))
        chk = is_transition_ok(k1, b1, k2, b2, semitone_limit=2, drift=0.15)
        trs.append({
            "i": i,
            "from": a.get("title"), "to": b.get("title"),
            "from_key": k1, "to_key": k2,
            "from_bpm": b1, "to_bpm": b2,
            "semitones": chk.semitones, "bpm_delta": chk.bpm_delta,
            "keys_ok": chk.keys_ok, "tempo_ok": chk.tempo_ok,
        })
    state.transitions_report = trs
    return f"Audited {len(trs)} transitions."

def tool_export_csv(state: AgentState, args: Dict[str, Any]) -> str:
    path = args.get("path", "setlist_export.csv")
    import pandas as pd
    pd.DataFrame(state.plan).to_csv(path, index=False)
    return f"Exported to {path}."

def tool_import_onsong(state: AgentState, args: Dict[str, Any]) -> str:
    txt = args.get("onsong_text", "")
    default_bpm = float(args.get("default_bpm", 120.0))
    row = import_onsong_to_catalog(
        onsong_text=txt, catalog_path=state.catalog_path, lyrics_dir="lyrics_private", default_bpm=default_bpm
    )
    return f"Imported '{row.get('title')}' ({row.get('bpm')} BPM)."


TOOLS: Dict[str, Tuple[ToolFn, str, Dict[str, Any]]] = {
    "retrieve_semantic": (
        tool_retrieve_semantic,
        "Retrieve top candidate songs semantically (metadata only, no lyrics).",
        {"query": "str (free text)", "top_k": "int (default 50)"},
    ),
    "plan_llm": (
        tool_plan_llm,
        "Use an LLM to propose an ordered setlist from candidates (agentic, not deterministic).",
        {"n_songs": "int (default from state)"},
    ),
    "reflect_and_repair": (
        tool_reflect_and_repair,
        "LLM critiques the plan and returns a revised order using the same songs.",
        {},
    ),
    "audit_metrics": (
        tool_audit_metrics,
        "Compute semitone/tempo metrics between adjacent songs (visibility only).",
        {},
    ),
    "export_csv": (
        tool_export_csv,
        "Export the current plan to CSV.",
        {"path": "str (default setlist_export.csv)"},
    ),
    "import_onsong": (
        tool_import_onsong,
        "Import an OnSong text into the catalog (stores plain lyrics locally).",
        {"onsong_text": "str", "default_bpm": "float"},
    ),
}

def tool_manifest() -> List[Dict[str, Any]]:
    return [{"name": k, "description": v[1], "args_schema": v[2]} for k, v in TOOLS.items()]


# =========================
# Supervisor (Agentic loop)
# =========================
SUPERVISOR_PROMPT = """You are a worship set planning supervisor.
You can call TOOLS to achieve the GOAL. Think step-by-step and choose the next best tool.
Typical flow: retrieve_semantic → plan_llm → reflect_and_repair (optional) → audit_metrics → export_csv (optional).

IMPORTANT:
- Do not assume candidates exist; retrieve first if needed.
- Use reflect_and_repair only if you believe the plan can improve.
- Always run audit_metrics before finishing.
- Return ONLY JSON: {"action": "<tool name or final>", "action_input": {...}}.
- If done, use action="final" and include a short 'summary' in action_input.
"""

def get_llm(model: str, temperature: float):
    if not _HAS_LLM:
        raise RuntimeError("langchain-ollama not installed. `pip install langchain-ollama` and run `ollama serve`.")
    return ChatOllama(model=model, temperature=temperature)

def run_agentic_graph(
    goal: str,
    theme: str = "",
    scripture: str = "",
    n_songs: int = 4,
    index_dir: str = ".rag_cache",
    catalog_path: str = "src/setlistgraph/data/catalog.csv",
    llm_model: str = "llama3.2",
    temperature: float = 0.4,
    max_steps: int = 8,
) -> AgentState:
    state = AgentState(
        goal=goal, theme=theme, scripture=scripture, n_songs=n_songs,
        index_dir=index_dir, catalog_path=catalog_path,
        llm_model=llm_model, temperature=temperature,
    )
    llm = get_llm(model=llm_model, temperature=temperature)

    for _ in range(max_steps):
        ctx = {
            "goal": state.goal,
            "theme": state.theme,
            "scripture": state.scripture,
            "n_songs": state.n_songs,
            "have_candidates": len(state.candidates),
            "have_plan": len(state.plan),
            "history": state.history[-4:],  # last few steps
        }
        msg = [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": "TOOLS=" + json.dumps(tool_manifest(), ensure_ascii=False)},
            {"role": "user", "content": "CONTEXT=" + json.dumps(ctx, ensure_ascii=False)},
        ]
        if state.plan:
            msg.append({"role": "user", "content": "PLAN_TITLES=" + json.dumps([s.get("title") for s in state.plan])})

        resp = llm.invoke(msg)
        try:
            decision = _extract_json(resp.content)
        except Exception:
            # Fallback heuristic: follow the typical flow
            if not state.candidates:
                decision = {"action": "retrieve_semantic", "action_input": {"top_k": 50}}
            elif not state.plan:
                decision = {"action": "plan_llm", "action_input": {"n_songs": state.n_songs}}
            elif not state.transitions_report:
                decision = {"action": "audit_metrics", "action_input": {}}
            else:
                decision = {"action": "final", "action_input": {"summary": "Completed via fallback policy."}}

        action = decision.get("action", "")
        if action == "final":
            state.notes = (decision.get("action_input") or {}).get("summary", "")
            break

        args = decision.get("action_input") or {}
        fn_tuple = TOOLS.get(action)
        if not fn_tuple:
            # If the model requests an unknown tool, try the next sensible step
            if not state.candidates:
                action, args = "retrieve_semantic", {"top_k": 50}
            elif not state.plan:
                action, args = "plan_llm", {"n_songs": state.n_songs}
            else:
                action, args = "audit_metrics", {}
            fn_tuple = TOOLS[action]

        fn = fn_tuple[0]
        try:
            result = fn(state, args)
        except Exception as e:
            result = f"ERROR: {e}"
        state.history.append({"action": action, "args": json.dumps(args), "result": result})

    return state
