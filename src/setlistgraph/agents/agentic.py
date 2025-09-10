from __future__ import annotations

import json, re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- Local LLM (Ollama) ---
try:
    from langchain_ollama import ChatOllama
    _HAS_LLM = True
except Exception:
    _HAS_LLM = False

# --- Your building blocks ---
from setlistgraph.pipelines.embed_catalog import load_index
from setlistgraph.retrievers.semantic import SemanticSongRetriever
from setlistgraph.scoring.compatibility import is_transition_ok
from setlistgraph.io.onsong_loader import import_onsong_to_catalog


# =========================
# Utilities (planning/judge)
# =========================
def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _energy(x: Any, default: int = 3) -> int:
    try:
        v = int(x)
        return v if 1 <= v <= 5 else default
    except Exception:
        return default

def _transition_ok(prev: Dict[str, Any], nxt: Dict[str, Any]) -> bool:
    k1 = (prev.get("key") or prev.get("default_key") or "").strip()
    k2 = (nxt.get("key") or nxt.get("default_key") or "").strip()
    b1 = _to_float(prev.get("bpm"))
    b2 = _to_float(nxt.get("bpm"))
    chk = is_transition_ok(k1, b1, k2, b2, semitone_limit=2, drift=0.15)
    return chk.keys_ok and chk.tempo_ok

def plan_flow(candidates: List[Dict[str, Any]], n_songs: int) -> List[Dict[str, Any]]:
    """Greedy planner: seed by semantic score (tiny tilt to faster opener), then add feasible next songs."""
    if not candidates:
        return []
    # Seed
    best_i, best = 0, -1e9
    for i, r in enumerate(candidates[:12]):  # keep it small
        base = float(r.get("score", 0) or 0)
        bpm_bonus = min(_to_float(r.get("bpm")) / 200.0, 1.0) * 0.08
        val = base + bpm_bonus
        if val > best:
            best, best_i = val, i
    planned = [candidates[best_i]]
    remaining = candidates[:best_i] + candidates[best_i + 1 :]

    while len(planned) < n_songs and remaining:
        prev = planned[-1]
        best_val, best_row = -1e9, None
        for r in remaining:
            if not _transition_ok(prev, r):
                continue
            base = float(r.get("score", 0.0) or 0.0)
            # light arc toward mid energy (2–3) in middle slots
            pos = len(planned) + 1
            target_e = 3 if 1 < pos < n_songs else _energy(prev.get("energy", 3))
            e_score = max(0.0, 1.0 - abs(_energy(r.get("energy", 3)) - target_e) / 2.0)
            val = base + 0.08 * e_score
            if val > best_val:
                best_val, best_row = val, r
        if best_row is None:
            # fallback: any feasible
            for r in remaining:
                if _transition_ok(prev, r):
                    best_row = r
                    break
        if best_row is None:
            # last resort: ignore feasibility (assume keyswitch/pad)
            best_row = remaining[0]
        planned.append(best_row)
        remaining = [x for x in remaining if x is not best_row]
    return planned[:n_songs]

def judge_transitions(plan: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for i in range(len(plan) - 1):
        a, b = plan[i], plan[i + 1]
        k1 = (a.get("key") or a.get("default_key") or "").strip()
        k2 = (b.get("key") or b.get("default_key") or "").strip()
        b1 = _to_float(a.get("bpm"))
        b2 = _to_float(b.get("bpm"))
        chk = is_transition_ok(k1, b1, k2, b2, semitone_limit=2, drift=0.15)
        rows.append({
            "i": i,
            "from_title": a.get("title"),
            "to_title": b.get("title"),
            "from_key": k1, "to_key": k2,
            "from_bpm": b1, "to_bpm": b2,
            "semitones": chk.semitones, "bpm_delta": chk.bpm_delta,
            "keys_ok": chk.keys_ok, "tempo_ok": chk.tempo_ok,
        })
    violations = [r for r in rows if not (r["keys_ok"] and r["tempo_ok"])]
    return rows, violations


# =========================
# Agent State & Tools
# =========================
@dataclass
class AgentState:
    goal: str
    theme: str = ""
    scripture: str = ""
    n_songs: int = 4
    index_dir: str = ".rag_cache"
    catalog_path: str = "src/setlistgraph/data/catalog.csv"

    # runtime
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    history: List[Dict[str, str]] = field(default_factory=list)  # for agent scratchpad

# --- Tool registry ---
ToolFn = Callable[[AgentState, Dict[str, Any]], str]

def tool_retrieve_semantic(state: AgentState, args: Dict[str, Any]) -> str:
    """args: {"query": str, "top_k": int}"""
    query = args.get("query") or f"{state.theme} {state.scripture}".strip()
    top_k = int(args.get("top_k", 50))
    meta, emb, _ = load_index(state.index_dir)
    retriever = SemanticSongRetriever(meta, emb)
    hits = retriever.search(query or "*", top_k=top_k)
    keep = ["song_id","title","artist","bpm","key","energy","scripture_refs","spotify_uri","score"]
    state.candidates = hits[keep].to_dict(orient="records")
    return f"Retrieved {len(state.candidates)} candidates."

def tool_plan_flow(state: AgentState, args: Dict[str, Any]) -> str:
    """args: {"n_songs": int}"""
    n = int(args.get("n_songs", state.n_songs))
    state.plan = plan_flow(state.candidates, n)
    return f"Planned {len(state.plan)} songs."

def tool_judge(state: AgentState, args: Dict[str, Any]) -> str:
    trs, vio = judge_transitions(state.plan)
    state.transitions, state.violations = trs, vio
    return f"Judged {len(trs)} transitions; violations={len(vio)}."

def tool_export_csv(state: AgentState, args: Dict[str, Any]) -> str:
    """args: {"path": str}"""
    import pandas as pd
    path = args.get("path", "setlist_export.csv")
    pd.DataFrame(state.plan).to_csv(path, index=False)
    return f"Exported CSV to {path}."

def tool_import_onsong(state: AgentState, args: Dict[str, Any]) -> str:
    """args: {"onsong_text": str, "default_bpm": float}"""
    txt = args.get("onsong_text", "")
    default_bpm = float(args.get("default_bpm", 120.0))
    row = import_onsong_to_catalog(
        onsong_text=txt,
        catalog_path=state.catalog_path,
        lyrics_dir="lyrics_private",
        default_bpm=default_bpm,
    )
    return f"Imported '{row.get('title')}' by {row.get('artist')} with bpm={row.get('bpm')}."

TOOLS: Dict[str, Tuple[ToolFn, str, Dict[str, Any]]] = {
    "retrieve_semantic": (
        tool_retrieve_semantic,
        "Retrieve top candidate songs semantically from the embeddings index.",
        {"query": "str (free text)", "top_k": "int (default 50)"},
    ),
    "plan_flow": (
        tool_plan_flow,
        "Create a feasible setlist order (BPM used only for placement; constraints ≤2 semitones & ≤15% drift).",
        {"n_songs": "int (default from service)"},
    ),
    "judge_transitions": (
        tool_judge,
        "Audit adjacent transitions and report violations.",
        {},
    ),
    "export_csv": (
        tool_export_csv,
        "Export the planned setlist to a CSV file.",
        {"path": "str (default setlist_export.csv)"},
    ),
    "import_onsong": (
        tool_import_onsong,
        "Import an OnSong text into the catalog and store plain lyrics locally.",
        {"onsong_text": "str", "default_bpm": "float (default 120.0)"},
    ),
}

def tool_manifest() -> List[Dict[str, Any]]:
    out = []
    for name, (_, desc, schema) in TOOLS.items():
        out.append({"name": name, "description": desc, "args_schema": schema})
    return out


# =========================
# Agent loop
# =========================
SYSTEM_PROMPT = """You are a worship set planning agent.
You can call TOOLS to retrieve candidates, plan an order, judge transitions, export CSV, and import OnSong.
Rules:
- Always start by ensuring you have candidates (use retrieve_semantic).
- Then plan the flow (plan_flow).
- Then judge transitions (judge_transitions).
- If asked, export via export_csv.
- NEVER invent song data; only use what you retrieved.
- Respond ONLY in JSON with keys: action (tool name or "final") and action_input (dict).
If you are done, return action="final" with action_input containing a short 'summary'.
"""

def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))

def get_llm(model: str = "llama3.2", temperature: float = 0.1):
    if not _HAS_LLM:
        raise RuntimeError("langchain-ollama not installed. `pip install langchain-ollama` and run `ollama serve`.")
    return ChatOllama(model=model, temperature=temperature)

def run_agent(
    goal: str,
    theme: str = "",
    scripture: str = "",
    n_songs: int = 4,
    index_dir: str = ".rag_cache",
    catalog_path: str = "src/setlistgraph/data/catalog.csv",
    llm_model: str = "llama3.2",
    max_steps: int = 6,
) -> AgentState:
    """
    Agentic executor loop. Returns the final AgentState (with plan, transitions, violations, notes).
    """
    state = AgentState(
        goal=goal, theme=theme, scripture=scripture, n_songs=n_songs,
        index_dir=index_dir, catalog_path=catalog_path
    )
    llm = get_llm(model=llm_model)

    for step in range(max_steps):
        # Build scratchpad
        tools_json = json.dumps(tool_manifest(), ensure_ascii=False)
        context = {
            "goal": state.goal, "theme": state.theme, "scripture": state.scripture,
            "n_songs": state.n_songs,
            "have_candidates": len(state.candidates),
            "have_plan": len(state.plan),
            "violations": len(state.violations),
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"TOOLS={tools_json}"},
            {"role": "user", "content": f"CONTEXT={json.dumps(context)}"},
        ]
        if state.plan:
            messages.append({"role": "user", "content": f"CURRENT_PLAN_TITLES={[r.get('title') for r in state.plan]}"} )

        resp = llm.invoke(messages)  # ChatOllama accepts message list
        try:
            decision = _extract_json(resp.content)
        except Exception as e:
            # fallback: force retrieve → plan → judge path
            if not state.candidates:
                decision = {"action": "retrieve_semantic", "action_input": {"top_k": 50}}
            elif not state.plan:
                decision = {"action": "plan_flow", "action_input": {"n_songs": state.n_songs}}
            elif not state.transitions:
                decision = {"action": "judge_transitions", "action_input": {}}
            else:
                decision = {"action": "final", "action_input": {"summary": "Completed with fallback policy."}}

        action = decision.get("action", "")
        if action == "final":
            state.notes = (decision.get("action_input") or {}).get("summary", "")
            break

        if action not in TOOLS:
            # Guard: if nonsense action, follow fixed pipeline next
            if not state.candidates:
                action = "retrieve_semantic"; args = {"top_k": 50}
            elif not state.plan:
                action = "plan_flow"; args = {"n_songs": state.n_songs}
            elif not state.transitions:
                action = "judge_transitions"; args = {}
            else:
                state.notes = "Completed (fallback)."; break
        else:
            args = decision.get("action_input") or {}

        # Execute tool
        fn, _, _ = TOOLS[action]
        try:
            tool_result = fn(state, args)
        except Exception as e:
            tool_result = f"ERROR: {e}"

        # Log step
        state.history.append({"action": action, "args": json.dumps(args), "result": tool_result})

    return state
