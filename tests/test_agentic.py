import json
import types
import setlistgraph.agents.agentic_graph as ag

class StubResp:
    def __init__(self, content): self.content = content

class StubChat:
    def __init__(self, *a, **k): pass
    def invoke(self, messages):
        # Decide tool based on context
        body = json.dumps({"action": "retrieve_semantic", "action_input": {"top_k": 10}})
        for m in messages:
            if isinstance(m, dict) and "CONTEXT=" in m.get("content",""):
                ctx = json.loads(m["content"].split("CONTEXT=")[-1])
                if ctx.get("have_candidates", 0) > 0 and ctx.get("have_plan", 0) == 0:
                    body = json.dumps({"action": "plan_llm", "action_input": {"n_songs": 2}})
                elif ctx.get("have_plan", 0) > 0:
                    body = json.dumps({"action": "audit_metrics", "action_input": {}})
        # When plan_llm is called, the tool will ask for an order; we return a valid JSON
        if any("Choose an order" in (m.get("content","") if isinstance(m, dict) else "") for m in messages):
            body = json.dumps({"order": [], "reasoning": "stub"})
        return StubResp(body)

def test_agentic_runs(monkeypatch):
    # monkeypatch ChatOllama
    monkeypatch.setattr(ag, "_HAS_LLM", True)
    monkeypatch.setattr(ag, "ChatOllama", StubChat)

    # monkeypatch retrieval to return two fake candidates
    def fake_tool_retrieve(state, args):
        state.candidates = [
            {"song_id": "A", "title": "Song A", "artist": "X", "bpm": 100, "key": "D", "energy": 4, "score": 0.9},
            {"song_id": "B", "title": "Song B", "artist": "Y", "bpm": 80, "key": "G", "energy": 2, "score": 0.8},
        ]
        return "ok"
    ag.TOOLS["retrieve_semantic"] = (fake_tool_retrieve, "stub", {})

    # monkeypatch plan_llm to produce deterministic order
    def fake_plan_llm(state, args):
        n = int(args.get("n_songs", 2))
        id2 = {c["song_id"]: c for c in state.candidates}
        order = ["A", "B"][:n]
        state.plan = [id2[i] for i in order]
        state.critique = "stub plan"
        return "ok"
    ag.TOOLS["plan_llm"] = (fake_plan_llm, "stub", {})

    state = ag.run_agentic_graph(
        goal="stub goal", theme="hope", scripture="1 Peter 1:3",
        n_songs=2, index_dir=".rag_cache", catalog_path="src/setlistgraph/data/catalog.csv",
        llm_model="llama3.2", temperature=0.2, max_steps=4,
    )

    assert len(state.plan) == 2
    assert isinstance(state.transitions_report, list)
