# tests/test_graph.py
from setlistgraph.agents.tools import plan_setlist

def test_plan_graph_runs():
    out = plan_setlist("hope", "1 Peter 1:3", "family", 4, "src/setlistgraph/data/song_catalog.sample.csv")
    assert "songs" in out and len(out["songs"]) >= 1
