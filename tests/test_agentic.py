from setlistgraph.io.loaders import load_catalog
from setlistgraph.retrievers.vdb import ChromaSongRetriever
import setlistgraph.agents.agentic_graph as ag

def test_agentic_runs_with_stub_llm(tmp_path, tiny_catalog_csv, chroma_dir, stub_sentence_transformer, stub_llm):
    # Seed vector DB
    df = load_catalog(tiny_catalog_csv, lyrics_dir=tmp_path / "lyrics_private")
    retr = ChromaSongRetriever(chroma_dir=chroma_dir)
    retr.upsert_rows(df.to_dict(orient="records"))

    # Run the agent with the stub LLM
    state = ag.run_agentic_graph(
        goal="Plan a hopeful 3-song set",
        theme="hope",
        scripture="1 Peter 1:3",
        n_songs=3,
        index_dir=chroma_dir,
        catalog_path=str(tiny_catalog_csv),
        llm_model="stubbed",
        temperature=0.1,
    )

    assert len(state.plan) == 3
    titles = [s.get("title") for s in state.plan]
    assert any("Hope" in (t or "") for t in titles)
    # audit may produce 0..2 transitions (3 songs => 2 transitions)
    assert len(state.transitions_report) in (0, 1, 2)
