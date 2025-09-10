import json
import streamlit as st

from setlistgraph.agents.agentic_graph import run_agentic_graph

st.header("Agent Console")

# Goal & context
goal = st.text_area("Goal", "Plan a hopeful 4-song set for a family service and export it.")
theme = st.text_input("Theme", "hope")
scripture = st.text_input("Scripture", "1 Peter 1:3")
n_songs = st.number_input("Number of songs", min_value=1, max_value=8, value=4, step=1)

# Backend & model
index_backend = st.selectbox("Index backend", [".chroma (Vector DB)", ".rag_cache (files)"], index=0)
index_dir = ".chroma" if index_backend.startswith(".chroma") else ".rag_cache"
llm_model = st.text_input("LLM model (Ollama)", "llama3.2")
temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

catalog_path = st.text_input("Catalog path", "src/setlistgraph/data/catalog.csv")

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("Run agent")
with col2:
    export_btn = st.button("Export CSV after run")

if run_btn:
    state = run_agentic_graph(
        goal=goal,
        theme=theme,
        scripture=scripture,
        n_songs=int(n_songs),
        index_dir=index_dir,
        catalog_path=catalog_path,
        llm_model=llm_model,
        temperature=float(temperature),
    )
    st.subheader("Plan")
    st.write([s.get("title") for s in state.plan])

    st.subheader("Agent Notes")
    st.write(state.notes or "—")

    st.subheader("LLM Critique (if any)")
    st.write(getattr(state, "critique", "") or "—")

    st.subheader("Transition Metrics")
    if state.transitions_report:
        st.dataframe(state.transitions_report, use_container_width=True)
    else:
        st.write("—")

    st.subheader("Tool Calls (last 10)")
    st.code(json.dumps(state.history[-10:], ensure_ascii=False, indent=2))

    if export_btn and state.plan:
        import pandas as pd, time
        out = f"setlist_{int(time.time())}.csv"
        pd.DataFrame(state.plan).to_csv(out, index=False)
        st.success(f"Exported {out}")
        st.download_button("Download CSV", data=pd.DataFrame(state.plan).to_csv(index=False), file_name=out, mime="text/csv")
