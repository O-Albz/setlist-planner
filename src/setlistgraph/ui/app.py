import os, json, requests, tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

from setlistgraph.io.loaders import load_catalog
from setlistgraph.retrievers.vdb import ChromaSongRetriever
from setlistgraph.agents.agentic_graph import run_agentic_graph
from setlistgraph.io.onsong_batch import batch_import_from_dir  # bulk OnSong

# Optional extra snippets if you have them
try:
    from setlistgraph.ui.onsong_batch_snippet import *  # noqa: F401,F403
    from setlistgraph.ui.onsong_import_snippet import *  # noqa: F401,F403
except Exception:
    pass

st.set_page_config(page_title="SetlistGraph", page_icon="🎶", layout="wide")
st.title("SetlistGraph — Scripture-Grounded Set Planner")
st.caption("Chroma vector search + Agentic planning (LLM)")

# ---------------- Sidebar config ----------------
with st.sidebar:
    st.header("Config")
    catalog_path = st.text_input("Catalog CSV", "src/setlistgraph/data/song_catalog.sample.csv")
    index_dir = st.text_input("Vector DB (Chroma dir)", ".chroma")

    # Ollama host + model
    ollama_host = st.text_input("Ollama host", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model = st.text_input("LLM (Ollama model)", os.getenv("LLM_MODEL", "llama3.2:3b"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    if st.button("Check Ollama"):
        try:
            r = requests.get(f"{ollama_host}/api/tags", timeout=3)
            r.raise_for_status()
            models = [m.get("name", "unknown") for m in r.json().get("models", [])]
            st.success(f"Ollama OK. Models: {', '.join(models) or '—'}")
        except Exception as e:
            st.error(f"Ollama unreachable: {e}")

    st.divider()
    st.subheader("Catalog actions")
    if st.button("Validate & show count"):
        try:
            df = load_catalog(catalog_path)  # also writes lyrics files if 'lyrics' exists
            st.success(f"Catalog OK • {len(df)} songs")
        except Exception as e:
            st.error(str(e))

    uploaded_csv = st.file_uploader("Upload CSV to ingest (min: title,artist,bpm,lyrics)", type=["csv"])
    if st.button("Ingest to Chroma"):
        try:
            if uploaded_csv is not None:
                # Save uploaded file to a temp path so load_catalog can normalize it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_csv.getvalue())
                    tmp_path = tmp.name
                df_full = load_catalog(tmp_path)
            else:
                df_full = load_catalog(catalog_path)

            retr = ChromaSongRetriever(index_dir)
            n = retr.upsert_rows(df_full.to_dict(orient="records"))
            st.success(f"Upserted {n} songs → {index_dir}")
        except Exception as e:
            st.error(str(e))

# ---------------- Tabs ----------------
tab_search, tab_plan, tab_onsong, tab_debug = st.tabs(["🔎 Search", "🎛️ Plan", "📥 Bulk OnSong", "🧪 Debug"])

# ---------------- Search tab ----------------
with tab_search:
    st.subheader("Semantic search")
    qcol, kcol = st.columns([3,1])
    with qcol:
        q = st.text_input("Query", "hope 1 Peter 1:3")
    with kcol:
        top_k = st.number_input("Top-K", 1, 100, 20, 1)

    if st.button("Search", type="primary"):
        try:
            retr = ChromaSongRetriever(index_dir)
            res = retr.search(q, top_k=int(top_k))
            show_cols = ["title", "artist", "key", "bpm", "scripture_refs", "score"]
            for c in show_cols:
                if c not in res.columns:
                    res[c] = None
            st.dataframe(res[show_cols].reset_index(drop=True), use_container_width=True)
        except Exception as e:
            st.error(str(e))

# ---------------- Plan tab ----------------
with tab_plan:
    st.subheader("Agentic planner")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        theme = st.text_input("Theme", "hope")
    with c2:
        scripture = st.text_input("Scripture (optional)", "1 Peter 1:3")
    with c3:
        n_songs = st.number_input("# Songs", 1, 8, 4, 1)

    if st.button("Run agent"):
        try:
            # Pass host via env for the agent
            os.environ["OLLAMA_HOST"] = ollama_host

            state = run_agentic_graph(
                goal=f"Plan a {int(n_songs)}-song set",
                theme=theme,
                scripture=scripture,
                n_songs=int(n_songs),
                index_dir=index_dir,
                catalog_path=catalog_path,
                llm_model=model,
                temperature=float(temperature),
            )
            st.success("Planned!")

            st.markdown("**Plan (order):**")
            if state.plan:
                st.write([f"{i+1}. {s.get('title')} — {s.get('artist')} (PE={s.get('perceived_energy','—')})"
                          for i, s in enumerate(state.plan)])
            else:
                st.info("No songs in plan. Make sure your Chroma index has entries (sidebar → Ingest).")

            st.markdown("**Transitions (metrics only):**")
            if state.transitions_report:
                st.dataframe(state.transitions_report, use_container_width=True)
            else:
                st.caption("No transitions to audit.")

            st.markdown("**LLM notes:**")
            st.write(state.critique or "—")

            if state.plan:
                csv = pd.DataFrame(state.plan).to_csv(index=False)
                st.download_button("Download plan CSV", csv, file_name="setlist.csv", mime="text/csv")

        except Exception as e:
            msg = str(e)
            if "ollama" in msg.lower() or "connect" in msg.lower():
                st.error("LLM unreachable. Start Ollama (`ollama serve`) and ensure the model is pulled (e.g., `ollama pull llama3.2:3b`).")
            else:
                st.error(msg)

# ---------------- Bulk OnSong tab ----------------
with tab_onsong:
    st.subheader("Bulk import OnSong files (local folder)")
    input_dir = st.text_input("Folder path (local)", "onsongs/")
    glob_str = st.text_input("Glob patterns", "*.txt,*.onsong")  # add *.pdf if your OCR pipeline is wired
    default_bpm = st.number_input("Default BPM (used if not present)", 40, 240, 120)

    if st.button("Import folder"):
        try:
            patterns = tuple([p.strip() for p in glob_str.split(",") if p.strip()])
            res = batch_import_from_dir(
                input_dir=input_dir,
                catalog_path=catalog_path,
                lyrics_dir="lyrics_private",
                glob_patterns=patterns,
                default_bpm=float(default_bpm),
                rebuild_embeddings=False,  # using Chroma; we will upsert below
            )
            st.success(f"Imported {res.successes} files; {res.failures} failed.")
            if res.errors:
                with st.expander("Errors"):
                    st.code("\n".join(res.errors[:200]))

            # Upsert newly added rows into Chroma
            if not res.added_rows.empty:
                retr = ChromaSongRetriever(index_dir)
                n = retr.upsert_rows(res.added_rows.fillna("").to_dict(orient="records"))
                st.info(f"Upserted {n} new songs into {index_dir}")

        except Exception as e:
            st.error(str(e))

# ---------------- Debug tab ----------------
with tab_debug:
    st.subheader("Diagnostics")
    st.code(json.dumps({
        "catalog_path": catalog_path,
        "index_dir": index_dir,
        "ollama_host": ollama_host,
        "model": model,
        "temperature": temperature,
    }, indent=2))
    st.caption("Tip: After changing the CSV, hit **Ingest to Chroma** to refresh the vector index.")
