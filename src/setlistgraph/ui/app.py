import sys
import os
import os, json, requests, tempfile, io
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

from setlistgraph.io.loaders import load_catalog
from setlistgraph.retrievers.vdb import ChromaSongRetriever
from setlistgraph.agents.agentic_graph import run_agentic_graph
from setlistgraph.io.onsong_batch import batch_import_from_dir  # bulk OnSong

# Optional snippets if present in your repo
try:
    from setlistgraph.ui.onsong_batch_snippet import *  # noqa: F401,F403
    from setlistgraph.ui.onsong_import_snippet import *  # noqa: F401,F403
except Exception:
    pass

# ---------------- Utilities ----------------
@st.cache_resource(show_spinner=False)
def cached_retriever(index_dir: str) -> ChromaSongRetriever:
    return ChromaSongRetriever(index_dir)

@st.cache_data(show_spinner=False)
def cached_load_catalog(path: str) -> pd.DataFrame:
    return load_catalog(path)

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

def normalize_plan_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    want = ["order","role","title","artist","key","bpm","meter","spotify_uri","why"]
    return ensure_cols(df, want)

def spotify_resolve_if_missing(title: str, artist: str) -> Optional[str]:
    """
    Stub for future Spotify auto-resolve.
    Implement with your Spotify client; return spotify:track:... or URL.
    """
    return None

def export_plan_pdf(service: Dict[str, Any], plan_df: pd.DataFrame, transitions_df: Optional[pd.DataFrame]) -> bytes:
    # Lightweight PDF export using reportlab
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf)
    styles = getSampleStyleSheet()
    story = []

    title = service.get("title") or "Setlist"
    subtitle = f"{service.get('gathering_type','').title()} • Theme: {service.get('theme','—')} • Scripture: {', '.join(service.get('scripture', [])) or '—'}"
    story += [Paragraph(title, styles["Title"]), Spacer(1, 6), Paragraph(subtitle, styles["Normal"]), Spacer(1, 12)]

    if not plan_df.empty:
        tbl_data = [["#", "Role", "Title", "Artist", "Key", "BPM", "Meter", "Spotify", "Why"]]
        for i, row in plan_df.reset_index(drop=True).iterrows():
            tbl_data.append([
                int(row.get("order", i+1)),
                row.get("role", "—"),
                row.get("title", "—"),
                row.get("artist", "—"),
                row.get("key", "—"),
                row.get("bpm", "—"),
                row.get("meter", "—"),
                row.get("spotify_uri", "—"),
                (row.get("why") or "—")[:140],
            ])
        t = Table(tbl_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story += [Paragraph("Setlist", styles["Heading2"]), t, Spacer(1, 12)]

    if transitions_df is not None and not transitions_df.empty:
        cols = ["from_title","to_title","key_move","tempo_move","method","notes"]
        transitions_df = ensure_cols(transitions_df.copy(), cols)
        tbl_data = [["From", "To", "Key move", "Tempo move", "Method", "Notes"]]
        for _, row in transitions_df.iterrows():
            tbl_data.append([
                row.get("from_title","—"),
                row.get("to_title","—"),
                row.get("key_move","—"),
                row.get("tempo_move","—"),
                row.get("method","—"),
                (row.get("notes") or "—")[:120],
            ])
        t = Table(tbl_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story += [Paragraph("Transitions", styles["Heading2"]), t]

    doc.build(story)
    return buf.getvalue()

# Import additional module for timing (optional)
import time
from datetime import datetime

# Define a callback function for agent progress
def create_progress_tracker(container):
    """Creates a progress tracking function that updates a Streamlit container."""
    start_time = datetime.now()
    current_step = {"tool": None, "start": None}
    
    def update_progress(tool_name=None, status="running"):
        """Update the progress display."""
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        
        if tool_name:
            current_step["tool"] = tool_name
            current_step["start"] = now
        
        if status == "running" and current_step["tool"]:
            # Calculate time spent on current step
            step_time = (now - current_step["start"]).total_seconds()
            
            # Display with spinning wheel and timing info
            container.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="margin-right: 10px;">
                    <span style="display: inline-block; width: 20px; height: 20px; border: 2px solid #3366ff; border-radius: 50%; border-top-color: transparent; animation: spin 1s linear infinite;"></span>
                </div>
                <div>
                    <b>Current tool:</b> {current_step["tool"]} <span style="color: gray; font-size: 0.8em;">({step_time:.1f}s)</span>
                </div>
            </div>
            <style>
                @keyframes spin {{ 
                    to {{ transform: rotate(360deg); }}
                }}
            </style>
            """, unsafe_allow_html=True)
        
        elif status == "complete":
            container.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="margin-right: 10px;">
                    <span style="color: green;">✓</span>
                </div>
                <div>
                    <b>Planning complete!</b> <span style="color: gray; font-size: 0.8em;">({elapsed:.1f}s total)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    return update_progress

# ---------------- App header ----------------
st.set_page_config(page_title="SetlistGraph", page_icon="🎶", layout="wide")
st.title("SetlistGraph — Scripture-Grounded Set Planner")
st.caption("Chroma vector search + Agentic planning (LLM)")

st.sidebar.write(f"Python: {sys.executable}")
st.sidebar.write(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

# ---------------- Sidebar config ----------------
with st.sidebar:
    st.header("Config")
    catalog_path = st.text_input("Catalog CSV", "src/setlistgraph/data/song_catalog.sample.csv")
    index_dir = st.text_input("Vector DB (Chroma dir)", ".chroma")

    # Ollama host + model
    ollama_host = st.text_input("Ollama host", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model = st.text_input("LLM (Ollama model)", os.getenv("LLM_MODEL", "llama3.2:3b"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    if st.button("Check Ollama", use_container_width=True):
        try:
            r = requests.get(f"{ollama_host}/api/tags", timeout=3)
            r.raise_for_status()
            models = [m.get("name", "unknown") for m in r.json().get("models", [])]
            st.success(f"Ollama OK. Models: {', '.join(models) or '—'}")
        except Exception as e:
            st.error(f"Ollama unreachable: {e}")

    st.divider()
    st.subheader("Catalog actions")
    if st.button("Validate & show count", use_container_width=True):
        try:
            df = cached_load_catalog(catalog_path)
            st.success(f"Catalog OK • {len(df)} songs")
        except Exception as e:
            st.error(str(e))

    uploaded_csv = st.file_uploader("Upload CSV to ingest (min: title,artist,bpm,lyrics)", type=["csv"])
    if st.button("Ingest to Chroma", use_container_width=True):
        try:
            if uploaded_csv is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_csv.getvalue())
                    tmp_path = tmp.name
                df_full = cached_load_catalog(tmp_path)
            else:
                df_full = cached_load_catalog(catalog_path)

            retr = cached_retriever(index_dir)
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
            retr = cached_retriever(index_dir)
            res = retr.search(q, top_k=int(top_k))
            show_cols = ["title", "artist", "key", "bpm", "scripture_refs", "score"]
            res = ensure_cols(res, show_cols)
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

    run_col, export_col = st.columns([1,1])
    with run_col:
        run_btn = st.button("Run agent", type="primary")
    with export_col:
        export_btn = st.button("Export PDF")

    if run_btn:
        try:
            # Try to verify langchain-ollama is installed
            try:
                import langchain_ollama
                st.sidebar.success(f"✓ langchain-ollama {langchain_ollama.__version__} installed")
            except ImportError:
                st.error("❌ langchain-ollama not installed in this environment")
                st.code("pip install langchain-ollama", language="bash")
                st.stop()
            
            # Create progress container
            progress_container = st.empty()
            
            # Initialize progress tracker
            update_progress = create_progress_tracker(progress_container)

            # Create a modified callback for the agentic graph
            def tool_callback(action, args=None, result=None, status="running"):
                if status == "start":
                    update_progress(tool_name=action)
                elif status == "complete":
                    # Could add specific completion handling here
                    pass
            
            # Set environment variables
            os.environ["OLLAMA_HOST"] = ollama_host
            os.environ["OLLAMA_BASE_URL"] = ollama_host
            
            # Call run_agentic_graph with the modified callback
            state = run_agentic_graph(
                goal=f"Plan a {n_songs}-song worship set",
                theme=theme,
                scripture=scripture,
                n_songs=n_songs,
                index_dir=index_dir,
                catalog_path=catalog_path,
                llm_model=model,
                temperature=float(temperature),
                tool_callback=tool_callback  # Pass the callback
            )
            
            # Update progress as complete
            update_progress(status="complete")
            
            st.session_state["last_state"] = state  # keep for export
            st.success("Planned!")

            # After successfully running the agent:
            if state:
                # 1. Display the song plan
                st.subheader("Plan (order):")
                
                # Create a nice formatted list of songs
                for i, song in enumerate(state.plan, 1):
                    st.markdown(f"**{i}. {song.get('title')}** — {song.get('artist')} (Key={song.get('key')}, BPM={song.get('bpm')})")
                
                # 2. Display the LLM's reasoning
                if state.critique:
                    st.subheader("Reasoning:")
                    st.markdown(state.critique)
                
                # 3. Display transition metrics
                if state.transitions_report:
                    st.subheader("Transitions (metrics):")
                    
                    for tr in state.transitions_report:
                        ok = "✅" if tr.get("keys_ok") and tr.get("tempo_ok") else "⚠️"
                        st.markdown(f"{ok} **{tr.get('from')}** → **{tr.get('to')}**")
                        st.markdown(f"- Key: {tr.get('from_key')} → {tr.get('to_key')} ({tr.get('semitones')} semitones)")
                        st.markdown(f"- BPM: {tr.get('from_bpm')} → {tr.get('to_bpm')} ({tr.get('bpm_delta'):.1f}% change)")
                
                # 4. Show the LLM's thought process (history)
                with st.expander("View agent's thought process", expanded=False):
                    st.subheader("Agent Steps:")
                    
                    for entry in state.history:
                        st.markdown(f"**Action:** {entry.get('action')}")
                        st.markdown(f"**Args:** `{entry.get('args')}`")
                        
                        # Format the result differently based on content
                        result = entry.get('result')
                        if result.startswith("ERROR:"):
                            st.error(result)
                        else:
                            st.success(result)
                        
                        # Add divider between entries
                        st.divider()

            else:
                st.info("No songs in plan. Make sure your Chroma index has entries (sidebar → Ingest).")

            st.markdown("**Transitions (metrics):**")
            if getattr(state, "transitions_report", None) is not None and not state.transitions_report.empty:
                st.dataframe(state.transitions_report, use_container_width=True)
            else:
                st.caption("No transitions to audit.")

            st.markdown("**LLM notes:**")
            st.write(getattr(state, "critique", None) or "—")

            # More detailed formatting
            if state.critique:
                st.subheader("Why This Order Works")
                
                # Add a nice card-like container
                with st.container():
                    st.markdown("*LLM Analysis:*")
                    
                    # Add a subtle background color to highlight the analysis
                    st.markdown(f"""
                    <div style="background-color: rgba(100,100,100,0.1); padding: 15px; border-radius: 5px;">
                    {state.critique}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add explanation about what this means
                    st.caption("This analysis explains how the songs flow together in terms of theme, energy, and message.")

            if state.plan:
                csv = pd.DataFrame(state.plan).to_csv(index=False)
                st.download_button("Download plan CSV", csv, file_name="setlist.csv", mime="text/csv")

        except Exception as e:
            import traceback
            st.error(f"Full error: {str(e)}")
            st.expander("Error details").code(traceback.format_exc(), language="python")
            
            if "ollama" in str(e).lower() or "connect" in str(e).lower():
                st.error("LLM unreachable. Start Ollama (`ollama serve`) and ensure the model is pulled (e.g., `ollama pull llama3.2:3b`).")
            else:
                st.error(str(e))

    if export_btn:
        state = st.session_state.get("last_state")
        if not state or not state.plan:
            st.warning("No planned set found. Run the agent first.")
        else:
            plan_df = normalize_plan_rows(state.plan)
            # try to pass scripture as list if present
            scripture_list = [s.strip() for s in (state.scripture or "").split(",")] if getattr(state, "scripture", None) else []
            service = {
                "title": f"Planned Set — {theme.title()}",
                "gathering_type": "youth" if "youth" in (theme+scripture).lower() else "service",
                "theme": theme,
                "scripture": scripture_list,
            }
            trans_df = getattr(state, "transitions_report", None)
            pdf_bytes = export_plan_pdf(service, plan_df, trans_df)
            st.download_button("Download PDF", data=pdf_bytes, file_name="setlist.pdf", mime="application/pdf")

            # Add after displaying the plan
            if state and state.plan:
                st.subheader("Quick Spotify Search Links")
                
                for i, song in enumerate(state.plan, 1):
                    title = song.get('title', '')
                    artist = song.get('artist', '')
                    
                    # Create a search URL for Spotify (no API needed)
                    search_query = f"{title} {artist}".replace(" ", "%20")
                    spotify_search_url = f"https://open.spotify.com/search/{search_query}"
                    
                    st.markdown(f"**{i}. {title}** — {artist} &nbsp; [🔍 Find on Spotify]({spotify_search_url})")

# ---------------- Bulk OnSong tab ----------------
with tab_onsong:
    st.subheader("Bulk import OnSong files (local folder)")
    input_dir = st.text_input("Folder path (local)", "onsongs/")
    glob_str = st.text_input("Glob patterns", "*.txt,*.onsong")  # add *.pdf when OCR pipeline is wired
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
                rebuild_embeddings=False,  # we upsert below
            )
            st.success(f"Imported {res.successes} files; {res.failures} failed.")
            if res.errors:
                with st.expander("Errors"):
                    st.code("\n".join(res.errors[:200]))

            # Upsert newly added rows into Chroma
            if not res.added_rows.empty:
                retr = cached_retriever(index_dir)
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
        "has_last_state": bool(st.session_state.get("last_state"))
    }, indent=2))
    st.caption("Tip: After changing the CSV, hit **Ingest to Chroma** to refresh the vector index.")
