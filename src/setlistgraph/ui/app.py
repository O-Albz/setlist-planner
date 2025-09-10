
import streamlit as st
import pandas as pd
from pathlib import Path

from setlistgraph.io.loaders import load_catalog, validate_catalog
from setlistgraph.retrievers.songs import SimpleSongRetriever, filter_by_gathering
from setlistgraph.ui.onsong_batch_snippet import *
from setlistgraph.ui.onsong_import_snippet import *
from setlistgraph.ui.agent_console import *

st.set_page_config(page_title="SetlistGraph", page_icon="🎶", layout="wide")
st.title("SetlistGraph — Scripture-Grounded Set Planner")
st.caption("Week 1 prototype: searchable catalog with TF-IDF retriever")

with st.sidebar:
    st.header("Catalog")
    catalog_path = st.text_input("Song catalog CSV path", "src/setlistgraph/data/song_catalog.sample.csv")
    if st.button("Validate catalog"):
        try:
            df = load_catalog(catalog_path)
            validate_catalog(df)
            st.success(f"Catalog OK • {len(df)} songs")
        except Exception as e:
            st.error(str(e))

col1, col2, col3 = st.columns([1,1,1])
with col1:
    theme = st.text_input("Theme", placeholder="hope, grace, resurrection")
with col2:
    scripture = st.text_input("Scripture (optional)", placeholder="John 3:16; Psalm 23")
with col3:
    gathering = st.selectbox("Gathering type", ["family", "youth", "traditional"], index=0)

n_songs = st.slider("How many candidates to show", 5, 50, 20, step=5)
run = st.button("Search")

from setlistgraph.agents.tools import plan_setlist

st.markdown("---")
st.subheader("Plan Setlist")
if st.button("Plan with graph"):
    try:
        result = plan_setlist(theme, scripture, gathering, n_songs=4, catalog_path=catalog_path)
        st.success("Planned!")
        st.json(result)
    except Exception as e:
        st.error(str(e))


if run:
    try:
        df = load_catalog(catalog_path)
        retriever = SimpleSongRetriever(df)
        query = f"{theme} {scripture}".strip()
        hits = retriever.search(query, top_k=n_songs)
        hits = filter_by_gathering(hits, gathering)
        show_cols = ["title","artist","default_key","bpm","energy","scripture_refs","spotify_uri","score"]
        st.dataframe(hits[show_cols].reset_index(drop=True), use_container_width=True)
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.caption("Tip: Use precise themes (e.g., 'resurrection; hope') and scripture (e.g., '1 Peter 1:3') for better matches.")
