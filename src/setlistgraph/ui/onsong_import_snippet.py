
# Streamlit snippet: paste/upload OnSong text and add to catalog
import streamlit as st
from pathlib import Path
from setlistgraph.io.onsong_loader import preview_onsong, import_onsong_to_catalog

st.subheader("Import OnSong")
catalog_path = st.text_input("Catalog path (CSV or Parquet)", "src/setlistgraph/data/catalog.csv")
lyrics_dir = st.text_input("Lyrics storage dir (gitignored)", "lyrics_private")
default_bpm = st.number_input("Default BPM (used if not in OnSong)", value=120.0, step=1.0)

onsong_text = st.text_area("Paste OnSong text here", height=300)

col1, col2 = st.columns(2)
with col1:
    if st.button("Preview"):
        try:
            info = preview_onsong(onsong_text)
            st.json(info)
        except Exception as e:
            st.error(str(e))
with col2:
    if st.button("Import into catalog"):
        try:
            row = import_onsong_to_catalog(onsong_text, catalog_path, lyrics_dir, default_bpm=default_bpm)
            st.success("Imported ✅")
            st.json(row)
        except Exception as e:
            st.error(str(e))
