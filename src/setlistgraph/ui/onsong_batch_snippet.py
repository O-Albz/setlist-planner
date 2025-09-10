
# Streamlit snippet: batch import folder or multiple uploads, then rebuild embeddings
import streamlit as st
import pandas as pd

from setlistgraph.io.onsong_batch import batch_import_from_dir, batch_import_from_uploads

st.subheader("Batch Import — OnSong")

tab1, tab2 = st.tabs(["Folder", "Upload files"])

with tab1:
    input_dir = st.text_input("Folder with OnSong files", "onsongs/")
    catalog_path = st.text_input("Catalog path (CSV or Parquet)", "src/setlistgraph/data/catalog.csv")
    lyrics_dir = st.text_input("Lyrics storage dir (gitignored)", "lyrics_private")
    default_bpm = st.number_input("Default BPM", value=120.0, step=1.0)
    rebuild = st.checkbox("Rebuild embeddings after import", value=True)
    if st.button("Import folder"):
        res = batch_import_from_dir(
            input_dir=input_dir,
            catalog_path=catalog_path,
            lyrics_dir=lyrics_dir,
            default_bpm=default_bpm,
            rebuild_embeddings=rebuild,
        )
        st.success(f"Imported {res.successes} file(s), {res.failures} failed.")
        if not res.added_rows.empty:
            st.dataframe(res.added_rows[["title","artist","bpm","key","lyrics_path"]], use_container_width=True)
        if res.errors:
            st.warning("Some errors occurred:")
            for e in res.errors:
                st.write("-", e)

with tab2:
    uploads = st.file_uploader("Upload .txt or .onsong files", type=["txt","onsong"], accept_multiple_files=True)
    catalog_path_u = st.text_input("Catalog path", "src/setlistgraph/data/catalog.csv", key="u_catalog")
    lyrics_dir_u = st.text_input("Lyrics storage dir", "lyrics_private", key="u_lyrics")
    default_bpm_u = st.number_input("Default BPM", value=120.0, step=1.0, key="u_bpm")
    rebuild_u = st.checkbox("Rebuild embeddings after import", value=True, key="u_rebuild")
    if st.button("Import uploads"):
        pairs = []
        for f in uploads or []:
            try:
                text = f.read().decode("utf-8")
            except Exception:
                text = f.read().decode(errors="ignore")
            pairs.append((f.name, text))
        res = batch_import_from_uploads(
            uploads=pairs,
            catalog_path=catalog_path_u,
            lyrics_dir=lyrics_dir_u,
            default_bpm=default_bpm_u,
            rebuild_embeddings=rebuild_u,
        )
        st.success(f"Imported {res.successes} file(s), {res.failures} failed.")
        if not res.added_rows.empty:
            st.dataframe(res.added_rows[["title","artist","bpm","key","lyrics_path"]], use_container_width=True)
        if res.errors:
            st.warning("Some errors occurred:")
            for e in res.errors:
                st.write("-", e)
