
#!/usr/bin/env python3
"""
CLI: Batch-import OnSong files and optionally rebuild embeddings.

Usage:
  python scripts/onsong_batch_import.py --input ./onsongs --catalog src/setlistgraph/data/catalog.csv --lyrics-dir lyrics_private --rebuild
"""
from __future__ import annotations
import argparse
from pathlib import Path
from setlistgraph.io.onsong_batch import batch_import_from_dir

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Folder containing .txt/.onsong files")
    p.add_argument("--catalog", required=True, help="Catalog CSV or Parquet")
    p.add_argument("--lyrics-dir", default="lyrics_private", help="Folder to store plain lyrics")
    p.add_argument("--bpm", type=float, default=120.0, help="Default BPM if missing")
    p.add_argument("--rebuild", action="store_true", help="Rebuild embeddings after import")
    p.add_argument("--emb-out", default=".rag_cache", help="Embeddings output dir")
    args = p.parse_args()

    res = batch_import_from_dir(
        input_dir=args.input,
        catalog_path=args.catalog,
        lyrics_dir=args.lyrics_dir,
        default_bpm=args.bpm,
        rebuild_embeddings=args.rebuild,
        embeddings_out_dir=args.emb_out,
    )
    print(f"Imported {res.successes} file(s), {res.failures} failed.")
    if not res.added_rows.empty:
        print(res.added_rows[["title","artist","bpm","key","lyrics_path"]].to_string(index=False))
    if res.errors:
        print("\nErrors:")
        for e in res.errors:
            print(" -", e)

if __name__ == "__main__":
    main()
