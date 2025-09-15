import sys, pandas as pd
from setlistgraph.retrievers.vdb import ChromaSongRetriever

if len(sys.argv) < 2:
    print("Usage: python scripts/build_chroma_from_csv.py <catalog_csv> [chroma_dir=.chroma]")
    sys.exit(1)

csv_path = sys.argv[1]
chroma_dir = sys.argv[2] if len(sys.argv) > 2 else ".chroma"

df = pd.read_csv(csv_path).fillna("")
retr = ChromaSongRetriever(chroma_dir=chroma_dir)
n = retr.upsert_rows(df.to_dict(orient="records"))
print(f"Upserted {n} songs into {chroma_dir}")
