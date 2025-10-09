"""
Check what's in the existing Chroma database without rebuilding it.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.setlistgraph.retrievers.vdb import ChromaSongRetriever

def main():
    # Path to existing Chroma DB
    index_dir = ".chroma"
    
    print(f"Checking existing Chroma DB at: {index_dir}")
    
    try:
        # Initialize retriever without ingestion
        retriever = ChromaSongRetriever(index_dir)
        
        # Try to count documents in the collection
        try:
            count = retriever.count()
            print(f"Found {count} songs in the database")
        except Exception as e:
            print(f"Couldn't count songs: {e}")
        
        # Test a simple query
        print("\nTesting a simple query...")
        results = retriever.search("hope", top_k=5)
        print(f"Query 'hope' returned {len(results)} results")
        if results:
            for i, r in enumerate(results, 1):
                print(f"{i}. {r.get('title')} - {r.get('artist')}")
        else:
            print("No results found, but database exists.")
            
    except Exception as e:
        print(f"Error accessing Chroma DB: {e}")
        print("The database might exist but could be corrupted or incompatible.")

if __name__ == "__main__":
    main()