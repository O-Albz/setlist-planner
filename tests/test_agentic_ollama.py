"""
Test script for agentic graph planning with Ollama.
Run this directly to test if the planner works outside of the Streamlit UI.
"""

import os
import sys
from pathlib import Path

print(f"Using Python interpreter: {sys.executable}")

# Make sure we can import from the project
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(f"Project root: {project_root}")

# Import the function we want to test
from src.setlistgraph.agents.agentic_graph import run_agentic_graph

def test_agentic_graph():
    """Test the agentic graph planning with Ollama."""
    
    # Set environment variables (both variants that might be used)
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
    os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
    
    print("\n--- Testing Agentic Graph Planning ---")
    print(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST')}")
    print(f"Using model: llama3.2:3b")
    
    # Use absolute paths to ensure correct file locations
    index_dir = str(project_root / ".chroma")
    catalog_path = str(project_root / "src" / "setlistgraph" / "data" / "song_catalog.sample.csv")
    
    print(f"Index dir: {index_dir}")
    print(f"Catalog path: {catalog_path}")
    
    try:
        # Run the planning function
        state = run_agentic_graph(
            goal="Plan a worship set",
            theme="hope",
            scripture="1 Peter 1:3",
            n_songs=3,  # Use a smaller number for faster testing
            index_dir=index_dir,
            catalog_path=catalog_path,
            llm_model="llama3.2:3b",
            temperature=0.4,
            max_steps=6
        )
        
        print("\n--- Planning Completed Successfully ---")
        print(f"Created plan with {len(state.plan)} songs")
        print("\nSong titles:")
        for i, song in enumerate(state.plan, 1):
            print(f"{i}. {song.get('title')} - {song.get('artist')}")
        
        print("\nCritique:")
        print(state.critique)
        
        if state.transitions_report:
            print("\nTransition report:")
            for tr in state.transitions_report:
                ok = "✅" if tr.get("keys_ok") and tr.get("tempo_ok") else "⚠️"
                print(f"{ok} {tr.get('from')} → {tr.get('to')} | "
                      f"Key: {tr.get('from_key')} → {tr.get('to_key')} ({tr.get('semitones')} semitones) | "
                      f"BPM: {tr.get('from_bpm')} → {tr.get('to_bpm')} ({tr.get('bpm_delta'):.1f}% change)")
        
        print("\n--- History (Tool Calls) ---")
        for entry in state.history:
            print(f"\n• Action: {entry.get('action')}")
            print(f"  Args: {entry.get('args')}")
            print(f"  Result: {entry.get('result')}")
            
        return True
    
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_agentic_graph()