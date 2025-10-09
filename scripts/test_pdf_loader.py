"""
Test script for OnSong PDF loader.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.setlistgraph.io.onsong_pdf_loader import load_pdf

def main():
    # Use the sample PDF
    pdf_path = project_root / "tests" / "data" / "sample.pdf"
    
    if not pdf_path.exists():
        print(f"Error: Test PDF not found at {pdf_path}")
        return
    
    print(f"Testing PDF loader with: {pdf_path}")
    try:
        song_data = load_pdf(str(pdf_path))
        print("\nExtracted Song Data:")
        print("-" * 40)
        for key, value in song_data.items():
            if key == "lyrics":
                # Truncate lyrics for display
                preview = value[:150] + "..." if len(value) > 150 else value
                print(f"{key}: {preview}")
            else:
                print(f"{key}: {value}")
        print("-" * 40)
        print("PDF loading successful!")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()