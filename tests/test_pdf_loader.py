import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import directly from the module (without 'src.' prefix)
from setlistgraph.io.onsong_pdf_loader import OnSongPDFLoader, load_pdf

# Create test directory if it doesn't exist
test_dir = Path(__file__).parent / "data"
test_dir.mkdir(exist_ok=True, parents=True)

# Create a test PDF file if it doesn't exist
test_pdf_path = test_dir / "sample.pdf"
if not test_pdf_path.exists():
    # Create a simple text file as a test PDF
    with open(test_pdf_path, "w") as f:
        f.write("Amazing Grace\n")
        f.write("Artist: John Newton\n")
        f.write("Key: G\n")
        f.write("BPM: 70\n\n")
        f.write("Amazing grace how sweet the sound\n")
    print(f"Created test file at {test_pdf_path}")

# Test functions
def test_pdf_loader_initialization():
    """Test that the loader initializes properly."""
    loader = OnSongPDFLoader(str(test_pdf_path))
    assert loader.file_path == str(test_pdf_path)
    print("✓ Loader initialization test passed")

def test_pdf_parsing():
    """Test that OnSong PDF content is parsed correctly."""
    result = load_pdf(str(test_pdf_path))
    assert "title" in result
    assert result["title"] == "Upphöj Jesus"
    assert "artist" in result
    assert result["artist"] == "Alfred Nygren"
    assert "key" in result
    assert result["key"] == "E"
    print("✓ PDF parsing test passed")
    return result

if __name__ == "__main__":
    print("Running PDF loader tests...")
    test_pdf_loader_initialization()
    result = test_pdf_parsing()
    print("\nExtracted data:")
    for key, value in result.items():
        if key == "lyrics":
            value = value[:50] + "..." if len(value) > 50 else value
        print(f"  {key}: {value}")
    print("\nAll tests passed!")