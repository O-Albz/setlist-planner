"""
Loader for OnSong PDF files.
Extracts song information including title, artist, key, BPM, and lyrics from PDF files.
"""

from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Try different PDF libraries in order of preference
try:
    import pypdf
    PDF_LIBRARY = "pypdf"
except ImportError:
    try:
        from pdfminer.high_level import extract_text
        PDF_LIBRARY = "pdfminer"
    except ImportError:
        PDF_LIBRARY = None

logger = logging.getLogger(__name__)

class OnSongPDFLoader:
    """Loads and parses OnSong PDF files."""
    
    def __init__(self, file_path: str):
        """Initialize with path to PDF file."""
        self.file_path = file_path
        if not PDF_LIBRARY:
            raise ImportError("No PDF library available. Install pypdf or pdfminer.six")
    
    def load(self) -> Dict[str, Any]:
        """Load and parse the PDF file."""
        text = self._extract_text()
        return self._parse_onsong_text(text)
    
    def _extract_text(self) -> str:
        """Extract text from the PDF file using available library."""
        try:
            if PDF_LIBRARY == "pypdf":
                return self._extract_with_pypdf()
            elif PDF_LIBRARY == "pdfminer":
                return extract_text(self.file_path)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            # Fall back to treating it as text file if PDF extraction fails
            try:
                with open(self.file_path, "r", encoding="utf-8", errors="replace") as file:
                    return file.read()
            except Exception:
                logger.error(f"Failed to read file as text: {self.file_path}")
                return ""
    
    def _extract_with_pypdf(self) -> str:
        """Extract text using pypdf library."""
        with open(self.file_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    
    def _parse_onsong_text(self, text: str) -> Dict[str, Any]:
        """Parse OnSong formatted text into structured data."""
        # Extract title, typically the first line
        lines = text.split('\n')
        title = None
        i = 0
        
        # Find the first non-empty line as the title
        while i < len(lines) and not title:
            if lines[i].strip():
                title = lines[i].strip()
            i += 1
                
        # Initialize metadata dictionary
        metadata = {}
        
        # Common OnSong metadata patterns
        patterns = {
            'artist': r'(?:Artist|Author):\s*(.+?)(?:\n|$)',
            'key': r'Key:\s*([A-G][#b]?m?)',
            'bpm': r'(?:Tempo|BPM):\s*(\d+)',
            'ccli': r'CCLI:\s*(\d+)',
            'capo': r'Capo:\s*(\d+)',
            'time_sig': r'Time(?:\s+Sig)?:\s*(\d+/\d+)',
        }
        
        # Search for metadata in the text
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()
        
        # Convert BPM to float if present
        if 'bpm' in metadata:
            try:
                metadata['bpm'] = float(metadata['bpm'])
            except ValueError:
                pass
        
        # Extract lyrics (everything after metadata section)
        lyrics_start = 0
        in_metadata = True
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for common metadata indicators
            if any(indicator in line_lower for indicator in ['key:', 'artist:', 'author:', 'bpm:', 'tempo:', 'ccli:', 'capo:', 'time:']):
                lyrics_start = i + 1
                continue
                
            # If we're past the metadata and find a chord or section marker, this is likely where lyrics start
            if i > 3 and (('[' in line and ']' in line) or  # Section marker like [Verse]
                         (line.strip() and any(c.isupper() for c in line) and  # Potential chord line
                          not line.startswith('CCLI') and 
                          not line.startswith('Key:'))):
                lyrics_start = i
                in_metadata = False
                break
        
        # Extract the lyrics part
        lyrics = "\n".join(lines[lyrics_start:]).strip()
        
        # Assemble the result
        result = {
            'title': title or "Unknown Title",
            'artist': metadata.get('artist', 'Unknown'),
            'key': metadata.get('key', ''),
            'bpm': metadata.get('bpm', 0),
            'ccli': metadata.get('ccli', ''),
            'capo': metadata.get('capo', ''),
            'time_sig': metadata.get('time_sig', ''),
            'lyrics': lyrics,
            'file_path': self.file_path,
            'file_name': Path(self.file_path).name
        }
        
        return result


def load_pdf(file_path: str) -> Dict[str, Any]:
    """Convenience function to load an OnSong PDF file."""
    loader = OnSongPDFLoader(file_path)
    return loader.load()
