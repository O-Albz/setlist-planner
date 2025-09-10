
from __future__ import annotations
import re, uuid, json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

# Regex to find chords like [G], [D/F#], [F#m], [Bb], etc.
CHORD_RE = re.compile(r"\[([A-G][#b]?m?(?:/[A-G][#b]?)?)\]")

@dataclass
class OnSongParseResult:
    title: str
    artist: str | None
    key: str | None
    bpm: float | None
    body_raw: str                 # original text
    lyrics_plain: str             # text with chords removed
    chords_by_line: list[list[str]]  # chords per line
    metadata: dict

def _parse_header(lines: list[str]) -> dict:
    meta = {}
    # OnSong headers typically look like: Title: ..., Artist: ..., Key: ..., Tempo: ...
    header_re = re.compile(r"^\s*([A-Za-zÄÖÅäöå]+)\s*:\s*(.+)$")
    for i, ln in enumerate(lines[:10]):
        m = header_re.match(ln.strip())
        if m:
            k, v = m.group(1).strip().lower(), m.group(2).strip()
            meta[k] = v
    # Support simple 1st/2nd lines as Title / Artist if not in key:value form
    if not meta.get("title") and lines:
        first = lines[0].strip()
        if first and ":" not in first and "[" not in first:
            meta["title"] = first
    if not meta.get("artist") and len(lines) > 1:
        second = lines[1].strip()
        if second and ":" not in second and "[" not in second:
            meta["artist"] = second
    return meta

def _strip_chords(line: str) -> tuple[str, list[str]]:
    chords = CHORD_RE.findall(line)
    text = CHORD_RE.sub("", line)
    # collapse double spaces created by removal
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text, chords

def parse_onsong_text(text: str) -> OnSongParseResult:
    lines = text.strip().splitlines()
    meta = _parse_header(lines)
    title = meta.get("title") or meta.get("namn") or ""
    artist = meta.get("artist") or None
    key = meta.get("key") or meta.get("tonart") or None
    bpm = None
    tempo = meta.get("tempo") or meta.get("bpm")
    if tempo:
        try:
            bpm = float(str(tempo).split()[0])
        except Exception:
            bpm = None

    # Remove pure header lines (those with "X: value")
    body_lines = []
    for ln in lines:
        if re.match(r"^\s*[A-Za-zÄÖÅäöå]+\s*:\s*.+$", ln.strip()):
            continue
        body_lines.append(ln)

    # Strip chords
    plain_lines = []
    chords_by_line = []
    for ln in body_lines:
        t, ch = _strip_chords(ln)
        plain_lines.append(t)
        chords_by_line.append(ch)

    lyrics_plain = "\n".join(plain_lines).strip()
    return OnSongParseResult(
        title=title or "Untitled",
        artist=artist,
        key=key,
        bpm=bpm,
        body_raw=text,
        lyrics_plain=lyrics_plain,
        chords_by_line=chords_by_line,
        metadata=meta,
    )

def _mode_from_key(k: str | None) -> str | None:
    if not k:
        return None
    k = k.strip()
    return "minor" if "m" in k.lower() and not k.lower().endswith("maj") else "major"

def _slug(s: str) -> str:
    import unicodedata
    s_norm = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s_norm = s_norm.lower()
    import re
    s_norm = re.sub(r"[^a-z0-9]+", "-", s_norm)
    return s_norm.strip("-") or "song"

def import_onsong_to_catalog(
    onsong_text: str,
    catalog_path: str | Path,
    lyrics_dir: str | Path = "lyrics_private",
    default_bpm: float = 120.0,
    energy: int | None = None,
    scripture_refs: str | None = None,
    spotify_uri: str | None = None,
    copyright_status: str = "copyrighted",
) -> dict:
    """
    Parse an OnSong text blob, save plain lyrics to a local file (gitignored),
    and append/merge a row into the catalog CSV/Parquet.
    """
    pr = parse_onsong_text(onsong_text)
    bpm = pr.bpm or default_bpm
    mode = _mode_from_key(pr.key)
    title = pr.title or "Untitled"
    artist = pr.artist or ""
    key = pr.key or ""

    # Prepare storage
    lyrics_dir = Path(lyrics_dir)
    lyrics_dir.mkdir(parents=True, exist_ok=True)
    song_id = str(uuid.uuid4())[:8]
    lyrics_path = lyrics_dir / f"{_slug(title)}_{song_id}.txt"
    lyrics_path.write_text(pr.lyrics_plain, encoding="utf-8")

    # Load catalog (CSV or Parquet); create if missing
    catalog_path = Path(catalog_path)
    if catalog_path.exists():
        if catalog_path.suffix.lower() == ".csv":
            df = pd.read_csv(catalog_path)
        else:
            df = pd.read_parquet(catalog_path)
    else:
        df = pd.DataFrame(columns=[
            "song_id","title","artist","bpm","key","mode","energy","scripture_refs",
            "spotify_uri","copyright_status","lyrics_path","theme_summary"
        ])

    row = {
        "song_id": song_id,
        "title": title,
        "artist": artist,
        "bpm": float(bpm),
        "key": key,
        "mode": mode,
        "energy": energy,
        "scripture_refs": scripture_refs,
        "spotify_uri": spotify_uri,
        "copyright_status": copyright_status,
        "lyrics_path": str(lyrics_path),
        "theme_summary": None,  # you can fill later
    }

    # Append and save
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    if catalog_path.suffix.lower() == ".csv":
        df.to_csv(catalog_path, index=False)
    else:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(catalog_path, index=False)

    return row

def preview_onsong(onsong_text: str) -> dict[str, Any]:
    pr = parse_onsong_text(onsong_text)
    return {
        "title": pr.title,
        "artist": pr.artist,
        "key": pr.key,
        "bpm": pr.bpm,
        "metadata": pr.metadata,
        "lyrics_preview": pr.lyrics_plain.splitlines()[:8],
        "first_line_chords": pr.chords_by_line[0] if pr.chords_by_line else [],
    }
