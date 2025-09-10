from __future__ import annotations
from dataclasses import dataclass

# Natural + sharps; we'll map flats to these
_CIRCLE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def _to_idx(key: str) -> int:
    k = key.strip().upper().replace("MIN", "M")
    k = k.replace("M", "")  # ignore minor for distance calc
    # flats → enharmonic sharps
    k = (k.replace("DB", "C#")
           .replace("EB", "D#")
           .replace("GB", "F#")
           .replace("AB", "G#")
           .replace("BB", "A#"))
    if k not in _CIRCLE:
        raise ValueError(f"Unknown key: {key}")
    return _CIRCLE.index(k)

def semitone_distance(k1: str, k2: str) -> int:
    i1, i2 = _to_idx(k1), _to_idx(k2)
    d = abs(i1 - i2) % 12
    return min(d, 12 - d)

def tempo_ok(bpm1: float, bpm2: float, drift: float = 0.15) -> bool:
    b1, b2 = float(bpm1), float(bpm2)
    return abs(b1 - b2) <= drift * max(b1, b2)

@dataclass
class TransitionCheck:
    keys_ok: bool
    tempo_ok: bool
    semitones: int
    bpm_delta: float

def is_transition_ok(k1: str, bpm1: float, k2: str, bpm2: float,
                     semitone_limit: int = 2, drift: float = 0.15) -> TransitionCheck:
    st = semitone_distance(k1, k2)
    bdelta = abs(float(bpm1) - float(bpm2))
    return TransitionCheck(
        keys_ok = st <= semitone_limit,
        tempo_ok = tempo_ok(bpm1, bpm2, drift=drift),
        semitones = st,
        bpm_delta = bdelta,
    )
