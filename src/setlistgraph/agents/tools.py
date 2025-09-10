# src/setlistgraph/agents/tools.py
from .graph import build_app

def plan_setlist(theme: str, scripture: str, gathering: str, n_songs: int, catalog_path: str):
    app = build_app()
    state = {
        "service": {"theme": theme, "scripture": scripture, "gathering": gathering, "n_songs": n_songs},
        "catalog_path": catalog_path,
        "candidates": [],
        "planned": [],
    }
    return app.invoke(state)
