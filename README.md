# SetlistGraph (Setlist Planner)

Scripture-grounded worship set planner (LangGraph + Streamlit).

## What it does
Input: service theme, scripture, gathering type →  
Output: 3–6 song setlist with title, artist, key, BPM, Spotify URI, rationale, and transitions.

## Quickstart
```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # put your real keys in .env

# run UI
make run
# or: streamlit run src/setlistgraph/ui/app.py
```

## Project layout
```
src/setlistgraph/        # package code
tests/                   # unit tests
notebooks/               # optional ETL/experiments
```

## Env vars (see .env.example)
- `OPENAI_API_KEY`
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`

## Common tasks
```bash
make setup     # install deps + pre-commit hooks
make lint      # ruff
make format    # black
make test      # pytest
make run       # streamlit UI
```

## License
MIT
