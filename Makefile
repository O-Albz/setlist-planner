PY=python3
VENV=.venv

setup:
\tbash scripts/setup_macos.sh

activate:
\t@echo "Run: source $(VENV)/bin/activate"

seed-chroma:
\t$(PY) scripts/build_chroma_from_csv.py src/setlistgraph/data/song_catalog.sample.csv .chroma

test:
\t$(PY) -m pytest -q

agent:
\tstreamlit run src/setlistgraph/ui/app.py
