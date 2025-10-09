# 🎶 Setlist Planner

An AI-powered tool for planning worship setlists. This application uses vector search and Large Language Models (LLMs) to create cohesive worship sets based on themes and scripture references.

## ✨ Features

* **AI-Driven Planning**: Generate worship setlists based on themes and scripture references.
* **Musical Coherence**: Analyze key and BPM transitions for smooth flow between songs.
* **Song Import**: Import OnSong format files (`.txt` and `.pdf`).
* **Vector Search**: Find thematically relevant songs using semantic search.
* **Spotify Integration**: Quick links to songs on Spotify.
* **Export Options**: Export setlists as PDF (with transitions/explanations) or CSV (for spreadsheet editing).

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/setlist-planner.git
cd setlist-planner

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Setup

### 1. Install Ollama

This project uses [Ollama](https://ollama.ai/) to run local language models.

Download and install Ollama from [https://ollama.ai/](https://ollama.ai/).

```bash
# Pull the Llama model
ollama pull llama3.2:3b

# Start the Ollama server
ollama serve
```

### 2. Build the Vector Database

```bash
python scripts/build_chroma_from_csv.py \
  --csv src/setlistgraph/data/song_catalog.sample.csv \
  --output .chroma
```

---

## ▶️ Running the Application

```bash
# Run the Streamlit app
streamlit run src/setlistgraph/ui/app.py
```

---

## 🎛️ Using the App

### 🔍 1. Search for Songs

Use the **Search** tab to find songs based on themes or scripture references using semantic search.

### 🎵 2. Plan a Worship Set

* Enter a theme (e.g., `hope`, `grace`)
* Add an optional scripture reference
* Select the number of songs
* Click **Run agent** to create a setlist

### 📂 3. Import Songs

Expand your catalog by importing OnSong format files:

* Use the **Bulk OnSong** tab to import from a directory
* Drag and drop individual `.txt` or `.pdf` files

### 📤 4. Export Your Setlist

* **PDF** (with transitions and explanations)
* **CSV** (for spreadsheet editing)

---

## 📦 Requirements

* Python 3.8+
* Ollama with `llama3.2:3b` model
* Internet connection (for Spotify links)

---

## 🛠️ Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.
