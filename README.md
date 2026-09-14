# Emotional Equivalence Assessment for Literary Translation

This repository contains an interpretable system for assessing how well an English-to-Russian literary translation preserves emotional structure and culture-specific context.

## Research scope

The system combines eight Plutchik emotion categories, a valence-arousal-dominance (VAD) representation, cultural-marker detection, multilingual sentence alignment, and optional machine-learning and LLM components. Three interpretable components are integrated into the EQA score:

- Emotional Distance (ED): normalized Euclidean distance between source and translation VAD profiles.
- Preservation of Dominant Emotion (PDE): agreement and intensity preservation for the dominant Plutchik emotion.
- Cultural Adaptation (K): mean emotional-profile similarity for aligned contexts containing cultural markers.

The current implementation is a proof of concept. Its scores support expert analysis and do not constitute an autonomous judgement of literary quality.

## Features

- Eight-emotion Plutchik profiles
- VAD profiles and normalized emotional distance
- Cultural-marker detection and cultural-density visualization
- EQA calculation with ED, PDE, and K weights of 0.5, 0.3, and 0.2
- Multilingual sentence embeddings and optional sentiment models
- Optional recommendations through Ollama and Qwen2.5:7b
- English-language Flask interface with detailed, interpretable results

## Installation

Python 3.9 or later is recommended.

```bash
git clone https://github.com/PetrNikitin20/Emotional_Equivalence.git
cd Emotional_Equivalence
python -m venv venv
```

Activate the environment on macOS or Linux:

```bash
source venv/bin/activate
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Optional Ollama setup

Install Ollama from [ollama.com](https://ollama.com), start the local service, and download a model:

```bash
ollama serve
ollama pull qwen2.5:7b
```

For lower-memory systems, `qwen2.5:0.5b` may be used. The optional `mistral:7b` model supports the alternative allusion-extraction script.

## Running the application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Paste the English source passage and its Russian translation, then select **Run analysis**.

## Reproducing the analysis

The `data` directory contains the lexicons, cultural-marker tables, and literary-text files used by the prototype. The Russian-language lexicon entries and Russian translations are intentionally retained in Russian because they are primary research inputs; translating them would change the experimental material. All software documentation, interface labels, comments, console messages, and generated recommendations are in English.

Aggregate expert-validation results and their scope are documented in `data/validation`. Individual expert rating sheets were not present in the source archive and have not been reconstructed or inferred.

To rebuild the cultural-marker collection:

1. Place an English source text in `data/books`.
2. Add its filename to `my_books` in `cultural_markers_ds.py` or `cultural_markers_ds2.py`.
3. Run the selected extraction script with Ollama available.
4. Run `python cultural_markers_uni.py` to merge and deduplicate markers.
5. Review the generated `data/cultural_markers_merged.csv` before copying validated entries into `data/cultural_markers.csv`.

## Repository structure

```text
.
|-- app.py                         Flask application
|-- lexicon_loader.py              Lexicon loading and tokenization
|-- preprocessor.py                Text preprocessing and sentence alignment
|-- emotion_analyzer.py            Eight-emotion Plutchik analysis
|-- vad_analyzer.py                VAD analysis
|-- cultural_analyzer.py           Cultural-marker and density analysis
|-- eqa_calculator.py              ED, PDE, K, and EQA calculations
|-- ml_emotion_analyzer.py         Machine-learning emotional analysis
|-- llm_advisor.py                 Optional Ollama recommendations
|-- cultural_markers_ds.py         Qwen-based allusion extraction
|-- cultural_markers_ds2.py        Mistral-based allusion extraction
|-- cultural_markers_uni.py        Marker merging and deduplication
|-- templates/index.html           English-language web interface
`-- data                             Lexicons, markers, research texts, and validation summary
```

## Data and rights

The repository preserves the source project's research files to support reproducibility. Literary texts and third-party lexicons remain subject to their respective copyright and licensing terms; inclusion here does not grant additional rights. Researchers should verify that their use and redistribution comply with the applicable terms.

## Provenance and citation

This reproducibility repository was prepared for the article **“Automated Assessment of Emotional Equivalence in Literary Translation with Explicit Cultural-Marker Analysis”** by Petr Nikitin. 

Permanent project URL: <https://github.com/PetrNikitin20/Emotional_Equivalence>
