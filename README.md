# AGH-AURA — hybrid RAG assistant

Asystent uczelniany AGH: porównanie trzech architektur RAG (wektorowy, hybrydowy GraphRAG, agentowy LangGraph).

## Wymagania

- Python 3.11+
- [Neo4j](https://neo4j.com/) (lokalnie, domyślnie `bolt://localhost:7687`)
- [Ollama](https://ollama.com/) z modelem `nomic-embed-text` (embeddingi)
- Klucz API [OpenRouter](https://openrouter.ai/)

## Quickstart (od zera)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # uzupełnij OPENROUTER_API_KEY i NEO4J_PASSWORD

# Uruchom Neo4j, następnie:
ollama pull nomic-embed-text

# Zbuduj bazę wiedzy z data/ (Neo4j + Chroma w ./db):
python ingest_hybrid.py

# Porównaj trzy pipeline'y na pytaniach z evaluation/questions.json:
python evaluation/run_compare.py --id terminy_01      # jedno pytanie
python evaluation/run_compare.py                        # pełny benchmark
```

## Wyniki

Po `run_compare.py` wyniki trafiają do:

- `evaluation/results/compare/results.json` — zbiorcze wyniki
- `evaluation/results/compare/report.md` — raport side-by-side
- `evaluation/results/compare/traces/{id}.json` — pełne trace per pytanie

Katalogi `db/` i `evaluation/results/` są generowane lokalnie i nie powinny trafiać do repozytorium.

## Struktura repozytorium

```
agents/          pipeline agentowy (LangGraph)
pipelines/       baseline wektorowy i hybrydowy
retrieval/       pobieranie z Neo4j i Chroma
evaluation/      run_compare.py + questions.json
data/            źródłowe CSV (graf) i PDF (wektory)
ingest_hybrid.py  ładowanie data/ → Neo4j + ./db
```

## Dane źródłowe

- `data/GraphRAG data/` — tabele CSV (progi, terminy, matury, egzaminy zawodowe itd.)
- `data/RAG data/` — dokumenty PDF do wyszukiwania wektorowego

Bez uruchomienia `ingest_hybrid.py` (przy pustym Neo4j / Chroma) compare nie przejdzie preflight.
