import os
import json
import re
import chromadb
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# ── Klienci ───────────────────────────────────────────────────────────────────

model = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", ""),
    refresh_schema=False,
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="pdfs",
    embedding_function=embeddings,
    persist_directory=os.getenv("DB_PATH", "./db"),
)


# ── Cypher queries ────────────────────────────────────────────────────────────

CYPHER_QUERIES = {
    "matury": """
        MATCH (k:Kierunek {id: $kierunek})-[r:WYMAGA_PRZEDMIOTU]->(p:Przedmiot_Maturalny)
        RETURN k.id AS kierunek,
               collect({przedmiot: p.id, kolumna: r.kolumna}) AS przedmioty
    """,

    "progi": """
        MATCH (k:Kierunek)-[r:OFEROWANY_JAKO]->(t:Tryb_Studiow)
        WHERE $kierunek IS NULL OR k.id = $kierunek
        RETURN k.id AS kierunek, t.id AS tryb,
               r.prog_min AS prog_min, r.prog_max AS prog_max,
               r.miejsca  AS miejsca
        ORDER BY r.prog_max DESC
    """,

    "egzaminy": """
        MATCH (k:Kierunek)-[r:HONORUJE_EGZAMIN]->(e:Egzamin_Zawodowy)
        WHERE $kierunek IS NULL OR k.id = $kierunek
        RETURN k.id AS kierunek,
               collect({egzamin: e.id, kolumna: r.kolumna}) AS egzaminy
    """,

    "osiagniecia": """
        MATCH (k:Kierunek)-[r:DAJE_PUNKTY_ZA]->(o:Osiagniecie)
        WHERE $kierunek IS NULL OR k.id = $kierunek
        RETURN k.id AS kierunek,
               collect({osiagniecie: o.id, max_punktow: r.max_punktow}) AS osiagniecia
    """,

    "wzory": """
        MATCH (w:Wzor_Rekrutacyjny)
        RETURN w.id          AS id,
               w.zastosowanie AS zastosowanie,
               w.postac       AS postac,
               w.zmienne      AS zmienne
    """,

    "terminy": """
        MATCH (t:Termin_Rekrutacji)
        WHERE $kontekst IS NULL
           OR toLower(t.kontekst) CONTAINS toLower($kontekst)
           OR toLower(t.etap)     CONTAINS toLower($kontekst)
        RETURN t.tabela   AS tabela,
               t.kontekst AS kontekst,
               t.cykl     AS cykl,
               t.etap     AS etap,
               t.termin   AS termin,
               t.uwagi    AS uwagi
        ORDER BY t.cykl, t.etap
    """,

    "kierunki": """
        MATCH (k:Kierunek)
        RETURN collect(k.id) AS kierunki
    """,

    # Zapytania pomocnicze do pobierania list encji z grafu
    "_lista_przedmiotow": """
        MATCH (p:Przedmiot_Maturalny)
        RETURN collect(p.id) AS przedmioty
    """,

    "_lista_egzaminow": """
        MATCH (e:Egzamin_Zawodowy)
        RETURN collect(e.id) AS egzaminy
    """,

    "_lista_trybow": """
        MATCH (t:Tryb_Studiow)
        RETURN collect(t.id) AS tryby
    """,
}

WYMAGA_KIERUNKU = {"matury"}


# ── Cache list encji (pobierane raz z grafu) ──────────────────────────────────

_entity_cache: dict[str, list[str]] = {}


def _get_entity_list(key: str, cypher_key: str, result_field: str) -> list[str]:
    """Pobierz listę encji z grafu — z prostym cache w pamięci."""
    if key not in _entity_cache:
        try:
            results = graph.query(CYPHER_QUERIES[cypher_key], {})
            _entity_cache[key] = results[0][result_field] if results else []
        except Exception:
            _entity_cache[key] = []
    return _entity_cache[key]


def get_kierunki() -> list[str]:
    return _get_entity_list("kierunki", "kierunki", "kierunki")

def get_przedmioty() -> list[str]:
    return _get_entity_list("przedmioty", "_lista_przedmiotow", "przedmioty")

def get_egzaminy() -> list[str]:
    return _get_entity_list("egzaminy", "_lista_egzaminow", "egzaminy")

def get_tryby() -> list[str]:
    return _get_entity_list("tryby", "_lista_trybow", "tryby")


# ── Normalizacja encji przez LLM ──────────────────────────────────────────────

NORMALIZE_PROMPT = ChatPromptTemplate.from_template("""
Masz listę wartości z bazy danych: {lista}

Użytkownik napisał: "{wartosc}"

Twoim zadaniem jest dopasować tekst użytkownika do najbliższej wartości z listy.
Weź pod uwagę:
- Odmianę gramatyczną (np. "Informatykę" → "Informatyka", "stacjonarnych" → "Stacjonarne")
- Literówki i drobne różnice pisowni
- Skróty i synonimy (np. "stacjonarne" → "Stacjonarne", "niestacjonarne" → "Niestacjonarne")

Zwróć TYLKO dokładną wartość z listy która najlepiej pasuje, bez żadnego dodatkowego tekstu.
Jeśli żadna wartość z listy nie pasuje nawet w przybliżeniu, zwróć: null
""")


def _normalize_via_llm(wartosc: str, dostepne: list[str]) -> str | None:
    """Znormalizuj wartość przez LLM — używaj tylko gdy dopasowanie bezpośrednie nie zadziałało."""
    if not dostepne:
        return wartosc
    chain = NORMALIZE_PROMPT | model
    result = chain.invoke({
        "lista": ", ".join(dostepne),
        "wartosc": wartosc,
    })
    normalized = result.content.strip().strip('"\'')
    return normalized if normalized in dostepne else None


def _normalize_entity(wartosc: str | None, dostepne: list[str]) -> str | None:
    """
    Normalizacja encji:
    1. Sprawdź dokładne dopasowanie (szybka ścieżka — bez LLM)
    2. Sprawdź dopasowanie case-insensitive
    3. Dopiero gdy poprzednie zawiodą — użyj LLM
    """
    if not wartosc:
        return None

    # 1. Dokładne dopasowanie
    if wartosc in dostepne:
        return wartosc

    # 2. Case-insensitive
    wartosc_lower = wartosc.lower()
    for d in dostepne:
        if d.lower() == wartosc_lower:
            return d

    # 3. LLM fallback
    return _normalize_via_llm(wartosc, dostepne)


def normalize_kierunek(kierunek: str | None) -> str | None:
    return _normalize_entity(kierunek, get_kierunki())

def normalize_przedmiot(przedmiot: str | None) -> str | None:
    return _normalize_entity(przedmiot, get_przedmioty())

def normalize_egzamin(egzamin: str | None) -> str | None:
    return _normalize_entity(egzamin, get_egzaminy())

def normalize_tryb(tryb: str | None) -> str | None:
    return _normalize_entity(tryb, get_tryby())


# ── Query analyzer ────────────────────────────────────────────────────────────

ANALYZER_PROMPT = ChatPromptTemplate.from_template("""
Jesteś analizatorem zapytań rekrutacyjnych. Na podstawie pytania użytkownika:

1. Określ kategorię (JEDNA lub KILKA z listy):
   - matury       → wymagane przedmioty maturalne na kierunek
   - progi        → progi punktowe, liczba miejsc
   - egzaminy     → honorowane egzaminy zawodowe
   - osiagniecia  → osiągnięcia sportowe / aktywności dające punkty
   - wzory        → wzory rekrutacyjne / formuły obliczania punktów
   - terminy      → terminy i harmonogram rekrutacji
   - kierunki     → lista dostępnych kierunków
   - ogolne       → inne pytania

2. Wyciągnij nazwę kierunku jeśli jest podana.
   WAŻNE: Zawsze podawaj nazwę kierunku w MIANOWNIKU liczby pojedynczej
   (np. "Informatykę" → "Informatyka", "Matematyki" → "Matematyka",
   "Informatykę i Systemy Inteligentne" → "Informatyka i Systemy Inteligentne",
   "na kierunku Elektronika" → "Elektronika").
   Jeśli kierunek nie jest podany, zwróć null.

3. Wyciągnij tryb studiów jeśli podany, w MIANOWNIKU.
   (np. "stacjonarnych" → "Stacjonarne", "niestacjonarnego" → "Niestacjonarne")
   Jeśli tryb nie jest podany, zwróć null.

4. Wyciągnij nazwę przedmiotu maturalnego jeśli podana, w MIANOWNIKU.
   (np. "matematyki" → "Matematyka", "z fizyki" → "Fizyka")
   Jeśli nie podano, zwróć null.

5. Wyciągnij egzamin zawodowy jeśli podany, w MIANOWNIKU.
   Jeśli nie podano, zwróć null.

6. Wyciągnij kontekst terminu TYLKO gdy pytanie dotyczy konkretnej grupy
   (np. "cudzoziemcy"). W przeciwnym razie zawsze zwracaj null.

Odpowiedz TYLKO w formacie JSON, bez żadnego dodatkowego tekstu:
{{"kategorie": ["..."], "kierunek": null, "tryb": null, "przedmiot": null, "egzamin": null, "kontekst": null}}

Pytanie: {question}
""")


def analyze_query(question: str) -> dict:
    chain = ANALYZER_PROMPT | model
    response = chain.invoke({"question": question})
    text = response.content.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"kategorie": ["ogolne"], "kierunek": None, "tryb": None,
            "przedmiot": None, "egzamin": None, "kontekst": None}


# ── Normalizacja wyników analizy ──────────────────────────────────────────────

def normalize_analysis(analysis: dict) -> dict:
    """
    Znormalizuj wszystkie wyekstrahowane encje do form z bazy danych.
    LLM jest używany tylko gdy szybkie dopasowanie zawiedzie (opcje 1+3).
    """
    normalized = analysis.copy()
    normalized["kierunek"] = normalize_kierunek(analysis.get("kierunek"))
    normalized["tryb"]     = normalize_tryb(analysis.get("tryb"))
    normalized["przedmiot"] = normalize_przedmiot(analysis.get("przedmiot"))
    normalized["egzamin"]  = normalize_egzamin(analysis.get("egzamin"))
    return normalized


# ── Graph retrieval ───────────────────────────────────────────────────────────

def retrieve_from_graph(analysis: dict) -> str:
    # Normalizacja encji przed zapytaniem do grafu
    analysis = normalize_analysis(analysis)

    kategorie = analysis.get("kategorie", [])
    kierunek  = analysis.get("kierunek")
    kontekst  = analysis.get("kontekst")

    graph_parts = []

    for kat in kategorie:
        query = CYPHER_QUERIES.get(kat)
        if not query:
            continue

        if kat in WYMAGA_KIERUNKU and not kierunek:
            graph_parts.append(
                f"[GRAPH — {kat.upper()}] Podaj nazwę kierunku, aby uzyskać szczegóły."
            )
            continue

        params = {"kierunek": kierunek, "kontekst": kontekst}

        try:
            results = graph.query(query, params)
            if results:
                formatted = f"[GRAPH — {kat.upper()}]\n"
                for row in results:
                    formatted += "  " + " | ".join(
                        f"{k}: {v}" for k, v in row.items() if v is not None
                    ) + "\n"
                graph_parts.append(formatted.strip())
            else:
                graph_parts.append(f"[GRAPH — {kat.upper()}] Brak wyników w bazie.")
        except Exception as e:
            graph_parts.append(f"[GRAPH — {kat.upper()}] Błąd zapytania: {e}")

    return "\n\n".join(graph_parts)


# ── Vector retrieval ──────────────────────────────────────────────────────────

def retrieve_from_vector(question: str, n_results: int = 3) -> tuple[str, list[dict]]:
    docs = vector_store.similarity_search_with_score(question, k=n_results)

    context_parts = []
    preview_data  = []

    for doc, score in docs:
        source = doc.metadata.get("source", "N/A")
        tag    = f"[PDF] {source}"
        context_parts.append(f"{tag}:\n{doc.page_content}")
        preview_data.append({
            "tag":     tag,
            "content": doc.page_content[:500].replace("\n", " ") + "...",
            "score":   round(float(score), 4),
        })

    return "\n\n".join(context_parts), preview_data


# ── Fusion + LLM ─────────────────────────────────────────────────────────────

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Jesteś inteligentnym asystentem uczelnianym. Odpowiadaj rzeczowo i zwięźle, \
wyłącznie na podstawie dostarczonego kontekstu.
Jeśli kontekst nie zawiera wystarczających informacji, poinformuj o tym użytkownika.

Pytanie: {question}

=== KONTEKST Z GRAFU WIEDZY (dane strukturalne) ===
{graph_context}

=== KONTEKST Z DOKUMENTÓW PDF (dane tekstowe) ===
{vector_context}

ODPOWIEDŹ:
""")


def generate_answer(question: str, verbose: bool = True) -> str:
    # 1. Analiza
    analysis = analyze_query(question)
    if verbose:
        print(f"\n[Analiza — surowa] {analysis}")

    # 2. Graph retrieval (normalizacja odbywa się wewnątrz)
    graph_context = retrieve_from_graph(analysis)
    if verbose:
        preview = graph_context[:600] if graph_context else "Brak wyników"
        print(f"\n[Graph context]\n{preview}")

    # 3. Vector retrieval
    vector_context, chunks = retrieve_from_vector(question)
    if verbose:
        print(f"\n[Vector sources]")
        for i, ch in enumerate(chunks, 1):
            print(f"  {i}. {ch['tag']} (score: {ch['score']})")
            print(f"     {ch['content'][:200]}\n")

    # 4. LLM
    chain = ANSWER_PROMPT | model
    response = chain.invoke({
        "question":       question,
        "graph_context":  graph_context  or "Brak danych strukturalnych.",
        "vector_context": vector_context or "Brak pasujących dokumentów.",
    })
    return response.content


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        # "Jakie przedmioty maturalne są wymagane na Informatykę i Systemy Inteligentne?",
        "Ile wynosi próg punktowy na Informatykę Stosowaną stacjonarną?",
        # "Czy egzamin zawodowy z informatyki jest honorowany?",
    ]

    q = questions[0]
    print(f"Pytanie: {q}\n{'=' * 60}")
    answer = generate_answer(q, verbose=True)
    print("\n--- ODPOWIEDŹ ---")
    print(answer)