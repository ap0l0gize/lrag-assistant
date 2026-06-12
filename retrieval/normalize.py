import json
import re

from langchain_core.prompts import ChatPromptTemplate

from retrieval.clients import graph, model
from retrieval.cypher import CYPHER_QUERIES

_entity_cache: dict[str, list[str]] = {}

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


def _get_entity_list(key: str, cypher_key: str, result_field: str) -> list[str]:
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


def _normalize_via_llm(wartosc: str, dostepne: list[str]) -> str | None:
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
    if not wartosc:
        return None
    if wartosc in dostepne:
        return wartosc
    wartosc_lower = wartosc.lower()
    for d in dostepne:
        if d.lower() == wartosc_lower:
            return d
    return _normalize_via_llm(wartosc, dostepne)


def normalize_kierunek(kierunek: str | None) -> str | None:
    return _normalize_entity(kierunek, get_kierunki())


def normalize_przedmiot(przedmiot: str | None) -> str | None:
    return _normalize_entity(przedmiot, get_przedmioty())


def normalize_egzamin(egzamin: str | None) -> str | None:
    return _normalize_entity(egzamin, get_egzaminy())


def normalize_tryb(tryb: str | None) -> str | None:
    return _normalize_entity(tryb, get_tryby())


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
    return {
        "kategorie": ["ogolne"],
        "kierunek": None,
        "tryb": None,
        "przedmiot": None,
        "egzamin": None,
        "kontekst": None,
    }


def normalize_analysis(analysis: dict) -> dict:
    normalized = analysis.copy()
    normalized["kierunek"] = normalize_kierunek(analysis.get("kierunek"))
    normalized["tryb"] = normalize_tryb(analysis.get("tryb"))
    normalized["przedmiot"] = normalize_przedmiot(analysis.get("przedmiot"))
    normalized["egzamin"] = normalize_egzamin(analysis.get("egzamin"))
    return normalized
