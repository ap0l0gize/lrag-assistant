import os
import ast
import csv
import logging
import asyncio
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = Config.from_env()
print(f"DEBUG: URL={config.neo4j_url}, User={config.neo4j_username}")

graph = Neo4jGraph(
    url=config.neo4j_url,
    username=config.neo4j_username,
    password=config.neo4j_password,
    refresh_schema=False,
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# ── Kanoniczne formy przedmiotów ─────────────────────────────────────────────
PRZEDMIOT_CANON = {
    "matematyka":              "Matematyka",
    "fizyka":                  "Fizyka",
    "informatyka":             "Informatyka",
    "chemia":                  "Chemia",
    "biologia":                "Biologia",
    "geografia":               "Geografia",
    "historia":                "Historia",
    "historia sztuki":         "Historia Sztuki",
    "język polski":            "Język Polski",
    "język angielski":         "Język Angielski",
    "język niemiecki":         "Język Niemiecki",
    "język francuski":         "Język Francuski",
    "język hiszpański":        "Język Hiszpański",
    "język rosyjski":          "Język Rosyjski",
    "język włoski":            "Język Włoski",
    "język obcy":              "Język Obcy",
    "filozofia":               "Filozofia",
    "wiedza o społeczeństwie": "Wiedza O Społeczeństwie",
}


def parse_list_field(value: str) -> list[str]:
    """Parsuj kolumnę zapisaną jako Python list string: "['a', 'b']" → ['a', 'b']."""
    try:
        result = ast.literal_eval(value.strip())
        return [str(x).strip() for x in result] if isinstance(result, list) else []
    except Exception:
        return []


def read_csv(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH INGESTION — bezpośrednio z CSV, bez LLM
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_matury(folder: str):
    """matury_kierunki.csv → węzły Kierunek + Przedmiot_Maturalny + relacje WYMAGA_PRZEDMIOTU."""
    path = os.path.join(folder, "matury_kierunki.csv")
    if not os.path.exists(path):
        logger.warning(f"Brak pliku: {path}")
        return

    rows = read_csv(path)
    logger.info(f"matury_kierunki.csv: {len(rows)} wierszy")

    for row in rows:
        kierunek = row["Nazwa kierunku"].strip()
        if not kierunek:
            continue

        # Utwórz węzeł Kierunek
        graph.query(
            "MERGE (k:Kierunek {id: $id})",
            {"id": kierunek}
        )

        # P1 i P2 — każdy przedmiot z obu kolumn
        for col in ("P1", "P2"):
            for przedmiot_raw in parse_list_field(row.get(col, "[]")):
                canon = PRZEDMIOT_CANON.get(przedmiot_raw.lower().strip())
                if not canon:
                    logger.info(f"  ✗ Nieznany przedmiot: {przedmiot_raw!r}")
                    continue

                graph.query(
                    "MERGE (p:Przedmiot_Maturalny {id: $id})",
                    {"id": canon}
                )
                graph.query(
                    """
                    MATCH (k:Kierunek {id: $kierunek})
                    MATCH (p:Przedmiot_Maturalny {id: $przedmiot})
                    MERGE (k)-[:WYMAGA_PRZEDMIOTU {kolumna: $kolumna}]->(p)
                    """,
                    {"kierunek": kierunek, "przedmiot": canon, "kolumna": col}
                )

    logger.info("  -> matury_kierunki: gotowe")


def ingest_tryby(folder: str):
    """progi_2025.csv → relacje OFEROWANY_JAKO (Stacjonarne/Niestacjonarne) + progi punktowe."""
    path = os.path.join(folder, "progi_2025.csv")
    if not os.path.exists(path):
        logger.warning(f"Brak pliku: {path}")
        return

    rows = read_csv(path)
    logger.info(f"progi_2025.csv: {len(rows)} wierszy")

    tryb_canon = {
        "stacjonarne":    "Stacjonarne",
        "niestacjonarne": "Niestacjonarne",
    }

    for row in rows:
        kierunek = row["Nazwa kierunku"].strip()
        tryb_raw = row["Tryb studiów"].strip().lower()
        tryb = tryb_canon.get(tryb_raw)
        if not kierunek or not tryb:
            continue

        graph.query("MERGE (k:Kierunek {id: $id})", {"id": kierunek})
        graph.query("MERGE (t:Tryb_Studiow {id: $id})", {"id": tryb})

        # Zbierz progi z wszystkich cykli (pomijaj puste)
        progi = []
        for col in ["Próg punktowy (Cykl 1)", "Próg punktowy (Cykl 2)",
                    "Próg punktowy (Cykl 3)", "Próg punktowy (Cykl 4)",
                    "Próg punktowy (Cykl 5)"]:
            val = row.get(col, "").strip()
            if val:
                try:
                    progi.append(int(val))
                except ValueError:
                    pass

        prog_min = min(progi) if progi else None
        prog_max = max(progi) if progi else None

        graph.query(
            """
            MATCH (k:Kierunek {id: $kierunek})
            MATCH (t:Tryb_Studiow {id: $tryb})
            MERGE (k)-[r:OFEROWANY_JAKO]->(t)
            SET r.prog_min = $prog_min,
                r.prog_max = $prog_max,
                r.miejsca  = $miejsca
            """,
            {
                "kierunek": kierunek,
                "tryb":     tryb,
                "prog_min": prog_min,
                "prog_max": prog_max,
                "miejsca":  row.get("Liczba podań / liczba miejsc", "").strip(),
            }
        )

    logger.info("  -> progi_2025: gotowe")


def ingest_egzaminy_zawodowe(folder: str):
    """egzaminy_zawodowe_kierunki.csv → węzły Egzamin_Zawodowy + relacje HONORUJE_EGZAMIN."""
    path = os.path.join(folder, "egzaminy_zawodowe_kierunki.csv")
    if not os.path.exists(path):
        logger.warning(f"Brak pliku: {path}")
        return

    rows = read_csv(path)
    logger.info(f"egzaminy_zawodowe_kierunki.csv: {len(rows)} wierszy")

    for row in rows:
        kierunek = row["Nazwa kierunku"].strip()
        if not kierunek:
            continue

        graph.query("MERGE (k:Kierunek {id: $id})", {"id": kierunek})

        kolumny = parse_list_field(row.get("Uwzględniany (tylko w P2, czy w P1 lub P2?)", "[]"))
        kolumna_str = ", ".join(kolumny) if kolumny else "P2"

        for egzamin_raw in parse_list_field(row.get("Egzaminy zawodowe", "[]")):
            egzamin = egzamin_raw.strip()
            if not egzamin:
                continue

            graph.query(
                "MERGE (e:Egzamin_Zawodowy {id: $id})",
                {"id": egzamin}
            )
            graph.query(
                """
                MATCH (k:Kierunek {id: $kierunek})
                MATCH (e:Egzamin_Zawodowy {id: $egzamin})
                MERGE (k)-[r:HONORUJE_EGZAMIN]->(e)
                SET r.kolumna = $kolumna
                """,
                {"kierunek": kierunek, "egzamin": egzamin, "kolumna": kolumna_str}
            )

    logger.info("  -> egzaminy_zawodowe: gotowe")


def ingest_osiagniecia(folder: str):
    """osiagniecia_aktywnosci_kierunki*.csv → węzły Osiagniecie + relacje DAJE_PUNKTY_ZA."""
    for filename, max_pkt in [
        ("osiagniecia_aktywnosci_kierunki.csv",    150),
        ("osiagniecia_aktywnosci_kierunki_v2.csv", 100),
    ]:
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            logger.warning(f"Brak pliku: {path}")
            continue

        rows = read_csv(path)
        logger.info(f"{filename}: {len(rows)} wierszy")

        col = next(iter(rows[0].keys()) if rows else iter([]))
        osiagniecia_col = [k for k in rows[0].keys() if k != "Nazwa kierunku"][0] if rows else None
        if not osiagniecia_col:
            continue

        for row in rows:
            kierunek = row["Nazwa kierunku"].strip()
            if not kierunek:
                continue

            graph.query("MERGE (k:Kierunek {id: $id})", {"id": kierunek})

            for osiagniecie in parse_list_field(row.get(osiagniecia_col, "[]")):
                osiagniecie = osiagniecie.strip()
                if not osiagniecie:
                    continue

                graph.query(
                    "MERGE (o:Osiagniecie {id: $id})",
                    {"id": osiagniecie}
                )
                graph.query(
                    """
                    MATCH (k:Kierunek {id: $kierunek})
                    MATCH (o:Osiagniecie {id: $osiagniecie})
                    MERGE (k)-[r:DAJE_PUNKTY_ZA]->(o)
                    SET r.max_punktow = $max_pkt
                    """,
                    {"kierunek": kierunek, "osiagniecie": osiagniecie, "max_pkt": max_pkt}
                )

    logger.info("  -> osiagniecia: gotowe")


def ingest_wzory(folder: str):
    """wzory_rekrutacyjne.csv → węzły Wzor_Rekrutacyjny jako osobny typ informacyjny."""
    path = os.path.join(folder, "wzory_rekrutacyjne.csv")
    if not os.path.exists(path):
        logger.warning(f"Brak pliku: {path}")
        return

    rows = read_csv(path)
    logger.info(f"wzory_rekrutacyjne.csv: {len(rows)} wierszy")

    for row in rows:
        graph.query(
            """
            MERGE (w:Wzor_Rekrutacyjny {id: $id})
            SET w.zastosowanie   = $zastosowanie,
                w.postac         = $postac,
                w.zmienne        = $zmienne
            """,
            {
                "id":           row["Identyfikator wzoru"].strip(),
                "zastosowanie": row.get("Zastosowanie", "").strip(),
                "postac":       row.get("Postać matematyczna", "").strip(),
                "zmienne":      row.get("Zmienne i objaśnienia", "").strip(),
            }
        )

    logger.info("  -> wzory_rekrutacyjne: gotowe")


def ingest_terminy(folder: str):
    """terminy_rekrutacji.csv → węzły Termin_Rekrutacji."""
    path = os.path.join(folder, "terminy_rekrutacji.csv")
    if not os.path.exists(path):
        logger.warning(f"Brak pliku: {path}")
        return

    rows = read_csv(path)
    logger.info(f"terminy_rekrutacji.csv: {len(rows)} wierszy")

    for i, row in enumerate(rows):
        graph.query(
            """
            MERGE (t:Termin_Rekrutacji {id: $id})
            SET t.tabela         = $tabela,
                t.kontekst       = $kontekst,
                t.cykl           = $cykl,
                t.etap           = $etap,
                t.termin         = $termin,
                t.uwagi          = $uwagi
            """,
            {
                "id":      f"termin_{i}",
                "tabela":  row.get("Tabela", "").strip(),
                "kontekst":row.get("Kontekst / Grupa docelowa", "").strip(),
                "cykl":    row.get("Cykl rekrutacyjny", "").strip(),
                "etap":    row.get("Etap rekrutacji", "").strip(),
                "termin":  row.get("Termin", "").strip(),
                "uwagi":   row.get("Uwagi / Przypisy", "").strip(),
            }
        )

    logger.info("  -> terminy_rekrutacji: gotowe")


def ingest_csvs_to_graph(folder: str):
    """Wejście do ekstrakcji grafu z CSV — bez LLM."""
    logger.info(f"== Graph ingestion z CSV: {folder} ==")
    ingest_matury(folder)
    ingest_tryby(folder)
    ingest_egzaminy_zawodowe(folder)
    ingest_osiagniecia(folder)
    ingest_wzory(folder)
    ingest_terminy(folder)
    logger.info("== Graph ingestion zakończony ==")


# ═══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE INGESTION — PDFy bez zmian
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(path: str) -> str | None:
    try:
        reader = PdfReader(path)
        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n".join(pages).strip() or None
    except Exception as e:
        logger.error(f"Błąd odczytu {path}: {e}")
        return None


def ingest_pdfs(folder_path: str, collection):
    """Wgraj PDFy do vector store (tylko RAG, bez grafu)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    logger.info(f"Znaleziono {len(pdf_files)} plików PDF w {folder_path}")

    for idx, filename in enumerate(pdf_files, 1):
        logger.info(f"[{idx}/{len(pdf_files)}] {filename}")
        path = os.path.join(folder_path, filename)
        text = extract_pdf_text(path)

        if not text:
            logger.warning(f"  Brak tekstu — pomijam {filename}")
            continue

        chunks = splitter.split_text(text)
        logger.info(f"  -> {len(chunks)} chunków")

        try:
            collection.add_texts(
                texts=chunks,
                metadatas=[{"source": filename, "type": "pdf"} for _ in chunks],
                ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))],
            )
        except Exception as e:
            logger.error(f"  Vector store error: {e}")
            continue

        logger.info(f"  -> Gotowe: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_ingest():
    logger.info("=== Hybrid RAG Ingestion START ===")
    os.makedirs(config.db_path, exist_ok=True)

    vector_store = Chroma(
        collection_name="pdfs",
        embedding_function=embeddings,
        persist_directory=config.db_path,
    )

    # 1. Graf — z CSV (bez LLM)
    graph_csv_folder = "data/GraphRAG data/"
    if os.path.exists(graph_csv_folder):
        ingest_csvs_to_graph(graph_csv_folder)
    else:
        logger.warning(f"Brak folderu: {graph_csv_folder}")

    # 2. Vector store — PDFy do RAG
    rag_folder = "data/RAG DATA/"
    if os.path.exists(rag_folder):
        ingest_pdfs(rag_folder, vector_store)
    else:
        logger.warning(f"Brak folderu: {rag_folder}")

    logger.info("=== [SUCCESS] Baza gotowa ===")


if __name__ == "__main__":
    run_full_ingest()