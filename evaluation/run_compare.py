"""
Compare vector / hybrid / agentic on simple questions from evaluation/questions.json.
"""

import argparse
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from compare_runner import run_compare, DEFAULT_QUESTIONS


def main():
    parser = argparse.ArgumentParser(
        description="Porównaj vector RAG vs hybrid GraphRAG vs agentic (side-by-side)"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help=f"Plik JSON z pytaniami (domyślnie {DEFAULT_QUESTIONS.name})",
    )
    parser.add_argument("--id", dest="case_id", help="Uruchom tylko jedno pytanie po id")
    parser.add_argument("--dry-run", action="store_true", help="Lista pytań bez wywołań API")
    parser.add_argument("--skip-preflight", action="store_true", help="Pomiń diagnostykę środowiska")
    args = parser.parse_args()

    run_compare(
        questions_path=args.questions,
        case_id=args.case_id,
        dry_run=args.dry_run,
        skip_preflight=args.skip_preflight,
    )


if __name__ == "__main__":
    main()
