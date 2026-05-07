import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    groq_api_key: str
    neo4j_url: str
    neo4j_username: str
    neo4j_password: str
    db_path: str = "./db"
    pdf_folder: str = "./documents_pdf"
    chunk_size: int = 1500
    chunk_overlap: int = 100
    batch_size: int = 5

    @classmethod
    def from_env(cls):
        required = {
            'GROQ_API_KEY':   os.getenv("GROQ_API_KEY"),
            'NEO4J_USERNAME': os.getenv("NEO4J_USERNAME"),
            'NEO4J_PASSWORD': os.getenv("NEO4J_PASSWORD"),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing env vars: {missing}")
        return cls(
            groq_api_key=required['GROQ_API_KEY'],
            neo4j_url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
            neo4j_username=required['NEO4J_USERNAME'],
            neo4j_password=required['NEO4J_PASSWORD'],
        )


ALLOWED_NODES = [
    "Kierunek",
    "Przedmiot_Maturalny",
    "Tryb_Studiow",
]

ALLOWED_RELATIONSHIPS = [
    "WYMAGA_PRZEDMIOTU",
    "OFEROWANY_JAKO",
]

NODE_PROPERTIES: list[str] = []
RELATIONSHIP_PROPERTIES: list[str] = []