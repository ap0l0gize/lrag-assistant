import os
import time
import logging
from pypdf import PdfReader
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from config import Config, ALLOWED_NODES, ALLOWED_RELATIONSHIPS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
config = Config.from_env()

# Initialize LLM and Graph
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)
graph = Neo4jGraph(
    url=config.neo4j_url,
    username=config.neo4j_username,
    password=config.neo4j_password,
    refresh_schema=False
)
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=ALLOWED_NODES,
    allowed_relationships=ALLOWED_RELATIONSHIPS
)

def ingest_to_graph(chunks, source_filename):
    """Extract and add graph from chunks."""
    docs = [Document(page_content=c, metadata={"source": source_filename}) for c in chunks]
    
    for i in range(0, len(docs), config.batch_size):
        batch = docs[i:i + config.batch_size]
        logger.info(f"  -> Graph extraction: chunks {i} to {i+config.batch_size}")
        
        try:
            graph_documents = transformer.convert_to_graph_documents(batch)
            graph.add_graph_documents(graph_documents)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in batch {i}: {e}")
            continue

def extract_pdf_text(path):
    """Extract text from PDF."""
    try:
        reader = PdfReader(path)
        full_text = ""
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        return full_text.strip() if full_text.strip() else None
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return None

def ingest_pdfs(folder_path, collection):
    """Process all PDFs in folder."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for idx, filename in enumerate(pdf_files, 1):
        logger.info(f"[{idx}/{len(pdf_files)}] Processing: {filename}")
        
        path = os.path.join(folder_path, filename)
        full_text = extract_pdf_text(path)
        
        if not full_text:
            logger.warning(f"Skipping {filename} - no text extracted")
            continue
        
        chunks = text_splitter.split_text(full_text)
        logger.info(f"  -> Created {len(chunks)} chunks")
        
        # 1. Vector Store
        try:
            ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "type": "pdf"} for _ in chunks]
            collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error(f"Vector store error: {e}")
            continue
        
        # 2. Graph Store
        ingest_to_graph(chunks=chunks, source_filename=filename)
        
        logger.info(f"  -> Completed: {filename}")

def run_full_ingest():
    """Main ingestion pipeline."""
    logger.info("=== Starting Hybrid RAG Ingestion ===")
    
    # Validate
    os.makedirs(config.db_path, exist_ok=True)
    if not os.path.exists(config.pdf_folder):
        raise FileNotFoundError(f"PDF folder not found: {config.pdf_folder}")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=config.db_path)
    collection = client.get_or_create_collection(name="pdfs")
    
    # Run ingestion
    ingest_pdfs(config.pdf_folder, collection)
    
    logger.info(f"\n[SUCCESS] Hybrid database ready!")
    logger.info(f"  Vector store: {collection.count()} chunks")

if __name__ == "__main__":
    run_full_ingest()