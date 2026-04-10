import os
import base64
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# path to db with pdf's
DB_PATH = "./db"
PDF_FOLDER = "./documents_pdf"

def ingest_pdfs(folder_path, collection):
    """Loads PDFs from folder and adds them to database."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Przetwarzanie PDF: {filename}...")
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)
            full_text = ""

            for page in reader.pages:
                text = page.extract_text()
                # check if text was found in pdf
                if text:
                    full_text += text + "\n"
            
            if not full_text.strip():
                print(f"Skipping {filename}: No extractable text found.")
                continue
            
            chunks = text_splitter.split_text(full_text)
            ids = [f"pdf_{filename}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "type": "pdf"} for _ in range(len(chunks))]
            
            collection.add(documents=chunks, metadatas=metadatas, ids=ids)
            print(f"Dodano {len(chunks)} fragmentów z pliku {filename}.")

def run_full_ingest():
    """Main function building database."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="pdfs")

    # PDFs
    ingest_pdfs(PDF_FOLDER, collection)
    
    print("\n Knowledge base is ready.")

if __name__ == "__main__":
    run_full_ingest()