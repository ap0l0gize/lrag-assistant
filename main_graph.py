import os
import chromadb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os.path
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
client = chromadb.PersistentClient(path="./db")
collection = client.get_collection(name="pdfs")

def get_context_and_preview(query_text):
    """Searches database, returns formatted context and list of found sources."""
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )
    
    context_parts = []
    preview_data = []
    
    # unpack results from chromadb
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0] if 'distances' in results else [0]*len(documents)

    for doc, meta, dist in zip(documents, metadatas, distances):
        source_type = meta.get('type', 'unknown')
        source_name = meta.get('source') if source_type == 'pdf' else f"Wątek: {meta.get('threadId', 'N/A')}"
        
        # create source tag
        source_tag = f"[{source_type.upper()}] {source_name}"
        context_parts.append(f"{source_tag}: {doc}")
        
        # save metadata to show it to user
        preview_data.append({
            "tag": source_tag,
            "content": doc[:1000].replace("\n", " ") + "...",
            "score": round(dist, 4) # vector distance (the less, the better)
        })
            
    return "\n\n".join(context_parts), preview_data

def generate_answer(new_question):
    # get context and data
    context, chunks = get_context_and_preview(new_question)
    
    # show metadata to user
    print("\nFOUND SOURCES IN DATABASE:")
    print("-" * 60)
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk['tag']} (Match score: {chunk['score']})")
        print(f"   Fragment: {chunk['content']}\n")
    print("-" * 60)

    prompt_template = ChatPromptTemplate.from_template("""
    Jesteś inteligentnym asystentem uczelnianym. Twoim celem jest przygotowanie odpowiedzi, bez zbędnego tekstu. Odpowiadaj tylko na bazie kontekstu, jeśli w kontekście z bazy wiedzy nie znajdziesz wystarczających informacji aby odpowiedzieć na pytanie, poinformuj o tym użytkownika.
    
    Treść zapytania: {question}

    Poniżej masz kontekst z bazy wiedzy:
    {context}

    ODPOWIEDŹ:
    """)

    chain = prompt_template | model
    response = chain.invoke({
        "question": new_question,
        "context": context
    })
    
    return response.content

if __name__ == "__main__":
    # test_question = "Przepis na sernik?"
    test_question = "Kiedy rozpoczyna się i kończy rekrutacja na pierwszy rok studiów?"
    print("Agent analyzes your question and prepares a response...")
    
    szkic = generate_answer(test_question)
    
    print("\n--- GENERATED RESPONSE ---")
    print(szkic)