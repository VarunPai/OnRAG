import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
RAW_DATA_PATH = "data/raw"
CHROMA_PATH = "data/chroma_db"


def initialize_database():
    """Load documents, split them, and save to local ChromaDB."""
    
    # 1. Load Documents (Looking for .md files)
    print("--- Loading documents... ---")
    loader = DirectoryLoader(RAW_DATA_PATH, glob="*.md", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No documents found in data/raw. Please add some .md files!")
        return

    # 2. Split Text
    # We use 500 character chunks with a small overlap so context isn't lost at the edges
    print(f"--- Splitting {len(documents)} documents... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"--- Created {len(chunks)} chunks. ---")

    # 3. Create Embeddings & Store in Chroma
    print("--- Generating embeddings and saving to ChromaDB... ---")
    
    # Note: This uses OpenAI embeddings (cheap and high quality). 
    # If you want 100% local, we can swap this for HuggingFace embeddings later.
    embeddings = OpenAIEmbeddings()
    
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"--- Database initialized and saved to {CHROMA_PATH} ---")
    return db


def query_database(query_text):
    """Simple search to verify the DB works."""
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Search for top 2 matches
    results = db.similarity_search_with_relevance_scores(query_text, k=2)
    return results


if __name__ == "__main__":
    # Day 1 Setup: Ensure directory exists
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        with open(f"{RAW_DATA_PATH}/test.md", "w") as f:
            f.write("# Project Aura\nProject Aura is a second brain built in April 2026.")

    initialize_database()
    
    # Test Query
    print("\n--- Testing Search ---")
    test_query = "What is Project Aura?"
    hits = query_database(test_query)
    for doc, score in hits:
        print(f"[Score: {score:.2f}] Content: {doc.page_content}")