import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from typing import List

# Import your Phase 2 & 3 modules
try:
    from app.audio import transcribe_audio_locally
except:
    from audio import transcribe_audio_locally

try:
    from app.database import CHROMA_PATH
except:
    from database import CHROMA_PATH

load_dotenv()

def get_rag_response_with_sources(user_query: str, vector_db: Chroma | None = None):
    """The RAG Chain: Connects context to the LLM and return sources."""

    db = check_and_create_chroma(vector_db=vector_db)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # 2. Add source tracking
    docs = retriever.invoke(user_query)
    sources = list(set([os.path.basename(doc.metadata.get("source", "Unknown")) for doc in docs]))

    # 2. Define the Prompt Template
    template = """
    You are a RAG application. Use the following pieces of retrieved context from the user's personal notes to answer the question. 
    If you don't know the answer based on the context, just say that you don't have a note about that yet.
    Keep the answer concise and conversational.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 3. Build the Chain
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {
        "answer" : chain.invoke(user_query),
        "sources" : sources,
    } 

def run_onrag_voice_command(audio_file_path: str, whisper_model = None, vector_db: Chroma | None = None):
    """Full Pipeline: Audio -> Text -> Context -> Answer + Sources"""

    print("\n--- Transcribing Voice... ---")
    if whisper_model:
        result = whisper_model.transcribe(audio_file_path, fp16=False)
        query_text = result["text"]
    else:
        query_text = transcribe_audio_locally(audio_file_path)
    print(f"You said: '{query_text}'")

    # RAG Logic
    print("--- Thinking... ---")
    response = get_rag_response_with_sources(query_text, vector_db)
    return query_text, response


def similarity_search_in_db(query: str, vector_db: Chroma | None = None, k : int = 4) -> List[Document]:
    """This function does a similarity search in chroma db and return the results"""
    db = check_and_create_chroma(vector_db)
    return db.similarity_search(query, k)


def check_and_create_chroma(vector_db: Chroma | None = None) -> Chroma:
    """Check if vector db is availbale, if not create a new one"""
    if not vector_db:
        # 1. Load the existing database
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    return vector_db



if __name__ == "__main__":
    # For testing, we'll assume you already recorded 'temp_recording.wav'
    # using your script from Phase 2.
    test_audio = "temp_recording.wav"
    
    if os.path.exists(test_audio):
        final_answer = run_onrag_voice_command(test_audio)
        print("\n" + "="*50)
        print(f"OnRAG: {final_answer}")
        print("="*50)
    else:
        print("Please record an audio file first using audio.py!")