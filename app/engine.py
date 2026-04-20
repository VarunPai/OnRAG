import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import your Phase 2 & 3 modules
from audio import transcribe_audio_locally
from database import CHROMA_PATH

load_dotenv()

def get_brain_response(user_query):
    """The RAG Chain: Connects context to the LLM."""
    
    # 1. Load the existing database
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

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

    return chain.invoke(user_query)

def run_onrag_voice_command(audio_file_path):
    """Full Pipeline: Audio -> Text -> Context -> Answer"""
    
    # Step A: Speech to Text (Local Whisper)
    print("\n--- Phase 2: Transcribing Voice... ---")
    query_text = transcribe_audio_locally(audio_file_path)
    print(f"You said: '{query_text}'")

    # Step B: RAG Logic
    print("--- Phase 4: Thinking... ---")
    response = get_brain_response(query_text)
    
    return response

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