from fastapi import FastAPI, UploadFile, File
import uvicorn
import whisper
from contextlib import asynccontextmanager
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from app.database import CHROMA_PATH
from app.engine import run_onrag_voice_command, similarity_search_in_db
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    OnRAG application start and shutdown logic
    """
    print("--- Starting OnRAG Engine ---")
    MODELS["whisper"] = whisper.load_model("base")

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    MODELS["whisper"] = whisper.load_model("base")
    
    MODELS["db"] = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    print("--- OnRAG Engine is Ready ---")

    yield

    # --- Shutdown Logic ---
    # Clean up
    MODELS.clear()
    print("--- OnRAG Engine Shutdown ---")

app = FastAPI(lifespan=lifespan)


@app.post("/query-audio")
async def voice_query(file: UploadFile = File(...)):
    """This is an api function for query using voice"""

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    user_text, result = run_onrag_voice_command(
        audio_file_path=temp_path,
        whisper_model=MODELS["whisper"],
        vector_db=MODELS["db"],
    )

    os.remove(temp_path)
    return {
        "user_text": user_text,
        "answer": result["answer"],
        "sources": result["sources"]
    }


@app.post("/similarity_search")
async def similarity_search(query: str, k: int) -> None:
    """Fast API function for similarity search"""
    return None



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
