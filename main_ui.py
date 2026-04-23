import streamlit as st
import requests

# --- Page Config ---
st.set_page_config(page_title="OnRAG (音RAG)", page_icon="🧠", layout="centered")

st.title("OnRAG (音RAG)")
st.write("Ask a question about your notes using your voice.")

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("History")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Native Audio Input ---
# This creates the new 2024/2025 native recording widget
audio_file = st.audio_input("Record your question")

if audio_file:
    with st.spinner("OnRAG is thinking..."):
        # Send to FastAPI
        files = {"file": ("audio.wav", audio_file, "audio/wav")}
        response = requests.post("http://localhost:8000/query-audio", files=files)
        
        if response.status_code == 200:
            data = response.json()
            user_text = data["user_text"]
            answer = data["answer"]
            sources = data['sources']

            # Store full result in state
            st.session_state.messages.append({
                "role": "user", 
                "content": user_text
            })
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources
            })

# --- Display Chat History ---
for msg in reversed(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # If assistant message has sources, show them in a clean caption
        if "sources" in msg and msg["sources"]:
            st.caption(f"📚 Sources: {', '.join(msg['sources'])}")