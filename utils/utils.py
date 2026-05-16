import streamlit as st
from langchain_ollama import ChatOllama
from utils.config import OLLAMA_MODEL

@st.cache_resource(show_spinner=False)
def load_llm():
    """Carica il modello LLM (Ollama) nella cache di sistema."""
    return ChatOllama(
        model=OLLAMA_MODEL, 
        temperature=0.1,
        num_ctx=8192 
    )

def format_context(docs):
    """Formatta i documenti per il prompt dell'LLM."""
    if not docs:
        return "Nessun documento rilevante trovato."
    
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Fonte sconosciuta")
        parts.append(f"[Documento {i+1} | Fonte: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)