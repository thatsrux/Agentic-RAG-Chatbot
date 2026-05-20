import os
import streamlit as st
from langchain_ollama import ChatOllama
from utils.config import OLLAMA_MODEL

from langchain_google_genai import ChatGoogleGenerativeAI

# Imposta la chiave API caricandola in modo sicuro dai secrets di Streamlit
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# @st.cache_resource(show_spinner=False)
# def load_llm():
#     """Carica il modello LLM (Ollama) nella cache di sistema."""
#     return ChatOllama(
#         model=OLLAMA_MODEL, 
#         temperature=0.1,
#         num_ctx=8192 
#     )

@st.cache_resource(show_spinner=False)
def load_llm():
    """Carica il modello LLM (Gemini 1.5 Flash) nella cache di sistema."""
    
   
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        temperature=0.1,
        max_output_tokens=1024 # Limita la lunghezza della risposta per mantenerlo conciso
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