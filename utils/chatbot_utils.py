import os
from langchain_ollama import ChatOllama
from crawling import CORSI_DIEM_URLS
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

SAMPLE_QUESTIONS = [
            "Quali sono i corsi offerti dal DIEM?",
            "Ho ottenuto un punteggio di 18 al TOLC. Posso iscrivermi?",
            "Quali sono i requisiti di ammissione per il Master in Ingegneria dell'Informazione per la Medicina Digitale?",
            "Qual è il programma del corso di Ingegneria del Software?",
            "Quali sono gli orari di ricevimento del Professor Capuano?",
            "Dove si trova l'aula 126?",
            "La mia media è 28,8. Qual è il voto finale massimo che posso ottenere?",
            "Chi è responsabile dell'internazionalizzazione presso DIEM?",
            "Quali opportunità di mobilità internazionale sono disponibili?",
            "Dove si trova il DIEM?",
            "Quali aree di ricerca sono attive al DIEM?",
            "Quali laboratori sono disponibili al DIEM?",
            "Quali attrezzature sono disponibili nel Laboratorio di Robotica?",
            "Chi sono i membri della Commissione Paritaria Studenti-Docenti?"
        ]

@st.cache_resource(show_spinner=False)
def load_llm(selected_model: str = "gemini-3.1-flash-lite"):
    """
    Carica il modello LLM in base alla scelta, costruendo la catena di fallback corretta.
    """
    gemini_31 = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.1,
        max_output_tokens=1024,
        timeout=30.0,
        max_retries=0
    )
    
    gemini_35 = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.1,
        max_output_tokens=1024,
        timeout=30.0,
        max_retries=0
    )
    
    gemini_25 = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1,
        max_output_tokens=1024,
        timeout=30.0,
        max_retries=0
    )
    
    mistral = ChatOllama(
        model="mistral-nemo", 
        temperature=0.1,
        num_ctx=8192 
    )

    llama = ChatOllama(
        model = "llama3.1",
        temperature=0.1,
        num_ctx=8192 
    )
        
    if selected_model == "gemini-3.1-flash-lite":
        return gemini_31.with_fallbacks([gemini_35, gemini_25, mistral])
    elif selected_model == "gemini-3.5-flash":
        return gemini_35.with_fallbacks([gemini_31, gemini_25, mistral])
    elif selected_model == "gemini-2.5-flash":
        return gemini_25.with_fallbacks([gemini_31, gemini_35, mistral])
    elif selected_model == "mistral-nemo":
        return mistral.with_fallbacks([llama])
    else: 
        return llama

def format_context(docs):
    """
    Formatta i documenti per il prompt dell'LLM.
    """
    if not docs:
        return "Nessun documento rilevante trovato."
    
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Fonte sconosciuta")
        parts.append(f"[Documento {i+1} | Fonte: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def is_valid_url(url: str) -> bool:
        """
        Determina se un URL è valido e rilevante per il dominio del DIEM, escludendo documenti non testuali o fonti non affidabili.
        """
        clean_url = url.replace("https://", "").replace("http://", "").replace("www.", "")
        
        if clean_url.lower().endswith((".xls", ".xlsx", ".pdf", ".doc", ".docx")):
            return False
        if "diem.unisa.it" in clean_url or "docenti.unisa.it" in clean_url:
            return True
        if "corsi.unisa.it" in clean_url:
            if "calendario-occupazione-spazi" in clean_url:
                return False
            if "strutture-didattiche" in clean_url:
                return True
            return any(corso in clean_url for corso in CORSI_DIEM_URLS)
        if "easycourse.unisa.it" in clean_url:
            if "Dipartimento_di_" in clean_url and "dellInformazione" not in clean_url:
                return False
            return True
            
        return False