import os
import streamlit as st

# --- PROTEZIONE CRASH (WINDOWS/STREAMLIT) ---
# Queste impostazioni devono essere in cima per evitare conflitti tra i thread
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import HybridRetriever

# --- CONFIGURAZIONE ---
OLLAMA_MODEL = "llama3.2" 

SYSTEM_PROMPT = """
Sei DIEMbot, l'assistente virtuale del DIEM (Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica applicata) dell'Università di Salerno.

REGOLE FONDAMENTALI:
1. Rispondi in italiano in modo professionale e cordiale.
2. Basati ESCLUSIVAMENTE sui documenti forniti nel "Contesto".
3. Se l'informazione non è presente nei documenti, rispondi onestamente che non disponi di quei dati. Non inventare nulla.
4. Indica sempre la fonte delle informazioni (es. "In base al sito dei docenti...") se disponibile nel contesto.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Contesto estratto dai documenti:\n{context}\n\nDomanda dell'utente: {question}")
])

# --- INIZIALIZZAZIONE RISORSE ---
@st.cache_resource(show_spinner=False)
def load_llm():
    """Carica il modello LLM (Ollama) nella cache di sistema."""
    return ChatOllama(
        model=OLLAMA_MODEL, 
        temperature=0.1,
        num_ctx=2048 # Dà a Ollama più respiro per la memoria del contesto
    )

# --- FUNZIONI DI SUPPORTO ---
def format_context(docs):
    """Formatta i documenti per il prompt dell'LLM."""
    if not docs:
        return "Nessun documento rilevante trovato."
    
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Fonte sconosciuta")
        parts.append(f"[Documento {i+1} | Fonte: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

# --- INTERFACCIA STREAMLIT ---
def main():
    st.set_page_config(
        page_title="DIEMbot — Università di Salerno",
        page_icon="🎓",
        layout="centered",
    )

    st.title("🎓 DIEMbot")
    st.caption("Assistente virtuale ufficiale del DIEM – Università di Salerno")

    # Inizializzazione Retriever (nella sessione per evitare ricaricamenti)
    if "retriever" not in st.session_state:
        with st.spinner("Caricamento del database della conoscenza (FAISS)..."):
            # Istanziamo la classe dal file retriever.py
            st.session_state.retriever = HybridRetriever()
            st.success("Database caricato con successo!")

    llm = load_llm()

    # Gestione cronologia chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("Impostazioni")
        show_sources = st.toggle("Mostra fonti estratte", value=True)
        if st.button("🗑️ Cancella cronologia chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.divider()
        st.markdown("**Esempi di domande:**")
        if st.button("Quali sono i corsi di laurea del DIEM?"):
            st.session_state.pending_question = "Quali sono i corsi di laurea del DIEM?"
        if st.button("Dove trovo gli orari delle lezioni?"):
            st.session_state.pending_question = "Dove trovo gli orari delle lezioni?"

    # Visualizzazione messaggi precedenti
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander("📄 Fonti consultate"):
                    for src in msg["sources"]:
                        st.caption(f"• {src}")

    # Input utente
    user_input = st.chat_input("Chiedimi qualcosa sul DIEM...")
    
    # Gestione domande dai pulsanti della sidebar
    if hasattr(st.session_state, "pending_question") and st.session_state.pending_question:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    if user_input:
        # Aggiungi domanda utente
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Risposta assistente
        with st.chat_message("assistant"):
            # 1. Retrieval
            with st.spinner("Consultazione documenti..."):
                docs = st.session_state.retriever.retrieve(user_input)
                context = format_context(docs)
                sources_list = list(set([d.metadata.get("source", "N/A") for d in docs]))
            
            # 2. Generazione con Streaming
            with st.spinner("Generazione risposta..."):
                chain = RAG_PROMPT | llm | StrOutputParser()
                
                response_placeholder = st.empty()
                full_response = ""
                
                # Streaming della risposta parola per parola
                for chunk in chain.stream({"context": context, "question": user_input}):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                # Risposta definitiva senza cursore
                response_placeholder.markdown(full_response)
                
                # Mostra fonti se abilitato
                if show_sources and sources_list:
                    with st.expander("📄 Fonti consultate"):
                        for src in sources_list:
                            st.caption(f"• {src}")

            # Salva in cronologia
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_list
            })

if __name__ == "__main__":
    main()