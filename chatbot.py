import os
import streamlit as st

# --- PROTEZIONE CRASH E OTTIMIZZAZIONE ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from retriever import HybridRetriever

# --- CONFIGURAZIONE ---
OLLAMA_MODEL = "llama3.2" # Modello ottimale per velocità/risorse

# --- PROMPT DEL ROUTER (Decide se serve il RAG) ---
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Sei un addetto all'accoglienza del dipartimento DIEM. 
    Analizza la domanda dell'utente. Se riguarda docenti, professori, aule, corsi di laurea, 
    esami o regole del dipartimento, rispondi SEMPRE 'SÌ'. 
    Altrimenti rispondi 'NO'. Rispondi SOLO con la parola 'SÌ' o 'NO'."""),
    ("human", "{question}")
])

# --- PROMPT DELLA RISPOSTA (Genera la risposta finale) ---
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Sei DIEMbot, l'assistente virtuale ufficiale del DIEM.
    
    REGOLE DI RISPOSTA:
    1. Usa il CONTESTO fornito per rispondere in modo professionale in italiano.
    2. Se il CONTESTO non contiene informazioni utili, scusa l'inconveniente e suggerisci 
       all'utente di essere più specifico (es. indicando il nome completo di un docente).
    3. Cita sempre la fonte se disponibile (es. [Fonte: sito docenti]).
    4. Se la domanda è un saluto generico, rispondi cordialmente senza cercare nel database.

    CONTESTO ESTRATTO DAL DATABASE:
    {context}"""),
    ("human", "{question}")
])

# --- INIZIALIZZAZIONE RISORSE ---
@st.cache_resource(show_spinner=False)
def load_llm():
    """Carica Ollama con un limite di contesto per evitare crash della VRAM."""
    return ChatOllama(
        model=OLLAMA_MODEL, 
        temperature=0.1, # Leggera creatività per la fluidità, ma resta ancorato ai dati
        num_ctx=3072     # Bilanciamento ideale per non saturare la GPU
    )

@st.cache_resource(show_spinner=False)
def load_retriever():
    """Carica il sistema di retrieval (FAISS + Reranker)."""
    return HybridRetriever()

def main():
    st.set_page_config(
        page_title="DIEMbot — Agentic Router",
        page_icon="🎓",
        layout="centered"
    )

    st.title("🎓 DIEMbot")
    st.caption("Sistema Agentic RAG con Routing Intelligente")

    # Caricamento backend
    llm = load_llm()
    retriever = load_retriever()

    # Gestione cronologia
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Utente
    user_input = st.chat_input("Chiedimi qualcosa sul DIEM...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # --- 1. LOGICA DI ROUTING ---
            # Liste di parole chiave per forzare la ricerca (Override manuale)
            keywords = ["aula", "dove", "chi è", "prof", "docente", "corso", "esame", "laurea", "vento", "saggese"]
            
            needs_rag = "NO"
            if any(k in user_input.lower() for k in keywords):
                needs_rag = "SÌ"
            else:
                # Se non ci sono keyword, chiediamo all'LLM di decidere
                router_chain = ROUTER_PROMPT | llm | StrOutputParser()
                decision = router_chain.invoke({"question": user_input}).strip().upper()
                if "SÌ" in decision or "SI" in decision:
                    needs_rag = "SÌ"

            # --- 2. ESECUZIONE RETRIEVAL (Se necessario) ---
            context = ""
            if needs_rag == "SÌ":
                with st.spinner("🔍 Interrogazione database del DIEM..."):
                    docs = retriever.retrieve(user_input)
                    if docs:
                        context = "\n\n".join([
                            f"[Fonte: {d.metadata.get('source', 'N/A')}] {d.page_content}" 
                            for d in docs
                        ])
                        st.caption("✅ Informazioni recuperate dal database.")
                    else:
                        st.caption("⚠️ Nessun documento trovato per questa query.")
                
                # Debug: Mostra cosa è stato trovato (opzionale)
                if context:
                    with st.expander("👀 Visualizza frammenti estratti (Debug)"):
                        st.text(context)
            else:
                st.caption("💬 Risposta basata su conoscenza generale (No RAG).")

            # --- 3. GENERAZIONE RISPOSTA FINALE ---
            with st.spinner("✍️ Elaborazione risposta..."):
                answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
                
                # Eseguiamo la risposta
                full_response = answer_chain.invoke({
                    "question": user_input, 
                    "context": context if context else "Nessuna informazione specifica trovata nel database."
                })
                
                st.markdown(full_response)

            # Salva in cronologia
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })

if __name__ == "__main__":
    main()