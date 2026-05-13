import os
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.data_anonymizer import PresidioAnonymizer
from presidio_analyzer import AnalyzerEngine
from retriever import HybridRetriever

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYSTEM_PROMPT = """
Sei DIEMbot, l'assistente virtuale del DIEM (Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica applicata) dell'Università di Salerno.

REGOLE FONDAMENTALI:
1. Rispondi in italiano in modo professionale e cordiale.
2. Basati ESCLUSIVAMENTE sui documenti forniti nel "Contesto".
3. Se l'informazione non è presente nei documenti, rispondi onestamente che non disponi di quei dati. Non inventare nulla.
4. NON inserire link o URL nel corpo della risposta (verranno visualizzati separatamente).
"""

@st.cache_resource(show_spinner=False)
def load_presidio():
    """Carica i motori di analisi e anonimizzazione di Presidio per l'italiano."""
    
    # 1. Creiamo la configurazione dicendo a Presidio di usare Spacy e il modello italiano
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "it", "model_name": "it_core_news_lg"}],
    }
    
    # 2. Inizializziamo il motore NLP con questa configurazione
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # 3. Passiamo il motore NLP all'Analyzer
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine, 
        supported_languages=["it"]
    )
    
    anonymizer = AnonymizerEngine()
    
    return analyzer, anonymizer

# --- INIZIALIZZAZIONE RISORSE ---
@st.cache_resource(show_spinner=False)
def load_anonymizer():
    """Inizializza l'anonimizzatore passando la configurazione per l'italiano."""
    return PresidioAnonymizer(
        languages_config={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "it", "model_name": "it_core_news_lg"}],
        }
    )

CURRENT_MODEL = "mistral-nemo"
CURRENT_MODEL = "llama3.2"
@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOllama(model=CURRENT_MODEL, temperature=0.1)

# --- 1. DEFINIZIONE DELLO STATO (La "Memoria" dell'Agente) ---
# Questo dizionario viaggerà da un nodo all'altro, accumulando dati.
class AgentState(TypedDict):
    question: str          # Domanda originale dell'utente
    safe_question: str     # Domanda dopo il filtro PII
    optimized_query: str   # Domanda riscritta per il database
    documents: List[str]   # Documenti recuperati dal Vector Store
    sources: List[str]     # Fonti estratte dai documenti
    generation: str        # Risposta generata dall'LLM
    loop_count: int        # Contatore per evitare loop infiniti

# --- 2. DEFINIZIONE DEI NODI (Le azioni) ---

def pii_guardrail_node(state: AgentState):
    """Nodo 1: Filtra i dati sensibili, ignorando i nomi propri dei docenti."""
    analyzer, anonymizer = load_presidio()
    
    # LISTA BIANCA DELLE ENTITÀ: 
    # Censuriamo SOLO queste. Omettiamo volutamente "PERSON".
    entities_to_catch = [
        "EMAIL_ADDRESS", 
        "PHONE_NUMBER", 
        "CREDIT_CARD", 
        "IBAN_CODE", 
        "IP_ADDRESS"
    ]
    
    # Analizza il testo cercando solo le entità specificate
    results = analyzer.analyze(
        text=state["question"], 
        language="it", 
        entities=entities_to_catch
    )
    
    # Applica l'anonimizzazione
    anonymized_result = anonymizer.anonymize(
        text=state["question"], 
        analyzer_results=results
    )
    safe_q = anonymized_result.text
    
    # (Opzionale) Stampa nel terminale per fare debug visivo
    print(f"[DEBUG PII] Originale: {state['question']} -> Sicura: {safe_q}")
    
    return {"safe_question": safe_q, "loop_count": state.get("loop_count", 0)}

def rewrite_query_node(state: AgentState):
    """Nodo 2: Riscrive la query con un prompt essenziale e minimalista."""
    llm = load_llm()
    
    prompt = ChatPromptTemplate.from_template(
        "Sei un assistente che ottimizza le query degli utenti per un motore di ricerca documentale del dipartimento DIEM.\n\n"
        "REGOLE TASSATIVE:\n"
        "1. OTTIMIZZAZIONE: Il tuo obiettivo è estrarre l'intento e le parole chiave. Non è necessario usare frasi grammaticalmente complete (evita articoli e preposizioni inutili se non servono).\n"
        "2. CONSERVAZIONE: Se la domanda è già espressa in modo chiaro o a parole chiave, DEVI ricopiarla ESATTAMENTE com'è, senza aggiungere articoli (es. 'il', 'del', 'di').\n"
        "3. DOMINIO: Limita il contesto a corsi, docenti, esami e attività del DIEM.\n"
        "4. FEDELTÀ: Non tradurre i nomi dei corsi e non espandere abbreviazioni se non sei sicuro.\n"
        "5. PRIVACY: Rimuovi eventuali tag come <PHONE_NUMBER>, <EMAIL_ADDRESS>.\n"
        "6. Restituisci SOLO ed ESCLUSIVAMENTE la query finale.\n\n"
        "Utente: {question}\n"
        "Riformulata:"
    )
    
    chain = prompt | llm | StrOutputParser()
    optimized = chain.invoke({"question": state["safe_question"]})
            
    optimized = optimized.strip('"\' \n')
    
    print(f"[DEBUG REWRITE] Originale: '{state['safe_question']}' -> Ottimizzata: '{optimized}'")
    
    return {"optimized_query": optimized}

def retrieve_node(state: AgentState):
    """Nodo 3: Cerca i documenti e popola la lista delle fonti."""
    query = state["optimized_query"]
    
    # 1. Chiamiamo il tuo retriever
    raw_docs = st.session_state.retriever.retrieve(query)
    
    doc_texts = []
    sources_list = []
    
    # 2. Iteriamo sui documenti estratti
    for doc in raw_docs:
        fonte = doc.metadata.get("source", "Fonte sconosciuta")
        doc_texts.append(f"[Fonte: {fonte}]\n{doc.page_content}")
        sources_list.append(fonte) # Raccogliamo il link/nome del file
        
    # 3. Gestione caso vuoto
    if not doc_texts:
        doc_texts = ["Nessuna informazione utile trovata."]
        
    # 4. Rimuoviamo i duplicati dalle fonti
    unique_sources = list(set(sources_list))
    
    print(f"[DEBUG RETRIEVER] Estratti {len(raw_docs)} documenti. Fonti: {unique_sources}")
    
    # IMPORTANTE: Restituiamo sia i documenti che le fonti
    return {"documents": doc_texts, "sources": unique_sources}


def generate_node(state: AgentState):
    llm = load_llm()
    # Organizzato come richiesto con SYSTEM e HUMAN message
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Contesto estratto dai documenti:\n{docs}\n\nDomanda dell'utente: {question}")
    ])
    
    docs_string = "\n\n---\n\n".join(state["documents"])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": state["safe_question"], "docs": docs_string})
    
    return {"generation": response, "loop_count": state["loop_count"] + 1}

def ethical_guardrail_node(state: AgentState):
    """Nodo 5 (Finale): Controlla che la risposta non sia offensiva o problematica."""
    # Qui potresti usare un classificatore veloce. Per semplicità usiamo l'LLM stesso:
    llm = load_llm()
    prompt = ChatPromptTemplate.from_template(
        "Il seguente testo contiene insulti, discriminazioni o contenuti pericolosi? "
        "Rispondi SOLO 'SI' o 'NO'. Testo: {text}"
    )
    chain = prompt | llm | StrOutputParser()
    is_offensive = chain.invoke({"text": state["generation"]}).strip().upper()
    
    if "SI" in is_offensive:
        return {"generation": "Mi dispiace, ma la risposta generata vìola le policy etiche del DIEM."}
    return {"generation": state["generation"]}


# --- 3. ARCHI CONDIZIONALI (Le decisioni) ---

def check_relevance(state: AgentState) -> str:
    """Valuta se la risposta generata risponde effettivamente alla domanda."""
    # Limite di sicurezza: se abbiamo fatto 3 giri, ci fermiamo comunque
    if state["loop_count"] >= 5:
        print(f"[DEBUG RELEVANCE] Raggiunto limite massimo di tentativi ({state['loop_count']}). Forzo uscita.")
        return "end"

    llm = load_llm()
    prompt = ChatPromptTemplate.from_template(
        "Questa risposta: '{response}' risponde in modo esauriente alla domanda: '{question}'? "
        "Rispondi SOLO 'SI' o 'NO'."
    )
    chain = prompt | llm | StrOutputParser()

    print(f"\n[DEBUG CHECKER] Sto valutando questa bozza:\n---")
    print(state["generation"])
    print("---\n")
    
    # Esecuzione valutazione
    eval_result = chain.invoke({
        "response": state["generation"], 
        "question": state["safe_question"]
    }).strip().upper()
    
    print(f"[DEBUG RELEVANCE] Valutazione risposta: {eval_result} (Giro n. {state['loop_count']})")
    
    if "SI" in eval_result or "SÌ" in eval_result:
        return "relevant" # Va al guardrail finale
    else:
        print(f"[DEBUG RELEVANCE] Risposta non pertinente. Re-instradamento verso rewrite_query...")
        return "not_relevant" # Torna a riscrivere la query


# --- 4. COSTRUZIONE DEL GRAFO (LangGraph) ---
@st.cache_resource(show_spinner=False)
def build_agentic_rag():
    workflow = StateGraph(AgentState)

    # Aggiungiamo tutti i nodi
    workflow.add_node("pii_guardrail", pii_guardrail_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("ethical_guardrail", ethical_guardrail_node)

    # Definiamo il flusso lineare di base
    workflow.set_entry_point("pii_guardrail")
    workflow.add_edge("pii_guardrail", "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "generate")

    # Aggiungiamo il NODO DECISIONALE (Il loop della Lezione 17)
    workflow.add_conditional_edges(
        "generate",
        check_relevance,
        {
            "relevant": "ethical_guardrail", # Se va bene, vai al filtro finale
            "not_relevant": "rewrite_query", # Se NON va bene, ricomincia dal rewrite
            "end": "ethical_guardrail"       # Se abbiamo fatto troppi giri, termina
        }
    )

    # Chiudiamo il grafo
    workflow.add_edge("ethical_guardrail", END)

    # Compiliamo l'applicazione
    return workflow.compile()


def main():
    st.set_page_config(page_title="DIEM Agentic RAG", page_icon="🎓")
    st.title("🎓 DIEMbot - Agentic RAG")
    
    if "retriever" not in st.session_state:
        with st.spinner("Caricamento del database della conoscenza..."):
            st.session_state.retriever = HybridRetriever()
            
    # --- 1. GESTIONE CRONOLOGIA CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- 2. SIDEBAR CON OPZIONE CANCELLA CRONOLOGIA ---
    with st.sidebar:
        st.header("Impostazioni")
        if st.button("🗑️ Cancella cronologia chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun() 
        
        st.divider()
        st.caption(f"Configurazione: LangGraph + ({CURRENT_MODEL})")

    # --- 3. COSTRUZIONE GRAFO ---
    app = build_agentic_rag()
    
    # Visualizzazione messaggi precedenti (Cronologia semplice, senza rieseguire la chain)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Fonti", expanded=False):
                    for src in msg["sources"]:
                        st.caption(f"• {src}")
            
    # --- 4. GESTIONE NUOVO INPUT UTENTE ---
    user_input = st.chat_input("Chiedimi qualcosa...")
    
    if user_input:
        # Mostra messaggio utente
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Generazione della risposta dell'assistente
        with st.chat_message("assistant"):
            final_answer = "Errore nella generazione."
            final_sources = []
            
            # CONTENITORE: L'esecuzione della pipeline (Agentic Chain)
            with st.expander("⚙️ Agentic Chain", expanded=False):
                # Prepariamo gli input per il grafo
                inputs = {"question": user_input, "loop_count": 0}
                
                # Esecuzione del grafo in streaming
                for output in app.stream(inputs):
                    for key, value in output.items():
                        st.caption(f"Passaggio completato: **{key}**")
                        
                        # Aggiorniamo i dati man mano che i nodi producono output
                        if "generation" in value:
                            final_answer = value["generation"]
                        if "sources" in value:
                            final_sources = value["sources"]
            
            # Visualizzazione risposta finale
            st.markdown(final_answer)
            
            # Visualizzazione fonti finale
            if final_sources:
                with st.expander("📚 Fonti", expanded=False):
                    for src in final_sources:
                        st.caption(f"• {src}")
                        
        # Salvataggio nella cronologia
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "sources": final_sources
        })

if __name__ == "__main__":
    main()