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

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOllama(model="llama3.2", temperature=0.1)

# --- 1. DEFINIZIONE DELLO STATO (La "Memoria" dell'Agente) ---
# Questo dizionario viaggerà da un nodo all'altro, accumulando dati.
class AgentState(TypedDict):
    question: str          # Domanda originale dell'utente
    safe_question: str     # Domanda dopo il filtro PII
    optimized_query: str   # Domanda riscritta per il database
    documents: List[str]   # Documenti recuperati dal Vector Store
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
        "Prepara la domanda dell'utente per la ricerca nei documenti del dipartimento DIEM.\n\n"
        "REGOLE TASSATIVE:\n"
        "1. LINGUAGGIO NATURALE: Mantieni sempre una frase discorsiva e di senso compiuto in italiano. È VIETATO ridurre la domanda a una lista di parole chiave (non fare 'nome cognome orari').\n"
        "2. FEDELTÀ: Non storpiare le parole e non espandere le abbreviazioni in modo strano. Se la domanda è già chiara (es. 'Qual è l'orario del prof Vento?'), RICOPILA ESATTAMENTE COM'È.\n"
        "3. Rimuovi eventuali tag di privacy come <PHONE_NUMBER> o simili.\n"
        "4. Restituisci SOLO ed ESCLUSIVAMENTE la frase finale, senza alcun prefisso o spiegazione.\n\n"
        "Domanda: {question}"
        "Riformulata:"
    )
    
    chain = prompt | llm | StrOutputParser()
    optimized = chain.invoke({"question": state["safe_question"]})
            
    optimized = optimized.strip('"\' \n')
    
    print(f"[DEBUG REWRITE] Originale: '{state['safe_question']}' -> Ottimizzata: '{optimized}'")
    
    return {"optimized_query": optimized}

def retrieve_node(state: AgentState):
    """Nodo 3: Cerca i documenti VERI nel database FAISS/Chroma."""
    query = state["optimized_query"]
    
    # 1. Chiamiamo il tuo vero Retriever
    raw_docs = st.session_state.retriever.retrieve(query)
    
    # 2. Estraiamo il testo e le fonti per l'LLM
    doc_texts = []
    for doc in raw_docs:
        fonte = doc.metadata.get("source", "Fonte sconosciuta")
        doc_texts.append(f"[Fonte: {fonte}]\n{doc.page_content}")
        
    # 3. Gestione caso "Nessun documento"
    if not doc_texts:
        doc_texts = ["Nessuna informazione utile trovata nel database ufficiale del DIEM per questa query."]
        
    # (Opzionale) Stampa di debug per farti vedere cosa estrae!
    print(f"[DEBUG RETRIEVER] Estratti {len(raw_docs)} documenti per la query: '{query}'")
    
    return {"documents": doc_texts}


def generate_node(state: AgentState):
    """Nodo 4: Genera la risposta usando i documenti."""
    llm = load_llm()
    
    # Prompt ottimizzato per risposte dirette e zero chiacchiere
    prompt = ChatPromptTemplate.from_template(
        "Sei l'assistente virtuale del DIEM. Rispondi alla seguente domanda basandoti ESCLUSIVAMENTE sui documenti forniti.\n\n"
        "--- INIZIO DOCUMENTI ---\n"
        "{docs}\n"
        "--- FINE DOCUMENTI ---\n\n"
        "Domanda: {question}\n\n"
        "REGOLE FONDAMENTALI DI STILE:\n"
        "1. RISPONDI DIRETTAMENTE: Non usare MAI formule di saluto (es. 'Ciao'), non presentarti (es. 'Sono l'assistente...') e non fare premesse (es. 'Per rispondere alla tua domanda...').\n"
        "2. Vai dritto al punto fornendo subito l'informazione richiesta in modo discorsivo ma conciso.\n"
        "3. Se l'informazione non è presente nei documenti, rispondi SOLO: 'Non ho trovato questa informazione nei documenti ufficiali.' Non inventare nulla.\n"
        "4. Aggiungi sempre la fonte alla fine della risposta."
    )
    
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
    if state["loop_count"] >= 3:
        return "end"

    llm = load_llm()
    prompt = ChatPromptTemplate.from_template(
        "Questa risposta: '{response}' risponde in modo esauriente alla domanda: '{question}'? "
        "Rispondi SOLO 'SI' o 'NO'."
    )
    chain = prompt | llm | StrOutputParser()
    eval_result = chain.invoke({"response": state["generation"], "question": state["safe_question"]}).strip().upper()
    
    if "SI" in eval_result:
        return "relevant" # Va al guardrail finale
    else:
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
            
    if "messages" not in st.session_state:
        st.session_state.messages = []

    app = build_agentic_rag()
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_input = st.chat_input("Chiedimi qualcosa...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("L'Agente sta ragionando..."):
                # Eseguiamo il grafo passandogli lo stato iniziale
                inputs = {"question": user_input, "loop_count": 0}
                
                # Usiamo stream() per vedere i passaggi (opzionale ma fantastico per il debug)
                for output in app.stream(inputs):
                    for key, value in output.items():
                        # Mostra a schermo in quale nodo si trova
                        st.caption(f"⚙️ Esecuzione nodo: **{key}**")
                        
                # Recuperiamo l'output finale dal nodo ethical_guardrail
                final_answer = value.get("generation", "Errore nella generazione.")
                
                st.markdown(final_answer)

if __name__ == "__main__":
    main()