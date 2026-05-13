import sqlite3
import operator
from typing import TypedDict, List, Annotated

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from retriever import HybridRetriever

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

OLLAMA_MODEL  = "llama3.2"
CHECKPOINT_DB = "checkpoints.db"
MAX_HISTORY   = 5     # turni (coppie domanda/risposta) da passare all'LLM
MAX_CHECKPOINT_MESSAGES = 30  # limite messaggi salvati nel checkpoint per prevenire "string too big"


# ==============================================================================
# REDUCER PERSONALIZZATO
# ==============================================================================

def _limit_history_reducer(old_history: List[dict], new_messages: List[dict]) -> List[dict]:
    """
    Reducer che aggiunge nuovi messaggi ma mantiene solo gli ultimi MAX_CHECKPOINT_MESSAGES.
    Evita che SQLite dia errore "string or blob too big" per checkpoint troppo grandi.
    """
    combined = old_history + new_messages
    return combined[-MAX_CHECKPOINT_MESSAGES:]


# ==============================================================================
# STATO DEL GRAFO
# ==============================================================================

class AgentState(TypedDict):
    # --- input ---
    query: str                       # domanda dell'utente

    # --- elaborazione interna ---
    documents: List[Document]        # documenti recuperati

    # --- output ---
    generation: str                  # risposta finale

    # --- memoria conversazione ---
    # Reducer personalizzato: aggiunge elementi ma limita la dimensione totale.
    # Il checkpointer SQLite salva solo gli ultimi MAX_CHECKPOINT_MESSAGES messaggi.
    chat_history: Annotated[List[dict], _limit_history_reducer]


# ==============================================================================
# MODELLO LLM
# ==============================================================================

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1, num_ctx=2048)


# ==============================================================================
# RETRIEVER (singleton) — inizializzato esternamente da build_graph()
# ==============================================================================

_retriever_instance = None

def get_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


# ==============================================================================
# UTILITY
# ==============================================================================

def _format_history(history: List[dict], max_turns: int = MAX_HISTORY) -> str:
    """
    Converte gli ultimi N turni di chat_history in stringa leggibile dall'LLM.
    Ogni turno è una coppia {"role": "user"|"assistant", "content": "..."}.
    """
    if not history:
        return ""
    recent = history[-(max_turns * 2):]
    lines  = []
    for msg in recent:
        label = "Utente" if msg["role"] == "user" else "DIEMbot"
        lines.append(f"{label}: {msg['content']}")
    return "\n".join(lines)


# ==============================================================================
# NODO 1 — Retrieval
# Passa la query dell'utente direttamente al retriever ibrido, senza riscrittura.
# ==============================================================================

def node_retrieve(state: AgentState) -> AgentState:
    docs = get_retriever().retrieve(state["query"])
    return {**state, "documents": docs}


# ==============================================================================
# NODO 2 — Generator
# Genera la risposta usando documenti + history. Usa la domanda ORIGINALE
# (non quella contestualizzata) per rispondere in modo naturale all'utente.
# ==============================================================================

SYSTEM_PROMPT = """
Sei DIEMbot, l'assistente virtuale del DIEM (Dipartimento di Ingegneria dell'Informazione 
ed Elettrica e Matematica applicata) dell'Università di Salerno.

REGOLE:
1. Rispondi in italiano in modo professionale e cordiale.
2. Basati ESCLUSIVAMENTE sui documenti forniti nel "Contesto".
3. Usa la "Conversazione precedente" per capire il contesto e risolvere riferimenti 
   impliciti, ma NON inventare informazioni da essa.
4. Se l'informazione non è nei documenti, dillo onestamente senza inventare nulla.
5. Indica la fonte (es. "In base al sito DIEM...") quando disponibile nel contesto.
"""

GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "Conversazione precedente:\n{history}\n\n"
     "Contesto estratto dai documenti:\n{context}\n\n"
     "Domanda: {question}")
])

def node_generate(state: AgentState) -> AgentState:
    context = "\n\n---\n\n".join([
        f"[Documento {i+1} | Fonte: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for i, d in enumerate(state["documents"])
    ]) if state["documents"] else "Nessun documento pertinente trovato."

    chain      = GEN_PROMPT | llm | StrOutputParser()
    generation = chain.invoke({
        "history":  _format_history(state.get("chat_history", [])),
        "context":  context,
        "question": state["query"],
    })
    return {**state, "generation": generation}


# ==============================================================================
# NODO 3 — Update History
# Appende il turno corrente alla chat_history tramite il reducer operator.add.
# ==============================================================================

def node_update_history(state: AgentState) -> dict:
    return {
        "chat_history": [
            {"role": "user",      "content": state["query"]},
            {"role": "assistant", "content": state["generation"]},
        ]
    }


# ==============================================================================
# COSTRUZIONE DEL GRAFO
# ==============================================================================

def build_graph():
    """
    Grafo lineare: retrieve → generate → update_history → END

    Contestualizzazione, router, grader e hallucination checker rimossi.
    Il retriever viene inizializzato qui, in modo che il caricamento
    di FAISS/reranker avvenga allo startup e non alla prima query.
    """
    # Inizializzazione eager: carica FAISS e reranker subito
    get_retriever()

    conn   = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    memory = SqliteSaver(conn)

    g = StateGraph(AgentState)

    g.add_node("retrieve",       node_retrieve)
    g.add_node("generate",       node_generate)
    g.add_node("update_history", node_update_history)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve",       "generate")
    g.add_edge("generate",       "update_history")
    g.add_edge("update_history", END)

    return g.compile(checkpointer=memory)
