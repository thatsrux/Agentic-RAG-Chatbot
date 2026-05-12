# agentic_rag.py
import os
from typing import TypedDict, List, Literal
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from retriever import HybridRetriever

# ── Stato condiviso del grafo ──────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str                  # domanda originale
    rewritten_query: str        # eventuale query riformulata
    documents: List[Document]   # docs recuperati
    generation: str             # risposta generata
    loop_count: int             # contatore cicli (anti-loop)
    route: str                  # "web" | "pdf" | "both"

# ── Modelli ────────────────────────────────────────────────────────────────────

llm = ChatOllama(model="llama3.2", temperature=0, num_ctx=2048)
llm_json = ChatOllama(model="llama3.2", temperature=0, format="json")

retriever = HybridRetriever()

# ── NODO 1: Router ─────────────────────────────────────────────────────────────
# Decide se cercare su web, PDF o entrambi in base alla natura della query

ROUTER_PROMPT = ChatPromptTemplate.from_template("""
Sei un router per un sistema RAG universitario (DIEM - Unisa).
Analizza la domanda e rispondi SOLO con un JSON: {{"route": "web"|"pdf"|"both"}}

Regole:
- "pdf"  → regolamenti, piani di studio, modulistica, bandi
- "web"  → docenti, corsi, orari, news, contatti
- "both" → domande generali o ambigue

Domanda: {query}
""")

def node_router(state: AgentState) -> AgentState:
    chain = ROUTER_PROMPT | llm_json | StrOutputParser()
    result = chain.invoke({"query": state["query"]})
    try:
        import json
        route = json.loads(result).get("route", "both")
    except Exception:
        route = "both"
    return {**state, "route": route, "loop_count": 0}

# ── NODO 2: Retrieval ──────────────────────────────────────────────────────────

def node_retrieve(state: AgentState) -> AgentState:
    q = state.get("rewritten_query") or state["query"]
    docs = retriever.retrieve(q)
    return {**state, "documents": docs}

# ── NODO 3: Grader di rilevanza ────────────────────────────────────────────────

GRADER_PROMPT = ChatPromptTemplate.from_template("""
Valuta se il documento è pertinente alla domanda.
Rispondi SOLO con JSON: {{"score": "yes"}} oppure {{"score": "no"}}

Documento: {document}
Domanda: {query}
""")

def node_grade_documents(state: AgentState) -> AgentState:
    chain = GRADER_PROMPT | llm_json | StrOutputParser()
    filtered = []
    for doc in state["documents"]:
        result = chain.invoke({
            "document": doc.page_content[:500],
            "query": state["query"]
        })
        try:
            import json
            if json.loads(result).get("score") == "yes":
                filtered.append(doc)
        except Exception:
            filtered.append(doc)  # in caso di errore, tieni il doc
    return {**state, "documents": filtered}

# ── NODO 4: Query Rewriter ─────────────────────────────────────────────────────

REWRITER_PROMPT = ChatPromptTemplate.from_template("""
La query originale non ha prodotto risultati utili.
Riformulala in modo più preciso per migliorare la ricerca vettoriale.
Rispondi SOLO con la nuova query, senza spiegazioni.

Query originale: {query}
""")

def node_rewrite(state: AgentState) -> AgentState:
    chain = REWRITER_PROMPT | llm | StrOutputParser()
    new_query = chain.invoke({"query": state["query"]})
    return {
        **state,
        "rewritten_query": new_query.strip(),
        "loop_count": state["loop_count"] + 1
    }

# ── NODO 5: Generator ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Sei DIEMbot, l'assistente virtuale del DIEM (Università di Salerno).
Rispondi in italiano in modo professionale.
Basati ESCLUSIVAMENTE sul contesto fornito.
Se l'informazione non è presente, dillo onestamente.
"""

GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Contesto:\n{context}\n\nDomanda: {question}")
])

def node_generate(state: AgentState) -> AgentState:
    context = "\n\n---\n\n".join([
        f"[Fonte: {d.metadata.get('source','?')}]\n{d.page_content}"
        for d in state["documents"]
    ]) or "Nessun documento pertinente trovato."
    
    chain = GEN_PROMPT | llm | StrOutputParser()
    generation = chain.invoke({
        "context": context,
        "question": state["query"]
    })
    return {**state, "generation": generation}

# ── NODO 6: Hallucination Checker ─────────────────────────────────────────────

HALLUC_PROMPT = ChatPromptTemplate.from_template("""
La risposta è supportata dai documenti forniti?
Rispondi SOLO con JSON: {{"grounded": "yes"}} oppure {{"grounded": "no"}}

Documenti: {documents}
Risposta: {generation}
""")

def node_check_hallucination(state: AgentState) -> AgentState:
    if not state["documents"]:
        return state  # nessun doc, non possiamo verificare
    
    chain = HALLUC_PROMPT | llm_json | StrOutputParser()
    result = chain.invoke({
        "documents": "\n".join([d.page_content[:300] for d in state["documents"]]),
        "generation": state["generation"]
    })
    # Il risultato viene usato dall'edge condizionale, non modifica lo stato
    try:
        import json
        grounded = json.loads(result).get("grounded", "yes")
    except Exception:
        grounded = "yes"
    return {**state, "_grounded": grounded}

# ── EDGE CONDIZIONALI ──────────────────────────────────────────────────────────

def edge_check_relevance(state: AgentState) -> Literal["generate", "rewrite"]:
    """Dopo il grading: se hai docs rilevanti genera, altrimenti riscrivi."""
    if state["documents"]:
        return "generate"
    if state["loop_count"] >= 2:  # stop anti-loop
        return "generate"
    return "rewrite"

def edge_check_hallucination(state: AgentState) -> Literal["end", "generate"]:
    """Dopo il check: se grounded termina, altrimenti rigenera."""
    if state.get("_grounded", "yes") == "yes" or state["loop_count"] >= 2:
        return "end"
    return "generate"

# ── COSTRUZIONE DEL GRAFO ──────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router",      node_router)
    g.add_node("retrieve",    node_retrieve)
    g.add_node("grade",       node_grade_documents)
    g.add_node("rewrite",     node_rewrite)
    g.add_node("generate",    node_generate)
    g.add_node("check_halluc",node_check_hallucination)

    g.set_entry_point("router")
    g.add_edge("router",   "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", edge_check_relevance, {
        "generate": "generate",
        "rewrite":  "rewrite"
    })
    g.add_edge("rewrite", "retrieve")  # ← il ciclo
    g.add_edge("generate", "check_halluc")
    g.add_conditional_edges("check_halluc", edge_check_hallucination, {
        "end":      END,
        "generate": "generate"
    })

    return g.compile()

# Istanza globale riusabile
rag_graph = build_graph()

def ask(query: str) -> dict:
    result = rag_graph.invoke({"query": query})
    return {
        "answer":    result["generation"],
        "sources":   [d.metadata.get("source") for d in result["documents"]],
        "rewrote":   result.get("rewritten_query"),
        "route":     result["route"],
    }