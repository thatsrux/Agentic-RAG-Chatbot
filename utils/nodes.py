import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.config import RAGState, RAG_PROMPT, DOMAIN_PROMPT, GRADER_PROMPT, REWRITE_PROMPT, MAX_RETRIES
from utils.utils import load_llm, format_context

def domain_guard_node(state: RAGState):
    question = state["question"]
    st.toast("🛡️ Controllo pertinenza della domanda...", icon="🔍")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", DOMAIN_PROMPT),
        ("human", f"Domanda utente: {question}")
    ])
    
    chain = prompt | load_llm() | JsonOutputParser()
    
    try:
        result = chain.invoke({})
        in_domain = result.get("in_domain", "si").lower()
    except Exception:
        in_domain = "si"

    if in_domain == "no":
        return {
            "is_in_domain": "no",
            "generation": "Mi dispiace, ma sono programmato per rispondere esclusivamente a domande riguardanti il dipartimento DIEM e l'Università di Salerno. Posso aiutarti con informazioni su corsi o docenti?",
            "sources": []
        }
        
    return {"is_in_domain": "si"}

def retrieve_node(state: RAGState):
    question = state["question"]
    docs = st.session_state.retriever.retrieve(question)

    with st.expander("🛠️ DEBUG: Ispeziona i Chunk estratti", expanded=False):
        if not docs:
            st.warning("Nessun documento recuperato.")
        else:
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}** — Fonte: `{doc.metadata.get('source', 'N/A')}`")
                st.info(doc.page_content)
    
    context = format_context(docs)
    sources = list(set([d.metadata.get("source", "N/A") for d in docs]))
    return {"context": context, "sources": sources}

def generate_node(state: RAGState):
    chain = RAG_PROMPT | load_llm() | StrOutputParser()
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"generation": response}

def grade_generation_node(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_PROMPT),
        ("human", f"Contesto: {state['context']}\n\nRisposta da valutare: {state['generation']}\n\nDomanda utente: {state['question']}")
    ])
    
    chain = prompt | load_llm() | JsonOutputParser()
    
    try:
        score = chain.invoke({})
        grade = score.get("binary_score", "no").lower()
    except Exception:
        grade = "no"

    return {"grade": grade}

def rewrite_node(state: RAGState):
    current_retries = state.get("retry_count", 0)
    st.toast(f"🔄 Tentativo {current_retries + 1}/{MAX_RETRIES}: Riformulazione domanda...", icon="🧠")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_PROMPT),
        ("human", f"Domanda originale: {state['question']}")
    ])
    
    rewritten_question = (prompt | load_llm() | StrOutputParser()).invoke({})
    return {"question": rewritten_question, "retry_count": current_retries + 1}

def fallback_node(state: RAGState):
    return {
        "generation": "Mi dispiace, ho cercato nei documenti a mia disposizione ma non ho trovato informazioni sicure per rispondere a questa domanda.",
        "sources": []
    }

# --- FUNZIONI DI ROUTING ---
def route_after_domain(state: RAGState):
    return "out_of_domain" if state.get("is_in_domain") == "no" else "in_domain"

def route_after_grade(state: RAGState):
    if state.get("grade") == "si":
        return "useful"
    return "rewrite" if state.get("retry_count", 0) < MAX_RETRIES else "max_retries"