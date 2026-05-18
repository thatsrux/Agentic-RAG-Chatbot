import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.config import *
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

def condense_question_node(state: RAGState):
    question = state["question"]
    chat_history = state.get("chat_history", "")
    
    if not chat_history.strip():
        return {"question": question} # Nessuna cronologia, salta
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_PROMPT),
        ("human", f"Cronologia:\n{chat_history}\n\nUltima domanda: {question}")
    ])
    
    new_question = (prompt | load_llm() | StrOutputParser()).invoke({})

    if new_question.strip() != question.strip():
        st.toast(f"🧠 Domanda contestualizzata: {new_question}", icon="🔗")

    return {"question": new_question}

def retrieve_node(state: RAGState):
    question = state["question"]
    docs = st.session_state.retriever.retrieve(question)

    with st.expander("📄 Chunk recuperati", expanded=False):
        if not docs:
            st.warning("Nessun documento recuperato.")
        else:
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}** — Fonte: `{doc.metadata.get('source', 'N/A')}`")
                st.info(doc.page_content)
    
    context = format_context(docs)
    sources = list(set([d.metadata.get("source", "N/A") for d in docs]))
    return {"context": context, "sources": sources}

def grade_documents_node(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", DOC_GRADER_PROMPT),
        ("human", f"Contesto:\n{state['context']}\n\nDomanda:\n{state['question']}")
    ])
    chain = prompt | load_llm() | JsonOutputParser()
    
    try:
        score = chain.invoke({})
        grade = score.get("binary_score", "no").lower()
    except Exception:
        grade = "si"

    return {"doc_grade": grade}

def generate_node(state: RAGState):
    chain = RAG_PROMPT | load_llm() | StrOutputParser()
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"generation": response}

def check_hallucination_node(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", HALLUCINATION_GRADER_PROMPT),
        ("human", f"Contesto: {state['context']}\n\nGenerazione: {state['generation']}")
    ])
    chain = prompt | load_llm() | JsonOutputParser()
    
    try:
        score = chain.invoke({})
        grade = score.get("binary_score", "no").lower()
    except Exception:
        grade = "si" 
    return {"hallucination_grade": grade}

def grade_answer_node(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_GRADER_PROMPT),
        ("human", f"Domanda: {state['question']}\n\nGenerazione: {state['generation']}")
    ])
    chain = prompt | load_llm() | JsonOutputParser()
    
    try:
        score = chain.invoke({})
        grade = score.get("binary_score", "no").lower()
    except Exception:
        grade = "si"
    return {"answer_grade": grade}

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

def route_after_domain(state: RAGState):
    return "out_of_domain" if state.get("is_in_domain") == "no" else "in_domain"

def route_after_doc_grade(state: RAGState):
    if state.get("doc_grade") == "si":
        return "generate"
    return "rewrite" if state.get("retry_count", 0) < MAX_RETRIES else "fallback"

def route_after_hallucination(state: RAGState):
    if state.get("hallucination_grade") == "si":
        return "grade_answer"
    return "rewrite" if state.get("retry_count", 0) < MAX_RETRIES else "fallback"

def route_after_answer(state: RAGState):
    if state.get("answer_grade") == "si":
        return "useful"
    return "rewrite" if state.get("retry_count", 0) < MAX_RETRIES else "fallback"