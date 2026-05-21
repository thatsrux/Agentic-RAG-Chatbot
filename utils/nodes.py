import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from utils.config import *
from utils.utils import load_llm, format_context

RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"    # CONDENSE
YELLOW  = "\033[93m"   # DOMAIN_GUARD
GREEN   = "\033[92m"   # RETRIEVE
BLUE    = "\033[94m"   # DOC_GRADE
MAGENTA = "\033[95m"   # GENERATE
RED     = "\033[91m"   # REWRITE / FALLBACK
ORANGE  = "\033[33m"   # ANSWER_GRADE
GRAY    = "\033[90m"   # ROUTE

def _log(color, tag, msg):
    print(f"{BOLD}{color}[{tag}]{RESET} {color}{msg}{RESET}")

def _sep(color, label):
    print(f"\n{BOLD}{color}{'━'*20} {label} {'━'*20}{RESET}")

def _safe_extract_string(output) -> str:
    """Previene l'errore 'list object has no attribute strip'."""
    if isinstance(output, list):
        if len(output) > 0:
            if isinstance(output[0], dict):
                return str(output[0].get("text", ""))
            return str(output[0])
        return ""
    return str(output)

# --- FIX GEMINI: Funzione di estrazione sicura ---
def _safe_extract_string(output) -> str:
    """Previene l'errore 'list object has no attribute strip'."""
    if isinstance(output, list):
        if len(output) > 0:
            if isinstance(output[0], dict):
                return str(output[0].get("text", ""))
            return str(output[0])
        return ""
    return str(output)
# -------------------------------------------------

def condense_question_node(state: RAGState):
    question = state["question"]
    history_list = state.get("chat_history", [])
    
    _sep(CYAN, "CONDENSE QUESTION")
    _log(CYAN, "CONDENSE", f"INPUT  question : {question}")

    if not history_list:
        _log(CYAN, "CONDENSE", "Nessuna history → skip LLM")
        return {"question": question}

    recent_history = history_list[-4:] if len(history_list) > 4 else history_list
    chat_history_str = ""
    for msg in recent_history:
        role = "Studente" if msg["role"] == "user" else "DIEMbot"
        chat_history_str += f"{role}: {msg['content']}\n"

    prompt = PromptTemplate.from_template(CONDENSE_PROMPT)
    chain = prompt | load_llm(state.get("current_model")) | StrOutputParser()

    try:
        raw_output = chain.invoke({
            "history": chat_history_str,
            "query": question
        })
        
        new_question = _safe_extract_string(raw_output).strip()
        
        _log(CYAN, "CONDENSE", f"OUTPUT rewritten : {new_question}")
        return {"question": new_question}
    except Exception as e:
        _log(CYAN, "CONDENSE", f"Errore → fallback: {e}")
        return {"question": question}


def domain_guard_node(state: RAGState):
    question = state["question"]
    _sep(YELLOW, "DOMAIN GUARD")
    _log(YELLOW, "DOMAIN_GUARD", f"INPUT  question : {question}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", DOMAIN_PROMPT),
        ("human", "Domanda utente: {question}")
    ])
    chain = prompt | load_llm(state.get("current_model")) | StrOutputParser()

    try:
        raw_output = chain.invoke({})
        result = _safe_extract_string(raw_output).strip().lower()
        in_domain = "no" if "no" in result[:5] else "si" 
    except Exception:
        in_domain = "si"

    _log(YELLOW, "DOMAIN_GUARD", f"OUTPUT in_domain: {in_domain}")

    if in_domain == "no":
        return {
            "is_in_domain": "no",
            "generation": "Mi dispiace, ma rispondo solo a domande sul dipartimento DIEM e l'Università di Salerno.",
            "sources": []
        }
    return {"is_in_domain": "si"}


def retrieve_node(state: RAGState):
    question = state["question"]
    _sep(GREEN, "RETRIEVE")
    _log(GREEN, "RETRIEVE", f"INPUT  question   : {question}")
    
    docs = st.session_state.retriever.retrieve(question)
    _log(GREEN, "RETRIEVE", f"OUTPUT docs count : {len(docs)}")
    
    context = format_context(docs)
    sources = list(set([d.metadata.get("source", "N/A") for d in docs]))
    return {"context": context, "sources": sources}

def generate_node(state: RAGState):
    _sep(MAGENTA, "GENERATE")
    _log(MAGENTA, "GENERATE", f"INPUT  question           : {state['question']}")
    
    chain = RAG_PROMPT | load_llm(state.get("current_model")) 
    
    ai_message = chain.invoke({"context": state["context"], "question": state["question"]})
    
    response_text = _safe_extract_string(ai_message.content)
    
    metadata = ai_message.response_metadata
    model_used = metadata.get("model_name") or metadata.get("model") or "Sconosciuto"
    
    _log(MAGENTA, "GENERATE", f"OUTPUT response (300 car) : {response_text[:300]}")
    _log(MAGENTA, "GENERATE", f"MODEL USED : {model_used}")
    
    return {"generation": response_text, "model_used": model_used}

def route_after_domain(state: RAGState):
    return "out_of_domain" if state.get("is_in_domain") == "no" else "in_domain"
