import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.config import *
from utils.utils import load_llm, format_context

# ── ANSI colors ────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"    # CONDENSE
YELLOW  = "\033[93m"   # DOMAIN_GUARD
GREEN   = "\033[92m"   # RETRIEVE
BLUE    = "\033[94m"   # DOC_GRADE
MAGENTA = "\033[95m"   # GENERATE
WHITE   = "\033[97m"   # HALLUCINATION
ORANGE  = "\033[33m"   # ANSWER_GRADE
RED     = "\033[91m"   # REWRITE / FALLBACK
GRAY    = "\033[90m"   # ROUTE

def _log(color, tag, msg):
    print(f"{BOLD}{color}[{tag}]{RESET} {color}{msg}{RESET}")

def _sep(color, label):
    print(f"\n{BOLD}{color}{'━'*20} {label} {'━'*20}{RESET}")
# ───────────────────────────────────────────────────────────


def condense_question_node(state: RAGState):
    question = state["question"]
    chat_history = state.get("chat_history", "")
    _sep(CYAN, "CONDENSE QUESTION")
    _log(CYAN, "CONDENSE", f"INPUT  question     : {question}")
    _log(CYAN, "CONDENSE", f"INPUT  chat_history : '{chat_history}'")

    if not chat_history.strip():
        _log(CYAN, "CONDENSE", "Nessuna history → skip LLM")
        return {"question": question}

    prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_PROMPT),
        ("human", f"Cronologia:\n{chat_history}\n\nUltima domanda: {question}")
    ])
    new_question = (prompt | load_llm() | StrOutputParser()).invoke({})
    _log(CYAN, "CONDENSE", f"OUTPUT new_question : {new_question}")

    if new_question.strip() != question.strip():
        st.toast(f"🧠 Domanda contestualizzata: {new_question}", icon="🔗")

    return {"question": new_question}


def domain_guard_node(state: RAGState):
    question = state["question"]
    chat_history = state.get("chat_history", "")
    _sep(YELLOW, "DOMAIN GUARD")
    _log(YELLOW, "DOMAIN_GUARD", f"INPUT  question : {question}")
    st.toast("🛡️ Controllo pertinenza della domanda...", icon="🔍")

    prompt = ChatPromptTemplate.from_messages([
        ("system", DOMAIN_PROMPT),
        ("human", f"Cronologia:\n{chat_history}\n\nDomanda utente: {question}")
    ])
    chain = prompt | load_llm() | JsonOutputParser()

    try:
        result = chain.invoke({})
        in_domain = result.get("in_domain", "si").lower()
    except Exception:
        in_domain = "si"

    _log(YELLOW, "DOMAIN_GUARD", f"OUTPUT in_domain: {in_domain}")

    if in_domain == "no":
        return {
            "is_in_domain": "no",
            "generation": "Mi dispiace, ma sono programmato per rispondere esclusivamente a domande riguardanti il dipartimento DIEM e l'Università di Salerno. Posso aiutarti con informazioni su corsi o docenti?",
            "sources": []
        }
    return {"is_in_domain": "si"}


def retrieve_node(state: RAGState):
    question = state["question"]
    _sep(GREEN, "RETRIEVE")
    _log(GREEN, "RETRIEVE", f"INPUT  question   : {question}")
    docs = st.session_state.retriever.retrieve(question)
    _log(GREEN, "RETRIEVE", f"OUTPUT docs count : {len(docs)}")
    for i, doc in enumerate(docs):
        score = doc.metadata.get("rerank_score", None)
        score_str = f"{score:.4f}" if score is not None else "N/A"
        _log(GREEN, "RETRIEVE", f"  [{i+1}] score={score_str}  source={doc.metadata.get('source', 'N/A')}")

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
    _sep(BLUE, "DOC GRADE")
    _log(BLUE, "DOC_GRADE", f"INPUT  question          : {state['question']}")
    _log(BLUE, "DOC_GRADE", f"INPUT  context (300 car) : {state['context'][:300]}")
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

    _log(BLUE, "DOC_GRADE", f"OUTPUT grade : {grade}")
    return {"doc_grade": grade}


def generate_node(state: RAGState):
    _sep(MAGENTA, "GENERATE")
    _log(MAGENTA, "GENERATE", f"INPUT  question           : {state['question']}")
    chain = RAG_PROMPT | load_llm() | StrOutputParser()
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    _log(MAGENTA, "GENERATE", f"OUTPUT response (300 car) : {response[:300]}")
    return {"generation": response}


def check_hallucination_node(state: RAGState):
    _sep(WHITE, "HALLUCINATION CHECK")
    _log(WHITE, "HALLUCINATION", f"INPUT  generation (300 car): {state['generation'][:300]}")
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

    _log(WHITE, "HALLUCINATION", f"OUTPUT grade : {grade}")
    return {"hallucination_grade": grade}


def grade_answer_node(state: RAGState):
    _sep(ORANGE, "ANSWER GRADE")
    _log(ORANGE, "ANSWER_GRADE", f"INPUT  question            : {state['question']}")
    _log(ORANGE, "ANSWER_GRADE", f"INPUT  generation (300 car): {state['generation'][:300]}")
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

    _log(ORANGE, "ANSWER_GRADE", f"OUTPUT grade : {grade}")
    return {"answer_grade": grade}


def rewrite_node(state: RAGState):
    current_retries = state.get("retry_count", 0)
    _sep(RED, "REWRITE")
    _log(RED, "REWRITE", f"INPUT  question    : {state['question']}")
    _log(RED, "REWRITE", f"       retry_count : {current_retries}")
    st.toast(f"🔄 Tentativo {current_retries + 1}/{MAX_RETRIES}: Riformulazione domanda...", icon="🧠")

    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_PROMPT),
        ("human", f"Domanda originale: {state['question']}")
    ])
    rewritten_question = (prompt | load_llm() | StrOutputParser()).invoke({})
    _log(RED, "REWRITE", f"OUTPUT rewritten   : {rewritten_question}")
    return {"question": rewritten_question, "retry_count": current_retries + 1}


def fallback_node(state: RAGState):
    _sep(RED, "FALLBACK")
    _log(RED, "FALLBACK", "Nessuna risposta trovata → fallback attivato.")
    return {
        "generation": "Mi dispiace, ho cercato nei documenti a mia disposizione ma non ho trovato informazioni sicure per rispondere a questa domanda.",
        "sources": []
    }


def route_after_domain(state: RAGState):
    result = "out_of_domain" if state.get("is_in_domain") == "no" else "in_domain"
    print(f"{BOLD}{GRAY}[ROUTE] after_domain → {result}{RESET}")
    return result

def route_after_doc_grade(state: RAGState):
    if state.get("doc_grade") == "si":
        result = "generate"
    else:
        result = "rewrite" if state.get("retry_count", 0) < MAX_RETRIES else "fallback"
    print(f"{BOLD}{GRAY}[ROUTE] after_doc_grade → {result}  (retry_count={state.get('retry_count', 0)}){RESET}")
    return result

def route_after_hallucination(state: RAGState):
    if state.get("hallucination_grade") == "si":
        result = "grade_answer"
    else:
        result = "rewrite" if state.get("retry_count", 0) < MAX_RETRIES else "fallback"
    print(f"{BOLD}{GRAY}[ROUTE] after_hallucination → {result}  (retry_count={state.get('retry_count', 0)}){RESET}")
    return result

def route_after_answer(state: RAGState):
    if state.get("answer_grade") == "si":
        result = "useful"
    else:
        result = "rewrite" if state.get("retry_count", 0) < MAX_RETRIES else "fallback"
    print(f"{BOLD}{GRAY}[ROUTE] after_answer → {result}  (retry_count={state.get('retry_count', 0)}){RESET}")
    return result
