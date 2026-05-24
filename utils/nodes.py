import json
import re
import requests
from bs4 import BeautifulSoup
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from utils.config import *
from utils.chatbot_utils import load_llm, format_context, is_valid_url
from utils.retriever import HybridRetriever
from utils.crawling_utils import KEYWORDS
from langgraph.graph import StateGraph, START, END
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
YELLOW  = "\033[93m"
GREEN   = "\033[92m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
GRAY    = "\033[90m"

RETRIEVER = HybridRetriever()

def _log(color, tag, msg):
    """
    Logga un messaggio con un tag colorato e in grassetto.
    """
    print(f"{BOLD}{color}[{tag}]{RESET} {color}{msg}{RESET}")

def _sep(color, label):
    """
    Stampa una linea di separazione colorata con un'etichetta al centro.
    """
    print(f"\n{BOLD}{color}{'━'*20} {label} {'━'*20}{RESET}")

def _safe_extract_string(output) -> str:
    """
    Estrae una stringa in modo sicuro da un output che potrebbe essere una lista.
    """
    if isinstance(output, list):
        if len(output) > 0:
            if isinstance(output[0], dict):
                return str(output[0].get("text", ""))
            return str(output[0])
        return ""
    return str(output)

def condense_question_node(state: RAGState):
    """
    Riformula la domanda dell'utente in modo più conciso, tenendo conto della cronologia recente della conversazione.
    """
    question = state["question"]
    history_list = state.get("chat_history", [])
    
    _sep(CYAN, "CONDENSE QUESTION")
    _log(CYAN, "CONDENSE", f"INPUT  question : {question}")

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
        
        _log(CYAN, "CONDENSE", f"OUTPUT rewritten: {new_question}")
        return {"question": new_question}
    except Exception as e:
        _log(CYAN, "CONDENSE", f"Errore → fallback: {e}")
        return {"question": question}


def domain_guard_node(state: RAGState):
    """
    Controlla se la domanda dell'utente è pertinente al dominio accademico del DIEM.
    Se non lo è, blocca la conversazione e restituisce un messaggio di cortesia.
    """
    question = state["question"]
    _sep(YELLOW, "DOMAIN GUARD")
    _log(YELLOW, "DOMAIN_GUARD", f"INPUT  question : {question}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", DOMAIN_PROMPT),
        ("human", f"Domanda utente: {question}")
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

def route_after_domain(state: RAGState):
    """
    Determina il percorso dopo il nodo di domain guard.
    """
    return "out_of_domain" if state.get("is_in_domain") == "no" else "in_domain"

def retrieve_node(state: RAGState):
    """
    Esegue la fase di retrieval, ottenendo i documenti più rilevanti per la domanda dell'utente e formattando il contesto da fornire al modello.
    """
    question = state["question"]
    _sep(GREEN, "RETRIEVE")
    _log(GREEN, "RETRIEVE", f"INPUT  question   : {question}")
    
    docs = RETRIEVER.retrieve(question)
    _log(GREEN, "RETRIEVE", f"OUTPUT docs count : {len(docs)}")
    
    context = format_context(docs)
    sources = list(set([d.metadata.get("source", "N/A") for d in docs]))
    return {"context": context, "sources": sources}

def generate_node(state: RAGState):
    """
    Esegue la fase di generazione, formulando la risposta dell'assistente basandosi esclusivamente sui documenti recuperati.
    """
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

def route_after_generation(state: RAGState):
    """
    Determina il percorso dopo il nodo di generazione, verificando se la risposta contiene il trigger per la ricerca web.
    """
    generation = state.get("generation", "")
    if "[TRIGGER_WEB_SEARCH]" in generation:
        return "go_to_web"
    return "go_to_end"

def web_search_node(state: RAGState):
    """
    Esegue una ricerca web mirata per trovare informazioni aggiuntive nei siti ufficiali del DIEM e dell'Università di Salerno,
    solo se la generazione precedente ha indicato che le informazioni nei documenti non erano sufficienti.
    """
    _sep(RED, "WEB SEARCH")
    question = state["question"]

    keyword_chain = KEYWORD_PROMPT | load_llm(state.get("current_model")) | StrOutputParser()
    
    try:
        raw_output = _safe_extract_string(keyword_chain.invoke({"question": question})).strip()
        clean_json = re.sub(r"```json|```", "", raw_output).strip()
        routing_data = json.loads(clean_json)
        
        search_query = routing_data.get("query", question)
        target_site = routing_data.get("site", "unisa.it")
        
    except Exception as e:
        search_query = f"{question} DIEM"
        target_site = "unisa.it"
        
    shielded_query = f"{search_query} site:{target_site}"
    _log(RED, "WEB_SEARCH", f"INPUT  Query: {shielded_query} (Target: {target_site})")
    
    web_contexts = []
    source_urls = []
    
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
        search_results = wrapper.results(shielded_query, max_results=25)
        
        valid_results_count = 0
        
        if search_results:
            for res in search_results:
                if valid_results_count >= 3:
                    break 
                    
                target_url = res.get("link", "")
                
                if not is_valid_url(target_url):
                    continue
                    
                snippet = res.get("snippet", "")
                
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    response = requests.get(target_url, headers=headers, timeout=5)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for noise in soup(["nav", "footer", "header", "script", "style", "aside"]):
                        noise.decompose()
                        
                    main_content = soup.find(id="unisa-content") or soup.find(id="content") or soup
                    page_text = main_content.get_text(separator=' ', strip=True)
                    
                    if "docenti.unisa.it" in target_url:
                        testo_upper = page_text.upper()
                        if not any(kw in testo_upper for kw in KEYWORDS):
                            continue 

                    web_contexts.append(f"[Fonte: {target_url}]\nTesto: {page_text[:4000]}\n")
                    
                except Exception as scrape_error:
                    if "docenti.unisa.it" in target_url:
                        continue
                        
                    web_contexts.append(f"[Fonte: {target_url}]\nSnippet: {snippet}\n")
                
                source_urls.append(f"Web: {target_url}")
                valid_results_count += 1
        
        if valid_results_count == 0:
            web_contexts.append("Nessun risultato utile trovato nei siti del DIEM autorizzati.")
            source_urls.append("Nessuna fonte web trovata")
            
    except Exception as e:
        web_contexts.append("Errore durante la connessione al motore di ricerca.")
        
    final_context = "\n---\n".join(web_contexts)
        
    chain = ChatPromptTemplate.from_template(WEB_GENERATE_PROMPT) | load_llm(state.get("current_model"))
    ai_message = chain.invoke({"context": final_context, "question": question})
    
    response_text = _safe_extract_string(ai_message.content)
    
    if "mi dispiace" in response_text.lower() or not response_text:
        response_text = "Mi dispiace, ma non trovo questa informazione nei documenti del DIEM né sui canali ufficiali autorizzati."

    _log(RED, "WEB_SEARCH", f"OUTPUT response: {response_text}")

    return {
        "generation": response_text, 
        "sources": state.get("sources", []) + source_urls
    }

def build_graph():
    """
    Costruisce e compila il grafo di stato per la pipeline RAG, definendo i nodi e le transizioni.
    """
    workflow = StateGraph(RAGState)

    workflow.add_node("condense_question", condense_question_node)
    workflow.add_node("domain_guard", domain_guard_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("web_search", web_search_node)

    workflow.add_edge(START, "condense_question")
    workflow.add_edge("condense_question", "domain_guard")

    workflow.add_conditional_edges(
        "domain_guard", route_after_domain,
        {"in_domain": "retrieve", "out_of_domain": END}
    )
    
    workflow.add_edge("retrieve", "generate")

    workflow.add_conditional_edges(
        "generate", route_after_generation,
        {
            "go_to_web": "web_search",
            "go_to_end": END
        }
    )
    
    workflow.add_edge("web_search", END)

    return workflow.compile()
