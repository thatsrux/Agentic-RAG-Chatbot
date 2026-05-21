import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from utils.config import *
from utils.utils import load_llm, format_context
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
        
        _log(CYAN, "CONDENSE", f"OUTPUT rewritten: {new_question}")
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

import json
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import json
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def web_search_node(state: RAGState):
    _sep(RED, "WEB SEARCH FALLBACK (SMART SCOPE)")
    question = state["question"]
    
    # --- KEYWORDS PER IL FILTRAGGIO DEL CONTENUTO ---
    KEYWORDS = ["DIEM", "DIPARTIMENTO DI INGEGNERIA DELL'INFORMAZIONE", "INGEGNERIA INFORMATICA"]
    
    # --- DEFINIZIONE DEI VINCOLI DI DOMINIO STRICT (DALLE TUE LISTE) ---
    CORSI_DIEM = [
        "corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
        "corsi.unisa.it/ingegneria-informatica",
        "corsi.unisa.it/electrical-engineering-for-digital-energy",
        "corsi.unisa.it/information-Engineering-for-digital-medicine",
        "corsi.unisa.it/0650107303300001",
        "corsi.unisa.it/DOT18CK8F9",
        "corsi.unisa.it/photovoltaics",
        "corsi.unisa.it/ingegneria-informatica-magistrale", 
        "corsi.unisa.it/ingegneria-dell-informazione"       
    ]
    
    def is_valid_url(url: str) -> bool:
        clean_url = url.replace("https://", "").replace("http://", "").replace("www.", "")
        
        # 1. SALVAGUARDIA SCRAPER: Blocchiamo i file binari
        if clean_url.lower().endswith((".xls", ".xlsx", ".pdf", ".doc", ".docx")):
            return False
            
        # 2. diem e docenti sono sempre sicuri come URL (il contenuto dei docenti lo filtriamo dopo)
        if "diem.unisa.it" in clean_url or "docenti.unisa.it" in clean_url:
            return True
            
        # 3. corsi.unisa.it
        if "corsi.unisa.it" in clean_url:
            if "calendario-occupazione-spazi" in clean_url:
                return False
            if "strutture-didattiche" in clean_url:
                return True
            return any(corso in clean_url for corso in CORSI_DIEM)
            
        # 4. easycourse.unisa.it
        if "easycourse.unisa.it" in clean_url:
            if "Dipartimento_di_" in clean_url and "dellInformazione" not in clean_url:
                return False
            return True
            
        return False

    # 1. ROUTING DINAMICO POTENZIATO
    keyword_prompt = PromptTemplate.from_template(
        """Sei un motore di routing avanzato per le ricerche del dipartimento DIEM dell'Università di Salerno.
        Devi analizzare la domanda e decidere la query migliore e il SITO SPECIFICO in cui cercare.
        
        REGOLE TASSATIVE:
        1. Aggiungi SEMPRE la parola "DIEM" oppure il nome del corso di laurea (es. "Ingegneria Informatica") alla query.
        2. Se cerchi un ORARIO (easycourse), aggiungi SEMPRE "Ingegneria Informatica".
        3. Se cerchi un'AULA o l'UBICAZIONE di un laboratorio, usa SEMPRE le parole "strutture didattiche".
        
        DOMINI A DISPOSIZIONE:
        - "diem.unisa.it" : per organi, responsabili, avvisi, bandi.
        - "corsi.unisa.it" : per aule, strutture didattiche, programmi.
        - "docenti.unisa.it" : SOLO per nome e cognome di un professore.
        - "easycourse.unisa.it" : per orari delle lezioni.
        
        Rispondi ESCLUSIVAMENTE con un JSON valido con questo formato:
        {{"query": "parole chiave pulite", "site": "dominio scelto"}}
        
        Domanda: {question}
        JSON:"""
    )
    
    keyword_chain = keyword_prompt | load_llm(state.get("current_model")) | StrOutputParser()
    
    try:
        raw_output = _safe_extract_string(keyword_chain.invoke({"question": question})).strip()
        clean_json = re.sub(r"```json|```", "", raw_output).strip()
        routing_data = json.loads(clean_json)
        
        search_query = routing_data.get("query", question)
        target_site = routing_data.get("site", "unisa.it")
        
    except Exception as e:
        _log(RED, "WEB_SEARCH", f"Errore parsing JSON: {e}. Fallback su unisa.it generico.")
        search_query = f"{question} DIEM"
        target_site = "unisa.it"
        
    shielded_query = f"{search_query} site:{target_site}"
    _log(RED, "WEB_SEARCH", f"Query inviata: {shielded_query} (Target forzato: {target_site})")
    
    web_contexts = []
    source_urls = []
    
    # 2. RICERCA E FILTRAGGIO PROGRAMMATICO
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
        search_results = wrapper.results(shielded_query, max_results=25)
        
        valid_results_count = 0
        
        if search_results:
            for res in search_results:
                if valid_results_count >= 3:
                    break 
                    
                target_url = res.get("link", "")
                
                # CONTROLLO 1: URL Valido?
                if not is_valid_url(target_url):
                    _log(GRAY, "WEB_FILTER", f"Scartato URL fuori perimetro: {target_url}")
                    continue
                    
                snippet = res.get("snippet", "")
                
                # --- SCRAPING INTELLIGENTE E CONTROLLO CONTENUTO ---
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    response = requests.get(target_url, headers=headers, timeout=5)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for noise in soup(["nav", "footer", "header", "script", "style", "aside"]):
                        noise.decompose()
                        
                    main_content = soup.find(id="unisa-content") or soup.find(id="content") or soup
                    page_text = main_content.get_text(separator=' ', strip=True)
                    
                    # --- CONTROLLO 2 (NOVITA'): Filtro Contenuto per i Docenti ---
                    if "docenti.unisa.it" in target_url:
                        testo_upper = page_text.upper()
                        if not any(kw in testo_upper for kw in KEYWORDS):
                            _log(GRAY, "WEB_FILTER", f"Scartato docente NON-DIEM (dal testo): {target_url}")
                            continue # Salta al prossimo risultato senza incrementare i count

                    web_contexts.append(f"[Fonte: {target_url}]\nTesto: {page_text[:4000]}\n")
                    
                except Exception as scrape_error:
                    # Se lo scraping fallisce per un docente, è meglio scartarlo del tutto 
                    # (perché lo snippet potrebbe fregarci)
                    if "docenti.unisa.it" in target_url:
                        _log(GRAY, "WEB_FILTER", f"Scraping fallito per {target_url}, scartato per sicurezza.")
                        continue
                        
                    _log(GRAY, "WEB_SCRAPE", f"Scraping fallito per {target_url}, uso lo snippet.")
                    web_contexts.append(f"[Fonte: {target_url}]\nSnippet: {snippet}\n")
                
                # Se arriviamo qui, il risultato ha superato TUTTI i controlli!
                source_urls.append(f"Web: {target_url}")
                _log(RED, "WEB_SEARCH", f"Risultato Valido {valid_results_count+1}: {target_url}")
                valid_results_count += 1
        
        if valid_results_count == 0:
            web_contexts.append("Nessun risultato utile trovato nei siti del DIEM autorizzati.")
            source_urls.append("Nessuna fonte web trovata")
            
    except Exception as e:
        _log(RED, "WEB_SEARCH", f"Errore ricerca DuckDuckGo: {e}")
        web_contexts.append("Errore durante la connessione al motore di ricerca.")
        
    final_context = "\n---\n".join(web_contexts)
        
    _sep(GRAY, "WEB SEARCH CONTEXT (TRONCATO)")
    _log(GRAY, "WEB_CONTEXT", f"\n{final_context[:800]}...\n") 
    _sep(GRAY, "------------------")
        
    # 3. GENERAZIONE RISPOSTA FINALE
    chain = ChatPromptTemplate.from_template(WEB_GENERATE_PROMPT) | load_llm(state.get("current_model"))
    ai_message = chain.invoke({"context": final_context, "question": question})
    
    response_text = _safe_extract_string(ai_message.content)
    
    _sep(MAGENTA, "LLM WEB GENERATION")
    _log(MAGENTA, "WEB_RESPONSE", f"\n{response_text}")
    _sep(MAGENTA, "------------------")
    
    if "mi dispiace" in response_text.lower() or not response_text:
        response_text = "Mi dispiace, ma non trovo questa informazione nei documenti del DIEM né sui canali ufficiali autorizzati."

    return {
        "generation": response_text, 
        "sources": state.get("sources", []) + source_urls
    }

def route_after_generation(state: RAGState):
    generation = state.get("generation", "")
    if "[TRIGGER_WEB_SEARCH]" in generation:
        return "go_to_web"
    return "go_to_end"