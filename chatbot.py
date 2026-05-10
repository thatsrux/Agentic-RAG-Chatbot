import os
import gradio as gr
import re
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_classic.agents.agent import AgentExecutor

load_dotenv()

CHROMA_PATH = "chroma_db"

print("⏳ Caricamento modello embedding...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

llm = ChatOllama(model="llama3.1", temperature=0)

print("⏳ Connessione al database Chroma in corso...")
vector_db = Chroma(
    collection_name="unisa_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

#print("=" * 20 + " PROVA " + "=" * 20)
#print(vector_db.get(where_document = {"$contains": "Mario Vento"}, limit=1))



def search_vector(query: str, source_filter: str = None, k: int = 5) -> str:
    """Ricerca vettoriale con filtro manuale in Python (più sicuro dei metadati Chroma)."""
    
    fetch_k = (k * 3) if source_filter else k
    results = vector_db.similarity_search(query, k=fetch_k)
    
    if source_filter:
        filtered_results = []
        for doc in results:
            meta_source = doc.metadata.get("source", "")
            meta_file = doc.metadata.get("local_file", "")
            
            if source_filter in meta_source or source_filter in meta_file:
                filtered_results.append(doc)
                
        results = filtered_results[:k]
        
    if not results:
        return f"Nessuna informazione trovata nel database per la query: {query}"
        
    formatted_results = []
    for doc in results:
        source_url = doc.metadata.get("source", doc.metadata.get("local_file", "Fonte sconosciuta"))
        found_in = doc.metadata.get("found_in", "")
        
        if "pdf" in doc.metadata.get("type", "") and found_in:
            meta_info = f"[Fonte PDF]({source_url}) (Trovato in: {found_in})"
        else:
            meta_info = f"[Fonte Web]({source_url})"
            
        formatted_results.append(f"{meta_info}\n{doc.page_content}")
        
    return "\n\n---\n\n".join(formatted_results)


@tool
def cerca_docente(nome_docente: str) -> str:
    """
    Cerca informazioni su un docente del DIEM: contatti, email,
    orari di ricevimento, ruolo, settore scientifico-disciplinare e curriculum.
    Accetta il nome e cognome o solo il cognome.
    """
    parti_nome_lower = [p.lower() for p in nome_docente.strip().split()]
    docs_validi = []
    
    def filtra_vero_docente(docs, metas):
        for doc, meta in zip(docs, metas):
            source = meta.get("source", meta.get("local_file", "Fonte sconosciuta"))
            
            if "docenti.unisa.it" in source:
                titolo_h1 = ""
                for riga in doc.split('\n')[:40]:
                    if riga.strip().startswith('# '):
                        titolo_h1 = riga.lower()
                        break
                
                if titolo_h1 and all(parte in titolo_h1 for parte in parti_nome_lower):
                    docs_validi.append(f"=== FRAMMENTO ===\nURL_DA_CITARE: {source}\nCONTENUTO:\n{doc}\n=================")

    parti = nome_docente.strip().split()
    varianti = [nome_docente]
    
    if len(parti) == 2:
        p1, p2 = parti[0], parti[1]
        varianti.extend([
            f"{p2} {p1}",
            f"{p2.title()} {p1.title()}",
            f"{p2.title()} {p1.upper()}",
            f"{p1.title()} {p2.title()}",
            f"{p1.title()} {p2.upper()}"
        ])
    elif len(parti) == 1:
        p1 = parti[0]
        varianti.extend([
            p1.title(),
            p1.upper(),
            p1.lower()
        ])

    varianti = list(set(varianti))

    for variante in varianti:
        esatti = vector_db.get(where_document={"$contains": variante})
        if esatti and esatti.get("documents"):
            filtra_vero_docente(esatti["documents"], esatti["metadatas"])

    if not docs_validi:
        ris_semantici = vector_db.similarity_search(nome_docente, k=15)
        filtra_vero_docente([d.page_content for d in ris_semantici], [d.metadata for d in ris_semantici])

    if docs_validi:
        docs_univoci = list(dict.fromkeys(docs_validi))
        
        homepage_docs = []
        curriculum_docs = []
        altri_docs = []
        
        for doc in docs_univoci:
            url_match = re.search(r'URL_DA_CITARE:\s*(https?://[^\s]+)', doc)
            url = url_match.group(1).lower() if url_match else ""
            
            if "/curriculum" in url:
                curriculum_docs.append(doc)
            elif re.search(r'docenti\.unisa\.it/\d+(?:/home|/)?$', url):
                homepage_docs.append(doc)
            else:
                altri_docs.append(doc)
                
        docs_finali = curriculum_docs + homepage_docs + altri_docs
        
        # Elimina testo_ritorno e reminder_finale attuali. Sostituisci il return con:
        istruzione = "Usa questi frammenti ufficiali per rispondere alla domanda. Non inventare nulla e cita l'URL_DA_CITARE.\n\n"
        return istruzione + "\n\n".join(docs_finali[:8])
        
    else:
        return f"ERRORE ASSOLUTO: Non esiste una pagina ufficiale per '{nome_docente}'. NON INVENTARE NULLA. NON INVENTARE LINK FALSI. Rispondi all'utente dicendo che non hai trovato il suo profilo nel database docenti."

@tool
def cerca_corso(query: str) -> str:
    """
    Cerca informazioni su corsi di laurea, piani di studio, syllabus,
    requisiti di ammissione, punteggi TOLC e immatricolazione al DIEM.
    """
    # Utilizziamo una ricerca espansa per catturare i vari moduli del corso
    risultati_raw = vector_db.similarity_search(query, k=10)
    
    docs_validi = []
    for doc in risultati_raw:
        source = doc.metadata.get("source", doc.metadata.get("local_file", ""))
        # Filtriamo per assicurarci che provenga dal portale corsi
        if "corsi.unisa.it" in source.lower():
            docs_validi.append(f"=== FRAMMENTO ===\nURL_DA_CITARE: {source}\nCONTENUTO:\n{doc.page_content}\n=================")

    if not docs_validi:
        return f"Nessuna informazione ufficiale trovata per la query: {query}"

    # Sostituisci il testo_ritorno e il return con:
    istruzione = "Usa questi frammenti ufficiali per rispondere alla domanda sui corsi. Non inventare nulla e cita l'URL_DA_CITARE.\n\n"
    return istruzione + "\n\n".join(docs_validi[:6])

@tool
def cerca_info_diem(query: str) -> str:
    """
    Cerca informazioni istituzionali sul DIEM: membri di commissioni, 
    aree di ricerca, organi collegiali, eventi, bandi e documenti ufficiali.
    NON USARE questo tool per cercare aule, laboratori o posizioni logistiche.
    """
    # Usiamo k alti per superare i menu, ma poi filtreremo drasticamente!
    vec_docs_web = vector_db.similarity_search(query, k=20, filter={"type": "webpage"})
    vec_docs_pdf = vector_db.similarity_search(query, k=5, filter={"type": "pdf"})
    
    docs_validi = []
    contenuti_visti = set()
    
    # 1. Recupero, filtraggio e deduplicazione
    for doc in vec_docs_web + vec_docs_pdf:
        source = doc.metadata.get("source", "Fonte sconosciuta")
        contenuto = doc.page_content.strip()
        
        if "diem.unisa.it" in source.lower() or doc.metadata.get("type") == "pdf":
            if contenuto in contenuti_visti:
                continue
            contenuti_visti.add(contenuto)
            
            # Usiamo un DIZIONARIO per poter elaborare facilmente i dati dopo
            docs_validi.append({
                "url": source,
                "text": contenuto
            })

    if not docs_validi:
        return "Nessuna informazione istituzionale trovata sul DIEM."

    # 2. LOGICA DI CLASSIFICAZIONE (Ranking Intelligente)
    pagine_prioritarie = []
    altre_pagine = []
    
    for item in docs_validi:
        txt = item["text"].lower()
        url = item["url"].lower()
        
        # VIP PASS: Se è la pagina della commissione e contiene parole della tabella, va al PRIMO posto!
        if "commissione-paritetica" in url and ("componente" in txt or "presidente" in txt):
            pagine_prioritarie.insert(0, item) 
        # Altre pagine della sezione dipartimento hanno priorità secondaria
        elif "dipartimento/" in url:
            pagine_prioritarie.append(item)
        else:
            altre_pagine.append(item)
            
    risultati_ordinati = pagine_prioritarie + altre_pagine

    # 3. Formattazione finale per Llama 3.1
    formatted_docs = []
    for item in risultati_ordinati:
        formatted_docs.append(f"=== FRAMMENTO ===\nURL_DA_CITARE: {item['url']}\nCONTENUTO:\n{item['text']}\n=================")

    # Sostituisci il testo_ritorno e il return con:
    istruzione = "Usa questi frammenti istituzionali per rispondere. Se ci sono tabelle, elenca chiaramente i dati. Cita l'URL_DA_CITARE.\n\n"
    return istruzione + "\n\n".join(formatted_docs[:5])

@tool
def cerca_internazionale(query: str) -> str:
    """
    Cerca informazioni su mobilità internazionale, programmi Erasmus+,
    accordi con università straniere e referenti per l'internazionalizzazione.
    """
    query_espansa = f"internazionale erasmus mobilità exchange {query}"
    risultati = vector_db.similarity_search(query_espansa, k=6)
    
    docs_validi = []
    for doc in risultati:
        source = doc.metadata.get("source", "")
        docs_validi.append(f"=== FRAMMENTO ===\nURL_DA_CITARE: {source}\nCONTENUTO:\n{doc.page_content}\n=================")

    istruzione = "Usa questi frammenti sull'internazionalizzazione per rispondere. Cita l'URL_DA_CITARE.\n\n"
    return istruzione + "\n\n".join(docs_validi)

@tool
def cerca_regolamento_voto_laurea(corso_di_laurea: str) -> str:
    """
    Cerca il regolamento didattico specifico per calcolare il voto di laurea, 
    inclusi bonus tesi, punti per carriera e criteri per la lode.
    """
    query = f"regolamento calcolo voto laurea bonus tesi lode {corso_di_laurea}"
    
    # Cerchiamo specificamente tra i PDF dei regolamenti e le pagine didattica
    risultati = vector_db.similarity_search(query, k=8)
    
    docs_validi = []
    for doc in risultati:
        source = doc.metadata.get("source", doc.metadata.get("local_file", ""))
        # Priorità ai PDF di regolamento e sezioni didattica/regolamenti
        if "regolamenti" in source.lower() or doc.metadata.get("type") == "pdf":
            docs_validi.append(f"=== FRAMMENTO ===\nURL_DA_CITARE: {source}\nCONTENUTO:\n{doc.page_content}\n=================")

    if not docs_validi:
        return f"Non ho trovato il regolamento specifico per il calcolo del voto di laurea di: {corso_di_laurea}."

    istruzione = "Usa questi frammenti per spiegare il regolamento di laurea. Riporta le formule con precisione. Cita l'URL_DA_CITARE.\n\n"
    return istruzione + "\n\n".join(docs_validi[:5])


@tool
def cerca_strutture(nome_struttura: str) -> str:
    """
    Cerca la posizione logistica, l'edificio, il piano o la stanza di aule, 
    laboratori, uffici docenti, biblioteche e segreterie del DIEM. 
    Restituisce anche informazioni su capienza, tipologia e attrezzature.
    """
    docs_validi = []
    
    nome_pulito = nome_struttura.strip()

    if len(nome_pulito) <= 3 and not any(kw in nome_pulito.lower() for kw in ["aula", "lab", "sala"]):
        termine_ricerca = f"Aula {nome_pulito.upper()}"
    else:
        termine_ricerca = nome_pulito
    
    varianti = [
        termine_ricerca,
        termine_ricerca.lower(),
        termine_ricerca.title(),
        termine_ricerca.upper()
    ]
    varianti = list(set(varianti))
    
    for variante in varianti:
            
        esatti = vector_db.get(where_document={"$contains": variante})
        if esatti and esatti.get("documents"):
            for doc_content, meta in zip(esatti["documents"], esatti["metadatas"]):
                source = meta.get("source", meta.get("local_file", "Fonte sconosciuta")).lower()
                doc_type = meta.get("type", "")
                
                if "diem.unisa.it" in source or "corsi.unisa.it" in source or doc_type == "pdf" or "pdf" in source:
                    url_da_citare = meta.get("source", meta.get("local_file", "Fonte sconosciuta"))
                    docs_validi.append(f"=== FRAMMENTO ===\nURL_DA_CITARE: {url_da_citare}\nCONTENUTO:\n{doc_content}\n=================")

    # --- 2. FALLBACK SEMANTICO ---
    if not docs_validi:
        query_espansa = f"posizione ubicazione stanza edificio piano aula laboratorio strutture-didattiche {nome_struttura}"
        risultati = vector_db.similarity_search(query_espansa, k=10)
        
        for doc in risultati:
            source = doc.metadata.get("source", doc.metadata.get("local_file", "Fonte sconosciuta")).lower()
            doc_type = doc.metadata.get("type", "")
            
            if "diem.unisa.it" in source or "corsi.unisa.it" in source or doc_type == "pdf" or "pdf" in source:
                url_da_citare = doc.metadata.get("source", doc.metadata.get("local_file", "Fonte sconosciuta"))
                docs_validi.append(f"=== FRAMMENTO ===\nURL_DA_CITARE: {url_da_citare}\nCONTENUTO:\n{doc.page_content}\n=================")

    # Rimuoviamo duplicati mantenendo l'ordine
    docs_validi = list(dict.fromkeys(docs_validi))

    if not docs_validi:
        return f"Non ho trovato informazioni logistiche sulla struttura '{nome_struttura}' nei database del dipartimento o dei corsi."

    # Sostituisci l'istruzione attuale con questa versione calibrata:
    istruzione = (
        "Usa questi frammenti per indicare la posizione e le caratteristiche della struttura cercata. "
        "ATTENZIONE: Se vedi lunghi elenchi di aule e laboratori, ignorali (sono menu del sito). Estrai i dati ESCLUSIVAMENTE per la struttura richiesta dall'utente, senza associarla alle altre presenti nella lista. "
        "Cita l'URL_DA_CITARE.\n\n"
    )
    return istruzione + "\n\n".join(docs_validi[:3])

TOOLS = [
    cerca_docente,
    cerca_corso,
    cerca_info_diem,
    cerca_internazionale,
    cerca_regolamento_voto_laurea,
    cerca_strutture,
]

SYSTEM_PROMPT = """Sei l'assistente ufficiale del DIEM dell'Università degli Studi di Salerno (UNISA).
Il tuo compito è rispondere in modo chiaro e diretto alle domande degli utenti, basandoti ESCLUSIVAMENTE sui frammenti di testo forniti dai tool.

REGOLE DI BASE:
1. NON INVENTARE: Non usare mai la tua conoscenza pregressa. Se un'informazione non è nei frammenti, non scriverla (niente atenei, email, telefoni o premi inventati).
2. STILE NATURALE: Rispondi direttamente. Non spiegare cosa stai facendo, non usare meta-frasi come "L'utente ha chiesto" o "Ecco i dati estratti".
3. FONTE: Concludi sempre la tua risposta con "Fonte: [URL_DA_CITARE]", usando l'URL presente nel frammento che hai utilizzato.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, TOOLS, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)


def to_langchain_history(gradio_history: list) -> list:
    """Converte la history di Gradio nel formato LangChain."""
    history = []
    if not gradio_history:
        return history

    if isinstance(gradio_history[0], dict):
        for msg in gradio_history:
            content = msg.get("content") or ""
            if not content: continue
            if msg["role"] == "user":
                history.append(HumanMessage(content=content))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=content))
    else:
        for user_msg, ai_msg in gradio_history:
            if user_msg: history.append(HumanMessage(content=user_msg))
            if ai_msg: history.append(AIMessage(content=ai_msg))

    return history

def respond(message: str, history: list):
    """Funzione di callback per Gradio."""
    result = agent_executor.invoke({
        "input": message,
        "chat_history": to_langchain_history(history),
    })
    
    yield result["output"]

chatbot = gr.ChatInterface(
    fn=respond,
    title="Assistente DIEM — Unisa 🎓",
    description="Chiedi informazioni su docenti, corsi, strutture o calcola il tuo voto di laurea. L'assistente cercherà i dati direttamente dal portale di Ateneo.",
    autoscroll=True,
)

if __name__ == "__main__":
    print("🚀 Avvio chatbot DIEM. Vai sul link locale fornito da Gradio!")
    chatbot.launch(share=True)