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
    orari di ricevimento, ruolo, settore scientifico-disciplinare.
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
        altri_docs = []
        for doc in docs_univoci:
            if re.search(r'docenti\.unisa\.it/\d+(?:/home)?\)', doc.lower()):
                homepage_docs.append(doc)
            else:
                altri_docs.append(doc)
                
        docs_finali = homepage_docs + altri_docs
        
        testo_ritorno = (
            "DATI TROVATI. Leggi i frammenti qui sotto. "
            "Se l'utente chiede una info specifica (es. curriculum), estrai i dati SOLO dal frammento con l'URL corrispondente. "
            "DEVI concludere la risposta inserendo l'URL_DA_CITARE esatto.\n\n"
        )
        return testo_ritorno + "\n\n".join(docs_finali[:8])
        
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

    testo_ritorno = (
        "DATI TROVATI SUI CORSI. Leggi i frammenti qui sotto. "
        "Se l'utente chiede dei requisiti di ammissione, usa i frammenti con '/iscriversi' o '/ammissione'. "
        "Se chiede del piano di studio, usa i frammenti con '/didattica'. "
        "REGOLA VITALE: Rispondi in modo ESTREMAMENTE MIRATO. Non aggiungere info su tasse o date se l'utente ha chiesto solo il syllabus. "
        "Assicurati che ogni punto dell'elenco sia una frase compiuta. "
        "Concludi con: Fonte: [URL_DA_CITARE].\n\n"
    )
    return testo_ritorno + "\n\n".join(docs_validi[:6])

@tool
def cerca_info_diem(query: str) -> str:
    """
    Cerca informazioni generali sul DIEM: membri di commissioni, sede, laboratori, 
    aree di ricerca, strutture, eventi e documenti ufficiali.
    """
    vec_docs_web = vector_db.similarity_search(query, k=5, filter={"type": "webpage"})
    vec_docs_pdf = vector_db.similarity_search(query, k=3, filter={"type": "pdf"})
    
    docs_validi = []
    for doc in vec_docs_web + vec_docs_pdf:
        source = doc.metadata.get("source", "Fonte sconosciuta")
        if "diem.unisa.it" in source.lower() or doc.metadata.get("type") == "pdf":
            docs_validi.append(f"=== FRAMMENTO ===\nURL_DA_CITARE: {source}\nCONTENUTO:\n{doc.page_content}\n=================")

    if not docs_validi:
        return "Nessuna informazione istituzionale trovata sul DIEM per questa richiesta."

    testo_ritorno = (
        "DATI DI DIPARTIMENTO TROVATI. "
        "ATTENZIONE TABELLE: Se i dati sono in una tabella (es. membri commissione), leggi riga per riga e riporta i nomi esatti. "
        "COERENZA ESTREMA: Se l'utente chiede i membri, elenca Nomi, Cognomi e Ruoli. Non inventare nomi generici. "
        "Concludi con: Fonte: [URL_DA_CITARE].\n\n"
    )
    return testo_ritorno + "\n\n".join(docs_validi[:6])

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

    testo_ritorno = (
        "DATI INTERNAZIONALIZZAZIONE. "
        "Distingui accuratamente tra istruzioni per studenti in uscita (Erasmus Outgoing) e studenti stranieri (Incoming). "
        "Sii mirato: se chiedono i referenti, elenca solo i nomi e i contatti senza spiegare come fare domanda. "
        "Controlla che i punti elenco non siano troncati. "
        "Concludi con: Fonte: [URL_DA_CITARE].\n\n"
    )
    return testo_ritorno + "\n\n".join(docs_validi)

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

    testo_ritorno = (
        "DATI REGOLAMENTO LAUREA TROVATI. "
        "Estrai con precisione la formula di calcolo, i punti bonus previsti per la tesi e eventuali premi carriera. "
        "CONTROLLO LOGICO: Assicurati di non troncare le formule matematiche. "
        "Se trovi informazioni contrastanti tra diversi anni accademici, specifica a quale coorte si riferiscono. "
        "Concludi con: Fonte: [URL_DA_CITARE].\n\n"
    )
    return testo_ritorno + "\n\n".join(docs_validi[:5])


@tool
def cerca_strutture(nome_struttura: str) -> str:
    """
    Cerca la posizione logistica, l'edificio, il piano o la stanza di aule, 
    laboratori, uffici docenti, biblioteche e segreterie del DIEM.
    """
    query_espansa = f"posizione ubicazione stanza edificio piano {nome_struttura}"
    
    risultati = vector_db.similarity_search(query_espansa, k=15)  # fetch più ampio per compensare il filtro
    
    docs_validi = []
    for doc in risultati:
        source = doc.metadata.get("source", doc.metadata.get("local_file", "Fonte sconosciuta"))
        
        if "diem.unisa.it" not in source.lower():
            continue
            
        docs_validi.append(f"=== FRAMMENTO ===\nURL_DA_CITARE: {source}\nCONTENUTO:\n{doc.page_content}\n=================")

    if not docs_validi:
        return f"Non ho trovato informazioni logistiche sulla struttura '{nome_struttura}' nel portale del DIEM."

    testo_ritorno = (
        "DATI LOGISTICI TROVATI. Leggi i frammenti qui sotto. "
        "COERENZA ESTREMA: Se l'utente chiede dove si trova un'aula, rispondi SOLO con edificio, piano e stanza. "
        "Se il frammento contiene indicazioni su come arrivare o orari di apertura della struttura, includili. "
        "Assicurati che i punti elenco siano frasi di senso compiuto. "
        "Concludi con: Fonte: [URL_DA_CITARE].\n\n"
    )
    return testo_ritorno + "\n\n".join(docs_validi[:6])

TOOLS = [
    cerca_docente,
    cerca_corso,
    cerca_info_diem,
    cerca_internazionale,
    cerca_regolamento_voto_laurea,
    cerca_strutture,
]

SYSTEM_PROMPT = """Sei l'assistente ufficiale del DIEM dell'Università di Salerno.

Regole FONDAMENTALI (da rispettare rigorosamente):
1. ZERO INVENZIONI: Non usare mai la tua memoria interna. Se un tool restituisce un errore, dichiara che non hai le informazioni e ASSOLUTAMENTE NON INVENTARE LINK.
2. FILTRO URL (FONDAMENTALE): Riceverai vari frammenti di testo separati da "===" con relativi URL. 
   - Se l'utente chiede il "curriculum", leggi e riassumi ESCLUSIVAMENTE il frammento che contiene "/curriculum" nell'URL. Ignora completamente il frammento "/home" (quindi niente stanze, telefoni o orari).
   - Se l'utente chiede "orari" o contatti, riassumi ESCLUSIVAMENTE il frammento "/home".
   - Se l'utente fa una domanda generale ("Chi è?"), unisci le informazioni.
3. CITAZIONE OBBLIGATORIA: È un obbligo assoluto concludere la tua risposta con: "Fonte: [URL]". Sostituisci URL con l'URL_DA_CITARE del frammento che hai effettivamente letto.
4. STRUTTURA E CONTROLLO LOGICO: Usa un paragrafo discorsivo iniziale e poi elenchi puntati. DEVI assicurarti che ogni punto dell'elenco sia una frase di senso compiuto. Non stampare MAI frasi troncate o monche.
5. VIETATO ARRENDERSI: È SEVERAMENTE VIETATO dire "Non ho trovato informazioni" se il tool ti ha passato del testo utile a rispondere. Rispondi in italiano.
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