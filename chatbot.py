import os
import gradio as gr
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
            meta_info = f"[Fonte PDF: {source_url} (Trovato in: {found_in})]"
        else:
            meta_info = f"[Fonte Web: {source_url}]"
            
        formatted_results.append(f"{meta_info}\n{doc.page_content}")
        
    return "\n\n---\n\n".join(formatted_results)


@tool
def cerca_docente(nome_docente: str) -> str:
    """
    Cerca informazioni su un docente del DIEM: contatti, email,
    orari di ricevimento, ruolo, settore scientifico-disciplinare.
    Accetta il nome e cognome.
    """
    risultati_nome = search_vector(nome_docente, source_filter="docenti.unisa.it", k=6)
    
    if "Nessuna informazione" in risultati_nome:
        parti = nome_docente.split()
        if len(parti) == 2:
            nome_invertito = f"{parti[1]} {parti[0]}"
            return search_vector(nome_invertito, source_filter="docenti.unisa.it", k=6)
            
    return risultati_nome

@tool
def cerca_corso(query: str) -> str:
    """
    Cerca informazioni su corsi di laurea, piani di studio, syllabus,
    requisiti di ammissione, punteggi TOLC e immatricolazione al DIEM.
    """
    return search_vector(query, source_filter="corsi.unisa.it", k=5)

@tool
def cerca_info_diem(query: str) -> str:
    """
    Cerca informazioni generali sul DIEM: sede, laboratori, dotazioni, aree di ricerca,
    commissioni, strutture, eventi, bandi e documenti ufficiali.
    """
    seen = set()
    docs = []
    
    vec_docs_web = vector_db.similarity_search(
        query, k=4,
        filter={"source": {"$contains": "diem.unisa.it"}}
    )
    
    vec_docs_pdf = vector_db.similarity_search(
        query, k=2,
        filter={"type": "pdf"}
    )
    
    for doc in vec_docs_web + vec_docs_pdf:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            source_url = doc.metadata.get("source", "Fonte sconosciuta")
            docs.append(f"[Fonte: {source_url}]\n{doc.page_content}")
            
    if not docs:
        return "Nessuna informazione trovata nel database."
        
    return "\n\n---\n\n".join(docs[:6])

@tool
def cerca_internazionale(query: str) -> str:
    """
    Cerca informazioni su mobilità internazionale, programmi Erasmus+,
    accordi con università straniere e referenti per l'internazionalizzazione.
    """
    return search_vector(f"internazionale erasmus mobilità {query}", k=5)

@tool
def calcola_voto_laurea(media_ponderata: float, bonus_tesi: float = 0.0) -> str:
    """
    Calcola il voto di laurea dalla media ponderata degli esami (in trentesimi).
    Parametri:
      - media_ponderata: media pesata degli esami (es. 27.5)
      - bonus_tesi: punti aggiuntivi della tesi, default 0 (max tipico: 7)
    """
    base = round((media_ponderata / 30) * 110, 2)
    finale = min(base + bonus_tesi, 110)
    lode = " cum laude" if finale >= 110 else ""
    return (
        f"Media in 30esimi:       {media_ponderata}\n"
        f"Conversione in 110esimi: {base:.1f}\n"
        f"Bonus tesi:             +{bonus_tesi}\n"
        f"Voto finale:            {finale:.0f}{lode}"
    )

TOOLS = [
    cerca_docente,
    cerca_corso,
    cerca_info_diem,
    cerca_internazionale,
    calcola_voto_laurea,
]

SYSTEM_PROMPT = """Sei l'assistente ufficiale del DIEM dell'Università di Salerno.

Regole FONDAMENTALI:
1. DEVI SEMPRE USARE I TOOL per cercare le informazioni prima di rispondere. Non usare la tua memoria interna per dati sui corsi o sui docenti.
2. Quando il tool ti restituisce delle informazioni, esse includeranno la fonte [Fonte: URL]. MENTZIONA SEMPRE IL LINK della fonte nella tua risposta finale per permettere all'utente di approfondire.
3. Se un tool non restituisce dati utili, dichiaralo chiaramente all'utente senza inventare nulla.
4. Usa la formattazione Markdown (elenchi, grassetti, tabelle) per rendere la lettura chiara e pulita.
5. Mantieni un tono istituzionale ma cordiale. Rispondi in italiano.
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
    chatbot.launch(share=False)