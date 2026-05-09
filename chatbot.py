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
    Accetta il nome e cognome.
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
                    docs_validi.append(f"FONTE: ({source})\n{doc}")

    parti = nome_docente.strip().split()
    varianti = [nome_docente]
    
    if len(parti) == 2:
        p1, p2 = parti[0], parti[1]
        varianti.extend([
            f"{p2} {p1}",                  # Invertito originale
            f"{p2.title()} {p1.title()}",  # Es. Alessia Saggese
            f"{p2.title()} {p1.upper()}",  # Es. Alessia SAGGESE
            f"{p1.title()} {p2.title()}",  # Es. Saggese Alessia
            f"{p1.title()} {p2.upper()}"   # Es. Saggese ALESSIA
        ])

    for variante in varianti:
        if docs_validi:
            break
        esatti = vector_db.get(where_document={"$contains": variante})
        if esatti and esatti.get("documents"):
            filtra_vero_docente(esatti["documents"], esatti["metadatas"])

    if not docs_validi:
        ris_semantici = vector_db.similarity_search(nome_docente, k=15)
        filtra_vero_docente([d.page_content for d in ris_semantici], [d.metadata for d in ris_semantici])

    if docs_validi:
        docs_univoci = list(dict.fromkeys(docs_validi))
        return "DATI TROVATI. Leggi attentamente queste informazioni, elabora una risposta discorsiva completa e poi aggiungi le fonti alla fine:\n\n" + "\n\n---\n\n".join(docs_univoci[:8])
    else:
        return f"ERRORE ASSOLUTO: Non esiste una pagina ufficiale per '{nome_docente}'. NON INVENTARE NULLA. NON INVENTARE LINK FALSI. Rispondi all'utente dicendo che non hai trovato il suo profilo nel database docenti."
    

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

Regole FONDAMENTALI (da rispettare rigorosamente):
1. ZERO INVENZIONI: Non usare mai la tua memoria interna. Se un tool restituisce un errore o non ci sono dati, dichiara che non hai le informazioni e ASSOLUTAMENTE NON INVENTARE LINK WEB FALSI (es. non usare www.unisa.it/pagina/...).2. SINTESI RICCA E DETTAGLIATA: Quando il tool ti fornisce dei dati su un docente o un corso, DEVI usare TUTTI i frammenti ricevuti per costruire un profilo testuale completo (ruolo, curriculum, progetti, orari).
3. STRUTTURA DELLA RISPOSTA: La tua risposta deve seguire questo ordine (SE POSSIBILE):
   - Paragrafo introduttivo e discorsivo (es. "Pierluigi Ritrovato è professore...").
   - Elenco dei dettagli rilevanti (usa elenchi puntati per orari, insegnamenti, o progetti, NON AGGIUNGERE UN PUNTO ALL'ELENCO SE NON COMPLETO).
   - Le fonti esatte fornite dal tool.
4. FONTI OBBLIGATORIE: Concludi SEMPRE copiando i link forniti dal tool usando il formato Markdown: (URL). Non ometterli mai e non usare altri link.
5. VIETATO ARRENDERSI: È SEVERAMENTE VIETATO dire "Non ho trovato informazioni" se il tool ti ha passato del testo, anche se parziale. Rispondi in italiano.
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