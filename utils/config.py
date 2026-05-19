from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate

import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#OLLAMA_MODEL = "llama3.1"
OLLAMA_MODEL = "mistral-nemo"
MAX_RETRIES = 2

class RAGState(TypedDict):
    question: str
    chat_history: str
    context: str
    sources: List[str]
    generation: str
    doc_grade: str
    answer_grade: str
    retry_count: int
    is_in_domain: str

SYSTEM_PROMPT = """
Sei DIEMbot, l'assistente virtuale del DIEM (Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica applicata) dell'Università di Salerno.

REGOLE FONDAMENTALI:
1. Rispondi in italiano in modo professionale e cordiale.
2. Basati ESCLUSIVAMENTE sui documenti forniti nel "Contesto".
3. Se l'informazione non è presente nei documenti, DEVI rispondere ESATTAMENTE: "Mi dispiace, ma non trovo questa informazione nei documenti a mia disposizione." Non tentare di indovinare.
4. NON usare mai espressioni come "In base al documento X" o "Il documento Y dice". Rispondi in modo diretto, discorsivo e fluido, assumendo le informazioni come tue conoscenze dirette.
5. Se noti che le informazioni nel contesto sono elenchi di dati, commissioni, orari o contatti, ricostruiscili se possibile mostrandoli all'utente sotto forma di tabella Markdown pulita e ben impaginata.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Contesto:\n{context}\n\nDomanda: {question}")
])

DOMAIN_PROMPT = """Sei un classificatore di sicurezza e pertinenza per DIEMbot, l'assistente virtuale del DIEM (Università di Salerno).
Devi determinare se la domanda dell'utente riguarda il dominio accademico: Dipartimento DIEM, Università di Salerno, didattica (corsi, esami, orari, CFU, bandi), ricerca, docenti, personale, strutture (aule, laboratori) o servizi universitari.

REGOLE:
- Se la domanda include un NOME PROPRIO (es. "Chi è Mario Vento?", "Ricevimento Capuano"), classificala SEMPRE come 'si'. Nel dubbio su un nome, fai passare la richiesta.
- Se la domanda riguarda l'università, lo studio, la carriera accademica, curriculum o il DIEM, rispondi 'si'.
- Se la domanda è totalmente fuori contesto (es. ricette di cucina, sport, gossip), rispondi 'no'.

Rispondi ESCLUSIVAMENTE si o no"""

CONDENSE_PROMPT = """Sei un ottimizzatore di query di ricerca. 
L'utente sta parlando con l'assistente virtuale del DIEM. Il database contiene GIA' E SOLO documenti del DIEM.

REGOLE TASSATIVE:
1. NESSUN CONTESTO AGGIUNTIVO: È SEVERAMENTE VIETATO aggiungere frasi come "al DIEM", "del Dipartimento...", "all'Università di Salerno". Queste aggiunte danneggiano la ricerca.
2. DOMANDE GIA' COMPLETE: Se la domanda cerca un luogo, una regola o un'entità specifica (es. "Dove si trova l'aula 126?", "Come funziona il tirocinio?"), COPIALA TESTUALMENTE senza aggiungere o togliere nulla.
3. RISOLUZIONE SOGGETTI SOTTINTESI: Modifica la domanda SOLO se si riferisce a una persona o cosa nominata prima tramite pronomi o verbi senza soggetto (es. "Cosa insegna?", "Dov'è il suo studio?"). In quel caso, prendi il NOME ESATTO dall'ultima risposta e inseriscilo (es. "Cosa insegna Vincenzo Auletta?"), rispettando sempre la regola 1.
4. Restituisci ESCLUSIVAMENTE la domanda riscritta, senza saluti o spiegazioni.

Cronologia della conversazione:
{history}

Ultima domanda: {query}

Query riscritta:"""