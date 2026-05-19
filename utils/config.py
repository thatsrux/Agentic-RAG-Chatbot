from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate

import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

OLLAMA_MODEL = "llama3.1"
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

REWRITE_PROMPT = """Sei un estrattore di parole chiave. Trasforma questa domanda in una stringa di ricerca per un database.
Togli i convenevoli. Togli le parole inutili. Mantieni solo nomi, concetti e numeri fondamentali.
NON rispondere all'utente. Rispondi ESCLUSIVAMENTE con le parole chiave estratte."""

CONDENSE_PROMPT = """Sei un modulo di elaborazione testuale. Il tuo unico compito è rendere l'Ultima Domanda indipendente, sostituendo i pronomi con il nome corretto preso dalla Cronologia.

REGOLE TASSATIVE:
1. Se l'Ultima Domanda contiene pronomi ("suo", "sua", "lo", "la") o omette il soggetto, INSERISCI il soggetto preso dalla Cronologia.
2. NON CANCELLARE L'ARGOMENTO: Se l'Ultima Domanda chiede il "curriculum" o gli "orari", la tua risposta DEVE contenere la parola "curriculum" o "orari". Non trasformare la frase in un semplice "Chi è [Nome]?".
3. Se l'Ultima Domanda è già chiara, contiene un soggetto esplicito, o cambia argomento (es. "Chi è il direttore?"), copiala e incollala IDENTICA, senza usare la cronologia.
4. NON INVENTARE NOMI. Se devi aggiungere un soggetto utilizza SOLTATNO i nomi presenti nella Cronologia.

ESEMPI:
Cronologia: "Chi è il Dottor Rossi?"
Ultima Domanda: "Qual è il suo curriculum?"
Risultato: Qual è il curriculum del Dottor Rossi?

Cronologia: "Dove si trova l'Aula A?"
Ultima Domanda: "Quanti posti ha?"
Risultato: Quanti posti ha l'Aula A?

Cronologia: "Chi è il Dottor Rossi?"
Ultima Domanda: "Chi è il direttore del DIEM?"
Risultato: Chi è il direttore del DIEM?

Rispondi ESCLUSIVAMENTE con il testo della domanda finale, senza altre parole."""

DOC_GRADER_PROMPT = """Il seguente 'Contesto' contiene informazioni utili per rispondere alla 'Domanda'?
Rispondi ESCLUSIVAMENTE con la parola "SI" oppure "NO". Niente altro."""

ANSWER_GRADER_PROMPT = """La seguente 'Generazione' risponde in modo sensato alla 'Domanda'?
Rispondi ESCLUSIVAMENTE con la parola "SI" oppure "NO". Niente altro."""