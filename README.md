# 🎓 DIEMbot - Assistente Virtuale Advanced RAG

**DIEMbot** è un assistente virtuale basato su Intelligenza Artificiale progettato per il DIEM (Dipartimento di Ingegneria dell'Informazione, Elettrica e Matematica applicata) dell'Università di Salerno. 

Il sistema sfrutta un'architettura **Advanced RAG** (Retrieval-Augmented Generation) orchestrata tramite un **Agentic Workflow** (LangGraph), permettendo agli studenti di interrogare in linguaggio naturale orari, esami, bandi, regolamenti e informazioni sui docenti, estraendo dati da database vettoriali locali e dai portali ufficiali dell'Ateneo.

---

## 🚀 Caratteristiche Principali

- **Agentic Workflow (LangGraph):** Macchina a stati deterministica che instrada la domanda attraverso nodi di riscrittura (Condense), guardrail di sicurezza (Domain Guard), generazione e ricerca web di fallback.
- **Hybrid Retrieval Avanzato:** Combina ricerca semantica densa (`FAISS` con strategia *Parent-Child Document Retriever*) e ricerca statistica sparsa (`BM25`), unendo i risultati con deduplicazione lineare.
- **Cross-Encoder Reranking:** Applica il modello `BAAI/bge-reranker-v2-m3` per riordinare e massimizzare la pertinenza dei documenti recuperati.
- **Model Fallback Chain:** Gestione robusta degli errori (es. *Rate Limits*). Se il modello Cloud principale fallisce, scala progressivamente su altre versioni o passa in locale su modelli `Ollama` (Mistral-Nemo, Llama 3.1).
- **Web Search on-the-fly:** Se il database locale non contiene la risposta, l'agente esegue autonomamente *Dorking* mirato sui domini `unisa.it` tramite DuckDuckGo, effettuando scraping e pulizia DOM in tempo reale.
- **Smart Web Crawling:** Crawler asincrono (`Crawl4AI`) dotato di difese contro *spider traps*, deduplicazione binaria dei PDF e parser personalizzati per decostruire calendari JS e tabelle HTML di esami in formato Markdown.

---

## 📂 Struttura della Repository

```text
.
├── chatbot.py                 # Entry point dell'interfaccia utente (Streamlit)
├── crawling.py                # Script di crawling asincrono e parsing PDF/HTML
├── db_ingest.py               # Script di indicizzazione (Vectorstores e BM25)
├── utils/                     # Moduli core del backend
│   ├── chatbot_utils.py       # Configurazione LLM, fallbacks e formattazione context
│   ├── config.py              # Definizione dei Prompt e variabili di Stato
│   ├── crawling_utils.py      # Logiche complesse di parsing (tabelle, calendari)
│   ├── ingest_utils.py        # Pulizia universale Markdown e modelli di Embedding
│   ├── nodes.py               # Nodi del grafo esecutivo (LangGraph)
│   ├── retriever.py           # Motore di Hybrid Retrieval e Reranking
│   └── style.py               # CSS e HTML personalizzato per l'interfaccia Streamlit
├── resources/                 # Materiale necessario per l'avvio
│   ├── knowledge.zip          # Archivio contenente i database estratti e vettorializzati
│   └── requirements.txt       # Dipendenze Python
└── .streamlit/                # Configurazione ambiente Streamlit
    ├── config.toml            # Settaggi tema e server
    └── secrets.toml           # Chiavi API (da configurare)
```

## ⚙️ Installazione e Configurazione

### 1. Clonare la repository e preparare l'ambiente

Assicurati di avere **Python 3.10+** installato. È caldamente consigliato l'uso di un ambiente virtuale (`venv` o `conda`).

```bash
# Clonazione e navigazione
git clone <url-della-tua-repo>
cd <nome-cartella-repo>

# Creazione e attivazione ambiente virtuale (Windows)
python -m venv venv
venv\Scripts\activate

# Installazione delle dipendenze
pip install -r resources/requirements.txt
```

### 2. Scompattare la Knowledge Base

Affinché il chatbot funzioni immediatamente senza dover eseguire il crawling da zero:

1. Vai nella cartella `resources/`.
2. Estrai il file `knowledge.zip`.
3. Sposta la cartella estratta `knowledge` nella root principale del progetto (allo stesso livello di `chatbot.py`).  
   La cartella dovrà contenere i file `.pkl`, lo stato JSON (`crawler_state.json`), e le sottocartelle `data` (con `pages` e `PDFs`) e `vectorstores`.

### 3. Ottenere e Configurare la Google API Key (Gemini)

Il chatbot utilizza **Google Gemini** (1.5 Flash / Flash-Lite) come LLM primario. Per utilizzarlo:

1. Visita [Google AI Studio](https://aistudio.google.com/).
2. Accedi con il tuo account Google.
3. Clicca su **"Get API key"** e genera una nuova chiave in un nuovo progetto.
4. Copia la chiave.
5. Nella root del tuo progetto, apri il file `.streamlit/secrets.toml` e incolla la chiave:

```toml
GOOGLE_API_KEY = "INCOLLA_QUI_LA_TUA_CHIAVE"
```

> ⚠️ **Attenzione:** Non condividere mai il file `secrets.toml` su repository pubbliche.

### 4. Modelli Locali (Opzionale)

Se desideri sfruttare i modelli di fallback locale (Llama 3.1 o Mistral-Nemo), installa [Ollama](https://ollama.com/) e scarica i modelli via terminale:

```bash
ollama run mistral-nemo
ollama run llama3.1
```

---

## 🚀 Avvio dell'Applicazione

Una volta configurato l'ambiente, avvia l'assistente virtuale con:

```bash
streamlit run chatbot.py
```

L'interfaccia sarà disponibile all'indirizzo `http://localhost:8501`.

---

## 🔄 Aggiornamento della Base di Conoscenza *(Solo Amministratori)*

Il sistema fornito contiene dati pre-elaborati. Per aggiornare la conoscenza con nuovi dati ufficiali:

**1. Eseguire il Crawling Incrementale**  
Scarica le nuove pagine web e PDF dai siti istituzionali. Grazie all'hashing MD5, lo script ignorerà i file non modificati.

```bash
python crawling.py
```

**2. Eseguire l'Indicizzazione (Ingestion)**  
Suddivide i file Markdown in strutture Parent-Child e rigenera gli indici vettoriali FAISS.

```bash
python db_ingest.py
```

---

> Sviluppato per supportare gli studenti del **DIEM - Università degli Studi di Salerno**. 🎓
