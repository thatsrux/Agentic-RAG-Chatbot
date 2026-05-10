"""
STEP 2 — INDICIZZAZIONE
Per i PDF usa ParentDocumentRetriever di LangChain:
  - child splitter: RecursiveCharacterTextSplitter (testo piatto)
  - parent splitter: RecursiveCharacterTextSplitter (chunk più grandi)
  - child embeddings → Chroma (vectorstore_pdf_child)
  - parent docs     → LocalFileStore (docstore_pdf)

Per le pagine web genera Hypothetical Questions tramite Llama 3.1 (Ollama)
e costruisce due FAISS:
  - vectorstore_hyp: domande ipotetiche
  - vectorstore_std: chunk web raw (fallback)

Prerequisito: chunking.py
Output: vectorstores/vectorstore_hyp, vectorstore_std, vectorstore_pdf_child,
        docstore_pdf/, hypothetical_questions.json
"""

import pickle
import json
import os
import time

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURAZIONE ---
CHUNKS_DIR = "chunks"
VS_DIR = "vectorstores"
HYP_FILE = "hypothetical_questions.json"
os.makedirs(VS_DIR, exist_ok=True)

OLLAMA_MODEL = "llama3.1"   # modello locale via Ollama
N_QUESTIONS = 3             # domande ipotetiche per chunk
MIN_CHUNK_LEN = 300         # chunk troppo corti vengono saltati
BATCH_SLEEP = 0.0           # pausa ogni 50 chunk (Ollama è locale, può essere bassa)

# Dimensioni chunking PDF
PDF_CHILD_SIZE = 400
PDF_CHILD_OVERLAP = 40
PDF_PARENT_SIZE = 2000
PDF_PARENT_OVERLAP = 200

# --- PROMPT HYPOTHETICAL QUESTIONS ---
HYP_PROMPT = PromptTemplate.from_template("""
Sei un assistente che aiuta a indicizzare contenuti del sito del DIEM \
(Dipartimento di Ingegneria dell'Informazione e Elettronica) \
dell'Università di Salerno.

Dato il seguente testo, genera esattamente {n} domande che uno studente, \
un docente o un visitatore potrebbe fare e a cui questo testo risponderebbe.
Le domande devono essere autonome e comprensibili senza contesto aggiuntivo.
Scrivi SOLO le domande, una per riga, senza numerazione o prefissi.
Se il testo non contiene informazioni utili rispondi con: NO_OUTPUT

Testo:
{chunk}

Domande:
""")


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    BAAI/bge-m3: multilingua IT/EN, ottimo per retrieval semantico.
    Preferibile a OllamaEmbeddings per qualità e velocità di retrieval.
    """
    print("[EMB] Caricamento BAAI/bge-m3...")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True},
    )


def load(filename: str):
    path = os.path.join(CHUNKS_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ── PDF: ParentDocumentRetriever ─────────────────────────────────────────────

def build_pdf_retriever(
    pdf_docs: list[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> ParentDocumentRetriever:
    """
    Usa ParentDocumentRetriever di LangChain per i PDF.
    - RecursiveCharacterTextSplitter perché il testo estratto da PDF
      è piatto: non ha titoli markdown su cui splittare.
    - child chunks → Chroma (retrieval preciso)
    - parent chunks → LocalFileStore (contesto ricco per l'LLM)
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHILD_SIZE,
        chunk_overlap=PDF_CHILD_OVERLAP,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_PARENT_SIZE,
        chunk_overlap=PDF_PARENT_OVERLAP,
    )

    child_vectorstore = Chroma(
        collection_name="pdf_child_chunks",
        embedding_function=embedding_model,
        persist_directory=os.path.join(VS_DIR, "vectorstore_pdf_child"),
    )
    parent_docstore = create_kv_docstore(
        LocalFileStore(os.path.join(VS_DIR, "docstore_pdf"))
    )

    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=parent_docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print(f"[PDF] Indicizzazione {len(pdf_docs)} documenti con ParentDocumentRetriever...")
    
    # Inserimento a blocchi (batching) per evitare il limite di 5461 di Chroma
    batch_size = 20  # Numero di Parent docs per batch. Modifica se necessario.
    for i in range(0, len(pdf_docs), batch_size):
        batch = pdf_docs[i : i + batch_size]
        print(f"  [PDF] Aggiunta batch {i//batch_size + 1} (da {i} a {i + len(batch) - 1})...")
        retriever.add_documents(batch)
        
    print("[PDF] Indicizzazione completata.")
    return retriever


# ── WEB: Hypothetical Questions + FAISS ──────────────────────────────────────

def generate_hypothetical_questions(
    chunks: list[Document],
    existing: dict,
    chain,
) -> tuple[list[Document], dict]:
    """
    Genera N domande ipotetiche per ogni chunk web.
    Salta chunk già processati (incrementale tramite 'existing').
    """
    hyp_docs = []
    updated = dict(existing)

    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        chunk_id = chunk.metadata.get("source", "") + "::" + content[:80]

        # Già processato: ricostruisci i Document senza chiamare l'LLM
        if chunk_id in existing:
            for q in existing[chunk_id]:
                hyp_docs.append(Document(
                    page_content=q,
                    metadata={**chunk.metadata, "original_content": content},
                ))
            continue

        if len(content) < MIN_CHUNK_LEN:
            continue

        try:
            output = chain.invoke({"chunk": content, "n": N_QUESTIONS})

            if "NO_OUTPUT" in output.strip():
                updated[chunk_id] = []
                continue

            questions = [
                q.strip() for q in output.strip().split("\n")
                if q.strip() and len(q.strip()) > 10
            ][:N_QUESTIONS]

            updated[chunk_id] = questions
            for q in questions:
                hyp_docs.append(Document(
                    page_content=q,
                    metadata={**chunk.metadata, "original_content": content},
                ))

        except Exception as e:
            print(f"  [!] Errore chunk {i}: {e}")
            continue

        if i % 50 == 0 and i > 0:
            print(f"  [HYP] {i}/{len(chunks)} chunk processati...")
            time.sleep(BATCH_SLEEP)

    return hyp_docs, updated


def build_faiss(docs: list[Document], embedding_model, name: str) -> FAISS:
    path = os.path.join(VS_DIR, name)
    print(f"[FAISS] Costruzione '{name}' con {len(docs)} doc...")
    vs = FAISS.from_documents(docs, embedding_model)
    vs.save_local(path)
    print(f"[FAISS] Salvato in {path}")
    return vs


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    web_chunks = load("chunks_web.pkl")
    pdf_docs = load("pdf_docs.pkl")

    embedding_model = get_embedding_model()

    # ── PDF: ParentDocumentRetriever ──
    build_pdf_retriever(pdf_docs, embedding_model)

    # ── WEB: Hypothetical Questions ──
    existing = {}
    if os.path.exists(HYP_FILE):
        with open(HYP_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[HYP] {len(existing)} entry già processate, salto rielaborazione.")

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    chain = HYP_PROMPT | llm | StrOutputParser()

    print(f"\n[HYP] Generazione domande ({N_QUESTIONS} per chunk) con {OLLAMA_MODEL}...")
    
    # --- NUOVA GESTIONE MULTITHREADING ---
    hyp_docs = []
    updated = dict(existing)
    
    # Filtriamo prima i chunk da elaborare per non sprecare thread
    chunks_to_process = []
    for chunk in web_chunks:
        content = chunk.page_content.strip()
        chunk_id = chunk.metadata.get("source", "") + "::" + content[:80]
        
        if chunk_id in existing:
            for q in existing[chunk_id]:
                hyp_docs.append(Document(page_content=q, metadata={**chunk.metadata, "original_content": content}))
        elif len(content) >= MIN_CHUNK_LEN:
            chunks_to_process.append((chunk_id, content, chunk))

    print(f"  [HYP] {len(chunks_to_process)} nuovi chunk da far elaborare al LLM.")

    def process_single_chunk(data):
        c_id, text, orig_chunk = data
        try:
            out = chain.invoke({"chunk": text, "n": N_QUESTIONS})
            if "NO_OUTPUT" in out:
                return c_id, [], orig_chunk, text
            
            qs = [q.strip() for q in out.strip().split("\n") if q.strip() and len(q.strip()) > 10][:N_QUESTIONS]
            return c_id, qs, orig_chunk, text
        except Exception as e:
            return c_id, None, orig_chunk, text

    # Usiamo 4 "lavoratori" in parallelo (se hai una GPU potente puoi salire a 8)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_chunk, item): item for item in chunks_to_process}
        
        completed = 0
        for future in as_completed(futures):
            c_id, qs, orig_chunk, text = future.result()
            completed += 1
            
            if qs is not None:
                updated[c_id] = qs
                for q in qs:
                    hyp_docs.append(Document(page_content=q, metadata={**orig_chunk.metadata, "original_content": text}))
            
            if completed % 20 == 0:
                print(f"  [HYP] {completed}/{len(chunks_to_process)} elaborati in parallelo...")

    # -------------------------------------

    with open(HYP_FILE, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
    print(f"[HYP] {len(hyp_docs)} domande salvate in {HYP_FILE}")

    # Vector store domande ipotetiche
    build_faiss(hyp_docs, embedding_model, "vectorstore_hyp")

    # Vector store chunk web raw (fallback)
    build_faiss(web_chunks, embedding_model, "vectorstore_std")

    print(f"\n{'='*40}")
    print("Indicizzazione completata.")
    print(f"  HypQ docs : {len(hyp_docs)}")
    print(f"  Std docs  : {len(web_chunks)}")
    print(f"  PDF docs  : {len(pdf_docs)} (ParentDocumentRetriever)")
    print(f"{'='*40}")
    print("Prossimo step: streamlit run chatbot.py")


if __name__ == "__main__":
    main()
