"""
STEP 1 — CHUNKING
Entrambi i tipi (web e pdf) sono salvati come .md dal crawler, ma con
contenuto diverso:
  - Web  → markdown strutturato (titoli, sezioni)  → MarkdownTextSplitter
  - PDF  → testo estratto piatto, senza struttura  → RecursiveCharacterTextSplitter

I PDF vengono salvati interi (pdf_docs.pkl): il chunking parent-child
viene eseguito da ParentDocumentRetriever in indexing.py.

Output: chunks/chunks_web.pkl, chunks/pdf_docs.pkl
"""

import pickle
import os
from langchain_core.documents import Document
from langchain.text_splitter import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

# --- CONFIGURAZIONE ---
KB_FILE = "knowledge_base.pkl"
OUTPUT_DIR = "chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEB_CHUNK_SIZE = 500
WEB_CHUNK_OVERLAP = 50


def load_knowledge_base(kb_file: str) -> list[Document]:
    with open(kb_file, "rb") as f:
        docs = pickle.load(f)
    print(f"[KB] Caricati {len(docs)} documenti da {kb_file}")
    return docs


def split_by_type(docs: list[Document]):
    web_docs = [d for d in docs if d.metadata.get("type") == "web"]
    pdf_docs = [d for d in docs if d.metadata.get("type") == "pdf"]
    print(f"[SPLIT] Web: {len(web_docs)} | PDF: {len(pdf_docs)}")
    return web_docs, pdf_docs


def chunk_web(web_docs: list[Document]) -> list[Document]:
    """
    Chunking per pagine web: usa MarkdownTextSplitter perché il
    crawler salva il contenuto con titoli e sezioni in markdown.
    """
    splitter = MarkdownTextSplitter(
        chunk_size=WEB_CHUNK_SIZE,
        chunk_overlap=WEB_CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(web_docs)
    # Filtra residui di navigazione o chunk troppo corti
    chunks = [c for c in chunks if len(c.page_content.strip()) > 80]
    print(f"[WEB] {len(chunks)} chunk da {len(web_docs)} pagine")
    return chunks


def save(obj, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVE] {path}")


def main():
    docs = load_knowledge_base(KB_FILE)
    web_docs, pdf_docs = split_by_type(docs)

    # Web: chunking fatto qui
    web_chunks = chunk_web(web_docs)
    save(web_chunks, "chunks_web.pkl")

    # PDF: salvati interi, il chunking parent-child avviene in indexing.py
    # tramite ParentDocumentRetriever (usa RecursiveCharacterTextSplitter
    # perché il testo estratto da PDF è piatto, senza markup)
    save(pdf_docs, "pdf_docs.pkl")

    print(f"\n{'='*40}")
    print(f"Web chunk pronti : {len(web_chunks)}")
    print(f"PDF doc salvati  : {len(pdf_docs)}")
    print(f"{'='*40}")
    print("Prossimo step: esegui indexing.py")


if __name__ == "__main__":
    main()
