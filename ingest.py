#!/usr/bin/env python3
"""
Ingestion script completo: crea Parent-Child vectorstores sia per PDF che per WEB.
Genera:
  - faiss_pdf_child + docstore_pdf (chunk: 400, parent: 2000)
  - faiss_web_child + docstore_web (chunk: 350, parent: 1200)
"""

import os
import pickle
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

VS_DIR = "vectorstores"
KB_FILE = "knowledge_base.pkl"

# Parametri PDF (chunk più grandi per documenti formali)
PDF_CHILD_SIZE = 400
PDF_CHILD_OVERLAP = 40
PDF_PARENT_SIZE = 2000
PDF_PARENT_OVERLAP = 200

# Parametri Web (chunk più piccoli per info specifiche)
WEB_CHILD_SIZE = 500
WEB_CHILD_OVERLAP = 35
WEB_PARENT_SIZE = 1500
WEB_PARENT_OVERLAP = 150

def ingest_vectorstores():
    print("="*60)
    print("INGEST COMPLETO: PDF + WEB Parent-Child")
    print("="*60)

    print("\n[INIT] Caricamento modello embedding BAAI/bge-m3...")
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("[INIT] Caricamento knowledge_base.pkl...")
    with open(KB_FILE, "rb") as f:
        all_docs = pickle.load(f)

    print(f"  ✓ Caricati {len(all_docs)} documenti totali")

    # Creiamo la cartella vectorstores
    os.makedirs(VS_DIR, exist_ok=True)

    # ===== INGEST PDF =====
    print("\n" + "="*60)
    print("FASE 1: INGEST PDF")
    print("="*60)

    pdf_docs = [d for d in all_docs if d.metadata.get("type") == "pdf"]
    print(f"[PDF] Trovati {len(pdf_docs)} documenti PDF")

    if pdf_docs:
        # Splittiamo i parent documents
        print("[PDF] Splitting parent documents...")
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PDF_PARENT_SIZE,
            chunk_overlap=PDF_PARENT_OVERLAP
        )
        pdf_parent_docs = parent_splitter.split_documents(pdf_docs)
        print(f"  ✓ {len(pdf_parent_docs)} parent chunks creati")

        # Splittiamo i child documents
        print("[PDF] Splitting child documents...")
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PDF_CHILD_SIZE,
            chunk_overlap=PDF_CHILD_OVERLAP
        )
        pdf_child_docs = []
        for i, parent in enumerate(tqdm(pdf_parent_docs, desc="  Processing parents", unit="doc")):
            children = child_splitter.split_documents([parent])
            for j, child in enumerate(children):
                child.metadata["parent_id"] = f"pdf_parent_{i}"
                pdf_child_docs.append(child)

        print(f"  ✓ {len(pdf_child_docs)} child chunks creati")

        # Creiamo FAISS vectorstore per PDF child documents
        print("[PDF] Creazione FAISS vectorstore...")
        pdf_child_texts = [doc.page_content for doc in pdf_child_docs]
        print("  [Step 1/2] Embedding dei documenti...")
        pdf_embeddings = []
        batch_size = 512
        for i in tqdm(range(0, len(pdf_child_texts), batch_size), desc="  Embedding batch", unit="batch"):
            batch = pdf_child_texts[i:i+batch_size]
            batch_embs = emb.embed_documents(batch)
            pdf_embeddings.extend(batch_embs)

        print("  [Step 2/2] Creazione indice FAISS...")
        pdf_child_vs = FAISS.from_embeddings(
            text_embeddings=list(zip(pdf_child_texts, pdf_embeddings)),
            embedding=emb,
            ids=[f"pdf_child_{i}" for i in range(len(pdf_child_docs))]
        )
        pdf_child_vs.docstore._dict = {f"pdf_child_{i}": doc for i, doc in enumerate(pdf_child_docs)}
        pdf_child_vs.save_local(os.path.join(VS_DIR, "faiss_pdf_child"))
        print(f"  ✓ Salvato: {os.path.join(VS_DIR, 'faiss_pdf_child')}")

        # Creiamo docstore per PDF parent documents
        print("[PDF] Creazione docstore...")
        pdf_docstore_path = os.path.join(VS_DIR, "docstore_pdf")
        os.makedirs(pdf_docstore_path, exist_ok=True)
        pdf_docstore = create_kv_docstore(LocalFileStore(pdf_docstore_path))

        for i, parent in enumerate(tqdm(pdf_parent_docs, desc="  Populating docstore", unit="doc")):
            parent.metadata["doc_id"] = f"pdf_parent_{i}"
            pdf_docstore.mset([(f"pdf_parent_{i}", parent)])

        print(f"  ✓ Salvato: {pdf_docstore_path}")
        print(f"\n  📊 PDF Summary:")
        print(f"     - Child vectorstore: {len(pdf_child_docs)} chunks")
        print(f"     - Parent docstore: {len(pdf_parent_docs)} documents")

    else:
        print("  ⚠️  Nessun documento PDF trovato, salto fase PDF")

    # ===== INGEST WEB =====
    print("\n" + "="*60)
    print("FASE 2: INGEST WEB")
    print("="*60)

    web_docs = [d for d in all_docs if d.metadata.get("type") == "web"]
    print(f"[WEB] Trovati {len(web_docs)} documenti WEB")

    if web_docs:
        # Splittiamo i parent documents
        print("[WEB] Splitting parent documents...")
        web_parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=WEB_PARENT_SIZE,
            chunk_overlap=WEB_PARENT_OVERLAP
        )
        web_parent_docs = web_parent_splitter.split_documents(web_docs)
        print(f"  ✓ {len(web_parent_docs)} parent chunks creati")

        # Splittiamo i child documents
        print("[WEB] Splitting child documents...")
        web_child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=WEB_CHILD_SIZE,
            chunk_overlap=WEB_CHILD_OVERLAP
        )
        web_child_docs = []
        for i, parent in enumerate(tqdm(web_parent_docs, desc="  Processing parents", unit="doc")):
            children = web_child_splitter.split_documents([parent])
            for j, child in enumerate(children):
                child.metadata["parent_id"] = f"web_parent_{i}"
                web_child_docs.append(child)

        print(f"  ✓ {len(web_child_docs)} child chunks creati")

        # Creiamo FAISS vectorstore per WEB child documents
        print("[WEB] Creazione FAISS vectorstore...")
        web_child_texts = [doc.page_content for doc in web_child_docs]
        print("  [Step 1/2] Embedding dei documenti...")
        web_embeddings = []
        batch_size = 512
        for i in tqdm(range(0, len(web_child_texts), batch_size), desc="  Embedding batch", unit="batch"):
            batch = web_child_texts[i:i+batch_size]
            batch_embs = emb.embed_documents(batch)
            web_embeddings.extend(batch_embs)

        print("  [Step 2/2] Creazione indice FAISS...")
        web_child_vs = FAISS.from_embeddings(
            text_embeddings=list(zip(web_child_texts, web_embeddings)),
            embedding=emb,
            ids=[f"web_child_{i}" for i in range(len(web_child_docs))]
        )
        web_child_vs.docstore._dict = {f"web_child_{i}": doc for i, doc in enumerate(web_child_docs)}
        web_child_vs.save_local(os.path.join(VS_DIR, "faiss_web_child"))
        print(f"  ✓ Salvato: {os.path.join(VS_DIR, 'faiss_web_child')}")

        # Creiamo docstore per WEB parent documents
        print("[WEB] Creazione docstore...")
        web_docstore_path = os.path.join(VS_DIR, "docstore_web")
        os.makedirs(web_docstore_path, exist_ok=True)
        web_docstore = create_kv_docstore(LocalFileStore(web_docstore_path))

        for i, parent in enumerate(tqdm(web_parent_docs, desc="  Populating docstore", unit="doc")):
            parent.metadata["doc_id"] = f"web_parent_{i}"
            web_docstore.mset([(f"web_parent_{i}", parent)])

        print(f"  ✓ Salvato: {web_docstore_path}")
        print(f"\n  📊 WEB Summary:")
        print(f"     - Child vectorstore: {len(web_child_docs)} chunks")
        print(f"     - Parent docstore: {len(web_parent_docs)} documents")

    else:
        print("  ⚠️  Nessun documento WEB trovato, salto fase WEB")

    print("\n" + "="*60)
    print("✅ INGESTION COMPLETATA")
    print("="*60)

if __name__ == "__main__":
    ingest_vectorstores()
