import pickle
import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore

from config import device

# --- CONFIGURAZIONE ---
KB_FILE = "knowledge_base.pkl"
VS_DIR = "vectorstores"

# WEB
WEB_CHILD_SIZE = 800
WEB_CHILD_OVERLAP = 80
WEB_PARENT_SIZE = 3000
WEB_PARENT_OVERLAP = 300

# PDF
PDF_CHILD_SIZE = 800
PDF_CHILD_OVERLAP = 80
PDF_PARENT_SIZE = 3000
PDF_PARENT_OVERLAP = 300

os.makedirs(VS_DIR, exist_ok=True)

def get_embedding_model():
    print("[EMB] Caricamento BAAI/bge-m3...")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", 
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 128
        }
    )

def main():
    if not os.path.exists(KB_FILE):
        print(f"[ERRORE] File {KB_FILE} non trovato.")
        return

    with open(KB_FILE, "rb") as f:
        all_docs = pickle.load(f)

    # Filtriamo a monte i documenti Web troppo corti 
    web_docs = [d for d in all_docs if d.metadata.get("type") == "web" and len(d.page_content.strip()) > 100]
    pdf_docs = [d for d in all_docs if d.metadata.get("type") == "pdf"]

    emb = get_embedding_model()

    # ==========================================
    # 1. Indicizzazione WEB (FAISS + Parent-Child)
    # ==========================================
    print("\n[WEB] Inizio indicizzazione Parent-Child per il Web...")
    
    web_child_splitter = MarkdownTextSplitter(
        chunk_size=WEB_CHILD_SIZE, 
        chunk_overlap=WEB_CHILD_OVERLAP
    )
    web_parent_splitter = MarkdownTextSplitter(
        chunk_size=WEB_PARENT_SIZE, 
        chunk_overlap=WEB_PARENT_OVERLAP
    )

    faiss_web_path = os.path.join(VS_DIR, "faiss_web_child")
    docstore_web_path = os.path.join(VS_DIR, "docstore_web")
    
    if os.path.exists(faiss_web_path): shutil.rmtree(faiss_web_path)
    if os.path.exists(docstore_web_path): shutil.rmtree(docstore_web_path)

    # Inizializza FAISS Web con il token __dummy__
    web_child_vs = FAISS.from_texts(["__dummy__"], emb)
    web_parent_store = create_kv_docstore(LocalFileStore(docstore_web_path))

    web_retriever = ParentDocumentRetriever(
        vectorstore=web_child_vs,
        docstore=web_parent_store,
        child_splitter=web_child_splitter,
        parent_splitter=web_parent_splitter,
    )

    batch_size_web = 5
    total_web_batches = (len(web_docs) + batch_size_web - 1) // batch_size_web
    print(f"  [WEB] Trovati {len(web_docs)} documenti originali. Suddivisi in {total_web_batches} blocchi.")

    for i in range(0, len(web_docs), batch_size_web):
        batch = web_docs[i : i + batch_size_web]
        current_batch = (i // batch_size_web) + 1
        print(f"  [WEB] Elaborazione batch {current_batch}/{total_web_batches}...")
        web_retriever.add_documents(batch)

    web_child_vs.save_local(faiss_web_path)
    print("\n[WEB] Indicizzazione Parent-Child completata con successo.")

    # ==========================================
    # 2. Indicizzazione PDF (FAISS + Parent-Child)
    # ==========================================
    print("\n[PDF] Inizio indicizzazione Parent-Child per i PDF...")
    
    pdf_child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHILD_SIZE, 
        chunk_overlap=PDF_CHILD_OVERLAP,
        separators=["\n\n", ". ", "\n", " ", ""]
    )
    pdf_parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_PARENT_SIZE, 
        chunk_overlap=PDF_PARENT_OVERLAP,
        separators=["\n\n", ". ", "\n", " ", ""]
    )

    faiss_pdf_path = os.path.join(VS_DIR, "faiss_pdf_child")
    docstore_pdf_path = os.path.join(VS_DIR, "docstore_pdf")
    
    if os.path.exists(faiss_pdf_path): shutil.rmtree(faiss_pdf_path)
    if os.path.exists(docstore_pdf_path): shutil.rmtree(docstore_pdf_path)

    child_vs = FAISS.from_texts(["__dummy__"], emb)
    parent_store = create_kv_docstore(LocalFileStore(docstore_pdf_path))

    pdf_retriever = ParentDocumentRetriever(
        vectorstore=child_vs,
        docstore=parent_store,
        child_splitter=pdf_child_splitter,
        parent_splitter=pdf_parent_splitter,
    )

    batch_size_pdf = 5
    total_pdf_batches = (len(pdf_docs) + batch_size_pdf - 1) // batch_size_pdf
    print(f"  [PDF] Trovati {len(pdf_docs)} documenti originali. Suddivisi in {total_pdf_batches} blocchi.")

    for i in range(0, len(pdf_docs), batch_size_pdf):
        batch = pdf_docs[i : i + batch_size_pdf]
        current_batch = (i // batch_size_pdf) + 1
        print(f"  [PDF] Elaborazione batch {current_batch}/{total_pdf_batches}...")
        pdf_retriever.add_documents(batch)

    child_vs.save_local(faiss_pdf_path)
    print("\n[FINISH] Indicizzazione FAISS completata con successo.")

if __name__ == "__main__":
    main()