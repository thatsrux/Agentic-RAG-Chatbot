import pickle
import os
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore

# --- CONFIGURAZIONE ---
KB_FILE = "knowledge_base.pkl"
VS_DIR = "vectorstores"

WEB_CHUNK_SIZE = 1000
WEB_CHUNK_OVERLAP = 100
PDF_CHILD_SIZE = 400
PDF_CHILD_OVERLAP = 40
PDF_PARENT_SIZE = 2000
PDF_PARENT_OVERLAP = 200

os.makedirs(VS_DIR, exist_ok=True)

def get_embedding_model():
    print("[EMB] Caricamento BAAI/bge-m3...")
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})

def main():
    if not os.path.exists(KB_FILE):
        print(f"[ERRORE] File {KB_FILE} non trovato.")
        return

    with open(KB_FILE, "rb") as f:
        all_docs = pickle.load(f)

    web_docs = [d for d in all_docs if d.metadata.get("type") == "web"]
    pdf_docs = [d for d in all_docs if d.metadata.get("type") == "pdf"]

    emb = get_embedding_model()

    # 1. Indicizzazione WEB (FAISS)
    print("\n[WEB] Inizio indicizzazione con FAISS...")
    web_splitter = MarkdownTextSplitter(chunk_size=WEB_CHUNK_SIZE, chunk_overlap=WEB_CHUNK_OVERLAP)
    web_chunks = web_splitter.split_documents(web_docs)
    web_chunks = [c for c in web_chunks if len(c.page_content.strip()) > 100]

    faiss_web_path = os.path.join(VS_DIR, "faiss_web")
    if os.path.exists(faiss_web_path): shutil.rmtree(faiss_web_path)

    # Inizializza FAISS e fa il batching per non intasare la RAM
    batch_size = 500
    total_batches = (len(web_chunks) + batch_size - 1) // batch_size
    print(f"  [WEB] Trovati {len(web_chunks)} chunk totali. Suddivisi in {total_batches} blocchi.")

    # Il primo batch serve a inizializzare l'indice FAISS vuoto
    print(f"  [WEB] Calcolo embeddings e salvataggio batch 1/{total_batches}...")
    vectorstore_web = FAISS.from_documents(web_chunks[:batch_size], emb)

    # I successivi li aggiungiamo all'indice
    for i in range(batch_size, len(web_chunks), batch_size):
        batch = web_chunks[i : i + batch_size]
        current_batch = (i // batch_size) + 1
        print(f"  [WEB] Calcolo embeddings e salvataggio batch {current_batch}/{total_batches}...")
        vectorstore_web.add_documents(batch)

    vectorstore_web.save_local(faiss_web_path)
    print(f"[WEB] Salvato FAISS Web con {len(web_chunks)} chunk.")

    # 2. Indicizzazione PDF (FAISS + Parent-Child)
    print("\n[PDF] Inizio indicizzazione Parent-Child...")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=PDF_CHILD_SIZE, chunk_overlap=PDF_CHILD_OVERLAP)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PDF_PARENT_SIZE, chunk_overlap=PDF_PARENT_OVERLAP)

    faiss_pdf_path = os.path.join(VS_DIR, "faiss_pdf_child")
    docstore_path = os.path.join(VS_DIR, "docstore_pdf")
    
    if os.path.exists(faiss_pdf_path): shutil.rmtree(faiss_pdf_path)
    if os.path.exists(docstore_path): shutil.rmtree(docstore_path)

    # Inizializza FAISS PDF con una stringa fittizia (richiesto da LangChain per FAISS vuoto)
    child_vs = FAISS.from_texts(["init"], emb)
    parent_store = create_kv_docstore(LocalFileStore(docstore_path))

    pdf_retriever = ParentDocumentRetriever(
        vectorstore=child_vs,
        docstore=parent_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    batch_size_pdf = 20
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