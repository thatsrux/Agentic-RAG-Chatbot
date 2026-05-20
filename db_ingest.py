import pickle
import os
import shutil
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from utils.config import device

# --- CONFIGURAZIONE ---
KB_FILE = "knowledge/knowledge_base.pkl"
VS_DIR = "knowledge/vectorstores"

WEB_CHILD_SIZE, WEB_CHILD_OVERLAP = 800, 80
WEB_PARENT_SIZE, WEB_PARENT_OVERLAP = 3000, 300
PDF_CHILD_SIZE, PDF_CHILD_OVERLAP = 800, 80
PDF_PARENT_SIZE, PDF_PARENT_OVERLAP = 3000, 300

os.makedirs(VS_DIR, exist_ok=True)

def get_embedding_model():
    print("[EMB] Caricamento BAAI/bge-m3...")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", 
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 128}
    )

def universal_markdown_cleaner(text: str) -> str:
    if not text: return ""
    text = re.sub(r'(?<!\!)\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'(\|\s*)\n([a-zA-Z0-9\[])', r'\1\n\n\2', text)
    text = re.sub(r'\n+#+\s+', r'\n\n# ', text)
    return re.sub(r'\n{3,}', r'\n\n', text)

def main():
    if not os.path.exists(KB_FILE):
        print(f"[ERRORE] File {KB_FILE} non trovato.")
        return

    with open(KB_FILE, "rb") as f:
        all_docs = pickle.load(f)

    web_docs, pdf_docs = [], []
    for d in all_docs:
        if d.metadata.get("type") == "web":
            if len(d.page_content.strip()) > 100:
                d.page_content = universal_markdown_cleaner(d.page_content)
                web_docs.append(d)
        elif d.metadata.get("type") == "pdf":
            pdf_docs.append(d)

    emb = get_embedding_model()

    # ==========================================
    # 1. Indicizzazione WEB
    # ==========================================
    print("\n[WEB] Indicizzazione")
    faiss_web_path, docstore_web_path = os.path.join(VS_DIR, "faiss_web_child"), os.path.join(VS_DIR, "docstore_web")
    if os.path.exists(faiss_web_path): shutil.rmtree(faiss_web_path)
    if os.path.exists(docstore_web_path): shutil.rmtree(docstore_web_path)

    web_retriever = ParentDocumentRetriever(
        vectorstore=FAISS.from_texts(["__dummy__"], emb),
        docstore=create_kv_docstore(LocalFileStore(docstore_web_path)),
        child_splitter=MarkdownTextSplitter(chunk_size=WEB_CHILD_SIZE, chunk_overlap=WEB_CHILD_OVERLAP),
        parent_splitter=MarkdownTextSplitter(chunk_size=WEB_PARENT_SIZE, chunk_overlap=WEB_PARENT_OVERLAP),
    )
    
    batch_size = 10
    for i in range(0, len(web_docs), batch_size):
        print(f"  [WEB] Batch {(i // batch_size) + 1}/{(len(web_docs) + batch_size - 1) // batch_size}...")
        web_retriever.add_documents(web_docs[i : i + batch_size])
    web_retriever.vectorstore.save_local(faiss_web_path)

    # ==========================================
    # 2. Indicizzazione PDF
    # ==========================================
    print("\n[PDF] Indicizzazione Parent-Child per i PDF...")
    faiss_pdf_path, docstore_pdf_path = os.path.join(VS_DIR, "faiss_pdf_child"), os.path.join(VS_DIR, "docstore_pdf")
    if os.path.exists(faiss_pdf_path): shutil.rmtree(faiss_pdf_path)
    if os.path.exists(docstore_pdf_path): shutil.rmtree(docstore_pdf_path)

    pdf_retriever = ParentDocumentRetriever(
        vectorstore=FAISS.from_texts(["__dummy__"], emb),
        docstore=create_kv_docstore(LocalFileStore(docstore_pdf_path)),
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=PDF_CHILD_SIZE, chunk_overlap=PDF_CHILD_OVERLAP, separators=["\n\n", ". ", "\n", " ", ""]),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=PDF_PARENT_SIZE, chunk_overlap=PDF_PARENT_OVERLAP, separators=["\n\n", ". ", "\n", " ", ""]),
    )
    
    for i in range(0, len(pdf_docs), batch_size):
        print(f"  [PDF] Batch {(i // batch_size) + 1}/{(len(pdf_docs) + batch_size - 1) // batch_size}...")
        pdf_retriever.add_documents(pdf_docs[i : i + batch_size])
    pdf_retriever.vectorstore.save_local(faiss_pdf_path)

    print("\n[FINISH] Tutte le indicizzazioni FAISS (Web e PDF) completate con successo!")

if __name__ == "__main__":
    main()