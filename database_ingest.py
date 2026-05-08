import os
import glob
import yaml
from uuid import uuid4
from tqdm import tqdm
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = r"data"
PAGES_DIR = os.path.join(DATA_PATH, "pages")
PDFS_DIR = os.path.join(DATA_PATH, "PDFs")
CHROMA_PATH = r"chroma_db"

def load_and_parse_markdown(directory):
    """
    Legge i file .md nella cartella, estrae i metadati (frontmatter YAML) 
    e restituisce una lista di oggetti Document di LangChain.
    """
    documents = []
    filepaths = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)
    
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        metadata = {"local_file": filepath}
        clean_text = content
        
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                    if isinstance(frontmatter, dict):
                        metadata.update(frontmatter)
                    clean_text = parts[2].strip()
                except yaml.YAMLError:
                    pass
                    
        if len(clean_text) > 10:
            documents.append(Document(page_content=clean_text, metadata=metadata))
            
    return documents

def main():
    print("📂 1. Lettura dei file Markdown in corso...")
    
    web_docs = load_and_parse_markdown(PAGES_DIR)
    pdf_docs = load_and_parse_markdown(PDFS_DIR)
    all_documents = web_docs + pdf_docs
    
    if not all_documents:
        print("❌ Nessun documento Markdown trovato. Hai eseguito crawling.py?")
        return

    print(f"✅ Trovati {len(web_docs)} pagine web e {len(pdf_docs)} PDF.")
    print("\n🔎 Anteprima Metadati dal primo documento:")
    print(all_documents[0].metadata)
    
    print("\n✂️ 2. Creazione dei chunk in corso...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(all_documents)
    print(f"✅ Generati {len(chunks)} frammenti di testo (chunk).")

    uuids = [str(uuid4()) for _ in range(len(chunks))]

    print("\n🧠 3. Inizializzazione modello di Embedding e ChromaDB...")

    embeddings_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    vector_store = Chroma(
        collection_name="unisa_collection",
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH,
    )

    print("\n💾 4. Ingestione nel Database...")
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Salvataggio su DB", unit="batch"):
        batch_chunks = chunks[i:i + batch_size]
        batch_uuids = uuids[i:i + batch_size]
        vector_store.add_documents(documents=batch_chunks, ids=batch_uuids)
        
    print("\n🎉 INGESTIONE COMPLETATA CON SUCCESSO!")
    print(f"Il tuo database è pronto all'uso nella cartella: {CHROMA_PATH}")

if __name__ == "__main__":
    main()