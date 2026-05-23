import re
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config import device

KB_FILE = "knowledge/knowledge_base.pkl"
VS_DIR = "knowledge/vectorstores"

WEB_CHILD_SIZE = 800
WEB_CHILD_OVERLAP = 80
WEB_PARENT_SIZE = 3000
WEB_PARENT_OVERLAP = 300

PDF_CHILD_SIZE = 800
PDF_CHILD_OVERLAP = 80
PDF_PARENT_SIZE = 3000
PDF_PARENT_OVERLAP = 300

def get_embedding_model():
    """
    Inizializza e restituisce il modello di embedding HuggingFace da utilizzare per l'indicizzazione.
    """
    print("[EMB] Caricamento BAAI/bge-m3.")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", 
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64
        }
    )

def universal_markdown_cleaner(text: str) -> str:
    """
    Pulisce e normalizza il Markdown generato dagli scraper per 
    aiutare il Text Splitter a tagliare nei punti giusti.
    """
    if not text:
        return ""

    text = re.sub(r'(?<!\!)\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'(\|\s*)\n([a-zA-Z0-9\[])', r'\1\n\n\2', text)
    text = re.sub(r'\n+#+\s+', r'\n\n# ', text)
    text = re.sub(r'\n{3,}', r'\n\n', text)
    
    return text