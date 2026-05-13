import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

VS_DIR = "vectorstores"
K_WEB = 15       # Candidati iniziali dal web (aumentato per migliore copertura)
K_PDF = 5        # Candidati iniziali dai PDF (aumentato per migliore copertura)
TOP_N = 5        # Documenti finali passati all'LLM (aumentato da 3 per miglior qualità)

PDF_CHILD_SIZE = 400
PDF_CHILD_OVERLAP = 40
PDF_PARENT_SIZE = 2000
PDF_PARENT_OVERLAP = 200

class HybridRetriever:
    def __init__(self):
        print("[RETRIEVER] Inizializzazione sistema FAISS-only...")
        self.emb = self._load_embedding_model()
        self.web_vs = self._load_web_vs()
        self.pdf_retriever = self._load_pdf_retriever()
        self.reranker = self._load_reranker()
        print("[RETRIEVER] Sistema pronto.")

    def _load_embedding_model(self):
        return HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})

    def _load_web_vs(self):
        print("  [VS] Caricamento FAISS Web...")
        return FAISS.load_local(
            os.path.join(VS_DIR, "faiss_web"), 
            self.emb, 
            allow_dangerous_deserialization=True # Richiesto dalle nuove versioni di Langchain
        )

    def _load_pdf_retriever(self):
        print("  [PDF] Caricamento FAISS Parent-Child...")
        child_vectorstore = FAISS.load_local(
            os.path.join(VS_DIR, "faiss_pdf_child"), 
            self.emb, 
            allow_dangerous_deserialization=True
        )
        parent_docstore = create_kv_docstore(LocalFileStore(os.path.join(VS_DIR, "docstore_pdf")))

        return ParentDocumentRetriever(
            vectorstore=child_vectorstore,
            docstore=parent_docstore,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=PDF_CHILD_SIZE, chunk_overlap=PDF_CHILD_OVERLAP),
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=PDF_PARENT_SIZE, chunk_overlap=PDF_PARENT_OVERLAP),
            search_kwargs={"k": K_PDF},
        )

    def _load_reranker(self):
        print("  [RERANKER] Caricamento mmarco-mMiniLMv2...")
        return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    def _dedup(self, docs):
        seen, out = set(), []
        for doc in docs:
            key = doc.page_content.strip()[:120]
            if key not in seen:
                seen.add(key)
                out.append(doc)
        return out

    def retrieve(self, query: str):
        web_chunks = self.web_vs.similarity_search(query, k=K_WEB)
        try:
            pdf_chunks = self.pdf_retriever.invoke(query)
        except Exception as e:
            print(f"  [PDF] Retrieval fallito: {e}")
            pdf_chunks = []

        all_candidates = self._dedup(web_chunks + pdf_chunks)
        if not all_candidates:
            return []

        pairs = [[query, d.page_content] for d in all_candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(all_candidates, scores), key=lambda x: x[1], reverse=True)

        # Filtriamo via la stringa "init" tecnica usata in fase di creazione PDF
        final_docs = [doc for doc, _ in ranked if doc.page_content != "init"]
        return final_docs[:TOP_N]