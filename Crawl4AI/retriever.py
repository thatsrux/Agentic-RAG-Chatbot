"""
STEP 3 — RETRIEVAL IBRIDO + RERANKING
Pipeline:
  1. FAISS vectorstore_hyp  → top-K dalle domande ipotetiche (web)
  2. FAISS vectorstore_std  → top-K dai chunk web raw (fallback)
  3. Chroma + LocalFileStore (ParentDocumentRetriever) → top-K PDF parent
  4. Merge e deduplicazione
  5. Cross-encoder reranking → top-N finali

Prerequisito: indexing.py
"""

import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# --- CONFIGURAZIONE ---
VS_DIR = "vectorstores"

K_HYP = 15       # candidati da vectorstore_hyp
K_STD = 8        # candidati da vectorstore_std (fallback)
K_PDF = 5        # candidati da ParentDocumentRetriever
TOP_N = 5        # chunk finali dopo reranking

PDF_CHILD_SIZE = 400
PDF_CHILD_OVERLAP = 40
PDF_PARENT_SIZE = 2000
PDF_PARENT_OVERLAP = 200


class HybridRetriever:

    def __init__(self):
        print("[RETRIEVER] Inizializzazione...")
        self.emb = self._load_embedding_model()
        self.hyp_vs = self._load_faiss("vectorstore_hyp")
        self.std_vs = self._load_faiss("vectorstore_std")
        self.pdf_retriever = self._load_pdf_retriever()
        self.reranker = self._load_reranker()
        print("[RETRIEVER] Pronto.")

    def _load_embedding_model(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_faiss(self, name: str) -> FAISS:
        path = os.path.join(VS_DIR, name)
        print(f"  [VS] Caricamento {name}...")
        return FAISS.load_local(
            path, self.emb, allow_dangerous_deserialization=True
        )

    def _load_pdf_retriever(self) -> ParentDocumentRetriever:
        """
        Ricostruisce il ParentDocumentRetriever collegandosi agli store
        già persistiti da indexing.py (Chroma + LocalFileStore).
        search_kwargs k=K_PDF: quanti child chunk recuperare prima
        di risalire al parent.
        """
        print("  [PDF] Caricamento ParentDocumentRetriever...")
        child_vectorstore = Chroma(
            collection_name="pdf_child_chunks",
            embedding_function=self.emb,
            persist_directory=os.path.join(VS_DIR, "vectorstore_pdf_child"),
        )
        parent_docstore = create_kv_docstore(
            LocalFileStore(os.path.join(VS_DIR, "docstore_pdf"))
        )
        return ParentDocumentRetriever(
            vectorstore=child_vectorstore,
            docstore=parent_docstore,
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=PDF_CHILD_SIZE, chunk_overlap=PDF_CHILD_OVERLAP
            ),
            parent_splitter=RecursiveCharacterTextSplitter(
                chunk_size=PDF_PARENT_SIZE, chunk_overlap=PDF_PARENT_OVERLAP
            ),
            search_kwargs={"k": K_PDF},
        )

    def _load_reranker(self) -> CrossEncoder:
        # Cross-encoder multilingua IT/EN
        print("  [RERANKER] Caricamento mmarco-mMiniLMv2...")
        return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    def _restore_original(self, doc: Document) -> Document:
        """
        I doc di vectorstore_hyp hanno page_content = domanda ipotetica.
        Questo metodo ripristina il chunk originale salvato nei metadata.
        """
        original = doc.metadata.get("original_content")
        if original:
            return Document(
                page_content=original,
                metadata={k: v for k, v in doc.metadata.items()
                          if k != "original_content"},
            )
        return doc

    def _dedup(self, docs: list[Document]) -> list[Document]:
        """Deduplicazione per contenuto (prime 120 lettere come chiave)."""
        seen, out = set(), []
        for doc in docs:
            key = doc.page_content.strip()[:120]
            if key not in seen:
                seen.add(key)
                out.append(doc)
        return out

    def retrieve(self, query: str) -> list[Document]:
        # 1. Hyp questions → chunk originali
        hyp_raw = self.hyp_vs.similarity_search(query, k=K_HYP)
        hyp_chunks = [self._restore_original(d) for d in hyp_raw]

        # 2. Std fallback
        std_chunks = self.std_vs.similarity_search(query, k=K_STD)

        # 3. PDF parent chunks via ParentDocumentRetriever
        try:
            pdf_chunks = self.pdf_retriever.invoke(query)
        except Exception as e:
            print(f"  [PDF] Retrieval fallito: {e}")
            pdf_chunks = []

        # 4. Merge e dedup
        all_candidates = self._dedup(hyp_chunks + std_chunks + pdf_chunks)

        if not all_candidates:
            return []

        # 5. Cross-encoder reranking
        pairs = [[query, d.page_content] for d in all_candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(all_candidates, scores), key=lambda x: x[1], reverse=True
        )
        return [doc for doc, _ in ranked[:TOP_N]]

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Versione debug con score visibili."""
        hyp_raw = self.hyp_vs.similarity_search(query, k=K_HYP)
        hyp_chunks = [self._restore_original(d) for d in hyp_raw]
        std_chunks = self.std_vs.similarity_search(query, k=K_STD)
        try:
            pdf_chunks = self.pdf_retriever.invoke(query)
        except Exception:
            pdf_chunks = []

        all_candidates = self._dedup(hyp_chunks + std_chunks + pdf_chunks)
        if not all_candidates:
            return []

        pairs = [[query, d.page_content] for d in all_candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(all_candidates, scores), key=lambda x: x[1], reverse=True
        )
        return [(doc, float(s)) for doc, s in ranked[:TOP_N]]


# Singleton: evita reload multipli con Streamlit
_instance = None

def get_retriever() -> HybridRetriever:
    global _instance
    if _instance is None:
        _instance = HybridRetriever()
    return _instance


if __name__ == "__main__":
    r = get_retriever()
    tests = [
        "Quali corsi offre il DIEM?",
        "Orari di ricevimento del professor Capuano",
        "Requisiti ammissione ingegneria informatica",
        "Chi è il re di Spagna?",   # out-of-domain
    ]
    for q in tests:
        print(f"\nQuery: {q}")
        for i, (doc, score) in enumerate(r.retrieve_with_scores(q)):
            src = doc.metadata.get("source", "?")[:55]
            typ = doc.metadata.get("type", "?")
            print(f"  [{i+1}] {score:+.3f} [{typ}] {src}")
            print(f"       {doc.page_content[:100]}...")
