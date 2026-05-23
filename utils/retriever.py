import os
import hashlib
import torch
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from utils.config import device
from rank_bm25 import BM25Okapi

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_TOKEN"] = "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"

VS_DIR = "knowledge/vectorstores"
K_WEB = 20
K_PDF = 10
TOP_N = 5

WEB_CHILD_SIZE = 800
WEB_CHILD_OVERLAP = 80
WEB_PARENT_SIZE = 3000
WEB_PARENT_OVERLAP = 300

PDF_CHILD_SIZE = 800
PDF_CHILD_OVERLAP = 80
PDF_PARENT_SIZE = 3000
PDF_PARENT_OVERLAP = 300

BM25_ENABLED = True

# =========================================================
# RETRIEVER
# =========================================================

class HybridRetriever:
    """
    Implementazione di un retriever ibrido che combina retrieval basato su embedding (FAISS) e retrieval basato su keyword (BM25).
    """
    def __init__(self):
        """
        Inizializza il retriever caricando i modelli di embedding, i vectorstore per web e PDF, il modello di reranking e costruendo l'indice BM25 se abilitato.
        """
        print("[RETRIEVER] Initializing system.")

        self.emb = self._load_embedding_model()
        self.web_retriever, self.web_vs = self._load_web_retriever()
        self.pdf_retriever = self._load_pdf_retriever()
        self.reranker = self._load_reranker()
        self.bm25 = None
        self.bm25_docs = []
        if BM25_ENABLED:
            self._initialize_bm25()

        print("[RETRIEVER] System ready.")

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    def _load_embedding_model(self):
        """
         Carica il modello di embedding BAAI/bge-m3, configurato per l'uso su GPU se disponibile, con normalizzazione e batch size ottimizzato.
        """
        print("  [EMB] Loading BAAI/bge-m3.")

        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={
                "device": device
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )

    # =====================================================
    # VECTORSTORES & RETRIEVERS
    # =====================================================

    def _load_web_retriever(self):
        """
        Carica il retriever per i documenti web, utilizzando una struttura parent-child con FAISS per i child e un docstore locale per i parent.
        """
        print("  [WEB] Loading Parent-Child FAISS Web.")

        child_vectorstore = FAISS.load_local(
            os.path.join(VS_DIR, "faiss_web_child"),
            self.emb,
            allow_dangerous_deserialization=True
        )
        parent_docstore = create_kv_docstore(
            LocalFileStore(
                os.path.join(VS_DIR, "docstore_web")
            )
        )
        retriever = ParentDocumentRetriever(
            vectorstore=child_vectorstore,
            docstore=parent_docstore,
            child_splitter=MarkdownTextSplitter(
                chunk_size=WEB_CHILD_SIZE,
                chunk_overlap=WEB_CHILD_OVERLAP,
            ),
            parent_splitter=MarkdownTextSplitter(
                chunk_size=WEB_PARENT_SIZE,
                chunk_overlap=WEB_PARENT_OVERLAP,
            ),
            search_kwargs={
                "k": K_WEB
            }
        )
        
        return retriever, child_vectorstore

    def _load_pdf_retriever(self):
        """
        Carica il retriever per i documenti PDF, utilizzando una struttura parent-child con FAISS per i child e un docstore locale per i parent.
        """
        print("  [PDF] Loading Parent-Child FAISS PDF.")

        child_vectorstore = FAISS.load_local(
            os.path.join(VS_DIR, "faiss_pdf_child"),
            self.emb,
            allow_dangerous_deserialization=True
        )
        parent_docstore = create_kv_docstore(
            LocalFileStore(
                os.path.join(VS_DIR, "docstore_pdf")
            )
        )

        return ParentDocumentRetriever(
            vectorstore=child_vectorstore,
            docstore=parent_docstore,
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=PDF_CHILD_SIZE,
                chunk_overlap=PDF_CHILD_OVERLAP,
                separators=["\n\n", ". ", "\n", " ", ""]
            ),
            parent_splitter=RecursiveCharacterTextSplitter(
                chunk_size=PDF_PARENT_SIZE,
                chunk_overlap=PDF_PARENT_OVERLAP,
                separators=["\n\n", ". ", "\n", " ", ""]
            ),
            search_kwargs={
                "k": K_PDF
            }
        )

    # =====================================================
    # RERANKER
    # =====================================================

    def _load_reranker(self):
        """
        Carica il modello di reranking BAAI/bge-reranker-v2-m3, configurato per l'uso su GPU se disponibile,
        con batch size ottimizzato e lunghezza massima aumentata per gestire meglio i documenti lunghi.
        """
        print("  [RERANKER] Loading BAAI/bge-reranker-v2-m3.")
        return CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            model_kwargs={"torch_dtype": torch.float16},
            max_length=1024,
            device=device
        )

    # =====================================================
    # BM25
    # =====================================================

    def _initialize_bm25(self):
        """
        Inizializza l'indice BM25 caricando i documenti web child, tokenizzandoli e costruendo l'indice BM25.
        """
        print("  [BM25] Building sparse index.")

        try:
            docs = []
            web_docs = self.web_vs.similarity_search(
                "test",
                k=100000
            )
            docs.extend(web_docs)

            self.bm25_docs = docs
            tokenized = [
                d.page_content.lower().split()
                for d in docs
            ]
            self.bm25 = BM25Okapi(tokenized)

            print(f"  [BM25] Indexed {len(docs)} documents.")

        except Exception as e:
            print(f"  [BM25] Failed: {e}")

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    def _dedup(self, docs):
        """
        Rimuove documenti duplicati basandosi su un hash del contenuto, mantenendo solo documenti unici.
        """
        seen = set()
        output = []

        for doc in docs:
            content = doc.page_content.strip()
            key = hashlib.md5(
                content.encode("utf-8")
            ).hexdigest()

            if key not in seen:
                seen.add(key)
                output.append(doc)

        return output

    # =====================================================
    # WEB RETRIEVAL
    # =====================================================

    def _retrieve_web(self, query):
        """
        Esegue il retrieval sui documenti web utilizzando il retriever parent-child, restituendo i documenti più rilevanti.
        """
        try:
            return self.web_retriever.invoke(query)

        except Exception as e:
            print(f"[WEB] Retrieval failed: {e}")
            return []

    # =====================================================
    # PDF RETRIEVAL
    # =====================================================

    def _retrieve_pdf(self, query):
        """
        Esegue il retrieval sui documenti PDF utilizzando il retriever parent-child, restituendo i documenti più rilevanti.
        """
        try:
            return self.pdf_retriever.invoke(query)

        except Exception as e:
            print(f"[PDF] Retrieval failed: {e}")
            return []

    # =====================================================
    # BM25 RETRIEVAL
    # =====================================================

    def _retrieve_bm25(self, query):
        """
        Esegue il retrieval sui documenti web utilizzando l'indice BM25, restituendo i documenti più rilevanti in base alla similarità keyword.
        """
        if self.bm25 is None:
            return []

        try:
            tokens = query.lower().split()
            scores = self.bm25.get_scores(tokens)
            ranked = sorted(
                zip(self.bm25_docs, scores),
                key=lambda x: x[1],
                reverse=True
            )

            return [
                doc
                for doc, _ in ranked[:K_WEB]
            ]

        except Exception as e:
            print(f"[BM25] Retrieval failed: {e}")
            return []

    # =====================================================
    # RETRIEVE FUNCTION
    # =====================================================

    def retrieve(self, query, route="both"):
        """
        Funzione principale di retrieval che esegue il retrieval parallelo sui documenti web, PDF e BM25 (se abilitato),
        """
        web_chunks = []
        pdf_chunks = []
        bm25_chunks = []

        # =================================================
        # PARALLEL RETRIEVAL
        # =================================================

        with ThreadPoolExecutor(max_workers=3) as executor:

            futures = {}

            # WEB
            if route in ["web", "both"]:

                futures["web"] = executor.submit(
                    self._retrieve_web,
                    query
                )

            # PDF
            if route in ["pdf", "both"]:

                futures["pdf"] = executor.submit(
                    self._retrieve_pdf,
                    query
                )

            # BM25
            if BM25_ENABLED:

                futures["bm25"] = executor.submit(
                    self._retrieve_bm25,
                    query
                )

            # COLLECT RESULTS
            if "web" in futures:
                web_chunks = futures["web"].result()

            if "pdf" in futures:
                pdf_chunks = futures["pdf"].result()

            if "bm25" in futures:
                bm25_chunks = futures["bm25"].result()

        # =================================================
        # MERGE
        # =================================================

        all_candidates = self._dedup(
            web_chunks +
            pdf_chunks +
            bm25_chunks
        )

        if not all_candidates:
            return []

        # =================================================
        # RERANK
        # =================================================

        pairs = [
            [query, d.page_content]
            for d in all_candidates
        ]

        scores = self.reranker.predict(
            pairs,
            batch_size=64,
            show_progress_bar=False
        )

        ranked = sorted(
            zip(all_candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # =================================================
        # FILTER
        # =================================================

        final_docs = []

        for doc, score in ranked:

            if doc.page_content == "__dummy__":
                continue

            doc.metadata["rerank_score"] = float(score)
            final_docs.append(doc)

        final_docs = sorted(
            final_docs,
            key=lambda d: d.metadata["rerank_score"],
            reverse=True
        )

        return final_docs[:TOP_N]

if __name__ == "__main__":

    retriever = HybridRetriever()

    while True:

        query = input("\nQuery > ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        docs = retriever.retrieve(query)

        print("\n================ RESULTS ================\n")

        for i, doc in enumerate(docs, start=1):

            print(f"[{i}] Score: {doc.metadata.get('rerank_score'):.4f}")
            print(f"Type: {doc.metadata.get('type')}")
            print(f"Source: {doc.metadata.get('source')}")
            print("\n")
            print(doc.page_content[:2000])
            print("\n----------------------------------------")