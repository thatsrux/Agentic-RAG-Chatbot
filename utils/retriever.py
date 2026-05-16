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
from rank_bm25 import BM25Okapi
from utils.config import device

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_TOKEN"] = "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"

# =========================================================
# CONFIG
# =========================================================

VS_DIR = "knowledge/vectorstores"

K_WEB = 10
K_PDF = 5
K_EXCEL = 5  # Numero di chunk da recuperare per gli Excel

TOP_N = 5

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

# EXCEL (Stesse dimensioni del Web, trattandosi di Markdown)
EXCEL_CHILD_SIZE = 800
EXCEL_CHILD_OVERLAP = 80
EXCEL_PARENT_SIZE = 3000
EXCEL_PARENT_OVERLAP = 300

BM25_ENABLED = True

# =========================================================
# RETRIEVER
# =========================================================

class HybridRetriever:

    def __init__(self):

        print("[RETRIEVER] Initializing system...")

        self.emb = self._load_embedding_model()

        self.web_retriever, self.web_vs = self._load_web_retriever()

        self.pdf_retriever = self._load_pdf_retriever()

        # Nuova inizializzazione per Excel
        self.excel_retriever, self.excel_vs = self._load_excel_retriever()

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

        print("  [EMB] Loading BAAI/bge-m3...")

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

        print("  [WEB] Loading Parent-Child FAISS Web...")

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

        print("  [PDF] Loading Parent-Child FAISS PDF...")

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

    def _load_excel_retriever(self):

        print("  [EXCEL] Loading Parent-Child FAISS Excel...")
        
        try:
            child_vectorstore = FAISS.load_local(
                os.path.join(VS_DIR, "faiss_excel_child"),
                self.emb,
                allow_dangerous_deserialization=True
            )

            parent_docstore = create_kv_docstore(
                LocalFileStore(
                    os.path.join(VS_DIR, "docstore_excel")
                )
            )

            retriever = ParentDocumentRetriever(
                vectorstore=child_vectorstore,
                docstore=parent_docstore,
                child_splitter=MarkdownTextSplitter(
                    chunk_size=EXCEL_CHILD_SIZE,
                    chunk_overlap=EXCEL_CHILD_OVERLAP,
                ),
                parent_splitter=MarkdownTextSplitter(
                    chunk_size=EXCEL_PARENT_SIZE,
                    chunk_overlap=EXCEL_PARENT_OVERLAP,
                ),
                search_kwargs={
                    "k": K_EXCEL
                }
            )
            
            return retriever, child_vectorstore
        except Exception as e:
            print(f"  [EXCEL] Failed to load Excel FAISS: {e}")
            # Fallback sicuro se il VectorStore Excel non esiste ancora
            return None, None

    # =====================================================
    # RERANKER
    # =====================================================

    def _load_reranker(self):
        print("  [RERANKER] Loading BAAI/bge-reranker-v2-m3...")
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

        print("  [BM25] Building sparse index...")

        try:

            docs = []

            # Carica i chunk Web Child per costruire l'indice delle keyword
            web_docs = self.web_vs.similarity_search(
                "test",
                k=100000
            )
            docs.extend(web_docs)
            
            # Carica anche i chunk Excel (molto utili per la ricerca esatta di aule/prof)
            if hasattr(self, 'excel_vs') and self.excel_vs is not None:
                excel_docs = self.excel_vs.similarity_search(
                    "test",
                    k=100000
                )
                docs.extend(excel_docs)

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
    # DEDUP
    # =====================================================

    def _dedup(self, docs):

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
    # RETRIEVAL METHODS
    # =====================================================

    def _retrieve_web(self, query):
        try:
            return self.web_retriever.invoke(query)
        except Exception as e:
            print(f"[WEB] Retrieval failed: {e}")
            return []

    def _retrieve_pdf(self, query):
        try:
            return self.pdf_retriever.invoke(query)
        except Exception as e:
            print(f"[PDF] Retrieval failed: {e}")
            return []
            
    def _retrieve_excel(self, query):
        if self.excel_retriever is None:
            return []
        try:
            return self.excel_retriever.invoke(query)
        except Exception as e:
            print(f"[EXCEL] Retrieval failed: {e}")
            return []

    def _retrieve_bm25(self, query):

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

            # Raccoglie i top documenti dal BM25 (limite K_WEB)
            return [
                doc
                for doc, score in ranked[:K_WEB]
            ]

        except Exception as e:

            print(f"[BM25] Retrieval failed: {e}")

            return []

    # =====================================================
    # RETRIEVE FUNCTION
    # =====================================================

    def retrieve(self, query, route="both"):

        web_chunks = []
        pdf_chunks = []
        excel_chunks = []
        bm25_chunks = []

        # =================================================
        # PARALLEL RETRIEVAL
        # =================================================

        # Aumentato il numero di worker per supportare anche la coda Excel
        with ThreadPoolExecutor(max_workers=4) as executor:

            futures = {}

            # WEB
            if route in ["web", "both", "all"]:

                futures["web"] = executor.submit(
                    self._retrieve_web,
                    query
                )

            # PDF
            if route in ["pdf", "both", "all"]:

                futures["pdf"] = executor.submit(
                    self._retrieve_pdf,
                    query
                )
                
            # EXCEL
            if route in ["excel", "both", "all"]:
                
                futures["excel"] = executor.submit(
                    self._retrieve_excel,
                    query
                )

            # BM25
            if BM25_ENABLED:

                futures["bm25"] = executor.submit(
                    self._retrieve_bm25,
                    query
                )

            # COLLECT

            if "web" in futures:
                web_chunks = futures["web"].result()

            if "pdf" in futures:
                pdf_chunks = futures["pdf"].result()
                
            if "excel" in futures:
                excel_chunks = futures["excel"].result()

            if "bm25" in futures:
                bm25_chunks = futures["bm25"].result()

        # =================================================
        # MERGE
        # =================================================

        all_candidates = self._dedup(
            web_chunks +
            pdf_chunks +
            excel_chunks +
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