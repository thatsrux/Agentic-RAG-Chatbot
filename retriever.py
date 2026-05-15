import os
import torch
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
K_WEB = 10       # Candidati iniziali web
K_PDF = 3        # Candidati iniziali PDF
TOP_N = 3        # Risultati finali dopo reranking

# Parametri PDF (chunk più grandi)
PDF_CHILD_SIZE = 400
PDF_CHILD_OVERLAP = 40
PDF_PARENT_SIZE = 2000
PDF_PARENT_OVERLAP = 200

# Parametri Web (chunk più piccoli per info specifiche)
WEB_CHILD_SIZE = 350
WEB_CHILD_OVERLAP = 35
WEB_PARENT_SIZE = 1500
WEB_PARENT_OVERLAP = 150

PARENT_ID_KEY = "parent_id"
# ──────────────────────────────────────────────────────────────────────────────


class HybridRetriever:
    def __init__(self):
        print("[RETRIEVER] Inizializzazione sistema Hybrid (Web Parent-Child + PDF Parent-Child)...")
        self.emb = self._load_embedding_model()
        self.web_retriever = self._load_web_retriever()
        self.pdf_retriever = self._load_pdf_retriever()
        self.reranker = self._load_reranker()
        print("[RETRIEVER] Sistema pronto.")

    def _load_embedding_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [EMBED] Dispositivo selezionato: {device}")
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

    def _load_web_retriever(self):
        print("  [WEB] Caricamento FAISS Parent-Child Web...")
        try:
            child_vectorstore = FAISS.load_local(
                os.path.join(VS_DIR, "faiss_web_child"),
                self.emb,
                allow_dangerous_deserialization=True
            )
            parent_docstore = create_kv_docstore(LocalFileStore(os.path.join(VS_DIR, "docstore_web")))

            return ParentDocumentRetriever(
                vectorstore=child_vectorstore,
                docstore=parent_docstore,
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=WEB_CHILD_SIZE, chunk_overlap=WEB_CHILD_OVERLAP),
                parent_splitter=RecursiveCharacterTextSplitter(chunk_size=WEB_PARENT_SIZE, chunk_overlap=WEB_PARENT_OVERLAP),
                search_kwargs={"k": K_WEB},
                id_key=PARENT_ID_KEY,
            )
        except Exception as e:
            print(f"  [WEB] Errore caricamento Parent-Child: {e}")
            print(f"  [WEB] Fallback: caricamento FAISS semplice...")
            return FAISS.load_local(
                os.path.join(VS_DIR, "faiss_web"),
                self.emb,
                allow_dangerous_deserialization=True
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
            id_key=PARENT_ID_KEY,
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

    def retrieve(self, query: str, tipo_fonte: str = "all"):
        web_chunks = []
        pdf_chunks = []

        # 1. Cerca nel database Web (se richiesto)
        if tipo_fonte in ["all", "web"]:
            try:
                if hasattr(self.web_retriever, 'invoke'):
                    web_chunks = self.web_retriever.invoke(query)
                else:
                    # Fallback a similarity search semplice (quando ParentChild non carica)
                    web_chunks = self.web_retriever.similarity_search(query, k=K_WEB)
                print(f"  [WEB] Trovati {len(web_chunks)} candidati")
            except Exception as e:
                print(f"  [WEB] Retrieval fallito: {e}")
                web_chunks = []

        # 2. Cerca nel database PDF (se richiesto)
        if tipo_fonte in ["all", "pdf"]:
            try:
                pdf_chunks = self.pdf_retriever.invoke(query)
                print(f"  [PDF] Trovati {len(pdf_chunks)} candidati")
            except Exception as e:
                print(f"  [PDF] Retrieval fallito: {e}")
                pdf_chunks = []

        # 3. Unisce e deduplica
        all_candidates = self._dedup(web_chunks + pdf_chunks)
        if not all_candidates:
            print("  [RETRIEVER] Nessun candidato trovato.")
            return []

        # 4. Reranking con Cross-Encoder
        pairs = [[query, d.page_content] for d in all_candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(all_candidates, scores), key=lambda x: x[1], reverse=True)

        # 5. Filtra il placeholder tecnico "init" e restituisce top-N
        final_docs = [doc for doc, _ in ranked if doc.page_content.strip() != "init"]
        print(f"  [RETRIEVER] Restituzione top-{min(TOP_N, len(final_docs))} documenti.")
        return final_docs[:TOP_N]