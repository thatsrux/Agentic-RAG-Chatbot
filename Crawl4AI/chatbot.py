"""
STEP 4 — CHATBOT STREAMLIT
Pipeline completa con:
  - Conversational Query Rewriting (Llama 3.1 via Ollama)
  - Retrieval ibrido (retriever.py)
  - Out-of-domain detection
  - Anti-allucinazione
  - Robustness ("Sei sicuro?", ecc.)
  - Visualizzazione fonti e query riscritta

Avvio: streamlit run chatbot.py
Requisito: Ollama in esecuzione con llama3.1 scaricato
  → ollama pull llama3.1
  → ollama serve
"""

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import get_retriever

# --- CONFIGURAZIONE ---
OLLAMA_MODEL = "llama3.1"

SYSTEM_PROMPT = """
Sei DIEMbot, l'assistente virtuale del DIEM (Dipartimento di Ingegneria \
dell'Informazione e Elettronica) dell'Università di Salerno.

REGOLE FONDAMENTALI:
1. Rispondi SOLO in base ai documenti forniti nel contesto. Non usare conoscenze esterne.
2. Se l'informazione non è presente nei documenti rispondi:
   "Non ho trovato informazioni su questo nelle fonti del DIEM."
3. Se la domanda non riguarda il DIEM, i suoi corsi, docenti o strutture rispondi:
   "Posso rispondere solo a domande riguardanti il DIEM e l'Università di Salerno."
4. Se l'utente dice "Sei sicuro?" o mette in dubbio la risposta, riconferma
   quanto presente nei documenti oppure ammetti l'incertezza se i doc non sono chiari.
5. Non inventare nomi, date, link o numeri non presenti nei documenti.
6. Rispondi in italiano, a meno che l'utente non scriva in inglese.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Domanda: {question}

Documenti di riferimento:
{context}

Risposta:"""),
])

REWRITE_PROMPT = PromptTemplate.from_template("""
Sei un assistente che ottimizza le query di ricerca.
Data la cronologia della conversazione e l'ultima domanda, riscrivila
come query autonoma risolvendo pronomi e riferimenti impliciti ai turni precedenti.
Se è già autonoma, riscrivila identica.
Rispondi con SOLO la domanda riscritta, senza spiegazioni.

Cronologia:
{history}

Ultima domanda: {query}

Domanda riscritta:
""")


@st.cache_resource
def load_resources():
    """Carica retriever e LLM una sola volta (Streamlit cache)."""
    retriever = get_retriever()
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    return retriever, llm


def rewrite_query(query: str, history: list[dict], llm) -> str:
    """Riscrive la query incorporando il contesto degli ultimi 4 turni."""
    if not history:
        return query

    history_text = ""
    for turn in history[-4:]:
        role = "Utente" if turn["role"] == "user" else "Assistente"
        history_text += f"{role}: {turn['content']}\n"

    chain = REWRITE_PROMPT | llm | StrOutputParser()
    try:
        return chain.invoke({"history": history_text, "query": query}).strip()
    except Exception:
        return query


def format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "fonte sconosciuta")
        doc_type = "PDF" if doc.metadata.get("type") == "pdf" else "Pagina web"
        parts.append(
            f"[Documento {i+1} — {doc_type}: {source}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def ask(
    query: str,
    history: list[dict],
    retriever,
    llm,
) -> tuple[str, list, str]:
    """
    1. Conversational rewriting
    2. Retrieval ibrido + reranking
    3. Generazione risposta
    """
    rewritten = rewrite_query(query, history, llm)
    docs = retriever.retrieve(rewritten)
    context = format_context(docs) if docs else "Nessun documento trovato."
    chain = RAG_PROMPT | llm | StrOutputParser()
    response = chain.invoke({"question": rewritten, "context": context})
    return response, docs, rewritten


# --- INTERFACCIA ---

def main():
    st.set_page_config(
        page_title="DIEMbot — Assistente DIEM UniSa",
        page_icon="🎓",
        layout="centered",
    )

    st.title("🎓 DIEMbot")
    st.caption("Assistente virtuale del DIEM – Università di Salerno")

    with st.spinner("Caricamento modelli in corso..."):
        retriever, llm = load_resources()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True

    # Sidebar
    with st.sidebar:
        st.header("Impostazioni")
        st.session_state.show_sources = st.toggle(
            "Mostra fonti", value=st.session_state.show_sources
        )
        if st.button("🗑️ Nuova conversazione"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown("**Esempi di domande:**")
        examples = [
            "Quali corsi offre il DIEM?",
            "Quali sono i requisiti di ammissione?",
            "Come contatto un docente?",
            "Dove si trova il DIEM?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state.pending = ex

    # Storico messaggi
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (msg["role"] == "assistant"
                    and st.session_state.show_sources
                    and msg.get("sources")):
                with st.expander("📄 Fonti"):
                    for i, src in enumerate(msg["sources"]):
                        st.caption(f"**[{i+1}]** {src}")
            if (msg["role"] == "assistant"
                    and msg.get("rewritten")
                    and msg["rewritten"] != msg.get("original")):
                with st.expander("🔍 Query interpretata"):
                    st.caption(f"*{msg['rewritten']}*")

    # Input
    user_input = st.chat_input("Scrivi la tua domanda sul DIEM...")
    if hasattr(st.session_state, "pending") and st.session_state.pending:
        user_input = st.session_state.pending
        st.session_state.pending = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Cerco nelle fonti DIEM..."):
                response, docs, rewritten = ask(
                    user_input,
                    st.session_state.messages[:-1],
                    retriever,
                    llm,
                )

            st.markdown(response)

            sources = [d.metadata.get("source", "?") for d in docs]
            if st.session_state.show_sources and sources:
                with st.expander("📄 Fonti"):
                    for i, src in enumerate(sources):
                        st.caption(f"**[{i+1}]** {src}")
            if rewritten != user_input:
                with st.expander("🔍 Query interpretata"):
                    st.caption(f"*{rewritten}*")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "rewritten": rewritten,
            "original": user_input,
        })


if __name__ == "__main__":
    main()
