import os
import uuid
import streamlit as st

# --- PROTEZIONE CRASH (WINDOWS/STREAMLIT) ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from agentic_rag import build_graph

# --- CONFIGURAZIONE ---
APP_TITLE = "DIEMbot — Università di Salerno"
APP_ICON  = "🎓"

# ==============================================================================
# INIZIALIZZAZIONE RISORSE (cached a livello di sistema)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_rag_graph():
    """
    Costruisce e compila il grafo LangGraph con checkpointer SQLite.
    Viene eseguito UNA SOLA VOLTA per tutta la durata del processo Streamlit.
    Il checkpointer è condiviso tra le sessioni: ogni sessione usa il proprio thread_id.
    """
    return build_graph()


# ==============================================================================
# UTILITY
# ==============================================================================

def new_thread_id() -> str:
    """Genera un ID univoco per una nuova sessione di conversazione."""
    return str(uuid.uuid4())

def render_message(msg: dict, show_sources: bool):
    """Renderizza un singolo messaggio della cronologia (utente o assistente)."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("📄 Fonti consultate"):
                for src in msg["sources"]:
                    st.caption(f"• {src}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="centered",
    )

    st.title("🎓 DIEMbot")
    st.caption("Assistente virtuale ufficiale del DIEM – Università di Salerno")

    # --- Caricamento grafo (una sola volta per processo) ---
    if "rag_graph" not in st.session_state:
        with st.spinner("Caricamento del sistema RAG agentivo..."):
            st.session_state.rag_graph = load_rag_graph()
        st.success("Sistema pronto!")

    # --- Thread ID: identifica univocamente la conversazione corrente ---
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = new_thread_id()

    # --- Cronologia messaggi (solo per la UI — la memoria vera è nel checkpointer) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ Impostazioni")

        show_sources = st.toggle("Mostra fonti estratte", value=True)

        st.divider()

        # --- Gestione conversazione ---
        st.markdown("**Conversazione**")
        st.caption(f"ID sessione: `{st.session_state.thread_id[:8]}…`")

        if st.button("✨ Nuova conversazione", use_container_width=True):
            st.session_state.thread_id = new_thread_id()
            st.session_state.messages  = []
            st.rerun()

        if st.button("🗑️ Cancella vista chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # --- Domande di esempio ---
        st.markdown("**Esempi di domande:**")
        example_questions = [
            "Quali sono i corsi di laurea del DIEM?",
            "Dove trovo gli orari delle lezioni?",
            "Chi è il direttore del dipartimento?",
            "Quali sono i requisiti per la laurea magistrale?",
        ]
        for q in example_questions:
            if st.button(q, use_container_width=True):
                st.session_state.pending_question = q

        st.divider()
        st.markdown(
            "<small>Powered by LangGraph · FAISS · llama3.2<br>"
            "Memoria persistente: SQLite</small>",
            unsafe_allow_html=True,
        )

    # ==========================================================================
    # CRONOLOGIA MESSAGGI
    # ==========================================================================
    for msg in st.session_state.messages:
        render_message(msg, show_sources)

    # ==========================================================================
    # INPUT UTENTE
    # ==========================================================================
    user_input = st.chat_input("Chiedimi qualcosa sul DIEM...")

    # Domande dai pulsanti della sidebar
    if hasattr(st.session_state, "pending_question") and st.session_state.pending_question:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    if not user_input:
        return

    # --- Mostra messaggio utente ---
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ==========================================================================
    # RISPOSTA AGENTIVA
    # ==========================================================================
    with st.chat_message("assistant"):
        with st.spinner("Ragionamento in corso..."):
            result = st.session_state.rag_graph.invoke(
                {
                    "query":        user_input,
                    "chat_history": [],   # delta iniziale vuoto; il reducer fa l'append
                },
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )

        full_response = result.get("generation", "Non ho trovato informazioni pertinenti.")
        sources_list  = list(set(
            d.metadata.get("source", "N/A")
            for d in result.get("documents", [])
        ))

        st.markdown(full_response)

        if show_sources and sources_list:
            with st.expander("📄 Fonti consultate"):
                for src in sources_list:
                    st.caption(f"• {src}")

    # Salva in cronologia UI
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "sources": sources_list,
    })


if __name__ == "__main__":
    main()
