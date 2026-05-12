import os
import streamlit as st

# --- PROTEZIONE CRASH (WINDOWS/STREAMLIT) ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from agentic_rag import build_graph

# --- CONFIGURAZIONE ---
APP_TITLE = "DIEMbot — Università di Salerno"
APP_ICON  = "🎓"

# --- INIZIALIZZAZIONE RISORSE ---
@st.cache_resource(show_spinner=False)
def load_rag_graph():
    """Costruisce e compila il grafo LangGraph (cached a livello di sistema)."""
    return build_graph()


# --- INTERFACCIA STREAMLIT ---
def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="centered",
    )

    st.title("🎓 DIEMbot")
    st.caption("Assistente virtuale ufficiale del DIEM – Università di Salerno")

    # Caricamento grafo agentico (con spinner solo al primo avvio)
    if "rag_graph" not in st.session_state:
        with st.spinner("Caricamento del sistema RAG agentivo..."):
            st.session_state.rag_graph = load_rag_graph()
        st.success("Sistema pronto!")

    # Gestione cronologia chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Impostazioni")

        show_sources = st.toggle("Mostra fonti estratte", value=True)
        show_debug   = st.toggle("Modalità debug (route + rewrite)", value=False)

        if st.button("🗑️ Cancella cronologia chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
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
            "<small>Powered by LangGraph · FAISS · llama3.2</small>",
            unsafe_allow_html=True,
        )

    # --- VISUALIZZAZIONE MESSAGGI PRECEDENTI ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # Debug info
                if show_debug and msg.get("debug"):
                    debug = msg["debug"]
                    cols = st.columns(2)
                    cols[0].caption(f"🔀 Route: `{debug.get('route', 'N/A')}`")
                    if debug.get("rewrote"):
                        cols[1].caption(f"🔄 Query riscritta: *{debug['rewrote']}*")

                # Fonti
                if show_sources and msg.get("sources"):
                    with st.expander("📄 Fonti consultate"):
                        for src in msg["sources"]:
                            st.caption(f"• {src}")

    # --- INPUT UTENTE ---
    user_input = st.chat_input("Chiedimi qualcosa sul DIEM...")

    # Domande dai pulsanti della sidebar
    if hasattr(st.session_state, "pending_question") and st.session_state.pending_question:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    if user_input:
        # Mostra messaggio utente
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # --- RISPOSTA AGENTIVA ---
        with st.chat_message("assistant"):
            with st.spinner("Ragionamento in corso..."):
                # Invoca il grafo LangGraph
                result = st.session_state.rag_graph.invoke({"query": user_input})

            full_response = result.get("generation", "Non ho trovato informazioni pertinenti.")
            sources_list  = list(set(
                d.metadata.get("source", "N/A")
                for d in result.get("documents", [])
            ))
            debug_info = {
                "route":   result.get("route"),
                "rewrote": result.get("rewritten_query"),
            }

            # Mostra info debug sopra la risposta (se attivo)
            if show_debug:
                cols = st.columns(2)
                cols[0].caption(f"🔀 Route: `{debug_info['route']}`")
                if debug_info.get("rewrote"):
                    cols[1].caption(f"🔄 Query riscritta: *{debug_info['rewrote']}*")

            # Risposta
            st.markdown(full_response)

            # Fonti
            if show_sources and sources_list:
                with st.expander("📄 Fonti consultate"):
                    for src in sources_list:
                        st.caption(f"• {src}")

        # Salva in cronologia con metadati
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_response,
            "sources": sources_list,
            "debug":   debug_info,
        })


if __name__ == "__main__":
    main()