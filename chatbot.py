import os
from utils.nodes import *
from utils.style import *
from utils.chatbot_utils import SAMPLE_QUESTIONS
import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

def main():
    """
    DIEMbot - Un chatbot intelligente per il Dipartimento di Ingegneria dell'Informazione, Elettrica e Matematica Applicata (DIEM) dell'Università di Salerno.
    """
    st.set_page_config(page_title="DIEMbot", page_icon="🎓", layout="centered")
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(get_header_html(), unsafe_allow_html=True)

    if "current_model" not in st.session_state:
        st.session_state.current_model = "gemini-3.1-flash-lite"

    if "pending_toast" in st.session_state:
        st.toast(st.session_state.pending_toast, icon="🔄")
        del st.session_state.pending_toast

    rag_app = build_graph()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.header("Impostazioni")
        model_options = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash", "mistral-nemo", "llama3.1"]
        current_idx = model_options.index(st.session_state.current_model) if st.session_state.current_model in model_options else 0
        
        selected_model = st.selectbox(
            "Modello in uso", 
            options=model_options, 
            index=current_idx
        )
        
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            st.rerun()
        
        show_sources = st.toggle("Mostra fonti estratte", value=True)
        
        if st.button("🗑️ Cancella chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.markdown("**Esempi di domande:**")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True):
                st.session_state.pending_question = q

    chat_input = st.chat_input("Chiedimi qualcosa sul DIEM...")
    user_input = None

    if chat_input:
        user_input = chat_input
    elif hasattr(st.session_state, "pending_question") and st.session_state.pending_question:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    should_generate = False
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        should_generate = True

    if not st.session_state.messages:
        st.markdown(get_welcome_screen_html(), unsafe_allow_html=True)
    else:
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    model_name = msg.get("model_used", "Sconosciuto")
                    st.markdown(get_info_icon_html(model_name), unsafe_allow_html=True)

                    if "steps" in msg and msg["steps"]:
                        with st.status("Elaborazione completata", state="complete", expanded=False):
                            for step in msg["steps"]:
                                st.write(step)
                
                st.markdown(msg["content"])
                
                if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                    with st.expander("📄 Fonti consultate", expanded=False, key=f"source_{idx}"):
                        for src in msg["sources"]:
                            st.caption(f"• {src}")

    if should_generate:
        with st.chat_message("assistant"):
            initial_state = {
                "question": user_input,
                "chat_history": st.session_state.messages[:-1],
                "retry_count": 0,
                "current_model": st.session_state.current_model
            }

            final_state = initial_state.copy()
            generation_error = None
            steps_log = []

            try:
                with st.status("✍️ Analisi della domanda...", expanded=True) as status:
                    for event in rag_app.stream(initial_state):
                        for node_name, node_state in event.items():
                            final_state.update(node_state)

                            if node_name == "condense_question":
                                step_msg = "✔️ Domanda contestualizzata"
                                st.write(step_msg)
                                steps_log.append(step_msg)
                                status.update(label="🛡️ Verifica del dominio in corso...")

                            elif node_name == "domain_guard":
                                if final_state.get("is_in_domain") == "no":
                                    step_msg = "🛑 Domanda fuori dominio"
                                    st.write(step_msg)
                                    steps_log.append(step_msg)
                                    status.update(label="Elaborazione completata", state="complete", expanded=False)
                                else:
                                    step_msg = "✔️ Dominio confermato"
                                    st.write(step_msg)
                                    steps_log.append(step_msg)
                                    status.update(label="🔍 Recupero informazioni dal database...")

                            elif node_name == "retrieve":
                                docs_count = len(final_state.get("sources", []))
                                step_msg = f"✔️ Recuperati {docs_count} documenti"
                                st.write(step_msg)
                                steps_log.append(step_msg)
                                status.update(label="🧠 Generazione della risposta...")

                            elif node_name == "generate":
                                generation = final_state.get("generation", "")
                                if "[TRIGGER_WEB_SEARCH]" in generation:
                                    step_msg = "⚠️ Informazioni non trovate nel database locale"
                                    st.write(step_msg)
                                    steps_log.append(step_msg)
                                    status.update(label="🌐 Ricerca sul Web in corso...")
                                else:
                                    step_msg = "✔️ Risposta generata dal database"
                                    st.write(step_msg)
                                    steps_log.append(step_msg)
                                    status.update(label="Elaborazione completata", state="complete", expanded=False)

                            elif node_name == "web_search":
                                step_msg = "✔️ Risposta generata dalle fonti web"
                                st.write(step_msg)
                                steps_log.append(step_msg)
                                status.update(label="Elaborazione completata", state="complete", expanded=False)

            except Exception as e:
                generation_error = e

        if generation_error is not None:
            error_msg = str(generation_error).lower()
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                st.warning("⏳ **Troppe richieste!** Il Dipartimento sta facendo molte domande a DIEMbot. Per favore, attendi circa un minuto e riprova.")
            else:
                st.error("❌ Ops, si è verificato un errore di connessione con il modello AI. Riprova tra un attimo.")
            st.session_state.messages.pop()
        else:
            full_response = final_state.get("generation", "")
            sources_list  = final_state.get("sources", [])
            used_model    = final_state.get("model_used", st.session_state.current_model)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_list,
                "model_used": used_model,
                "steps": steps_log
            })

            if used_model != st.session_state.current_model:
                st.session_state.pending_toast = f"Fallback attivato! Passaggio da {st.session_state.current_model} a {used_model}"
                st.session_state.current_model = used_model

        st.rerun()

if __name__ == "__main__":
    main()