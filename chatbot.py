import os
import streamlit as st
from utils.style import get_info_icon_html, get_global_css, get_welcome_screen_html, get_header_html
from utils.utils import sample_questions
from utils.nodes import build_graph

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

def main():
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
        for q in sample_questions:
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
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    model_name = msg.get("model_used", "Sconosciuto")
                    st.markdown(get_info_icon_html(model_name), unsafe_allow_html=True)
                
                st.markdown(msg["content"])
                
                if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                    with st.expander("📄 Fonti consultate"):
                        for src in msg["sources"]:
                            st.caption(f"• {src}")

    if should_generate:
        with st.chat_message("assistant"):
            with st.spinner("DIEMbot sta elaborando la richiesta..."):
                initial_state = {
                    "question": user_input, 
                    "chat_history": st.session_state.messages[:-1], 
                    "retry_count": 0,
                    "current_model": st.session_state.current_model
                }
                
                try:
                    final_state = rag_app.invoke(initial_state)
                    
                    full_response = final_state["generation"]
                    sources_list = final_state.get("sources", [])
                    used_model = final_state.get("model_used", st.session_state.current_model)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources_list,
                        "model_used": used_model
                    })
                    
                    if used_model != st.session_state.current_model:
                        st.session_state.pending_toast = f"Fallback attivato! Passaggio da {st.session_state.current_model} a {used_model}"
                        st.session_state.current_model = used_model
                        st.rerun()
                    else:
                        st.markdown(get_info_icon_html(used_model), unsafe_allow_html=True)
                        st.markdown(full_response)
                        
                        if show_sources and sources_list:
                            with st.expander("📄 Fonti consultate"):
                                for src in sources_list:
                                    st.caption(f"• {src}")

                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        st.warning("⏳ **Troppe richieste!** Il Dipartimento sta facendo molte domande a DIEMbot. Per favore, attendi circa un minuto e riprova.")
                    else:
                        st.error("❌ Ops, si è verificato un errore di connessione con il modello AI. Riprova tra un attimo.")
                    
                    st.session_state.messages.pop()
                    st.rerun()

if __name__ == "__main__":
    main()