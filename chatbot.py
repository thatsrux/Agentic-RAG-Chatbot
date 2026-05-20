import os
import streamlit as st
from langgraph.graph import StateGraph, START, END
from utils.config import RAGState
from utils.nodes import *
from utils.retriever import HybridRetriever
from utils.style import get_info_icon_html, get_global_css

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("condense_question", condense_question_node)
    workflow.add_node("domain_guard", domain_guard_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "condense_question")
    workflow.add_edge("condense_question", "domain_guard")

    workflow.add_conditional_edges(
        "domain_guard", route_after_domain,
        {"in_domain": "retrieve", "out_of_domain": END}
    )
    
    workflow.add_edge("retrieve", "generate")
    
    workflow.add_edge("generate", END)

    return workflow.compile()

def main():
    st.set_page_config(page_title="DIEMbot", page_icon="🎓", layout="centered")
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.title("🎓 DIEMbot")
    st.caption("Assistente virtuale ufficiale del DIEM – Università di Salerno")

    if "current_model" not in st.session_state:
        st.session_state.current_model = "gemini-3.1-flash-lite"

    if "pending_toast" in st.session_state:
        st.toast(st.session_state.pending_toast, icon="🔄")
        del st.session_state.pending_toast

    if "retriever" not in st.session_state:
        with st.spinner("Caricamento del database della conoscenza..."):
            st.session_state.retriever = HybridRetriever()
            st.success("Database caricato con successo!")

    rag_app = build_graph()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.header("Impostazioni")
        model_options = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash", "mistral-nemo"]
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
        sample_questions = [
            "Quali sono i corsi offerti dal DIEM?",
            "Ho ottenuto un punteggio di 18 al TOLC. Posso iscrivermi?",
            "Quali sono i requisiti di ammissione per il Master in Ingegneria dell'Informazione per la Medicina Digitale?",
            "Qual è il programma del corso di Ingegneria del Software?",
            "Quali sono gli orari di ricevimento del Professor Capuano?",
            "Dove si trova l'aula 126?",
            "La mia media è 28,8. Qual è il voto finale massimo che posso ottenere?",
            "Chi è responsabile dell'internazionalizzazione presso DIEM?",
            "Quali opportunità di mobilità internazionale sono disponibili?",
            "Dove si trova il DIEM?",
            "Quali aree di ricerca sono attive al DIEM?",
            "Quali laboratori sono disponibili al DIEM?",
            "Quali attrezzature sono disponibili nel Laboratorio di Robotica?",
            "Chi sono i membri della Commissione Paritaria Studenti-Docenti?"
        ]
        for q in sample_questions:
            if st.button(q, use_container_width=True):
                st.session_state.pending_question = q

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

    chat_input = st.chat_input("Chiedimi qualcosa sul DIEM...")
    user_input = None

    if chat_input:
        user_input = chat_input
    elif hasattr(st.session_state, "pending_question") and st.session_state.pending_question:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

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
                    
                    # 2. Controllo Fallback
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

if __name__ == "__main__":
    main()