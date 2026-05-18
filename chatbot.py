import os
import streamlit as st
from langgraph.graph import StateGraph, START, END
from utils.config import RAGState
from utils.nodes import *
from utils.retriever import HybridRetriever

# --- PROTEZIONE CRASH ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("domain_guard", domain_guard_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_generation", grade_generation_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("fallback", fallback_node)

    workflow.add_edge(START, "domain_guard")

    workflow.add_conditional_edges(
        "domain_guard", route_after_domain,
        {"in_domain": "retrieve", "out_of_domain": END}
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade_generation")

    workflow.add_conditional_edges(
        "grade_generation", route_after_grade,
        {"useful": END, "rewrite": "rewrite", "max_retries": "fallback"}
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("fallback", END)

    return workflow.compile()

def main():
    st.set_page_config(page_title="DIEMbot", page_icon="🎓", layout="centered")
    st.title("🎓 DIEMbot")
    st.caption("Assistente virtuale ufficiale del DIEM – Università di Salerno")

    if "retriever" not in st.session_state:
        with st.spinner("Caricamento del database della conoscenza..."):
            st.session_state.retriever = HybridRetriever()
            st.success("Database caricato con successo!")

    rag_app = build_graph()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.header("Impostazioni")
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
            "Chi è il delegato all'Internazionalizzazione del DIEM?",
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
                initial_state = {"question": user_input, "retry_count": 0}
                final_state = rag_app.invoke(initial_state)
                
                full_response = final_state["generation"]
                sources_list = final_state.get("sources", [])

                st.markdown(full_response)
                
                if show_sources and sources_list:
                    with st.expander("📄 Fonti consultate"):
                        for src in sources_list:
                            st.caption(f"• {src}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_list
            })

if __name__ == "__main__":
    main()