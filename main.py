import os
import re
import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from retriever import HybridRetriever

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource(show_spinner=False)
def get_retriever() -> HybridRetriever:
    """Carica e mantiene in cache l'istanza del retriever."""
    return HybridRetriever()

@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOllama:
    """Carica e mantiene in cache l'istanza del LLM."""
    return ChatOllama(model="llama3.2", temperature=0.1)

@tool
def search_diem_documents(testo_ricerca: str, tipo_fonte: str = "all") -> str:
    """Usa questo strumento per cercare nei documenti ufficiali del DIEM.
    MOLTO IMPORTANTE: Inserisci SOLO PAROLE CHIAVE essenziali.

    REGOLE PER IL PARAMETRO 'tipo_fonte':
    - Usa "web" se l'utente chiede di: aule, orari, contatti, docenti o informazioni di servizio.
    - Usa "pdf" se l'utente chiede di: regolamenti, bandi, tesi, manifesti degli studi o burocrazia.
    - Usa "all" in tutti gli altri casi o se la prima ricerca fallisce.
    """
    raw_docs = get_retriever().retrieve(testo_ricerca, tipo_fonte=tipo_fonte)
    if not raw_docs:
        return "Nessuna informazione trovata nei documenti ufficiali."
    results = []
    for doc in raw_docs:
        fonte = doc.metadata.get("source", "Fonte sconosciuta")
        results.append(f"[Fonte: {fonte}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

_SYS_AGENT = """Sei DIEMbot, l'assistente virtuale del DIEM (Università di Salerno).
Rispondi sempre in italiano professionale.

REGOLE PER L'USO DELLO STRUMENTO DI RICERCA:
1. Cerca SOLO ed ESCLUSIVAMENTE l'argomento dell'ULTIMA domanda dell'utente.
2. NON mescolare MAI argomenti o parole chiave di domande precedenti con quella attuale.
3. Estrai solo 1 o 2 parole chiave."""

_SYS_REWRITE = """Sei un estrattore automatico di parole chiave.
Il tuo UNICO scopo è estrarre 2-3 parole chiave essenziali dalla domanda.
REGOLE RIGIDE:
- NON rispondere alla domanda.
- NON aggiungere spiegazioni, saluti o contesto.
- Restituisci ESCLUSIVAMENTE il testo da cercare.

Esempi:
Domanda: Chi è Mario Vento?
Risultato: Mario Vento

Domanda: Quali sono gli orari per la segreteria studenti?
Risultato: orari segreteria studenti"""

_SYS_GENERATE = """Sei DIEMbot, l'assistente virtuale istituzionale del DIEM (Università di Salerno).
Il tuo compito è rispondere alle domande degli utenti basandoti ESCLUSIVAMENTE sul contesto fornito, in italiano corretto e professionale.

REGOLE CRITICHE DI COMPORTAMENTO:
1. PRIORITÀ INFORMATIVA PER LE AULE E STRUTTURE: Se la domanda riguarda un'aula, un laboratorio o un ufficio, cerca nel contesto le parole chiave "Ubicazione", "Piano", "Edificio" e "Capienza". La tua risposta DEVE contenere la posizione fisica esatta prima di ogni altra cosa.
2. COMPLETEZZA UTILE: Se disponibili nel contesto, aggiungi dettagli come la capienza o le attrezzature presenti, rendendo la frase discorsiva.
3. DIVIETO DI META-LINGUAGGIO: Non iniziare mai le frasi con "Nel contesto fornito...", "In base ai documenti..." o "Il testo dice...". Rispondi direttamente.
4. GESTIONE LINK E URL: Non incollare mai URL grezzi o link Markdown nel testo della risposta.
5. TERZA PERSONA: Riferisciti sempre a docenti, personale e strutture in terza persona."""

def agent_node(state):
    llm_with_tools = get_llm().bind_tools([search_diem_documents])
    response = llm_with_tools.invoke([SystemMessage(content=_SYS_AGENT)] + state["messages"])
    return {"messages": [response]}

def rewrite_node(state):
    question = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    response = get_llm().invoke([
        SystemMessage(content=_SYS_REWRITE),
        HumanMessage(content=f"Domanda: {question}\nRisultato:")
    ])
    new_query = response.content.strip().replace('"', '').replace('\n', ' ')
    return {"messages": [HumanMessage(content=new_query)]}

def generate_node(state):
    context  = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, ToolMessage))
    question = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    response = get_llm().invoke([
        SystemMessage(content=_SYS_GENERATE),
        HumanMessage(content=f"CONTESTO:\n{context}\n\nDOMANDA: {question}")
    ])
    return {"messages": [response]}

def grade_documents(state):
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage):
        return "generate"
    if "Nessuna informazione trovata" in last_msg.content:
        st.session_state.last_grade = {"score": "no",  "reasoning": "Nessun documento trovato dal database vettoriale."}
        return "rewrite"
    st.session_state.last_grade = {"score": "yes", "reasoning": "Documenti trovati e validati dal Cross-Encoder."}
    return "generate"

@st.cache_resource(show_spinner=False)
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("agent",    agent_node)
    workflow.add_node("retrieve", ToolNode([search_diem_documents]))
    workflow.add_node("rewrite",  rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        lambda x: "retrieve" if hasattr(x["messages"][-1], "tool_calls") and x["messages"][-1].tool_calls else END
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)
    return workflow.compile()

def display_sources(sources):
    if sources:
        with st.expander("📚 Fonti utilizzate"):
            for src in sources:
                if src.startswith("http"):
                    st.markdown(f"- [{src}]({src})")
                else:
                    st.markdown(f"- {src}")

def main():
    st.set_page_config(page_title="DIEM Agentic RAG", page_icon="🎓")
    st.title("🎓 DIEMbot")

    # Pre-caricamento al lancio
    get_retriever()
    get_llm()
    build_graph()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("🔄 Reset"):
        st.cache_resource.clear()
        st.rerun()

    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if role == "assistant":
                display_sources(msg.additional_kwargs.get("sources", []))

    user_input = st.chat_input("Fai una domanda al DIEMbot...")

    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.expander("⚙️ Processo decisionale", expanded=True):
                st.caption("⏳ Elaborazione in corso...")

                final_answer = ""
                collected_sources = []

                try:
                    for output in build_graph().stream({"messages": st.session_state.messages}, config={"recursion_limit": 15}):
                        for key, value in output.items():
                            st.caption(f"--- Nodo: **{key}** ---")

                            if key == "retrieve" and hasattr(st.session_state, 'last_grade'):
                                score = st.session_state.last_grade['score']
                                icon  = "✅" if score == "yes" else "❌"
                                st.info(f"🔍 **Valutazione:** {icon} {score.upper()} — {st.session_state.last_grade['reasoning']}")

                            if "messages" in value:
                                last_msg = value["messages"][-1]

                                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                    for tc in last_msg.tool_calls:
                                        st.warning(f"🛠️ **Tool:** `{tc['name']}` | **Input:** `{tc['args']}`")

                                elif isinstance(last_msg, ToolMessage):
                                    with st.expander("📄 Contenuto recuperato dal DB", expanded=False):
                                        st.markdown(last_msg.content)
                                    found_sources = re.findall(r"\[Fonte: (.*?)\]", last_msg.content)
                                    collected_sources.extend(found_sources)

                                elif key == "rewrite":
                                    st.error(f"🔄 **Nuova ricerca:** {last_msg.content}")

                                if key == "generate" or (key == "agent" and not hasattr(last_msg, "tool_calls")):
                                    final_answer = last_msg.content
                                    st.write("✨ **Risposta generata!**")

                except Exception as e:
                    st.error(f"🚨 **Errore critico nel grafo:** {str(e)}")
                    st.stop()

            if final_answer:
                st.markdown(final_answer)
                unique_sources = list(dict.fromkeys(collected_sources))
                display_sources(unique_sources)
                st.session_state.messages.append(
                    AIMessage(content=final_answer, additional_kwargs={"sources": unique_sources})
                )

if __name__ == "__main__":
    main()