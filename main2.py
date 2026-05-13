import os
import re
import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# LangChain & LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# Modulo locale
from retriever import HybridRetriever

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURAZIONE LLM ---
CURRENT_MODEL = "llama3.2"

def get_llm():
    return ChatOllama(model=CURRENT_MODEL, temperature=0.1)

@st.cache_resource(show_spinner=False)
def get_retriever():
    return HybridRetriever()

# --- DEFINIZIONE TOOL ---
@tool
def search_diem_documents(testo_ricerca: str, tipo_fonte: str = "all") -> str:
    """Usa questo strumento per cercare nei documenti ufficiali del DIEM.
    MOLTO IMPORTANTE: Inserisci SOLO PAROLE CHIAVE essenziali.
    
    REGOLE PER IL PARAMETRO 'tipo_fonte':
    - Usa "web" se l'utente chiede di: aule (es. Aula delle lauree, Aula 126), orari, contatti, docenti o informazioni di servizio.
    - Usa "pdf" se l'utente chiede di: regolamenti, bandi, tesi, manifesti degli studi o burocrazia.
    - Usa "all" in tutti gli altri casi o se la prima ricerca fallisce.
    """
    retriever = get_retriever()
    
    # Passiamo il filtro al retriever locale!
    raw_docs = retriever.retrieve(testo_ricerca, tipo_fonte=tipo_fonte)
    
    if not raw_docs:
        return "Nessuna informazione trovata nei documenti ufficiali."
    
    results = []
    for doc in raw_docs:
        fonte = doc.metadata.get("source", "Fonte sconosciuta")
        results.append(f"[Fonte: {fonte}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)

# --- STATO E NODI (Invariati) ---
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

def create_agent_node(tools):
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    sys_prompt = """Sei DIEMbot, l'assistente virtuale del DIEM (Università di Salerno).
Rispondi sempre in italiano professionale. 

REGOLE PER L'USO DELLO STRUMENTO DI RICERCA:
1. Cerca SOLO ed ESCLUSIVAMENTE l'argomento dell'ULTIMA domanda dell'utente.
2. NON mescolare MAI argomenti o parole chiave di domande precedenti con quella attuale.
3. Estrai solo 1 o 2 parole chiave."""
    
    def agent(state):
        messages = state["messages"]
        full_messages = [SystemMessage(content=sys_prompt)] + messages
        response = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}
    
    return agent

def create_rewrite_node():
    llm = get_llm()
    
    def rewrite(state):
        question = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
        
        # PROMPT BLINDATO CON ESEMPI (Few-Shot)
        sys_msg = SystemMessage(content="""Sei un estrattore automatico di parole chiave. 
Il tuo UNICO scopo è estrarre 2-3 parole chiave essenziali dalla domanda.
REGOLE RIGIDE:
- NON rispondere alla domanda.
- NON aggiungere spiegazioni, saluti o contesto.
- Restituisci ESCLUSIVAMENTE il testo da cercare.

Esempi:
Domanda: Chi è Mario Vento?
Risultato: Mario Vento

Domanda: Quali sono gli orari per la segreteria studenti?
Risultato: orari segreteria studenti""")

        user_msg = HumanMessage(content=f"Domanda: {question}\nRisultato:")
        
        response = llm.invoke([sys_msg, user_msg])
        
        new_query = response.content.strip().replace('"', '').replace('\n', ' ')
        
        return {"messages": [HumanMessage(content=new_query)]}
    
    return rewrite

def create_generate_node():
    llm = get_llm()
    def generate(state):
        # Recupera l'ultimo contesto e l'ultima domanda
        context = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, ToolMessage))
        question = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
        
        # SYSTEM PROMPT POTENZIATO
        sys_prompt = """Sei DIEMbot, l'assistente virtuale istituzionale del DIEM (Università di Salerno).
Il tuo compito è rispondere alle domande degli utenti basandoti ESCLUSIVAMENTE sul contesto fornito, in italiano corretto e professionale.

REGOLE CRITICHE DI COMPORTAMENTO:
1. PRIORITÀ INFORMATIVA PER LE AULE E STRUTTURE: Se la domanda riguarda un'aula, un laboratorio o un ufficio, cerca nel contesto le parole chiave "Ubicazione", "Piano", "Edificio" e "Capienza". La tua risposta DEVE contenere la posizione fisica esatta prima di ogni altra cosa. Ignora i link ai "Calendari Occupazione" a meno che l'utente non chieda specificamente gli orari.
2. COMPLETEZZA UTILE: Se disponibili nel contesto, aggiungi dettagli come la capienza o le attrezzature presenti, rendendo la frase discorsiva.
3. DIVIETO DI META-LINGUAGGIO: Non iniziare mai le frasi con "Nel contesto fornito...", "In base ai documenti..." o "Il testo dice...". Rispondi direttamente.
4. GESTIONE LINK E URL: Non incollare mai URL grezzi o link Markdown nel testo della risposta. Il sistema gestisce le fonti separatamente.
5. TERZA PERSONA: Riferisciti sempre a docenti, personale e strutture in terza persona.

"""
        
        sys_msg = SystemMessage(content=sys_prompt)
        user_msg = HumanMessage(content=f"CONTESTO:\n{context}\n\nDOMANDA: {question}")
        
        return {"messages": [llm.invoke([sys_msg, user_msg])]}
        
    return generate

# --- GRADER E LOGICA GRAFO (Invariati) ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Rilevante? 'yes'/'no'")

def create_grade_documents():
    def grade_documents(state):
        last_msg = state["messages"][-1]
        
        # 1. Sicurezza: Se non è un messaggio del tool, vai a generate
        if not isinstance(last_msg, ToolMessage): 
            return "generate"
            
        # 2. Controllo testuale: Il database è vuoto?
        if "Nessuna informazione trovata" in last_msg.content:
            st.session_state.last_grade = {
                "score": "no", 
                "reasoning": "Nessun documento trovato dal database vettoriale."
            }
            return "rewrite"
            
        # 3. Bypass LLM: Se il Tool ha trovato documenti, CI FIDIAMO DEL CROSS-ENCODER!
        # Il Cross-Encoder in retriever.py è molto più bravo di Llama a valutare la pertinenza.
        st.session_state.last_grade = {
            "score": "yes", 
            "reasoning": "Documenti trovati e validati dal Cross-Encoder."
        }
        return "generate"
        
    return grade_documents

@st.cache_resource(show_spinner=False)
def build_graph():
    tools = [search_diem_documents]
    workflow = StateGraph(GraphState)
    workflow.add_node("agent", create_agent_node(tools))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", create_rewrite_node())
    workflow.add_node("generate", create_generate_node())
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", lambda x: "retrieve" if hasattr(x["messages"][-1], "tool_calls") and x["messages"][-1].tool_calls else END)
    workflow.add_conditional_edges("retrieve", create_grade_documents())
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)
    return workflow.compile()

# --- INTERFACCIA STREAMLIT AGGIORNATA ---

def display_sources(sources):
    """Funzione helper per visualizzare i link in modo cliccabile."""
    if sources:
        with st.expander("📚 Fonti utilizzate"):
            for src in sources:
                # Se è un URL (inizia con http), crea un link Markdown
                if src.startswith("http"):
                    st.markdown(f"- [{src}]({src})")
                else:
                    # Altrimenti mostra il testo normale (es. nome del file)
                    st.markdown(f"- {src}")

def main():
    st.set_page_config(page_title="DIEM Agentic RAG", page_icon="🎓")
    st.title("🎓 DIEMbot")
    
    get_retriever()
    if "messages" not in st.session_state: st.session_state.messages = []
    
    # Visualizza storico
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if role == "assistant":
                display_sources(msg.additional_kwargs.get("sources", []))

    # Input Utente
    user_input = st.chat_input("Fai una domanda al DIEMbot...")
    
    if user_input:
        user_msg = HumanMessage(content=user_input)
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.expander("⚙️ Processo decisionale (Agentic Graph)", expanded=True):
                st.caption("⏳ Contatto l'LLM (Ollama)...") # Feedback visivo immediato
                
                app = build_graph()
                final_answer = ""
                collected_sources = [] 
                
                try:
                    # FIX: Passiamo tutto lo storico (st.session_state.messages) per mantenere il contesto
                    inputs = {"messages": st.session_state.messages}
                    
                    for output in app.stream(inputs, config={"recursion_limit": 15}):
                        for key, value in output.items():
                            st.caption(f"--- Nodo completato: **{key}** ---")
                            
                            if key == "retrieve" and hasattr(st.session_state, 'last_grade'):
                                score = st.session_state.last_grade['score']
                                reason = st.session_state.last_grade['reasoning']
                                icon = "✅" if score.lower() == "yes" else "❌"
                                st.info(f"🔍 **Valutazione:** {icon} {score.upper()}\n\n💭 *{reason}*")
                            
                            if "messages" in value:
                                last_msg = value["messages"][-1]
                                
                                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                    for tc in last_msg.tool_calls:
                                        st.warning(f"🛠️ **Tool:** `{tc['name']}` | **Input:** `{tc['args']}`")
                                        
                                elif isinstance(last_msg, ToolMessage):
                                    st.success(f"📄 **Dati trovati (anteprima):**\n{last_msg.content[:150]}...")
                                    found_sources = re.findall(r"\[Fonte: (.*?)\]", last_msg.content)
                                    collected_sources.extend(found_sources)
                                    
                                elif key == "rewrite":
                                    st.error(f"🔄 **Nuova ricerca:** {last_msg.content}")
                                
                                if key == "generate" or (key == "agent" and not hasattr(last_msg, "tool_calls")):
                                    final_answer = last_msg.content
                                    st.write("✨ **Risposta generata!**")
                                    
                except Exception as e:
                    # Se l'LLM o il Grafo crashano, mostriamo l'errore in grande nell'interfaccia!
                    st.error(f"🚨 **Errore critico nel grafo:** {str(e)}")
                    st.stop() # Ferma l'esecuzione di Streamlit qui

            # VISUALIZZAZIONE FINALE FUORI DALL'EXPANDER
            if final_answer:
                st.markdown(final_answer)
                
                unique_sources = list(dict.fromkeys(collected_sources))
                display_sources(unique_sources)
                
                st.session_state.messages.append(
                    AIMessage(
                        content=final_answer, 
                        additional_kwargs={"sources": unique_sources}
                    )
                )

if __name__ == "__main__":
    main()