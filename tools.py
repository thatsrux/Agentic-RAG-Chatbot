from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# --- CONFIGURAZIONE LLM PER I TOOL ---
# Creiamo un'istanza leggera del client qui per evitare importazioni circolari dal file principale.
def get_tool_llm():
    return ChatOllama(
        model="llama3.2", 
        temperature=0.1,
        num_ctx=2048 
    )

@tool
def rewrite_query_tool(query: str) -> str:
    """
    Riscrive la domanda dell'utente per renderla più efficace,
    specifica e adatta al semantic retrieval.
    """
    llm = get_tool_llm()
    
    prompt = PromptTemplate.from_template(
        "Sei un ottimizzatore di query di ricerca. "
        "Riscrivi la seguente domanda dell'utente per renderla più efficace "
        "per il recupero semantico dei documenti. Rendila specifica, auto-contenuta, "
        "e usa la terminologia tecnica appropriata dove necessario."
        "Rispondi soltanto con la query riscritta, senza spiegazioni o testo aggiuntivo come 'Ritorna' o 'Restituisci'.\n\n"
        "Domanda utente: {query}\n\n"
        "Query riscritta:"
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})

@tool
def multi_query_tool(query: str, n: int = 3) -> list[str]:
    """
    Genera {n} varianti della query originale per esplorare
    diverse angolazioni dell'argomento.
    """
    llm = get_tool_llm()
    
    prompt = PromptTemplate.from_template(
        "Sei un ottimizzatore di query di ricerca. "
        "Genera {n} diverse riformulazioni della seguente domanda, "
        "ognuna esplorando un'angolazione o un aspetto diverso dell'argomento. "
        "Non allontanarti dalla domanda originale e non inventare informazioni non presenti nella query."
        "Se la query originale chiede COSA, non trasformarla in un COME o PERCHÉ, ma esplora invece diversi modi di chiedere COSA. "
        "Restituisci SOLO la lista di domande, una per riga.\n\n"
        "Domanda originale: {query}\n\n"
        "Riformulazioni:"
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query, "n": n})
    
    variations = [line.lstrip("- *1234567890. ").strip() for line in response.split("\n") if line.strip()]
    return variations