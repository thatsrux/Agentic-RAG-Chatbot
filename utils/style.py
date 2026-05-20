def get_global_css() -> str:
    """CSS globale per migliorare UX e UI dell'intera app."""
    return """
    <style>
    /* Importa un font moderno (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Stile della barra di Input Chat */
    [data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid #334155 !important;
        background-color: #0f172a !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2), 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* Stile dei bottoni "Esempi di domande" nella sidebar */
    div[data-testid="stSidebar"] button {
        border-radius: 10px !important;
        border: 1px solid #1e293b !important;
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        text-align: left !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease-in-out !important;
        font-size: 13px !important;
        line-height: 1.4 !important;
    }
    
    /* Effetto Hover sui bottoni della sidebar */
    div[data-testid="stSidebar"] button:hover {
        border-color: #38bdf8 !important;
        background-color: #0f172a !important;
        transform: translateX(4px); /* Leggero scivolamento a destra */
        color: #38bdf8 !important;
    }
    
    /* Evidenzia il bottone "Cancella chat" */
    div[data-testid="stSidebar"] button:first-of-type {
        background-color: #7f1d1d !important; /* Rosso scuro */
        border-color: #991b1b !important;
        text-align: center !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSidebar"] button:first-of-type:hover {
        background-color: #991b1b !important;
        border-color: #b91c1c !important;
        color: white !important;
        transform: none; /* Annulla lo scivolamento per questo bottone */
    }

    /* Stile dei messaggi dell'assistente (per distinguerli meglio) */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """

def get_info_icon_html(model_name: str) -> str:
    """Genera l'HTML e il CSS per l'icona 'i' con il tooltip."""
    return f"""
    <style>
    .tooltip-container {{
        position: relative;
        display: inline-block;
        float: right;
        cursor: help;
        margin-left: 10px;
    }}
    .info-icon {{
        color: #94a3b8;
        font-size: 14px;
        border: 1px solid #94a3b8;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-style: italic;
        font-family: serif;
        transition: all 0.2s ease;
    }}
    .info-icon:hover {{
        color: #38bdf8;
        border-color: #38bdf8;
        background-color: rgba(56, 189, 248, 0.1);
    }}
    .tooltip-container .tooltip-text {{
        visibility: hidden;
        width: max-content;
        background-color: #1e293b;
        color: #e2e8f0;
        text-align: center;
        border-radius: 6px;
        padding: 6px 12px;
        position: absolute;
        z-index: 1;
        bottom: 135%; 
        left: 50%;
        transform: translateX(-50%) translateY(5px);
        opacity: 0;
        transition: all 0.3s ease;
        font-size: 12px;
        font-family: 'Inter', sans-serif;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }}
    .tooltip-container .tooltip-text::after {{
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #334155 transparent transparent transparent;
    }}
    .tooltip-container:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }}
    </style>
    <div class="tooltip-container">
        <div class="info-icon">i</div>
        <span class="tooltip-text">Generato con: <b style="color: #38bdf8;">{model_name}</b></span>
    </div>
    """