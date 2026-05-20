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
    }}
    .tooltip-container .tooltip-text {{
        visibility: hidden;
        width: max-content;
        background-color: #1e293b;
        color: #e2e8f0;
        text-align: center;
        border-radius: 6px;
        padding: 5px 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%; 
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        font-family: sans-serif;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.2);
    }}
    .tooltip-container .tooltip-text::after {{
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #1e293b transparent transparent transparent;
    }}
    .tooltip-container:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    <div class="tooltip-container">
        <div class="info-icon">i</div>
        <span class="tooltip-text">Generato con: {model_name}</span>
    </div>
    """