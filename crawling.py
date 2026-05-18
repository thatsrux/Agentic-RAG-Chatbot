import os
import requests
import pickle
import asyncio
import logging
import re
import hashlib
import json
import io
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Parsing PDF al volo
from pypdf import PdfReader

# Importiamo Crawl4AI e i Documenti di LangChain
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from langchain_core.documents import Document

logging.getLogger("pypdf").setLevel(logging.ERROR)

# --- CONFIGURAZIONE ---
DATA_PATH = r"knowledge/data"
PAGES_DIR = os.path.join(DATA_PATH, "pages")     
PDF_MD_DIR = os.path.join(DATA_PATH, "PDFs")   
STATE_FILE = "knowledge/crawler_state.json"
KB_FILE = "knowledge/knowledge_base.pkl"

os.makedirs(PAGES_DIR, exist_ok=True)
os.makedirs(PDF_MD_DIR, exist_ok=True)

KEYWORDS = ["DIEM", "DIPARTIMENTO DI INGEGNERIA DELL'INFORMAZIONE", "INGEGNERIA INFORMATICA"]
CORSI_DIEM_URLS = [
    "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
    "https://corsi.unisa.it/ingegneria-informatica",
    "https://corsi.unisa.it/electrical-engineering-for-digital-energy",
    "https://corsi.unisa.it/information-Engineering-for-digital-medicine",
    "https://corsi.unisa.it/0650107303300001", 
    "https://corsi.unisa.it/DOT18CK8F9",       
    "https://corsi.unisa.it/photovoltaics"
]
CORSI_ALIASES = {
    "https://corsi.unisa.it/0650107303300001": "https://corsi.unisa.it/ingegneria-informatica-magistrale",
    "https://corsi.unisa.it/DOT18CK8F9": "https://corsi.unisa.it/ingegneria-dell-informazione"
}

# --- GESTIONE DELLO STATO (INCREMENTAL SCRAPING) ---

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"web": {}, "pdfs": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

def load_knowledge_base():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "rb") as f:
            docs = pickle.load(f)
            return {doc.metadata.get("source"): doc for doc in docs}
    return {}

# --- FUNZIONI DI SUPPORTO ---

def generate_safe_filename(url):
    """Genera un nome file leggibile e sicuro per il SO, identico a crawling.py"""
    clean_url = url.replace("https://", "").replace("http://", "")
    safe_name = re.sub(r'[\/\\?%*:|"<>&=]+', '_', clean_url)
    safe_name = safe_name.strip('_')
    
    if len(safe_name) > 200:
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
        safe_name = safe_name[:190] + "_" + url_hash
        
    return f"{safe_name}.md"

def clean_md(md_content):
    """Pulisce il markdown rimuovendo menu laterali, intestazioni e footer."""
    if not md_content:
        return ""
    clean_text = md_content.strip()
    clean_text = re.sub(r'\[skip to main content\].*?Condividi\s*(?:\d+\.\s*\[\]\(.*?\)\s*)*', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'\* \[Home \]\(.*?\).*?\[Contatti \]\(.*?\)', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'\* \[Presentazione \]\(.*?\).*?\[Strutture \]\(.*?\)', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'\[Vai al Contenuto della Pagina\].*', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    return clean_text.strip()

def is_relevant(text, url):
    text_upper = text.upper()
    is_valid = any(kw in text_upper for kw in KEYWORDS)
    if is_valid:
        print(f"  [FILTRO] ✅ ACCETTATO: {url}")
    else:
        print(f"  [FILTRO] ❌ SCARTATO: {url}")
    return is_valid

def is_recent_pdf(pdf_url):
    year_matches = re.findall(r'(?<!\d)20\d{2}(?!\d)', pdf_url)
    if year_matches:
        return any(int(y) >= 2020 for y in year_matches)
    return True

def download_and_parse_pdf(pdf_url, state, doc_dict):
    if pdf_url in state["pdfs"]:
        return

    try:
        res = requests.get(pdf_url, timeout=10)
        if res.status_code == 200 and 'application/pdf' in res.headers.get('Content-Type', '').lower():
            
            if not is_recent_pdf(pdf_url):
                print(f"  [SKIP] PDF antecedente al 2020: {pdf_url}")
                return

            file_hash = hashlib.md5(res.content).hexdigest()
            old_info = state["pdfs"].get(pdf_url)

            if old_info and old_info["hash"] == file_hash:
                return

            existing_hashes = [info["hash"] for info in state["pdfs"].values() if info.get("hash")]
            if file_hash in existing_hashes and not old_info:
                state["pdfs"][pdf_url] = {"hash": file_hash, "filename": "DUPLICATO_CONTENUTO"}
                print(f"  [DEDUPLICAZIONE] Contenuto PDF già presente, salto: {pdf_url}")
                return

            if not res.content.startswith(b'%PDF'):
                return
                
            reader = PdfReader(io.BytesIO(res.content))
            extracted_text = "\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            if len(extracted_text.strip()) > 50:
                filename = generate_safe_filename(pdf_url)
                filepath = os.path.join(PDF_MD_DIR, filename)
                
                # --- RIPRISTINO YAML FRONTMATTER ---
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"---\nsource: {pdf_url}\ntype: pdf\n---\n\n")
                    f.write(extracted_text)
                
                doc = Document(page_content=extracted_text, metadata={"source": pdf_url, "type": "pdf"})
                doc_dict[pdf_url] = doc
                
                state["pdfs"][pdf_url] = {"hash": file_hash, "filename": filename}
                print(f"  [PDF->MD] {'AGGIORNATO' if old_info else 'NUOVO PARSING'}: {filename}")
                
    except Exception as e:
        print(f"  [!] Errore parsing PDF: {e}")

def extract_links_and_pdfs(html, current_url, start_url):
    soup = BeautifulSoup(html, "html.parser")
    links_to_visit = []
    pdfs_to_download = []
    
    boundary = start_url.lower().rstrip('/')
    allowed_domain = urlparse(boundary).netloc

    valid_boundaries = [boundary]

    for key, alias in CORSI_ALIASES.items():
        if boundary == key.lower():
            valid_boundaries.append(alias.lower())
        elif boundary == alias.lower():
            valid_boundaries.append(key.lower())

    for a in soup.find_all('a', href=True):
        href = a['href']

        if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
            continue

        if not href.startswith(('http', '/', '#')):
            href = '/' + href

        full_url = urljoin(current_url, href).split('#')[0]
        parsed = urlparse(full_url)
        link_domain = parsed.netloc

        if full_url.lower().endswith('.pdf') or '/pdf/' in full_url.lower():
            if link_domain == allowed_domain:
                pdfs_to_download.append(full_url)

        elif any(full_url.lower().startswith(b) for b in valid_boundaries):  # ← confronto lowercase unificato
            if allowed_domain == "docenti.unisa.it":
                path_parts = [p for p in parsed.path.split('/') if p]
                if path_parts and '.' in path_parts[0]:
                    continue

            if not any(full_url.lower().endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.jpeg']):
                if '/en/' not in full_url and not full_url.endswith('/en'):
                    links_to_visit.append(full_url)

    return list(set(links_to_visit)), list(set(pdfs_to_download))

async def crawl_task(task, crawler, state, doc_dict):
    name = task["name"]
    start_urls = task["urls"]
    max_depth = task["depth"]
    use_filter = task["filter"]

    print(f"\n{'='*20} INIZIO TASK: {name} {'='*20}")
    visited = set()

    for start_url in start_urls:
        print(f"\n>>> Analisi per: {start_url}")
        current_queue = [start_url]
        visited.add(start_url)

        for depth in range(max_depth + 1):
            if not current_queue:
                break

            print(f"\n  [Depth {depth}/{max_depth}] Crawling {len(current_queue)} URL in parallelo...")

            js_cleanup = """
            const selectors = [
                '#off-search', '#off-language', '#off-servizi-on-line', '#off-profili',
                '#box-agenda', '.homeBox', '#logo-footer',
                '.modal', '#blueimp-gallery', '.blueimp-gallery-controls',
                '.carousel-control', '.carousel-indicators', '.control-box', '#go_down', '#pause', '#resize',
                '#scrollUp_div', '.fc-toolbar', '.fc-header-toolbar', '#scrollUp',
                '#cookie-bar', '#unisa-utilities-bar', '.bg-footer', '.sub-footer'
            ].join(', ');
            document.querySelectorAll(selectors).forEach(el => { if(el) el.remove(); });
            """

            results = await crawler.arun_many(
                urls=current_queue,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=0,
                    excluded_tags=['footer', 'header', 'form', 'noscript'],
                    js_code=[js_cleanup] 
                    ),
                max_concurrent=10
            )

            next_queue = []

            for result in results:
                if not result.success:
                    continue

                if result.status_code and result.status_code != 200:
                    print(f"  [{result.status_code}] Salto: {result.url}")
                    continue

                cleaned_markdown = clean_md(result.markdown)
                page_hash = hashlib.md5(cleaned_markdown.encode('utf-8')).hexdigest()
                old_hash = state["web"].get(result.url)
                content_changed = (old_hash != page_hash)
                
                page_is_relevant = True
                should_explore = True
                
                if use_filter:
                    if depth == 0:
                        page_is_relevant = is_relevant(cleaned_markdown, result.url)
                    elif depth == 1:
                        page_is_relevant = is_relevant(cleaned_markdown, result.url)
                        if not page_is_relevant:
                            should_explore = False 
                    else:
                        page_is_relevant = True

                if page_is_relevant:
                    if not content_changed:
                        print(f"  [CACHE] Invariato, salto salvataggio: {result.url}")
                    else:
                        if len(cleaned_markdown) > 10:
                            doc = Document(page_content=cleaned_markdown, metadata={"source": result.url, "type": "web"})
                            doc_dict[result.url] = doc
                            
                            filename = generate_safe_filename(result.url)
                            nome_file = os.path.join(PAGES_DIR, filename)
                            
                            # --- RIPRISTINO YAML FRONTMATTER ---
                            with open(nome_file, "w", encoding="utf-8") as f:
                                f.write(f"---\nsource: {result.url}\ntype: webpage\n---\n\n")
                                f.write(cleaned_markdown)
                            
                            state["web"][result.url] = page_hash
                            print(f"  [MD] {'AGGIORNATO' if old_hash else 'NUOVO'}: {filename}")
                        else:
                            print(f"  [SKIPPED] Testo troppo corto dopo la pulizia: {result.url}")

                if depth < max_depth and should_explore:
                    new_links, pdfs = extract_links_and_pdfs(result.html, result.url, start_url)

                    # --- NUOVA STAMPA VISIVA ---
                    if not content_changed:
                         print(f"    ↳ [ESPLORAZIONE] Estratti {len(new_links)} link e {len(pdfs)} PDF dalla pagina in cache.")
                    # --------------------------

                    for pdf_url in pdfs:
                        download_and_parse_pdf(pdf_url, state, doc_dict)

                    for link in new_links:
                        if link not in visited:
                            visited.add(link)
                            next_queue.append(link)

            current_queue = next_queue

async def main():
    state = load_state()
    doc_dict = load_knowledge_base()

    SEARCH_TASKS = [
        {"name": "Sito DIEM", 
         "urls": ["https://www.diem.unisa.it/"], 
         "depth": 3, 
         "filter": False},
        {"name": "Docenti", 
         "urls": ["https://docenti.unisa.it/"], 
         "depth": 2, 
         "filter": True},
        {"name": "Corsi DIEM", 
         "urls": CORSI_DIEM_URLS, 
         "depth": 3, 
         "filter": False}
    ]

    async with AsyncWebCrawler(verbose=False) as crawler:
        for task in SEARCH_TASKS:
            await crawl_task(task, crawler, state, doc_dict)

    print(f"\nCompilazione Knowledge Base finale in corso...")
    
    knowledge_base = list(doc_dict.values())
    
    with open(KB_FILE, "wb") as f:
        pickle.dump(knowledge_base, f)
        
    save_state(state)

    pdf_files_count = len(state["pdfs"])
    web_files_count = len(state["web"])

    print(f"\n{'='*12} RESOCONTO FINALE {'='*12}")
    print(f"🌍 Pagine Web uniche tracciate: {web_files_count}")
    print(f"📁 Documenti PDF unici tracciati (convertiti): {pdf_files_count}")
    print(f"🧠 Totale 'Documenti' salvati nella Knowledge Base: {len(knowledge_base)}")
    print(f"{'='*40}")
    print("Stato incrementale salvato. Al prossimo avvio verificherò solo le variazioni!")

if __name__ == "__main__":
    asyncio.run(main())