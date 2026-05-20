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
from pypdf import PdfReader
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

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return {"web": {}, "pdfs": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(state, f, indent=4)

def load_knowledge_base():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "rb") as f:
            return {doc.metadata.get("source"): doc for doc in pickle.load(f)}
    return {}

def generate_safe_filename(url):
    safe_name = re.sub(r'[\/\\?%*:|"<>&=]+', '_', url.replace("https://", "").replace("http://", "")).strip('_')
    if len(safe_name) > 200: safe_name = safe_name[:190] + "_" + hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
    return f"{safe_name}.md"

def clean_md(md_content):
    if not md_content: return ""
    txt = md_content.strip()
    txt = re.sub(r'\[skip to main content\].*?Condividi\s*(?:\d+\.\s*\[\]\(.*?\)\s*)*', '', txt, flags=re.DOTALL|re.IGNORECASE)
    txt = re.sub(r'\* \[Home \]\(.*?\).*?\[Contatti \]\(.*?\)', '', txt, flags=re.DOTALL|re.IGNORECASE)
    txt = re.sub(r'\* \[Presentazione \]\(.*?\).*?\[Strutture \]\(.*?\)', '', txt, flags=re.DOTALL|re.IGNORECASE)
    return re.sub(r'\[Vai al Contenuto della Pagina\].*', '', txt, flags=re.DOTALL|re.IGNORECASE).strip()

def is_relevant(text, url):
    is_valid = any(kw in text.upper() for kw in KEYWORDS)
    print(f"  [FILTRO] {'✅ ACCETTATO' if is_valid else '❌ SCARTATO'}: {url}")
    return is_valid

def is_recent_pdf(pdf_url):
    year_matches = re.findall(r'(?<!\d)20\d{2}(?!\d)', pdf_url)
    return any(int(y) >= 2020 for y in year_matches) if year_matches else True

# --- PARSER CALENDARI FULLCALENDAR ---
def parse_unisa_calendar_to_sentences(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    frasi = []
    titolo_calendario = soup.find('h3').get_text(strip=True) if soup.find('h3') else "Calendario Aule"
    
    fc_view = soup.find("div", class_=lambda c: c and "fc-view" in c)
    if not fc_view: return ""

    giorni = [th.get_text(strip=True) for th in fc_view.find_all("th", class_=lambda c: c and "fc-day-header" in c)]
    skeleton = fc_view.find("div", class_="fc-content-skeleton")
    if not giorni or not skeleton or not skeleton.find("tr"): return ""

    colonne = [td for td in skeleton.find("tr").find_all("td") if "fc-axis" not in td.get("class", [])]
    for idx, colonna in enumerate(colonne):
        if idx >= len(giorni): break
        data_corrente = giorni[idx]
        for evento in colonna.find_all("a", class_=lambda c: c and "fc-event" in c):
            time_div = evento.find("div", class_="fc-time")
            orario = time_div.get("data-full", time_div.get_text(strip=True)) if time_div else "Orario non definito"
            title_div = evento.find("div", class_="fc-title")
            dettaglio = re.sub(r'\s+', ' ', title_div.get_text(separator=" ", strip=True)).strip() if title_div else "Dettagli non definiti"
            if len(dettaglio) > 3:
                frasi.append(f"In {titolo_calendario}, per il giorno {data_corrente}, la fascia oraria {orario} prevede: {dettaglio}.")
    return "\n".join(frasi) if frasi else ""

def extract_links_and_pdfs(html, current_url, start_url):
    soup, links_to_visit, pdfs_to_download = BeautifulSoup(html, "html.parser"), [], []
    base_boundary = start_url.lower().rstrip('/')
    valid_boundaries = [base_boundary] + [alias.lower() for k, alias in CORSI_ALIASES.items() if base_boundary in (k.lower(), alias.lower())] + [k.lower() for k, alias in CORSI_ALIASES.items() if base_boundary in (k.lower(), alias.lower())]
    allowed_domain = urlparse(start_url).netloc

    for a in soup.find_all('a', href=True):
        href = a['href']
        if not href or href.startswith(('javascript:', 'mailto:', 'tel:')): continue
        if not href.startswith(('http', '/', '#')): href = '/' + href
        full_url = urljoin(current_url, href).split('#')[0]
        parsed = urlparse(full_url)
        
        if full_url.lower().endswith('.pdf') or '/pdf/' in full_url.lower():
            if parsed.netloc == allowed_domain: pdfs_to_download.append(full_url)
        elif any(full_url.lower().startswith(b) for b in valid_boundaries):
            if allowed_domain == "docenti.unisa.it" and [p for p in parsed.path.split('/') if p] and '.' in [p for p in parsed.path.split('/') if p][0]: continue
            if not any(full_url.lower().endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.jpeg']):
                if '/en/' not in full_url and not full_url.endswith('/en'): links_to_visit.append(full_url)
    return list(set(links_to_visit)), list(set(pdfs_to_download))

def download_and_parse_pdf(pdf_url, state, doc_dict):
    if pdf_url in state["pdfs"]: return
    try:
        res = requests.get(pdf_url, timeout=10)
        if res.status_code == 200 and 'application/pdf' in res.headers.get('Content-Type', '').lower() and is_recent_pdf(pdf_url):
            file_hash = hashlib.md5(res.content).hexdigest()
            if state["pdfs"].get(pdf_url, {}).get("hash") == file_hash: return
            if file_hash in [i["hash"] for i in state["pdfs"].values() if i.get("hash")]:
                state["pdfs"][pdf_url] = {"hash": file_hash, "filename": "DUPLICATO"}
                return
            if not res.content.startswith(b'%PDF'): return
            
            extracted_text = "\n\n".join([page.extract_text() for page in PdfReader(io.BytesIO(res.content)).pages if page.extract_text()])
            if len(extracted_text.strip()) > 50:
                filename = generate_safe_filename(pdf_url)
                with open(os.path.join(PDF_MD_DIR, filename), "w", encoding="utf-8") as f:
                    f.write(f"---\nsource: {pdf_url}\ntype: pdf\n---\n\n{extracted_text}")
                doc_dict[pdf_url] = Document(page_content=extracted_text, metadata={"source": pdf_url, "type": "pdf"})
                state["pdfs"][pdf_url] = {"hash": file_hash, "filename": filename}
                print(f"  [PDF->MD] Salvato: {filename}")
    except Exception as e:
        print(f"  [!] Errore PDF: {e}")

async def crawl_task(task, crawler, state, doc_dict):
    name, start_urls, max_depth, use_filter = task["name"], task["urls"], task["depth"], task["filter"]
    print(f"\n{'='*20} INIZIO TASK: {name} {'='*20}")
    visited = set()

    for start_url in start_urls:
        current_queue = [start_url]
        visited.add(start_url)

        for depth in range(max_depth + 1):
            if not current_queue: break
            print(f"\n  [Depth {depth}/{max_depth}] Crawling {len(current_queue)} URL in parallelo...")

            js_cleanup = "const selectors = ['#off-search', '#off-language', '#off-servizi-on-line', '#off-profili', '#box-agenda', '.homeBox', '#logo-footer', '.modal', '#blueimp-gallery', '.blueimp-gallery-controls', '.carousel-control', '.carousel-indicators', '.control-box', '#go_down', '#pause', '#resize', '#scrollUp_div', '.fc-toolbar', '.fc-header-toolbar', '#scrollUp', '#cookie-bar', '#unisa-utilities-bar', '.bg-footer', '.sub-footer'].join(', '); document.querySelectorAll(selectors).forEach(el => { if(el) el.remove(); });"
            
            results = await crawler.arun_many(
                urls=current_queue,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS, word_count_threshold=0,
                    excluded_tags=['footer', 'header', 'form', 'noscript'],
                    js_code=[js_cleanup], wait_until="networkidle", delay_before_return_html=1.0
                ),
                max_concurrent=10
            )

            next_queue = []
            for result in results:
                if not result.success or (result.status_code and result.status_code != 200): continue
                
                # INTERCETTAZIONE CALENDARI
                testo_calendario = parse_unisa_calendar_to_sentences(result.html) if "calendario-occupazione-spazi" in result.url.lower() else ""
                if testo_calendario:
                    testo_da_salvare = f"DATI ORARIO UFFICIALE ESTRATTI:\n\n{testo_calendario}"
                    print(f"  [CALENDARIO PARSATO] {result.url}")
                else:
                    testo_da_salvare = clean_md(result.markdown)

                page_hash = hashlib.md5(testo_da_salvare.encode('utf-8')).hexdigest()
                content_changed = (state["web"].get(result.url) != page_hash)
                page_is_relevant, should_explore = True, True
                
                if use_filter and depth in [0, 1]:
                    page_is_relevant = is_relevant(testo_da_salvare, result.url)
                    if depth == 1 and not page_is_relevant: should_explore = False 

                if page_is_relevant and content_changed and len(testo_da_salvare) > 10:
                    doc_dict[result.url] = Document(page_content=testo_da_salvare, metadata={"source": result.url, "type": "web"})
                    state["web"][result.url] = page_hash
                    
                    filename = generate_safe_filename(result.url)
                    with open(os.path.join(PAGES_DIR, filename), "w", encoding="utf-8") as f:
                        f.write(f"---\nsource: {result.url}\ntype: webpage\n---\n\n{testo_da_salvare}")
                    print(f"  [MD] Salvato locale: {filename}")

                if depth < max_depth and should_explore:
                    new_links, pdfs = extract_links_and_pdfs(result.html, result.url, start_url)
                    for pdf_url in pdfs: download_and_parse_pdf(pdf_url, state, doc_dict)
                    for link in new_links:
                        if link not in visited:
                            visited.add(link)
                            next_queue.append(link)
            current_queue = next_queue

async def main():
    state, doc_dict = load_state(), load_knowledge_base()
    SEARCH_TASKS = [
        {"name": "Sito DIEM", "urls": ["https://www.diem.unisa.it/"], "depth": 3, "filter": False},
        {"name": "Docenti", "urls": ["https://docenti.unisa.it/"], "depth": 2, "filter": True},
        {"name": "Corsi DIEM", "urls": CORSI_DIEM_URLS, "depth": 3, "filter": False}
    ]

    async with AsyncWebCrawler(verbose=False) as crawler:
        for task in SEARCH_TASKS:
            await crawl_task(task, crawler, state, doc_dict)

    print("\nCompilazione Knowledge Base finale in corso...")
    with open(KB_FILE, "wb") as f: pickle.dump(list(doc_dict.values()), f)
    save_state(state)

    print(f"\n{'='*12} RESOCONTO FINALE {'='*12}\n🌍 Pagine Web uniche: {len(state['web'])}\n📁 Documenti PDF unici: {len(state['pdfs'])}\n🧠 Totale 'Documenti': {len(doc_dict)}\n{'='*40}")

if __name__ == "__main__":
    asyncio.run(main())