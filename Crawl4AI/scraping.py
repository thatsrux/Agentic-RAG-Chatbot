import os
import requests
import pickle
import asyncio
import logging
import re
import hashlib
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Importiamo Crawl4AI e i Documenti di LangChain
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader

# --- CONFIGURAZIONE ---
DATA_PATH = r"data"
PDF_DIR = os.path.join(DATA_PATH, "PDFs")
PAGES_DIR = os.path.join(DATA_PATH, "pages")
STATE_FILE = "crawler_state.json"
KB_FILE = "knowledge_base.pkl"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PAGES_DIR, exist_ok=True)

KEYWORDS = ["DIEM", "DIPARTIMENTO DI INGEGNERIA DELL'INFORMAZIONE", "INGEGNERIA INFORMATICA"]
# Link specifici dei corsi DIEM
CORSI_DIEM_URLS = [
    "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
    "https://corsi.unisa.it/ingegneria-informatica",
    #"https://corsi.unisa.it/electrical-engineering-for-digital-energy",
    #"https://corsi.unisa.it/information-Engineering-for-digital-medicine",
    #"https://corsi.unisa.it/ingegneria-informatica-magistrale",
    #"https://corsi.unisa.it/ingegneria-dell-informazione",
    #"https://corsi.unisa.it/photovoltaics"
]

# --- GESTIONE DELLO STATO (INCREMENTAL SCRAPING) ---

def load_state():
    """Carica la memoria delle run precedenti."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # Struttura iniziale se è la prima run
    return {"web": {}, "pdfs": {}}

def save_state(state):
    """Salva lo stato aggiornato su disco."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

def load_knowledge_base():
    """Carica la KB esistente e la converte in un dizionario per facili aggiornamenti."""
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "rb") as f:
            docs = pickle.load(f)
            # Mappiamo URL -> Documento (Escludiamo i PDF perché li ricarichiamo sempre freschi dalla cartella)
            return {doc.metadata.get("source"): doc for doc in docs if not doc.metadata.get("source", "").endswith(".pdf")}
    return {}

# --- FUNZIONI DI SUPPORTO ---

def is_relevant(text, url):
    text_upper = text.upper()
    is_valid = any(kw in text_upper for kw in KEYWORDS)
    if is_valid:
        print(f"  [FILTRO] ✅ ACCETTATO: {url}")
    else:
        print(f"  [FILTRO] ❌ SCARTATO: {url}")
    return is_valid

def is_recent_pdf(pdf_url, headers):
    year_matches = re.findall(r'(?<!\d)20\d{2}(?!\d)', pdf_url)
    if year_matches:
        return any(int(y) >= 2020 for y in year_matches)
    return True

def download_pdf(pdf_url, state):
    """Scarica i PDF evitando duplicati sia per URL che per Contenuto (Hash)."""
    # 1. Controllo veloce: abbiamo già processato questo ESATTO URL?
    if pdf_url in state["pdfs"]:
        return

    try:
        res = requests.get(pdf_url, timeout=10)
        if res.status_code == 200 and 'application/pdf' in res.headers.get('Content-Type', '').lower():
            if not is_recent_pdf(pdf_url, res.headers):
                return

            file_hash = hashlib.md5(res.content).hexdigest()
            
            # --- NUOVO FIX: Controllo se l'hash esiste già sotto QUALSIASI altro URL ---
            existing_hashes = [info["hash"] for info in state["pdfs"].values()]
            if file_hash in existing_hashes:
                state["pdfs"][pdf_url] = {"hash": file_hash, "filename": "DUPLICATO_CONTENUTO"}
                print(f"  [DEDUPLICAZIONE] Contenuto già presente (da altro link), salto: {pdf_url}")
                return
            # -----------------------------------------------------------------------

            # --- RECUPERO DEL NOME FILE ---
            filename = None
            cd = res.headers.get('content-disposition')
            if cd and 'filename=' in cd:
                fnames = re.findall('filename=["\']?([^"\';]+)["\']?', cd)
                if fnames:
                    filename = fnames[0]
            
            if not filename:
                base_name = os.path.basename(pdf_url).split("?")[0]
                if not base_name or base_name.lower() == 'pdf' or base_name.isnumeric():
                    base_name = f"doc_{file_hash[:6]}"
                filename = base_name if base_name.lower().endswith('.pdf') else f"{base_name}.pdf"

            filepath = os.path.join(PDF_DIR, filename)
            name_part, ext_part = os.path.splitext(filename)
            counter = 1
            
            # Anti-sovrascrittura per nomi uguali ma contenuti diversi
            while os.path.exists(filepath):
                filename = f"{name_part}_{counter}{ext_part}"
                filepath = os.path.join(PDF_DIR, filename)
                counter += 1
            # ------------------------------

            # Salvataggio fisico
            with open(filepath, "wb") as f:
                f.write(res.content)
            
            # Aggiornamento stato
            state["pdfs"][pdf_url] = {"hash": file_hash, "filename": filename}
            print(f"  [PDF] Nuovo file unico scaricato: {filename}")
            
    except Exception as e:
        print(f"  [!] Errore download PDF: {e}")

def extract_links_and_pdfs(html, current_url, start_url):
    soup = BeautifulSoup(html, "html.parser")
    links_to_visit = []
    pdfs_to_download = []
    allowed_domain = urlparse(start_url).netloc  
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        if not href.startswith(('http', '/', '#')):
            href = '/' + href

        full_url = urljoin(current_url, href).split('#')[0]
        link_domain = urlparse(full_url).netloc
        
        if full_url.lower().endswith('.pdf') or '/pdf/' in full_url.lower():
            if link_domain == allowed_domain:
                pdfs_to_download.append(full_url)
                
        elif full_url.startswith(start_url):
            if allowed_domain == "docenti.unisa.it":
                path_parts = [p for p in urlparse(full_url).path.split('/') if p]
                if path_parts:
                    prof_id_or_name = path_parts[0]
                    if '.' in prof_id_or_name:
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
                '#off-rubrica', '#off-search', '#off-language', '#off-servizi-on-line', '#off-profili',
                '#menu-bar', '#box-agenda',
                '.homeBox', '#logo-footer',
                '.modal', '#blueimp-gallery', '.blueimp-gallery', '.blueimp-gallery-controls',
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
                    excluded_tags=['nav', 'footer', 'header', 'aside', 'form', 'noscript'],
                    js_code=[js_cleanup] 
                    ),
                max_concurrent=10
            )

            next_queue = []

            for result in results:
                if not result.success:
                    continue

                # --- CONTROLLO HASH (MODIFICA PAGINA) ---
                page_hash = hashlib.md5(result.markdown.encode('utf-8')).hexdigest()
                old_hash = state["web"].get(result.url)
                
                content_changed = (old_hash != page_hash)
                
                # --- VALUTAZIONE FILTRO ---
                page_is_relevant = True
                should_explore = True
                
                if use_filter:
                    if depth == 0:
                        page_is_relevant = is_relevant(result.markdown, result.url)
                    elif depth == 1:
                        page_is_relevant = is_relevant(result.markdown, result.url)
                        if not page_is_relevant:
                            should_explore = False 
                    else:
                        page_is_relevant = True

                # --- AGGIORNAMENTO DATI (Solo se rilevante E se il contenuto è cambiato) ---
                if page_is_relevant:
                    if not content_changed:
                        print(f"  [CACHE] Invariato, salto salvataggio: {result.url}")
                    else:
                        # 1. Aggiorna la Base di Conoscenza
                        doc = Document(page_content=result.markdown, metadata={"source": result.url})
                        doc_dict[result.url] = doc
                        
                        # 2. Salva il file Markdown (Nome basato sull'Hash dell'URL, stabile!)
                        url_hash_name = hashlib.md5(result.url.encode()).hexdigest()[:10]
                        nome_file = os.path.join(PAGES_DIR, f"page_{url_hash_name}.md")
                        
                        with open(nome_file, "w", encoding="utf-8") as f:
                            f.write(f"SOURCE: {result.url}\n{'='*50}\n\n{result.markdown}")
                        
                        # 3. Aggiorna lo stato
                        state["web"][result.url] = page_hash
                        print(f"  [MD] {'AGGIORNATO' if old_hash else 'NUOVO'}: {result.url}")

                # --- ESTRAZIONE LINK (Deve essere fatta sempre per poter navigare) ---
                if depth < max_depth and should_explore:
                    new_links, pdfs = extract_links_and_pdfs(result.html, result.url, start_url)

                    for pdf_url in pdfs:
                        download_pdf(pdf_url, state)

                    for link in new_links:
                        if link not in visited:
                            visited.add(link)
                            next_queue.append(link)

            current_queue = next_queue

async def main():
    # 1. Carica la memoria precedente
    state = load_state()
    doc_dict = load_knowledge_base()

    SEARCH_TASKS = [
        #{"name": "Sito DIEM", "urls": ["https://www.diem.unisa.it/"], "depth": 3, "filter": False},
        {"name": "Docenti", "urls": ["https://docenti.unisa.it/"], "depth": 2, "filter": True},
        #{"name": "Corsi DIEM", "urls": CORSI_DIEM_URLS, "depth": 3, "filter": False}
    ]

    # 2. Esegui lo scraping (che aggiornerà 'state' e 'doc_dict')
    async with AsyncWebCrawler(verbose=False) as crawler:
        for task in SEARCH_TASKS:
            await crawl_task(task, crawler, state, doc_dict)

    # 3. Gestione PDF finali
    print(f"\nCompilazione Knowledge Base finale...")
    logging.getLogger("pypdf").setLevel(logging.ERROR) 
    
    knowledge_base = list(doc_dict.values())
    
    pdf_loader = PyPDFDirectoryLoader(PDF_DIR)
    pdf_docs = pdf_loader.load()
    knowledge_base.extend(pdf_docs) 

    # 4. Salva tutto su disco
    with open(KB_FILE, "wb") as f:
        pickle.dump(knowledge_base, f)
        
    save_state(state)

    # --- IL NUOVO CONTO FINALE ---
    pdf_files_count = len([f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')])

    print(f"\n{'='*12} RESOCONTO FINALE {'='*12}")
    print(f"🌍 Pagine Web uniche (Markdown): {len(doc_dict)}")
    print(f"📁 Documenti PDF unici scaricati: {pdf_files_count}")
    print(f"📑 Pagine testuali lette dentro i PDF: {len(pdf_docs)}")
    print(f"🧠 Totale 'Documenti' in Knowledge Base: {len(knowledge_base)}")
    print(f"{'='*40}")
    print("Stato incrementale salvato. Al prossimo avvio scaricherò solo le novità!")

if __name__ == "__main__":
    asyncio.run(main())