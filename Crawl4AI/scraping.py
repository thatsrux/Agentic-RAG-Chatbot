import os
import requests
import pickle
import asyncio
import logging
import re
import hashlib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Importiamo Crawl4AI e i Documenti di LangChain
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader

# --- CONFIGURAZIONE DIRECTORY ---
PDF_DIR = "./diem_pdfs"
MD_DIR = "./markdown_estratti"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)

# Parole chiave per il filtraggio 
KEYWORDS = ["DIEM", "DIPARTIMENTO DI INGEGNERIA DELL'INFORMAZIONE", "INGEGNERIA INFORMATICA"]

# Link specifici dei corsi DIEM
CORSI_DIEM_URLS = [
    #"https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
    "https://corsi.unisa.it/ingegneria-informatica",
    #"https://corsi.unisa.it/electrical-engineering-for-digital-energy",
    #"https://corsi.unisa.it/information-Engineering-for-digital-medicine",
    #"https://corsi.unisa.it/ingegneria-informatica-magistrale",
    #"https://corsi.unisa.it/ingegneria-dell-informazione",
    #"https://corsi.unisa.it/photovoltaics"
]

def is_relevant(text, url):
    """Verifica se il testo è pertinente al DIEM."""
    text_upper = text.upper()
    is_valid = any(kw in text_upper for kw in KEYWORDS)
    
    if is_valid:
        print(f"  [FILTRO] ✅ ACCETTATO: {url}")
    else:
        print(f"  [FILTRO] ❌ SCARTATO: {url}")
        
    return is_valid

def is_recent_pdf(pdf_url, headers):
    """Restituisce False solo se troviamo esplicitamente un anno < 2020."""
    year_matches = re.findall(r'(?<!\d)20\d{2}(?!\d)', pdf_url)
    if year_matches:
        return any(int(y) >= 2020 for y in year_matches)
    return True

def download_pdf(pdf_url, downloaded_urls, downloaded_hashes):
    """Scarica i PDF, evita duplicati e sovrascritture."""
    if pdf_url in downloaded_urls:
        return

    try:
        res = requests.get(pdf_url, timeout=10)
        ctype = res.headers.get('Content-Type', '').lower()
        
        if res.status_code == 200 and 'application/pdf' in ctype:
            if not is_recent_pdf(pdf_url, res.headers):
                print(f"  [SKIP] PDF antecedente al 2020: {pdf_url}")
                return
            file_hash = hashlib.md5(res.content).hexdigest()
            
            if file_hash in downloaded_hashes:
                downloaded_urls.add(pdf_url)
                return

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
            while os.path.exists(filepath):
                filename = f"{name_part}_{counter}{ext_part}"
                filepath = os.path.join(PDF_DIR, filename)
                counter += 1

            with open(filepath, "wb") as f:
                f.write(res.content)
            
            downloaded_urls.add(pdf_url)
            downloaded_hashes.add(file_hash)
            print(f"  [PDF] Scaricato: {filename}")
    except Exception:
        pass

def extract_links_and_pdfs(html, current_url, start_url):
    """Estrae i link bloccando gli alias nome.cognome e gestendo PDF esterni."""
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

async def crawl_task(task, crawler, downloaded_urls, downloaded_hashes, knowledge_base):
    name = task["name"]
    start_urls = task["urls"]
    max_depth = task["depth"]
    use_filter = task["filter"]

    print(f"\n{'='*20} INIZIO TASK: {name} {'='*20}")

    visited = set()

    for start_url in start_urls:
        print(f"\n>>> Analisi specifica per: {start_url}")
        
        current_queue = [start_url]
        visited.add(start_url)

        for depth in range(max_depth + 1):
            if not current_queue:
                break

            print(f"\n  [Depth {depth}/{max_depth}] Crawling {len(current_queue)} URL in parallelo...")

            results = await crawler.arun_many(
                urls=current_queue,
                config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
                max_concurrent=10
            )

            next_queue = []

            for result in results:
                if not result.success:
                    print(f"  [!] Errore su {result.url}: {result.error_message}")
                    continue

                # --- VALUTAZIONE FILTRO BASATA SULLA PROFONDITÀ ---
                page_is_relevant = True
                should_explore = True
                
                if use_filter:
                    if depth == 0:
                        # Root: Testiamo per vedere se c'è testo rilevante, ma esploriamo a prescindere.
                        page_is_relevant = is_relevant(result.markdown, result.url)
                        should_explore = True 
                        
                    elif depth == 1:
                        # Depth 1: Il VERO POSTO DI BLOCCO
                        page_is_relevant = is_relevant(result.markdown, result.url)
                        if not page_is_relevant:
                            should_explore = False # PRUNING! Tagliamo il ramo
                            print(f"    ↳ Ramo potato: Il filtro ha bloccato {result.url}")
                            
                    else:
                        # Depth > 1: Disattiviamo il filtro. Se siamo qui, il Depth 1 era valido.
                        page_is_relevant = True
                        should_explore = True
                        print(f"  [FILTRO] BYPASS (Depth {depth} interno): {result.url}")
                # ----------------------------------------------------

                # SALVATAGGIO
                if page_is_relevant:
                    doc = Document(page_content=result.markdown, metadata={"source": result.url})
                    knowledge_base.append(doc)

                    parsed_path = urlparse(result.url).path.rstrip('/')
                    page_name = parsed_path.split('/')[-1] or "home"
                    nome_file = os.path.join(MD_DIR, f"{page_name}.md")
                    counter = 1
                    while os.path.exists(nome_file):
                        nome_file = os.path.join(MD_DIR, f"{page_name}_{counter}.md")
                        counter += 1

                    with open(nome_file, "w", encoding="utf-8") as f:
                        f.write(f"SOURCE: {result.url}\n{'='*50}\n\n{result.markdown}")

                # ESTRAZIONE LINK
                if depth < max_depth and should_explore:
                    new_links, pdfs = extract_links_and_pdfs(result.html, result.url, start_url)

                    for pdf_url in pdfs:
                        download_pdf(pdf_url, downloaded_urls, downloaded_hashes)

                    for link in new_links:
                        if link not in visited:
                            visited.add(link)
                            next_queue.append(link)

            current_queue = next_queue

    print(f"\n{'='*20} FINE TASK: {name} {'='*20}")

async def main():
    knowledge_base = []
    downloaded_urls = set()
    downloaded_hashes = set()

    SEARCH_TASKS = [
        #{"name": "Sito DIEM", "urls": ["https://www.diem.unisa.it/"], "depth": 3, "filter": False},
        {"name": "Docenti", "urls": ["https://docenti.unisa.it/"], "depth": 2, "filter": True},
        #{"name": "Corsi DIEM", "urls": CORSI_DIEM_URLS, "depth": 4, "filter": False}
    ]

    async with AsyncWebCrawler(verbose=False) as crawler:
        for task in SEARCH_TASKS:
            await crawl_task(task, crawler, downloaded_urls, downloaded_hashes, knowledge_base)

    print(f"\nParsing dei {len(downloaded_hashes)} PDF unici...")
    logging.getLogger("pypdf").setLevel(logging.ERROR) 
    
    pdf_loader = PyPDFDirectoryLoader(PDF_DIR)
    pdf_docs = pdf_loader.load()
    knowledge_base.extend(pdf_docs) 

    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(knowledge_base, f)

    print(f"\n--- FINE ---")
    print(f"Totale pagine HTML salvate: {len(knowledge_base) - len(pdf_docs)}")
    print(f"Totale pagine PDF: {len(pdf_docs)}")
    print(f"Totale PDF unici scaricati: {len(downloaded_hashes)}")

if __name__ == "__main__":
    asyncio.run(main())