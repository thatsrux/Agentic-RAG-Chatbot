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

def is_relevant(text):
    """Verifica se il testo è pertinente al DIEM."""
    text_upper = text.upper()
    return any(kw in text_upper for kw in KEYWORDS)

# def is_recent_pdf(pdf_url, headers):
#     # Controllo 1: anno 4 cifre come segmento directory (es. /2025/)
#     year_match = re.search(r'/(20\d{2})/', pdf_url)
#     if year_match:
#         return int(year_match.group(1)) >= 2020

#     # Controllo 2: anno 4 cifre nel nome file (es. _2019_, -2018-)
#     year_match = re.search(r'[\-_](20\d{2})[\-_\.]', pdf_url)
#     if year_match:
#         return int(year_match.group(1)) >= 2020

#     # Controllo 3: anno 2 cifre nel nome file (es. -14-, -17-, -19-)
#     # Interpreta XX < 50 come 20XX (es. 17 → 2017)
#     # year_match = re.search(r'[\-_](\d{2})[\-_\.]', os.path.basename(pdf_url))
#     # if year_match:
#     #     year_2digit = int(year_match.group(1))
#     #     full_year = 2000 + year_2digit
#     #     return full_year >= 2020

#     # Controllo 4: header Last-Modified
#     last_modified = headers.get('Last-Modified', '')
#     if last_modified:
#         try:
#             from email.utils import parsedate
#             date_tuple = parsedate(last_modified)
#             if date_tuple:
#                 return date_tuple[0] >= 2020
#         except Exception:
#             pass

#     # Nessuna info: includi per sicurezza
#     return True

def is_recent_pdf(pdf_url, headers):
    """Restituisce False solo se troviamo esplicitamente un anno < 2020."""
    
    # Cerca qualsiasi anno 4 cifre nell'URL (directory o nome file)
    year_matches = re.findall(r'(?<!\d)20\d{2}(?!\d)', pdf_url)
    if year_matches:
        # Basta che almeno uno degli anni trovati sia >= 2020
        return any(int(y) >= 2020 for y in year_matches)

    # Nessun anno trovato → includi per sicurezza
    return True

def download_pdf(pdf_url, downloaded_urls, downloaded_hashes):
    """Scarica i PDF, evita i duplicati tramite Hash ed evita le sovrascritture."""
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

            # Determinazione del nome file
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
    """Estrae i link validi e individua i PDF."""
    soup = BeautifulSoup(html, "html.parser")
    links_to_visit = []
    pdfs_to_download = []
    
    # Identifichiamo il dominio base per i PDF (es. "corsi.unisa.it" o "www.diem.unisa.it")
    allowed_domain = urlparse(start_url).netloc
    
    for a in soup.find_all('a', href=True):
        href = a['href']

        if not href.startswith(('http', '/', '#')):
            href = '/' + href

        full_url = urljoin(current_url, href).split('#')[0]
        link_domain = urlparse(full_url).netloc
        
        # 1. LOGICA PDF: "Maglia Larga" -> Basta che sia nello stesso dominio base (cattura gli /uploads/)
        if full_url.lower().endswith('.pdf') or '/pdf/' in full_url.lower():
            if link_domain == allowed_domain:
                pdfs_to_download.append(full_url)
                
        # 2. LOGICA NAVIGAZIONE HTML: "Maglia Stretta" -> Deve iniziare ESATTAMENTE con lo start_url
        elif full_url.startswith(start_url):
            if not any(full_url.lower().endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.jpeg']):
                # Escludiamo le versioni inglesi se non ci interessano
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
        
        # Primo livello: URL di partenza
        current_queue = [start_url]
        visited.add(start_url)

        for depth in range(max_depth + 1):
            if not current_queue:
                break

            print(f"\n  [Depth {depth}/{max_depth}] Crawling {len(current_queue)} URL in parallelo...")

            # Esegui tutti gli URL del livello corrente in parallelo
            results = await crawler.arun_many(
                urls=current_queue,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                ),
                max_concurrent=10
            )

            next_queue = []

            for result in results:
                if not result.success:
                    print(f"  [!] Errore su {result.url}: {result.error_message}")
                    continue

                print(f"  [OK] {result.url}")

                if not use_filter or is_relevant(result.markdown):
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
                    print(f"  [MD] Salvato: {os.path.basename(nome_file)}")

                # Esplora link solo se non siamo all'ultimo livello
                if depth < max_depth:
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
        #{"name": "Docenti", "urls": ["https://docenti.unisa.it/"], "depth": 2, "filter": True},
        {"name": "Corsi DIEM", "urls": CORSI_DIEM_URLS, "depth": 3, "filter": False}
    ]

    async with AsyncWebCrawler(verbose=False) as crawler:
        for task in SEARCH_TASKS:
            await crawl_task(task, crawler, downloaded_urls, downloaded_hashes, knowledge_base)

    # Caricamento PDF e salvataggio finale
    print(f"\nParsing dei {len(downloaded_hashes)} PDF unici...")
    logging.getLogger("pypdf").setLevel(logging.ERROR) 
    
    pdf_loader = PyPDFDirectoryLoader(PDF_DIR)
    pdf_docs = pdf_loader.load()
    knowledge_base.extend(pdf_docs) 

    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(knowledge_base, f)

    print(f"\n--- FINE ---")
    print(f"Totale pagine HTML: {len(knowledge_base) - len(pdf_docs)}")
    print(f"Totale pagine PDF: {len(pdf_docs)}")
    print(f"Totale PDF unici: {len(downloaded_hashes)}")

if __name__ == "__main__":
    asyncio.run(main())