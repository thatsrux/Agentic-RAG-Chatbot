import os
import requests
import pickle
import asyncio
import logging
import re
import hashlib
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Importiamo Crawl4AI e i Documenti di LangChain
from crawl4ai import AsyncWebCrawler
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
    #"https://corsi.unisa.it/ingegneria-informatica",
    #"https://corsi.unisa.it/electrical-engineering-for-digital-energy",
    #"https://corsi.unisa.it/information-Engineering-for-digital-medicine",
    "https://corsi.unisa.it/ingegneria-informatica-magistrale",
    #"https://corsi.unisa.it/ingegneria-dell-informazione",
    #"https://corsi.unisa.it/photovoltaics"
]

def is_relevant(text):
    """Verifica se il testo è pertinente al DIEM."""
    text_upper = text.upper()
    return any(kw in text_upper for kw in KEYWORDS)

def download_pdf(pdf_url, downloaded_urls, downloaded_hashes):
    """Scarica i PDF, evita i duplicati tramite Hash ed evita le sovrascritture."""
    if pdf_url in downloaded_urls:
        return

    try:
        res = requests.get(pdf_url, timeout=10)
        ctype = res.headers.get('Content-Type', '').lower()
        
        if res.status_code == 200 and 'application/pdf' in ctype:
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
    
    for a in soup.find_all('a', href=True):
        href = a['href']

        if not href.startswith(('http', '/', '#')):
            href = '/' + href

        full_url = urljoin(current_url, href).split('#')[0]
        
        if full_url.lower().endswith('.pdf') or '/pdf/' in full_url.lower():
            pdfs_to_download.append(full_url)
        elif full_url.startswith(start_url):
            if not any(full_url.lower().endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.jpeg']):
                links_to_visit.append(full_url)
                
    return list(set(links_to_visit)), list(set(pdfs_to_download))

async def crawl_task(task, crawler, downloaded_urls, downloaded_hashes, knowledge_base):
    """Esegue lo scraping con un loop esplicito per ogni URL di partenza."""
    name = task["name"]
    start_urls = task["urls"]
    max_depth = task["depth"]
    use_filter = task["filter"]

    print(f"\n{'='*20} INIZIO TASK: {name} {'='*20}")
    
    # visited è condiviso tra tutti gli URL dello stesso task per non ri-visitare pagine comuni
    visited = set()

    # Iteriamo esplicitamente uno alla volta i link dei corsi (o del sito principale)
    for start_url in start_urls:
        print(f"\n>>> Analisi specifica per: {start_url}")
        
        # Coda BFS per l'URL corrente
        queue = [(start_url, 0)]
        
        while queue:
            current_url, current_depth = queue.pop(0)

            if current_url in visited:
                continue
            visited.add(current_url)

            print(f"  [Depth {current_depth}/{max_depth}] Analizzando: {current_url}")

            result = await crawler.arun(url=current_url)
            
            if not result.success:
                print(f"  [!] Errore su {current_url}: {result.error_message}")
                continue

            # Filtro e aggiunta documento
            if not use_filter or is_relevant(result.markdown):
                doc = Document(page_content=result.markdown, metadata={"source": current_url})
                knowledge_base.append(doc)
                
                # Salvataggio Markdown Live
                doc_id = len(knowledge_base)
                nome_file = os.path.join(MD_DIR, f"pagina_{doc_id}.md")
                with open(nome_file, "w", encoding="utf-8") as f:
                    f.write(f"SOURCE: {current_url}\n{'='*50}\n\n{result.markdown}")
                print(f"  [MD] Salvato: pagina_{doc_id}.md")

            # Esplorazione nuovi link
            if current_depth < max_depth:
                new_links, pdfs = extract_links_and_pdfs(result.html, current_url, start_url)
                
                for pdf_url in pdfs:
                    download_pdf(pdf_url, downloaded_urls, downloaded_hashes)
                    
                for link in new_links:
                    if link not in visited:
                        queue.append((link, current_depth + 1))

    print(f"\n{'='*20} FINE TASK: {name} {'='*20}")

async def main():
    knowledge_base = []
    downloaded_urls = set()
    downloaded_hashes = set()

    SEARCH_TASKS = [
        #{"name": "Sito DIEM", "urls": ["https://www.diem.unisa.it/"], "depth": 3, "filter": False},
        #{"name": "Docenti", "urls": ["https://docenti.unisa.it/"], "depth": 2, "filter": True},
        {"name": "Corsi DIEM", "urls": CORSI_DIEM_URLS, "depth": 4, "filter": False}
    ]

    async with AsyncWebCrawler(verbose=False) as crawler:
        for task in SEARCH_TASKS:
            await crawl_task(task, crawler, downloaded_urls, downloaded_hashes, knowledge_base)

    # Caricamento PDF e salvataggio finale
    print(f"\nParsing dei {len(downloaded_hashes)} PDF unici...")
    logging.getLogger("pypdf").setLevel(logging.ERROR) 
    
    pdf_loader = PyPDFDirectoryLoader(PDF_DIR)
    knowledge_base.extend(pdf_loader.load()) 

    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(knowledge_base, f)

    print(f"\n--- FINE ---")
    print(f"Totale pagine HTML: {len(knowledge_base) - len(downloaded_hashes)}")
    print(f"Totale PDF unici: {len(downloaded_hashes)}")
    print("Database 'knowledge_base.pkl' aggiornato.")

if __name__ == "__main__":
    asyncio.run(main())