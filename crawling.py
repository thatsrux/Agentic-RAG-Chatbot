import logging
import os
import asyncio
import hashlib
import aiohttp
import io
import re
from pypdf import PdfReader
from urllib.parse import urlparse, urljoin
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from tqdm.asyncio import tqdm

logging.getLogger("pypdf").setLevel(logging.ERROR)

DATA_PATH = r"data"
PAGES_DIR = os.path.join(DATA_PATH, "pages")
PDFS_DIR = os.path.join(DATA_PATH, "PDFs")

os.makedirs(PAGES_DIR, exist_ok=True)
os.makedirs(PDFS_DIR, exist_ok=True)

TARGET_SITES = [
    {"url": "https://www.diem.unisa.it/", "max_depth": 3},
    {"url": "https://docenti.unisa.it/", "max_depth": 2},
    {"url": "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale", "max_depth": 3},
    {"url": "https://corsi.unisa.it/ingegneria-informatica", "max_depth": 3},
    {"url": "https://corsi.unisa.it/electrical-engineering-for-digital-energy", "max_depth": 3},
    {"url": "https://corsi.unisa.it/information-Engineering-for-digital-medicine", "max_depth": 3},
    {"url": "https://corsi.unisa.it/ingegneria-informatica-magistrale", "max_depth": 3},
    {"url": "https://corsi.unisa.it/ingegneria-dell-informazione", "max_depth": 3},
    {"url": "https://corsi.unisa.it/photovoltaics", "max_depth": 3}
]

def get_domain(url):
    """Estrae il dominio da un URL per limitare la navigazione."""
    return urlparse(url).netloc

def generate_safe_filename(url, is_pdf):
    """Genera un nome file usando l'URL completo, sostituendo i caratteri non ammessi dall'OS."""
    clean_url = url.replace("https://", "").replace("http://", "")
    
    safe_name = re.sub(r'[\/\\?%*:|"<>&=]+', '_', clean_url)
    
    safe_name = safe_name.strip('_')
    
    if len(safe_name) > 200:
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
        safe_name = safe_name[:190] + "_" + url_hash
        
    ext = ".md"
    
    return f"{safe_name}{ext}"

def clean_md(md_content):
    """Pulisce il markdown rimuovendo menu laterali, intestazioni e footer."""
    clean_text = md_content.strip()
    
    clean_text = re.sub(r'\[skip to main content\].*?Condividi\s*(?:\d+\.\s*\[\]\(.*?\)\s*)*', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    
    clean_text = re.sub(r'\* \[Home \]\(.*?\).*?\[Contatti \]\(.*?\)', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'\* \[Presentazione \]\(.*?\).*?\[Strutture \]\(.*?\)', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    
    clean_text = re.sub(r'\[Vai al Contenuto della Pagina\].*', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    
    return clean_text.strip()

async def process_pdf(url, session, global_visited, source_page, depth, max_depth):
    """Scarica ed estrae il testo da un PDF in modo asincrono."""
    if url in global_visited:
        return
    global_visited.add(url)
    
    print(f"   ➔ [{depth}/{max_depth}] [PDF] ↓ {url} (Trovato in: {source_page})")
    
    try:
        async with session.get(url) as response:
            if response.status == 200:
                pdf_bytes = await response.read()
                
                if not pdf_bytes.startswith(b'%PDF'):
                    print(f"   ➔ [{depth}/{max_depth}] [!] Ignorato falso PDF: {url}")
                    return
                    
                reader = PdfReader(io.BytesIO(pdf_bytes))
                
                text = "\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                
                if len(text.strip()) > 50:
                    filename = generate_safe_filename(url, is_pdf=True)
                    save_path = os.path.join(PDFS_DIR, filename)
                    
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(f"---\nsource: {url}\nfound_in: {source_page}\ntype: pdf\n---\n\n")
                        f.write(text)
    except Exception as e:
        print(f"   ➔ [{depth}/{max_depth}] [!] Errore PDF {url}: {e}")

async def process_site(crawler, session, site, global_visited):
    base_url = site["url"]
    max_depth = site["max_depth"]
    base_domain = get_domain(base_url)
    
    current_level_urls = [base_url]
    valid_professors = set()
    
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

    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=0,
        exclude_external_links=True,
        excluded_tags=['nav', 'footer', 'header', 'aside', 'form', 'noscript'],
        js_code=[js_cleanup] 
    )

    print(f"\n🔍 [0/{max_depth}] Esploro: {base_url} (Profondità: {max_depth})")

    for depth in range(max_depth + 1):
        urls_to_crawl = list(set([u for u in current_level_urls if u not in global_visited]))
        
        if not urls_to_crawl:
            break
            
        print(f"   ➔ [{depth}/{max_depth}] Livello {depth}: elaborazione parallela di {len(urls_to_crawl)} URL...")
        
        for u in urls_to_crawl:
            global_visited.add(u)

        # Crawling parallelo delle pagine web
        results = await crawler.arun_many(urls=urls_to_crawl, config=config)
        
        next_level_urls = set()
        pdf_urls_to_process = set()
        
        for result in results:
            if not result.success:
                continue

            if "docenti.unisa.it" in result.url and result.url.rstrip("/") != "https://docenti.unisa.it":
                # Estraiamo l'ID del professore dall'URL (es. '005501')
                prof_match = re.search(r'docenti\.unisa\.it/([^/]+)', result.url)
                if prof_match:
                    prof_id = prof_match.group(1)
                    
                    # Se siamo sulla homepage del docente, valutiamo a quale dipartimento appartiene
                    if result.url.endswith(f"/{prof_id}") or result.url.endswith(f"/{prof_id}/") or "/home" in result.url:
                        html_lower = (result.html or "").lower()
                        is_diem = "diem" in html_lower or "informazione ed elettrica" in html_lower
                        
                        if is_diem:
                            valid_professors.add(prof_id) # Promosso! È del DIEM
                        else:
                            print(f"   ➔ [{depth}/{max_depth}] [SKIPPED] Docente non DIEM: {result.url}")
                            continue
                    else:
                        # Se è una sottopagina (come /pubblicazioni o /didattica)
                        # passa SOLO se il professore è già stato promosso sulla home
                        if prof_id not in valid_professors:
                            continue
            
            md_content = result.markdown
            
            if md_content:
                cleaned_text = clean_md(md_content)

                if len(cleaned_text) > 10:
                    filename = generate_safe_filename(result.url, is_pdf=False)
                    save_path = os.path.join(PAGES_DIR, filename)
                    
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(f"---\nsource: {result.url}\ntype: webpage\n---\n\n")
                        f.write(cleaned_text)
                else:
                    print(f"   ➔ [{depth}/{max_depth}] [SKIPPED] Testo troppo corto ({len(cleaned_text)} char): {result.url}")
            else:
                print(f"   ➔ [{depth}/{max_depth}] [SKIPPED] Markdown vuoto: {result.url}")
                
            
            if depth < max_depth:
                internal_links = result.links.get("internal", [])
                external_links = result.links.get("external", [])
                all_links = internal_links + external_links
                
                for link_obj in all_links:
                    href = link_obj.get("href")
                    if href:
                        absolute_url = urljoin(result.url, href).split("#")[0] 
                        
                        if ".pdf" in absolute_url.lower():
                            pdf_urls_to_process.add((absolute_url, result.url))
                        else:
                            if absolute_url.startswith(base_url):
                                next_level_urls.add(absolute_url)
        
        if pdf_urls_to_process:
            print(f"   ➔ [{depth}/{max_depth}] Trovati {len(pdf_urls_to_process)} PDF. Download e parsing in corso...")
            pdf_tasks = [process_pdf(pdf_url, session, global_visited, source_page, depth, max_depth) for pdf_url, source_page in pdf_urls_to_process]
            await asyncio.gather(*pdf_tasks)

        current_level_urls = list(next_level_urls)

async def main():
    print("🚀 Inizio il web crawling parallelo con Crawl4AI (Boilerplate rimosso)...\n")
    
    browser_config = BrowserConfig(
        headless=True,
        text_mode=True
    )
    
    global_visited = set()
    
    async with aiohttp.ClientSession() as session:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for site in TARGET_SITES:
                await process_site(crawler, session, site, global_visited)
            
    print("\n✅ Crawling completato!")
    print(f"📄 Pagine (pulite) salvate in: {PAGES_DIR}")
    print(f"📕 Testo dei PDF salvato in: {PDFS_DIR}")

if __name__ == "__main__":
    asyncio.run(main())