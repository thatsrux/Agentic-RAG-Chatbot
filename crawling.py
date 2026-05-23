import os
import pickle
import asyncio
import logging
import hashlib
from langchain_core.documents import Document
from utils.crawling_utils import *
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

logging.getLogger("pypdf").setLevel(logging.ERROR)

os.makedirs(PAGES_DIR, exist_ok=True)
os.makedirs(PDF_MD_DIR, exist_ok=True)

async def crawl_task(task, crawler, state, doc_dict):
    """
    Esegue un'attività di crawling definita da un dizionario contenente il nome del task, gli URL di partenza, la profondità massima e se applicare un filtro di rilevanza.
    """
    name, start_urls, max_depth, use_filter = task["name"], task["urls"], task["depth"], task["filter"]
    print(f"\n{'='*20} INIZIO TASK: {name} {'='*20}")
    visited = set()

    for start_url in start_urls:
        current_queue = [start_url]
        visited.add(start_url)

        for depth in range(max_depth + 1):
            if not current_queue: break
            print(f"\n  [Depth {depth}/{max_depth}] Crawling {len(current_queue)} URL in parallelo...")

            js_cleanup = "const selectors = ['#cookie-bar', '#unisa-utilities-bar', '.bg-footer', '.sub-footer'].join(', '); document.querySelectorAll(selectors).forEach(el => { if(el) el.remove(); });"
            
            results = await crawler.arun_many(
                urls=current_queue,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS, word_count_threshold=0,
                    excluded_tags=['footer', 'header', 'form', 'noscript'],
                    js_code=[js_cleanup], wait_until="networkidle"
                ),
                max_concurrent=10
            )

            next_queue = []
            for result in results:
                if not result.success or (result.status_code and result.status_code != 200 and result.status_code != 301 and result.status_code != 302): continue
                
                if "calendario-occupazione-spazi" in result.url.lower():
                    testo_calendario = parse_unisa_calendar_to_sentences(result.html)
                    if testo_calendario:
                        testo_da_salvare = f"DATI ORARIO UFFICIALE ESTRATTI:\n\n{testo_calendario}"
                        print(f"  [CALENDARIO PARSATO] {result.url}")
                    else:
                        testo_da_salvare = clean_md(result.markdown)
                        
                elif "easycourse" in result.url.lower():
                    testo_easycourse = parse_easycourse_table(result.html)
                    if testo_easycourse:
                        testo_da_salvare = f"DATI ORARI/ESAMI ESTRATTI:\n\n{testo_easycourse}"
                        print(f"  [DATI ESTRATTI] {result.url}")
                    else:
                        testo_da_salvare = clean_md(result.markdown)
                        
                else:
                    testo_da_salvare = clean_md(result.markdown)

                page_hash = hashlib.md5(testo_da_salvare.encode('utf-8')).hexdigest()
                content_changed = (state["web"].get(result.url) != page_hash)

                if not content_changed:
                    print(f"  [CACHE] ⏩ Invariato: {result.url} (Nessuna modifica rilevata, salto scrittura)")

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
    """
    Punto di ingresso principale del crawler asincrono. Definisce le attività di crawling, gestisce lo stato e la knowledge base, e stampa un resoconto finale al termine dell'esecuzione.
    """
    state, doc_dict = load_state(), load_knowledge_base()
    SEARCH_TASKS = [
        {"name": "Sito DIEM", "urls": ["https://www.diem.unisa.it/"], "depth": 3, "filter": False},
        {"name": "Docenti", "urls": ["https://docenti.unisa.it/"], "depth": 3, "filter": True},
        {"name": "Corsi DIEM", "urls": CORSI_DIEM_URLS, "depth": 3, "filter": False},
        {"name": "EasyCourse Orari", "urls": EASYCOURSE, "depth": 2, "filter": False},
        {"name": "EasyTest Esami", "urls": EASYTEST_ESAMI_URLS, "depth": 1, "filter": False}
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