import os
import requests
import pickle
import asyncio
import logging
import re
import hashlib
import json
import io
import pandas as pd

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
EXCEL_DIR = os.path.join(DATA_PATH, "excels") 

STATE_FILE = "knowledge/crawler_state.json"
KB_FILE = "knowledge/knowledge_base.pkl"

os.makedirs(PAGES_DIR, exist_ok=True)
os.makedirs(PDF_MD_DIR, exist_ok=True)
os.makedirs(EXCEL_DIR, exist_ok=True)

# URL Base
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

EASYCOURSE_ORARI_URLS = [
    "https://easycourse.unisa.it/EasyCourse/Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2025-2026/index.html",
    "https://easycourse.unisa.it/EasyCourse/Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2024-2025/index.html",
    "https://easycourse.unisa.it/EasyCourse/Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2023-2024/index.html",
    "https://easycourse.unisa.it/EasyCourse/Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2022-2023/index.html",
    "https://easycourse.unisa.it/EasyCourse/Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2021-2022/index.html",
]

# URL di partenza di EasyRoom
EASYROOM_URLS = [
    "https://easycourse.unisa.it/EasyRoom/index.php?content=view_prenotazioni&vista=day&area=2&_lang=it",   # Edificio E
    "https://easycourse.unisa.it/EasyRoom/index.php?content=view_prenotazioni&vista=day&area=37&_lang=it",  # Edificio E1
    "https://easycourse.unisa.it/EasyRoom/index.php?content=view_prenotazioni&vista=day&area=36&_lang=it",  # Edificio E2
]

# --- CREAZIONE SESSIONE HTTP GLOBALE ---
http_session = requests.Session()
http_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://easycourse.unisa.it/"
})
try:
    http_session.get("https://easycourse.unisa.it/EasyRoom/index.php", timeout=10)
except:
    pass

# --- GESTIONE DELLO STATO ---

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
            if "excels" not in state: state["excels"] = {}
            return state
    return {"web": {}, "pdfs": {}, "excels": {}}

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
    clean_url = url.replace("https://", "").replace("http://", "")
    safe_name = re.sub(r'[\/\\?%*:|"<>&=]+', '_', clean_url)
    safe_name = safe_name.strip('_')
    
    if len(safe_name) > 200:
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
        safe_name = safe_name[:190] + "_" + url_hash
        
    return f"{safe_name}.md"

def clean_md(md_content):
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
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"---\nsource: {pdf_url}\ntype: pdf\n---\n\n")
                    f.write(extracted_text)
                
                doc = Document(page_content=extracted_text, metadata={"source": pdf_url, "type": "pdf"})
                doc_dict[pdf_url] = doc
                
                state["pdfs"][pdf_url] = {"hash": file_hash, "filename": filename}
                print(f"  [PDF->MD] {'AGGIORNATO' if old_info else 'NUOVO PARSING'}: {filename}")
                
    except Exception as e:
        print(f"  [!] Errore parsing PDF: {e}")

# --- PARSING EXCEL IN MEMORIA (100% DINAMICO E BLINDATO) ---
# --- PARSING EXCEL IN MEMORIA (IBRIDO: VECCHIO EASYROOM + NUOVO EASYCOURSE) ---
def download_and_parse_excel(excel_url, state, doc_dict):
    """Scarica l'Excel in RAM. Usa il parsing hardcodato per EasyRoom, e quello dinamico per gli altri."""
    if excel_url in state["excels"]:
        return

    try:
        res = http_session.get(excel_url, timeout=15)
        if res.status_code != 200:
            return

        file_hash = hashlib.md5(res.content).hexdigest()
        old_info = state["excels"].get(excel_url)

        if old_info and old_info["hash"] == file_hash:
            return

        safe_name = generate_safe_filename(excel_url).replace('.md', '')
        
        # --- BIVIO: EASYROOM vs EASYCOURSE ---
        is_easyroom = "easyroom" in excel_url.lower() and "area=" in excel_url.lower()

        if is_easyroom:
            # =====================================================================
            # 1. VECCHIO PARSING ORIGINALE (SOLO PER EASYROOM)
            # =====================================================================
            try:
                # Legge l'HTML in RAM esattamente come faceva dal disco
                dfs = pd.read_html(io.BytesIO(res.content))
                df = dfs[0]
            except Exception as e:
                print(f"  [!] Impossibile leggere EasyRoom come HTML: {e}")
                return
            
            # Assegnazione nome Edificio (come nel tuo vecchio script)
            url_lower = excel_url.lower()
            if "area=2&" in url_lower or url_lower.endswith("area=2"):
                edificio = "Edificio E"
            elif "area=37" in url_lower:
                edificio = "Edificio E1"
            elif "area=36" in url_lower:
                edificio = "Edificio E2"
            else:
                edificio = "Sede Non Specificata"
            
            data_orario = "Aggiornato all'ultimo download"
            
            cleaned_columns = []
            for col_name in df.columns:
                nome_stringa = str(col_name)
                nome_pulito = re.split(r'\d+\s*posti', nome_stringa, flags=re.IGNORECASE)[0].strip()
                cleaned_columns.append(nome_pulito)
            df.columns = cleaned_columns
            
            testo_documento = f"# Orari e Prenotazioni Aule - {edificio} ({data_orario})\n\n"
            testo_documento += "| Fascia Oraria | Aula | Edificio | Data | Dettagli Lezione/Evento |\n"
            testo_documento += "|---|---|---|---|---|\n"
            
            righe_valide = 0
            for index, row in df.iterrows():
                orario = str(row.iloc[0]).strip()
                if not orario or orario.lower() == 'nan':
                    continue
                for i in range(1, len(df.columns)):
                    if i >= len(row): continue
                    cella = str(row.iloc[i]).strip()
                    
                    # Ignoriamo il testo JavaScript che finisce nelle celle e i "non autorizzato"
                    if cella and cella.lower() not in ['nan', 'non autorizzato'] and "document.cookie" not in cella and len(cella) > 3:
                        aula = str(df.columns[i]).strip()
                        if aula.lower() == 'nan': aula = f"Colonna {i}"
                        cella_pulita = cella.replace('\n', ' ').replace('\r', ' ')
                        testo_documento += f"| {orario} | {aula} | {edificio} | {data_orario} | {cella_pulita} |\n"
                        righe_valide += 1

        else:
            # =====================================================================
            # 2. NUOVO PARSING DINAMICO (PER EASYCOURSE E TUTTO IL RESTO)
            # =====================================================================
            try:
                soup = BeautifulSoup(res.content, "html.parser")
                for tag in soup(["script", "style", "noscript", "meta", "link"]):
                    tag.extract()
                cleaned_html = str(soup)
                dfs = pd.read_html(io.StringIO(cleaned_html))
                df = max(dfs, key=lambda x: x.size)
            except Exception:
                try:
                    content_io = io.BytesIO(res.content)
                    df = pd.read_excel(content_io)
                except Exception as e2:
                    print(f"  [!] Impossibile parsare l'Excel in memoria {excel_url}: {e2}")
                    return

            metadata_estratta = ""
            
            if any("Unnamed" in str(c) for c in df.columns):
                prima_colonna = str(df.columns[0])
                if "Corso di laurea" in prima_colonna or "Curriculum" in prima_colonna:
                    metadata_estratta = prima_colonna.replace('\n', ' | ')

                for idx, row in df.iterrows():
                    row_str = " ".join(str(v).lower() for v in row)
                    if "luned" in row_str and "marted" in row_str:
                        df.columns = [str(c).strip() if pd.notna(c) else "" for c in row]
                        df = df.iloc[idx+1:].reset_index(drop=True)
                        break
            
            cleaned_columns = []
            for col_name in df.columns:
                nome_stringa = str(col_name)
                if nome_stringa.endswith('.0'): nome_stringa = nome_stringa[:-2]
                nome_pulito = re.split(r'\d+\s*posti', nome_stringa, flags=re.IGNORECASE)[0].strip()
                cleaned_columns.append(nome_pulito)
            df.columns = cleaned_columns

            is_timetable = False
            if len(df) > 0 and len(df.columns) > 1:
                for check_idx in range(min(10, len(df))):
                    if re.search(r'\d{2}:\d{2}', str(df.iloc[check_idx, 0])):
                        is_timetable = True
                        break

            testo_documento = f"# Estrazione Dati: {safe_name}\n\n"
            if metadata_estratta:
                testo_documento += f"**Info Corso:** {metadata_estratta}\n\n"

            righe_valide = 0

            if is_timetable:
                testo_documento += "| Fascia Oraria | Giorno / Aula | Dettagli Lezione/Evento |\n"
                testo_documento += "|---|---|---|\n"
                
                for index, row in df.iterrows():
                    orario = str(row.iloc[0]).strip()
                    if not re.search(r'\d{2}:\d{2}', orario):
                        continue
                    
                    for i in range(1, len(df.columns)):
                        if i >= len(row): continue
                        cella = str(row.iloc[i]).strip()
                        if cella and cella.lower() not in ['nan', 'non autorizzato'] and len(cella) > 3:
                            intestazione_colonna = str(df.columns[i]).strip()
                            if intestazione_colonna.lower() == 'nan' or not intestazione_colonna:
                                intestazione_colonna = f"Colonna {i}"
                            cella_pulita = cella.replace('\n', ' ').replace('\r', ' ')
                            testo_documento += f"| {orario} | {intestazione_colonna} | {cella_pulita} |\n"
                            righe_valide += 1
            else:
                headers = " | ".join(df.columns)
                separators = " | ".join(["---"] * len(df.columns))
                testo_documento += f"| {headers} |\n| {separators} |\n"
                
                for index, row in df.iterrows():
                    if row.isna().all(): continue 
                    row_str = " | ".join(str(cell).replace('\n', ' ').replace('\r', '').replace('|', '').strip() if pd.notna(cell) else "" for cell in row)
                    testo_documento += f"| {row_str} |\n"
                    righe_valide += 1

        # =====================================================================
        # 3. SALVATAGGIO IN MARKDOWN COMUNE A ENTRAMBI I METODI
        # =====================================================================
        if righe_valide == 0:
            return

        md_filename = f"{safe_name}.md"
        md_filepath = os.path.join(EXCEL_DIR, md_filename)
        source_id = f"excel_{md_filename}"
        
        doc = Document(
            page_content=testo_documento, 
            metadata={"source": source_id, "type": "excel"}
        )
        doc_dict[source_id] = doc
        
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(f"---\nsource: {source_id}\ntype: excel\n---\n\n")
            f.write(testo_documento)
            
        state["excels"][excel_url] = {"hash": file_hash, "filename": md_filename}
        print(f"  [EXCEL->MD] {'AGGIORNATO' if old_info else 'NUOVO PARSING IN-MEMORY'}: {md_filename} ({righe_valide} righe)")

    except Exception as e:
        print(f"  [!] Errore parsing in-memory da URL {excel_url}: {e}")

def extract_links_and_pdfs(html, current_url, start_url):
    soup = BeautifulSoup(html, "html.parser")
    links_to_visit = []
    pdfs_to_download = []
    excels_to_download = []
    
    base_boundary = start_url.lower().rstrip('/')
    allowed_domain = urlparse(start_url).netloc
    is_easycourse = (allowed_domain == "easycourse.unisa.it")
    
    if is_easycourse:
        if base_boundary.endswith('.html'):
            base_boundary = base_boundary.rsplit('/', 1)[0]
        elif '/easyroom/index.php' in base_boundary:
            base_boundary = base_boundary.split('?')[0]
    
    valid_boundaries = [base_boundary]
    
    for key, alias in CORSI_ALIASES.items():
        if base_boundary == key.lower():
            valid_boundaries.append(alias.lower())
        elif base_boundary == alias.lower():
            valid_boundaries.append(key.lower())

    for tag in soup.find_all(['a', 'frame', 'iframe', 'option']):
        href = tag.get('href') or tag.get('src') or tag.get('value')
        
        if not href or href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
            continue
        if href in ['export_xls', 'null', '-- scegli --']:
            continue

        if not is_easycourse:
            if not href.startswith(('http', '/', '#')):
                href = '/' + href

        full_url = urljoin(current_url, href).split('#')[0]
        parsed = urlparse(full_url)
        link_domain = parsed.netloc
        
        url_l = full_url.lower()

        if url_l.endswith('.pdf') or '/pdf/' in url_l:
            if link_domain == allowed_domain:
                pdfs_to_download.append(full_url)

        elif url_l.endswith(('.xls', '.xlsx')) or 'esporta=xls' in url_l:
            if link_domain == allowed_domain:
                excels_to_download.append(full_url)

        elif any(url_l.startswith(b) for b in valid_boundaries):

            if allowed_domain == "docenti.unisa.it":
                path_parts = [p for p in parsed.path.split('/') if p]
                if path_parts and '.' in path_parts[0]:
                    continue

            if is_easycourse:
                if any(t in url_l for t in ['date=', 'data=', 'day=', 'week=', 'month=', 'year=', 'periodo=', 'settimana=']):
                    continue
                
                export_traps = ['esporta=', 'print=', 'ical=', 'view=pdf']
                if any(t in url_l for t in export_traps):
                    continue

                if '_lang=' in url_l and '_lang=it' not in url_l:
                    continue

                is_whitelisted = False
                
                if url_l.endswith('/index.html') or url_l.endswith('/tree.html') or url_l.endswith('/main.html'):
                    is_whitelisted = True
                elif any(url_l.endswith(view) for view in ['/tthtml.html', '/ttcdlhtml.html', '/ttteacherhtml.html', '/ttdayhtml.html']):
                    is_whitelisted = True
                elif '/curricula/' in url_l:
                    if '_1_comune_' not in url_l and '_2_' not in url_l:
                        is_whitelisted = True
                elif '/easyroom/' in url_l:
                    if any(f"area={a}&" in url_l or url_l.endswith(f"area={a}") for a in ['2', '36', '37']):
                        is_whitelisted = True
                        
                if not is_whitelisted:
                    continue
            
            if not any(url_l.endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.jpeg']):
                if '/en/' not in full_url and not full_url.endswith('/en'):
                    links_to_visit.append(full_url)

    return list(set(links_to_visit)), list(set(pdfs_to_download)), list(set(excels_to_download))

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

            if "easycourse" in start_url.lower():
                js_cleanup = """
                const selectors = [
                    '#off-search', '#off-language', '#logo-footer',
                    '.modal', '#cookie-bar', '.bg-footer', '.sub-footer'
                ].join(', ');
                document.querySelectorAll(selectors).forEach(el => { if(el) el.remove(); });
                """
            else:
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

                if result.status_code and result.status_code != 200 and result.status_code != 301 and result.status_code != 302:
                    print(f"  [{result.status_code}] Salto: {result.url}")
                    continue

                skip_webpage_save = False

                # --- MAGIA EASYROOM: INIEZIONE DOWNLOAD ---
                if "easycourse" in result.url.lower() and "/easyroom/" in result.url.lower():
                    if "esporta=xls" not in result.url.lower():
                        if "export_xls" in result.html or "esporta in excel" in result.html.lower():
                            excel_url = result.url + ("&esporta=xls" if "?" in result.url else "?esporta=xls")
                            download_and_parse_excel(excel_url, state, doc_dict)
                            # Per EasyRoom salta il salvataggio della pagina web, abbiamo l'Excel pulito
                            skip_webpage_save = True

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

                if page_is_relevant and not skip_webpage_save:
                    if not content_changed:
                        print(f"  [CACHE] Invariato, salto salvataggio: {result.url}")
                    else:
                        if len(cleaned_markdown) > 10:
                            doc = Document(page_content=cleaned_markdown, metadata={"source": result.url, "type": "web"})
                            doc_dict[result.url] = doc
                            
                            filename = generate_safe_filename(result.url)
                            nome_file = os.path.join(PAGES_DIR, filename)
                            
                            with open(nome_file, "w", encoding="utf-8") as f:
                                f.write(f"---\nsource: {result.url}\ntype: webpage\n---\n\n")
                                f.write(cleaned_markdown)
                            
                            state["web"][result.url] = page_hash
                            print(f"  [MD] {'AGGIORNATO' if old_hash else 'NUOVO'}: {filename}")
                        else:
                            print(f"  [SKIPPED] Testo troppo corto dopo la pulizia: {result.url}")
                elif skip_webpage_save:
                    print(f"  [SKIP WEB] Pagina EasyRoom ignorata perché convertita in Excel MD: {result.url}")

                if depth < max_depth and should_explore:
                    new_links, pdfs, excels = extract_links_and_pdfs(result.html, result.url, start_url)

                    if not content_changed and not skip_webpage_save:
                         print(f"    ↳ [ESPLORAZIONE] Estratti {len(new_links)} link, {len(pdfs)} PDF e {len(excels)} Excel.")

                    for pdf_url in pdfs:
                        download_and_parse_pdf(pdf_url, state, doc_dict)
                        
                    # Qui vengono processati tutti gli Excel trovati in siti "non-EasyRoom"
                    for excel_url in excels:
                        download_and_parse_excel(excel_url, state, doc_dict)

                    for link in new_links:
                        if link not in visited:
                            visited.add(link)
                            next_queue.append(link)

            current_queue = next_queue

# --- MAIN ---

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
         "filter": False},
        {"name": "EasyCourse Orari",
         "urls": EASYCOURSE_ORARI_URLS,
         "depth": 3,
         "filter": False},
        {"name": "EasyRoom Aule",
         "urls": EASYROOM_URLS,
         "depth": 1,
         "filter": False},
    ]

    async with AsyncWebCrawler(verbose=False) as crawler:
        for task in SEARCH_TASKS:
            await crawl_task(task, crawler, state, doc_dict)

    print(f"\nCompilazione Knowledge Base finale in corso...")
    
    knowledge_base = list(doc_dict.values())
    
    with open(KB_FILE, "wb") as f:
        pickle.dump(knowledge_base, f)
        
    save_state(state)

    web_files_count = len(state["web"])
    pdf_files_count = len(state["pdfs"])
    md_xls_files_count = len(state.get("excels", {}))

    print(f"\n{'='*12} RESOCONTO FINALE {'='*12}")
    print(f"🌍 Pagine Web uniche tracciate: {web_files_count}")
    print(f"📁 Documenti PDF unici tracciati: {pdf_files_count}")
    print(f"📊 Tabelle Excel estratte (in memoria -> .md): {md_xls_files_count}")
    print(f"🧠 Totale 'Documenti' salvati nella Knowledge Base: {len(knowledge_base)}")
    print(f"{'='*40}")
    print("Stato incrementale salvato. Al prossimo avvio verificherò solo le variazioni!")

if __name__ == "__main__":
    asyncio.run(main())