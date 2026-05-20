import os
import requests
import pickle
import asyncio
import logging
import re
import hashlib
import json
import io
from datetime import datetime, timedelta
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
EASYCOURSE= [
    "https://easycourse.unisa.it/EasyCourse//Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2025-2026/index.html",
    "https://easycourse.unisa.it/EasyCourse//Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2024-2025/index.html",
    "https://easycourse.unisa.it/EasyCourse//Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2023-2024/index.html",
    "https://easycourse.unisa.it/EasyCourse//Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2022-2023/index.html",
    "https://easycourse.unisa.it/EasyCourse//Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2021-2022/index.html"
]

EASYROOM = [
    "https://easycourse.unisa.it/EasyRoom/index.php?vista=week&content=view_prenotazioni&area=2&_lang=it&room=6",
    "https://easycourse.unisa.it/EasyRoom/index.php?vista=week&content=view_prenotazioni&area=37&_lang=it&room=18",
    "https://easycourse.unisa.it/EasyRoom/index.php?vista=week&content=view_prenotazioni&area=36&_lang=it&room=15"
]
    



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

def parse_easyroom_table(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    frasi = []

    # 1. Trova il nome dell'aula in modo sicuro e definitivo
    nome_aula = "Aula Sconosciuta"
    
    # STRATEGIA A: Estrazione dall'opzione selezionata nel menu delle aule
    # (Escludiamo il select con id="function_options")
    select_aula = soup.find("select", attrs={"name": "url", "id": lambda x: x != "function_options"})
    if select_aula:
        option_selected = select_aula.find("option", selected=True)
        if option_selected:
            text_opzione = option_selected.get_text(strip=True)
            # Pulisce "Aula 133 (40 posti - ...)" lasciando solo "Aula 133"
            nome_aula = text_opzione.split("(")[0].strip()

    # STRATEGIA B: Di riserva, se la precedente fallisce, cerca l'intestazione esatta della tabella
    if nome_aula == "Aula Sconosciuta":
        for td in soup.find_all("td"):
            # Usiamo il confronto ESATTO per evitare il match con "Cambia aula:"
            if td.get_text(strip=True) == "Aula:":
                sibling = td.find_next_sibling("td")
                if sibling:
                    raw_text = sibling.get_text(separator=" ", strip=True)
                    nome_aula = raw_text.split("Dettagli aula")[0].strip()
                    break

    # 2. Individua la griglia del calendario
    timegrid = soup.find("table", class_="timegrid")
    if not timegrid: 
        return ""

    # 3. Estrai le intestazioni dei giorni (colonne 1-7)
    giorni = {}
    thead = timegrid.find("thead")
    if thead:
        headers = thead.find("tr").find_all("td", recursive=False)
        for i, th in enumerate(headers):
            if 0 < i < len(headers) - 1: 
                giorni[i] = th.get_text(separator=" ", strip=True)

    # 4. Leggi la griglia tenendo traccia dei rowspan
    tbody = timegrid.find("tbody")
    if not tbody: 
        return ""

    active_rowspans = {}  
    rows = tbody.find_all("tr", recursive=False)

    for row in rows:
        col_idx = 0
        tds = row.find_all("td", recursive=False)
        td_iter = iter(tds)

        while col_idx <= 7:  
            if active_rowspans.get(col_idx, 0) > 0:
                active_rowspans[col_idx] -= 1
                col_idx += 1
                continue

            try:
                td = next(td_iter)
            except StopIteration:
                break

            if col_idx == 0:
                current_time = td.get_text(strip=True)
            elif 1 <= col_idx <= 7:
                entry = td.find("table", class_="entry")
                if entry:
                    materia_td = entry.find(lambda tag: tag.name == "td" and "font-weight:bold" in tag.get("style", ""))
                    prof_td = entry.find(lambda tag: tag.name == "td" and "#990000" in tag.get("style", ""))
                    tipo_span = entry.find(lambda tag: tag.name == "span" and "entry_type_name" in tag.get("id", ""))

                    materia = materia_td.get_text(strip=True) if materia_td else "Materia Sconosciuta"
                    prof = prof_td.get_text(strip=True) if prof_td else "Docente Sconosciuto"
                    tipo = tipo_span.get_text(strip=True) if tipo_span else "Evento"

                    rowspan = int(td.get("rowspan", 1))
                    active_rowspans[col_idx] = rowspan - 1
                    
                    # Calcolo orario di inizio e fine
                    start_time = current_time.split("-")[0] if "-" in current_time else current_time
                    durata_minuti = rowspan * 30
                    
                    try:
                        start_dt = datetime.strptime(start_time, "%H:%M")
                        end_dt = start_dt + timedelta(minutes=durata_minuti)
                        end_time = end_dt.strftime("%H:%M")
                    except ValueError:
                        end_time = "Fine non definita"

                    giorno_str = giorni.get(col_idx, f"Giorno {col_idx}")

                    frasi.append(f"In {nome_aula}, il giorno {giorno_str}, dalle {start_time} alle {end_time}, è prevista l'attività: {tipo} di {materia} (Docente: {prof}).")
            
            col_idx += 1

    return "\n".join(frasi) if frasi else ""

def parse_easycourse_table(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    frasi = set()

    # =======================================================
    # TIPO 1: Tabella "COMPATTO" (lista insegnamenti - tblchk)
    # =======================================================
    tblchk = soup.find("table", class_="tblchk")
    if tblchk:
        rows = tblchk.find_all("tr")[1:]  # Salta l'intestazione
        for row in rows:
            cols = row.find_all("td", recursive=False)
            if len(cols) >= 6:
                insegnamento = cols[1].get_text(separator=" ", strip=True)
                
                # Il corso è nascosto dentro il div dei dettagli (il tooltip JS)
                corso_div = cols[2].find("div", class_="details")
                corso = corso_div.get_text(separator=" - ", strip=True) if corso_div else "Corso non specificato"
                
                docente = cols[4].get_text(separator=" / ", strip=True)
                
                # Le singole lezioni sono separate da tag <br> all'interno del <td>
                lezioni_td = cols[5]
                lezioni = [line.strip() for line in lezioni_td.stripped_strings if line.strip()]
                
                for lezione in lezioni:
                    frasi.add(f"Per il corso di {insegnamento} ({corso}, Docente: {docente}) è prevista lezione: {lezione}.")
        
        return "\n".join(sorted(list(frasi))) if frasi else ""

    # =======================================================
    # TIPO 2: Tabella GRIGLIA "DOCENTE/CURRICULA" (timegrid)
    # =======================================================
    timegrid = soup.find("table", class_="timegrid")
    if timegrid:
        # Invece di mappare la griglia (soggetta a bug), estraiamo i dati dai popup "details" nascosti!
        details_divs = soup.find_all("div", class_="details")
        for detail in details_divs:
            text_blocks = detail.find_all("div", recursive=False)
            
            ins, doc, percorsi = "Sconosciuto", "Sconosciuto", "Percorso non specificato"
            orari = []
            
            for block in text_blocks:
                b_tag = block.find("b", recursive=False)
                if not b_tag: continue
                
                key = b_tag.get_text(strip=True).replace(":", "")
                val = b_tag.next_sibling
                val_text = val.strip(": \n") if isinstance(val, str) else ""
                
                if "Insegnamento" in key: 
                    ins = val_text
                elif "Docenti titolari" in key or "Docente" in key: 
                    doc = val_text
                elif "Percorsi" in key:
                    ul = block.find("ul")
                    if ul: percorsi = " | ".join([li.get_text(separator=" ", strip=True) for li in ul.find_all("li")])
                elif "Orario" in key:
                    ul = block.find("ul")
                    if ul: orari = [li.get_text(strip=True) for li in ul.find_all("li")]
            
            # Formatta e salva le frasi solo se abbiamo l'orario
            if ins != "Sconosciuto" and orari:
                for orario in orari:
                    frasi.add(f"Insegnamento: {ins} (Docente: {doc}, Percorsi: {percorsi}) -> Orario Ufficiale: {orario}")

        # Fallback (emergenza): Se per qualche motivo mancano i popup, analizziamo la griglia colorata
        if not frasi:
            giorni = {}
            thead = timegrid.find("tr")
            if thead:
                headers = thead.find_all("td", recursive=False)
                for i, th in enumerate(headers):
                    if i > 0: giorni[i] = th.get_text(strip=True)

            rows = timegrid.find_all("tr", recursive=False)[1:]
            for row in rows:
                tds = row.find_all("td", recursive=False)
                if not tds: continue
                orario = tds[0].get_text(strip=True)
                for i in range(1, len(tds)):
                    # Escludiamo le celle bianche (vuote)
                    if "FFFFFF" not in tds[i].get("bgcolor", "") and tds[i].get_text(strip=True):
                        materia_td = tds[i].find(class_="subject_pos1")
                        aula_td = tds[i].find(class_="subject_pos2")
                        if materia_td:
                            mat = materia_td.get_text(strip=True)
                            au = aula_td.get_text(strip=True) if aula_td else "Aula non specificata"
                            giorno = giorni.get(i, f"Giorno {i}")
                            frasi.add(f"Il giorno {giorno} dalle {orario} si terrà {mat} in {au}.")
                            
        return "\n".join(sorted(list(frasi))) if frasi else ""

    return ""

def extract_links_and_pdfs(html, current_url, start_url):
    soup = BeautifulSoup(html, "html.parser")
    links_to_visit = []
    pdfs_to_download = []
    
    # --- FIX 1: Pulisce i doppi slash che confondono il calcolo dei confini ---
    start_url = re.sub(r'(?<!:)//+', '/', start_url)
    current_url = re.sub(r'(?<!:)//+', '/', current_url)
    
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

    # --- FIX 2: INIEZIONE DIRETTA ---
    # Poiché index.html è un frameset vuoto, forziamo l'aggiunta delle pagine 
    # che contengono le vere liste di link, saltando il problema del DOM vuoto.
    if is_easycourse and current_url.lower().endswith('index.html'):
        base_dir = current_url.rsplit('/', 1)[0]
        links_to_visit.append(f"{base_dir}/tree.html")
        links_to_visit.append(f"{base_dir}/ttteacherhtml.html")
        links_to_visit.append(f"{base_dir}/ttcdlhtml.html")

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
                # --- FIX 3: Whitelist allargata per includere TUTTI i Curricula e i Docenti ---
                elif '/curricula/' in url_l:
                    is_whitelisted = True
                elif '/docenti/' in url_l:
                    is_whitelisted = True
                elif '/easyroom/' in url_l:
                    if any(f"area={a}&" in url_l or url_l.endswith(f"area={a}") for a in ['2', '36', '37']):
                        is_whitelisted = True
                        
                if not is_whitelisted:
                    continue
            
            if not any(url_l.endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.jpeg']):
                if '/en/' not in full_url and not full_url.endswith('/en'):
                    links_to_visit.append(full_url)

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

            # Rimuoviamo classi generiche come .control-box o toolbar per non distruggere EasyRoom
            js_cleanup = "const selectors = ['#cookie-bar', '#unisa-utilities-bar', '.bg-footer', '.sub-footer'].join(', '); document.querySelectorAll(selectors).forEach(el => { if(el) el.remove(); });"
            
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
                if not result.success or (result.status_code and result.status_code != 200 and result.status_code != 301 and result.status_code != 302): continue
                
                # --- INTERCETTAZIONE CALENDARI E EASYROOM ---
                if "calendario-occupazione-spazi" in result.url.lower():
                    testo_calendario = parse_unisa_calendar_to_sentences(result.html)
                    if testo_calendario:
                        testo_da_salvare = f"DATI ORARIO UFFICIALE ESTRATTI:\n\n{testo_calendario}"
                        print(f"  [CALENDARIO PARSATO] {result.url}")
                    else:
                        testo_da_salvare = clean_md(result.markdown)
                        
                elif "easyroom" in result.url.lower():
                    # Usa il nuovo parser avanzato per tabelle con rowspan
                    testo_easyroom = parse_easyroom_table(result.html)
                    if testo_easyroom:
                        testo_da_salvare = f"DATI ORARIO EASYROOM:\n\n{testo_easyroom}"
                        print(f"  [EASYROOM PARSATO] {result.url}")
                    else:
                        testo_da_salvare = clean_md(result.markdown)
                        
                elif "easycourse" in result.url.lower():
                    testo_easycourse = parse_easycourse_table(result.html)
                    if testo_easycourse:
                        testo_da_salvare = f"DATI ORARI EASYCOURSE:\n\n{testo_easycourse}"
                        print(f"  [EASYCOURSE PARSATO] {result.url}")
                    else:
                        testo_da_salvare = clean_md(result.markdown)
                        
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
        {"name": "Sito DIEM", "urls": ["https://www.diem.unisa.it/"], "depth": 4, "filter": False},
        {"name": "Docenti", "urls": ["https://docenti.unisa.it/"], "depth": 3, "filter": True},
        {"name": "Corsi DIEM", "urls": CORSI_DIEM_URLS, "depth": 3, "filter": False},
        {"name": "EasyCourse", "urls": EASYCOURSE, "depth": 2, "filter": False},
        {"name": "EasyRoom", "urls": EASYROOM, "depth": 1, "filter": False}
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