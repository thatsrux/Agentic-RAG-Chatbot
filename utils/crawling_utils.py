import os
import requests
import re
import json
import io
import hashlib
import pickle
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_core.documents import Document

DATA_PATH = r"knowledge/data"
PAGES_DIR = os.path.join(DATA_PATH, "pages")     
PDF_MD_DIR = os.path.join(DATA_PATH, "PDFs")   
STATE_FILE = "knowledge/crawler_state.json"
KB_FILE = "knowledge/knowledge_base.pkl"

KEYWORDS = ["DIEM", "DIPARTIMENTO DI INGEGNERIA DELL'INFORMAZIONE ED ELETTRICA E MATEMATICA APPLICATA"]
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

EASYTEST_ESAMI_URLS = [
    "https://easycourse.unisa.it/EasyTest//Calendario/Facolta_di_Ingegneria_-_Esami/1309/index.html",
    "https://easycourse.unisa.it/EasyTest//Calendario/Facolta_di_Ingegneria_-_Esami/1354/index.html"
]

EASYTEST_TARGET_CURRICULA = [
    "ELECTRICALENGINEERINGFORDIGITALENERGY_CORSODILAUREAMAGISTRALE_06233",
    "ELECTRICALENGINEERINGFORDIGITALENERGY-primoanno_CORSODILAUREAMAGISTRALE_IE233",
    "INFORMATIONENGINEERINGFORDIGITALMEDICINE_CORSODILAUREAMAGISTRALE_06232",
    "INFORMATIONENGINEERINGFORDIGITALMEDICINE-primoanno_CORSODILAUREAMAGISTRALE_IE232",
    "INGEGNERIAINFORMATICA_CORSODILAUREAMAGISTRALE_06227",
    "INGEGNERIAINFORMATICA-primoanno_CORSODILAUREAMAGISTRALE_IE227",
    "INGEGNERIADELLINFORMAZIONEPERLAMEDICINADIGITALE_CORSODILAUREA_06128",
    "INGEGNERIADELLINFORMAZIONEPERLAMEDICINADIGITALE-primoanno_CORSODILAUREA_IE128",
    "INGEGNERIAINFORMATICA_CORSODILAUREA_06127",
    "INGEGNERIAINFORMATICA-primoanno_CORSODILAUREA_IE127"
]

def load_state():
    """
    Carica lo stato del crawler da un file JSON. Se il file non esiste, restituisce uno stato iniziale vuoto.
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return {"web": {}, "pdfs": {}}

def save_state(state):
    """
    Salva lo stato del crawler in un file JSON.
    """
    with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(state, f, indent=4)

def load_knowledge_base():
    """
    Carica la knowledge base da un file pickle. Restituisce un dizionario con URL come chiavi e oggetti Document come valori.
    """
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "rb") as f:
            return {doc.metadata.get("source"): doc for doc in pickle.load(f)}
    return {}

def generate_safe_filename(url):
    """
    Genera un nome di file sicuro a partire da un URL, rimuovendo o sostituendo caratteri non validi e limitando la lunghezza.
    """
    safe_name = re.sub(r'[\/\\?%*:|"<>&=]+', '_', url.replace("https://", "").replace("http://", "")).strip('_')
    if len(safe_name) > 200: safe_name = safe_name[:190] + "_" + hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
    return f"{safe_name}.md"

def clean_md(md_content):
    """
    Pulisce il contenuto markdown rimuovendo elementi non informativi o ridondanti specifici dei siti target, come banner, link di navigazione, e inviti a condividere.
    """
    if not md_content: return ""
    txt = md_content.strip()
    txt = re.sub(r'\[skip to main content\].*?Condividi\s*(?:\d+\.\s*\[\]\(.*?\)\s*)*', '', txt, flags=re.DOTALL|re.IGNORECASE)
    txt = re.sub(r'\* \[Home \]\(.*?\).*?\[Contatti \]\(.*?\)', '', txt, flags=re.DOTALL|re.IGNORECASE)
    txt = re.sub(r'\* \[Presentazione \]\(.*?\).*?\[Strutture \]\(.*?\)', '', txt, flags=re.DOTALL|re.IGNORECASE)
    return re.sub(r'\[Vai al Contenuto della Pagina\].*', '', txt, flags=re.DOTALL|re.IGNORECASE).strip()

def is_relevant(text, url):
    """
    Determina se una pagina web è rilevante per il dominio DIEM basandosi sulla presenza di parole chiave specifiche nel testo e nell'URL.
    Restituisce True se la pagina è considerata rilevante, altrimenti False.
    """
    is_valid = any(kw in text.upper() for kw in KEYWORDS)
    print(f"  [FILTRO] {'✅ ACCETTATO' if is_valid else '❌ SCARTATO'}: {url}")
    return is_valid

def is_recent_pdf(pdf_url):
    """
    Determina se un PDF è recente basandosi sulla presenza di un anno (2020 o successivo) nell'URL. Se non viene trovato alcun anno, assume che il PDF sia recente.
    """
    year_matches = re.findall(r'(?<!\d)20\d{2}(?!\d)', pdf_url)
    return any(int(y) >= 2020 for y in year_matches) if year_matches else True

def parse_unisa_calendar_to_sentences(html_content):
    """
    Parsa il contenuto HTML del calendario di occupazione spazi dell'Università di Salerno e restituisce una stringa formattata con le informazioni estratte.
    """
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

def parse_easycourse_table(html_content):
    """
    Parsa il contenuto HTML di una pagina di EasyCourse (orari o esami) e restituisce una stringa formattata con le informazioni estratte.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    frasi = set()

    tblchk = soup.find("table", class_="tblchk")
    if tblchk:
        rows = tblchk.find_all("tr")[1:]  
        for row in rows:
            cols = row.find_all("td", recursive=False)
            if len(cols) >= 6:
                insegnamento = cols[1].get_text(separator=" ", strip=True)
                corso_div = cols[2].find("div", class_="details")
                corso = corso_div.get_text(separator=" - ", strip=True) if corso_div else "Corso non specificato"
                docente = cols[4].get_text(separator=" / ", strip=True)
                lezioni_td = cols[5]
                lezioni = [line.strip() for line in lezioni_td.stripped_strings if line.strip()]
                for lezione in lezioni:
                    frasi.add(f"Per il corso di {insegnamento} ({corso}, Docente: {docente}) è prevista lezione/esame: {lezione}.")
        return "\n".join(sorted(list(frasi))) if frasi else ""

    timegrid = soup.find("table", class_="timegrid")
    if timegrid:
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
                
                if "Insegnamento" in key: ins = val_text
                elif "Docenti titolari" in key or "Docente" in key: doc = val_text
                elif "Percorsi" in key:
                    ul = block.find("ul")
                    if ul: percorsi = " | ".join([li.get_text(separator=" ", strip=True) for li in ul.find_all("li")])
                elif "Orario" in key:
                    ul = block.find("ul")
                    if ul: orari = [li.get_text(strip=True) for li in ul.find_all("li")]
            
            if ins != "Sconosciuto" and orari:
                for orario in orari:
                    frasi.add(f"Insegnamento: {ins} (Docente: {doc}, Percorsi: {percorsi}) -> Orario Ufficiale: {orario}")

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
                    if "FFFFFF" not in tds[i].get("bgcolor", "") and tds[i].get_text(strip=True):
                        materia_td = tds[i].find(class_="subject_pos1")
                        aula_td = tds[i].find(class_="subject_pos2")
                        if materia_td:
                            mat = materia_td.get_text(strip=True)
                            au = aula_td.get_text(strip=True) if aula_td else "Aula non specificata"
                            giorno = giorni.get(i, f"Giorno {i}")
                            frasi.add(f"Il giorno {giorno} dalle {orario} si terrà {mat} in {au}.")
                            
        return "\n".join(sorted(list(frasi))) if frasi else ""

    exam_tables = soup.find_all("table", title=re.compile(r"Esame", re.IGNORECASE))
    
    if exam_tables:
        testo_esami = "| Data | Insegnamento | Docente | Tipo Esame | Appello | Orario e Aula | Info Aggiuntive |\n"
        testo_esami += "|---|---|---|---|---|---|---|\n"
        esami_visti = set()
        righe_valide = 0
        
        for exam in exam_tables:
            data_esame = "Data non definita"
            parent = exam.parent
            while parent:
                parent_text = parent.get_text(separator=" ", strip=True)
                # Cerca un pattern data (es. 5-1-2026 o 05-01-2026)
                date_match = re.search(r'\b(\d{1,2}-\d{1,2}-\d{4})\b', parent_text)
                if date_match:
                    data_esame = date_match.group(1)
                    break
                parent = parent.parent
                
            rows = exam.find_all("tr")
            if len(rows) >= 7:
                corso = rows[0].get_text(strip=True).replace('\n', ' ').replace('|', '')
                codice = rows[1].get_text(strip=True)
                anno_sem = rows[2].get_text(strip=True)
                docente = rows[3].get_text(strip=True).replace('\n', ' ').replace('|', '')
                orario_aula = rows[4].get_text(separator=" ", strip=True).replace('\n', ' ').replace('|', '')
                appello = rows[5].get_text(strip=True)
                tipo = rows[6].get_text(strip=True)
                
                info = f"{anno_sem} - {codice}"
                
                firma = f"{data_esame}-{codice}-{appello}-{tipo}"
                
                if firma not in esami_visti:
                    testo_esami += f"| {data_esame} | {corso} | {docente} | {tipo} | {appello} | {orario_aula} | {info} |\n"
                    esami_visti.add(firma)
                    righe_valide += 1
        
        if righe_valide > 0:
            return testo_esami

    return ""

def extract_links_and_pdfs(html, current_url, start_url):
    """
    Estrae i link e i PDF da una pagina HTML, filtrando quelli che appartengono al dominio e ai confini specificati.
    Restituisce due liste: una con i link da visitare e una con i PDF da scaricare.
    """
    soup = BeautifulSoup(html, "html.parser")
    links_to_visit = []
    pdfs_to_download = []
    
    start_url = re.sub(r'(?<!:)//+', '/', start_url)
    current_url = re.sub(r'(?<!:)//+', '/', current_url)
    
    base_boundary = start_url.lower().rstrip('/')
    allowed_domain = urlparse(start_url).netloc
    is_easycourse = (allowed_domain == "easycourse.unisa.it")
    
    if is_easycourse:
        if base_boundary.endswith('.html'):
            base_boundary = base_boundary.rsplit('/', 1)[0]
    
    valid_boundaries = [base_boundary]
    
    for key, alias in CORSI_ALIASES.items():
        if base_boundary == key.lower():
            valid_boundaries.append(alias.lower())
        elif base_boundary == alias.lower():
            valid_boundaries.append(key.lower())

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
                
                if '/easytest/' in url_l:
                    if '/curricula/' in url_l and any(url_l.endswith(target.lower() + ".html") for target in EASYTEST_TARGET_CURRICULA):
                        is_whitelisted = True
                else:
                    if url_l.endswith('/index.html') or url_l.endswith('/tree.html') or url_l.endswith('/main.html'):
                        is_whitelisted = True
                    elif any(url_l.endswith(view) for view in ['/tthtml.html', '/ttcdlhtml.html', '/ttteacherhtml.html', '/ttdayhtml.html']):
                        is_whitelisted = True
                    elif '/curricula/' in url_l:
                        is_whitelisted = True
                    elif '/docenti/' in url_l:
                        is_whitelisted = True
                        
                if not is_whitelisted:
                    continue
            
            if not any(url_l.endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.jpeg']):
                if '/en/' not in full_url and not full_url.endswith('/en'):
                    links_to_visit.append(full_url)

    return list(set(links_to_visit)), list(set(pdfs_to_download))

def download_and_parse_pdf(pdf_url, state, doc_dict):
    """
    Scarica un PDF da un URL, verifica se è recente e se il suo contenuto è cambiato rispetto alla versione salvata in cache.
    Se il PDF è valido e contiene testo estraibile, lo salva in formato markdown e aggiorna la knowledge base.
    """
    if pdf_url in state["pdfs"]: return
    try:
        res = requests.get(pdf_url, timeout=10)
        if res.status_code == 200 and 'application/pdf' in res.headers.get('Content-Type', '').lower() and is_recent_pdf(pdf_url):
            file_hash = hashlib.md5(res.content).hexdigest()
            if state["pdfs"].get(pdf_url, {}).get("hash") == file_hash: 
                print(f"  [CACHE] ⏩ PDF Invariato: {pdf_url} (Hash coincidente, nessun download)")
                return
            if file_hash in [i["hash"] for i in state["pdfs"].values() if i.get("hash")]:
                state["pdfs"][pdf_url] = {"hash": file_hash, "filename": "DUPLICATO"}
                print(f"  [CACHE] ⏩ PDF Duplicato: {pdf_url} (Hash già presente, salto scrittura)")
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