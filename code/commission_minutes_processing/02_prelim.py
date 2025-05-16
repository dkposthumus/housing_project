import pandas as pd
import fitz  # PyMuPDF
import re
from pathlib import Path
import json

# === Set up paths ===
home = Path.home()
work_dir = home / "housing_project"
data = work_dir / "data"
minutes = data / "meeting_minutes"
minutes_raw = minutes / "raw"
minutes_clean = minutes / "processed"
text_dir = minutes / "text"
text_dir.mkdir(parents=True, exist_ok=True)

# === Section names map ===
canonical_names = {
    "Consideration Items Proposed For Continuance": "continuance",
    "Consent Calendar": "consent_calendar",
    "Commission Matters": "commission_matters",
    "Department Matters": "dept_matters",
    "General Public Comment": "general_public_comment",
    "Regular Calendar": "regular_calendar"
}

# === Extract project entries from section text ===
def extract_projects(text):
    pattern = r"(\d{1,2}[a-zA-Z]?\.\s+\d{4}-\d{6,}[A-Z]*)"
    matches = list(re.finditer(pattern, text))
    blocks = []

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        blocks.append(text[start:end].strip())

    records = []
    for block in blocks:
        lines = block.strip().splitlines()
        header = lines[0]
        m = re.match(r"(\d{1,2}[a-zA-Z]?)\.\s+(\d{4}-\d{6,}[A-Z]*)", header)
        if not m: continue
        item_num, code = m.group(1), m.group(2)

        address_match = re.search(r"(\d+ .+?(STREET|AVENUE|BOULEVARD|ROAD|DRIVE|COURT|PLACE))", block, re.IGNORECASE)
        address = address_match.group(1).title() if address_match else None

        action_match = re.search(r"ACTION:\s+([^\n]+)", block)
        aye_match = re.search(r"AYES:\s+([^\n]+)", block)
        nay_match = re.search(r"NAYS:\s+([^\n]+)", block)
        res_match = re.search(r"(MOTION|RESOLUTION):\s+(\d+)", block)

        speakers_match = re.search(r"SPEAKERS:(.*?)(?:ACTION:|AYES:|NAYS:|RESOLUTION:|MOTION:|$)", block, re.DOTALL)
        if speakers_match:
            raw_speakers = speakers_match.group(1).strip()
            speakers_list = re.findall(r"[+=-]\s*.*?(?=\n|$)", raw_speakers)
            speakers_count = len(speakers_list)
        else:
            speakers_count = 0

        records.append({
            "item_num": item_num,
            "code": code,
            "address": address,
            "action": action_match.group(1).strip() if action_match else None,
            "aye_votes": aye_match.group(1).strip() if aye_match else None,
            "nay_votes": nay_match.group(1).strip() if nay_match else None,
            "motion_or_resolution": res_match.group(2) if res_match else None,
            "speakers_count": speakers_count
        })

    return pd.DataFrame(records)

for year in range(1998, 2025):
    minutes_subdir = minutes_raw / str(year)
    
    # PDF extraction for 1998–2000 and 2015–2024
    if year in range(1998, 2001) or year in range(2015, 2025):
        for pdf_file in minutes_subdir.glob("*.pdf"):
            try:
                doc = fitz.open(pdf_file)
                full_text = "\n".join(page.get_text() for page in doc)
                # Do something with full_text (e.g., save, analyze, etc.)
            except Exception as e:
                print(f"Failed to process {pdf_file}: {e}")
    
    # TXT format matching for 2001–2014
    elif year in range(2001, 2015):
        for txt_file in minutes_subdir.glob("*.txt"):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    full_text = f.read()
                    # Use full_text to compare format to PDF-extracted content
            except Exception as e:
                print(f"Failed to read {txt_file}: {e}")

        # Split into sections
        section_headers = list(re.finditer(r"\n([A-F])\.\s+([^\n]+)", full_text))
        section_texts = {}
        for i, match in enumerate(section_headers):
            title = match.group(2).strip().title()
            start = match.end()
            end = section_headers[i+1].start() if i+1 < len(section_headers) else len(full_text)
            section_texts[title] = full_text[start:end].strip()

        # Save each section’s text
        for title, body in section_texts.items():
            canonical = canonical_names.get(title)
            if not canonical:
                continue
            section_dir = text_dir / canonical
            section_dir.mkdir(parents=True, exist_ok=True)
            with open(section_dir / f"{date_str}.txt", "w", encoding="utf-8") as f:
                f.write(body)

        # Extract project data from A/B/F
        project_sections = {
            "Consideration Items Proposed For Continuance": "continuance",
            "Consent Calendar": "consent_calendar",
            "Regular Calendar": "regular_calendar"
        }

        project_dfs = []
        for display_title, canonical_name in project_sections.items():
            body = section_texts.get(display_title, "")
            df = extract_projects(body)
            if not df.empty:
                df["section"] = canonical_name
                df["meeting_date"] = meeting_date
                project_dfs.append(df)

        if project_dfs:
            df_all = pd.concat(project_dfs, ignore_index=True)
            df_all.to_csv(minutes_clean / f"planning_commission_projects_{date_str}.csv", index=False)
            print(f"✓ Processed {pdf_file.name}")
        else:
            print(f"⚠ No projects found in {pdf_file.name}")
