import pandas as pd
import fitz  # PyMuPDF
import re
from pathlib import Path
from datetime import datetime

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
    "Regular Calendar": "regular_calendar",
}

# === Regex pattern for meeting date (handles missing weekday + missing space before time) ===
date_pattern = re.compile(
    r"(?:(Monday|Tuesday|Wednesday|Thursday|Friday),?\s+)?"  # optional weekday
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"  # month
    r"(\d{1,2}),\s*"  # day
    r"(\d{4})",  # year
    re.IGNORECASE,
)

# === Extract project entries from section text ===

def extract_projects(text: str) -> pd.DataFrame:
    pattern = r"(\d{1,2}[a-zA-Z]?\.\s+\d{4}-\d{6,}[A-Z]*)"
    matches = list(re.finditer(pattern, text))
    blocks: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(text[start:end].strip())

    records = []
    for block in blocks:
        header = block.splitlines()[0]
        m_header = re.match(r"(\d{1,2}[a-zA-Z]?)\.\s+(\d{4}-\d{6,}[A-Z]*)", header)
        if not m_header:
            continue
        item_num, code = m_header.groups()

        address_match = re.search(
            r"(\d+ .+?(STREET|AVENUE|BOULEVARD|ROAD|DRIVE|COURT|PLACE))",
            block,
            re.IGNORECASE,
        )
        address = address_match.group(1).title() if address_match else None

        def grab(label):
            m = re.search(fr"{label}:\s+([^\n]+)", block)
            return m.group(1).strip() if m else None

        action = grab("ACTION")
        aye_votes = grab("AYES")
        nay_votes = grab("NAYS")
        motion_or_resolution = (
            re.search(r"(MOTION|RESOLUTION):\s+(\d+)", block).group(2)
            if re.search(r"(MOTION|RESOLUTION):\s+(\d+)", block)
            else None
        )

        speakers_match = re.search(
            r"SPEAKER(S)?:?(.*?)(?:ACTION:|AYES:|NAYS:|RESOLUTION:|MOTION:|$)",
            block,
            re.DOTALL,
        )
        speakers_count = 0
        if speakers_match:
            raw = speakers_match.group(2).strip()
            speakers_count = len(re.findall(r"[+=-]", raw))

        records.append(
            {
                "item_num": item_num,
                "code": code,
                "address": address,
                "action": action,
                "aye_votes": aye_votes,
                "nay_votes": nay_votes,
                "motion_or_resolution": motion_or_resolution,
                "speakers_count": speakers_count,
            }
        )

    return pd.DataFrame(records)

# === Split text into multiple meetings ===

def split_meetings(text: str):
    meetings = []
    matches = list(date_pattern.finditer(text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Build a canonical date string "Month DD, YYYY"
        month = m.group(2)
        day = m.group(3)
        year = m.group(4)
        date_str_for_parse = f"{month} {day}, {year}"
        try:
            dt = datetime.strptime(date_str_for_parse, "%B %d, %Y")
        except ValueError:
            continue
        meetings.append({"date": dt, "text": text[start:end].strip()})
    return meetings

# === Main processing ===

for year in range(1998, 2025):
    minutes_year_dir = minutes_raw / str(year)
    for file_path in list(minutes_year_dir.glob("*.pdf")) + list(minutes_year_dir.glob("*.txt")):
        # --- STEP 1: read file ---
        if file_path.suffix.lower() == ".pdf":
            try:
                with fitz.open(str(file_path)) as doc:
                    full_text = "\n".join(page.get_text() for page in doc)
            except Exception:
                # Fallback to reading as text
                try:
                    full_text = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    print(f"⨯ Skipping {file_path.name}: {e}")
                    continue
        else:
            try:
                full_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"⨯ Skipping {file_path.name}: {e}")
                continue

        # --- STEP 2: split into meetings ---
        for meeting in split_meetings(full_text):
            meeting_date = meeting["date"]
            date_str = meeting_date.strftime("%Y-%m-%d")
            meeting_text = meeting["text"]

            # --- STEP 3: break the meeting text into sections ---
            section_headers = list(re.finditer(r"\n([A-F])\.\s+([^\n]+)", meeting_text))
            section_texts = {}
            for i, mh in enumerate(section_headers):
                title = mh.group(2).strip().title()
                start_idx = mh.end()
                end_idx = section_headers[i + 1].start() if i + 1 < len(section_headers) else len(meeting_text)
                section_texts[title] = meeting_text[start_idx:end_idx].strip()

            # --- STEP 4: save each canonical section ---
            for title, body in section_texts.items():
                canonical = canonical_names.get(title)
                if not canonical:
                    continue
                out_dir = text_dir / canonical
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{date_str}.txt").write_text(body, encoding="utf-8")

            # --- STEP 5: extract and save project rows ---
            project_sections = {
                "Consideration Items Proposed For Continuance": "continuance",
                "Consent Calendar": "consent_calendar",
                "Regular Calendar": "regular_calendar",
            }

            dfs = []
            for display_title, canonical_name in project_sections.items():
                body = section_texts.get(display_title, "")
                df = extract_projects(body)
                if not df.empty:
                    df["section"] = canonical_name
                    df["meeting_date"] = meeting_date
                    dfs.append(df)

            if dfs:
                out_df = pd.concat(dfs, ignore_index=True)
                out_df.to_csv(minutes_clean / f"planning_commission_projects_{date_str}.csv", index=False)
                print(f"✓ {file_path.name} → {date_str}")
            else:
                print(f"⚠ No project rows in {file_path.name} ({date_str})")
