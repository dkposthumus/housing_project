import re
import pandas as pd
import fitz  # PyMuPDF
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

# === Section names map (for saving raw section text) ===
canonical_names = {
    "Consideration Items Proposed For Continuance": "continuance",
    "Consent Calendar": "consent_calendar",
    "Commission Matters": "commission_matters",
    "Department Matters": "dept_matters",
    "General Public Comment": "general_public_comment",
    "Regular Calendar": "regular_calendar",
}

# === 1) Date‐splitting regex ===
date_pattern = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday),?\s+"  # optional weekday
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(\d{1,2}),\s*(\d{4})",
    re.IGNORECASE,
)

def split_meetings(text: str):
    """Return list of dicts {'date':datetime, 'text':section_text}."""
    meetings = []
    for i, m in enumerate(date_pattern.finditer(text)):
        start = m.start()
        end = (list(date_pattern.finditer(text))[i+1].start()
               if i+1 < len(list(date_pattern.finditer(text))) else len(text))
        month, day, year = m.groups()
        try:
            dt = datetime.strptime(f"{month} {day}, {year}", "%B %d, %Y")
        except ValueError:
            continue
        meetings.append({"date": dt, "text": text[start:end].strip()})
    return meetings

# === 2) Enhanced extractor ===
def extract_projects(text: str) -> pd.DataFrame:
    """
    For each numbered project block, extract:
      item_num, case_code, commissioner, phone_ext,
      address, project_summary,
      prelim_recommendation, continuance_to,
      speakers_count, action, ayes, nays, motion_no
    """
    header_re = re.compile(
        r"^(\d{1,2})\.\s+"                   # item number
        r"(\d{4}\.\d+[A-Z]?)\s*"             # case code
        r"\(([^:]+):\s*([\d-]+)\)",          # (Commissioner: phone_ext)
        re.MULTILINE
    )

    # Slice into blocks
    blocks = []
    for mh in header_re.finditer(text):
        start = mh.start()
        next_m = next((nh for nh in header_re.finditer(text) if nh.start() > start), None)
        end = next_m.start() if next_m else len(text)
        blocks.append(text[start:end])

    rows = []
    for block in blocks:
        h = header_re.match(block)
        if not h:
            continue
        item_num, case_code, commissioner, phone_ext = h.groups()

        # Remainder of first line => address + summary
        first_line = block.splitlines()[0]
        remainder = first_line[h.end():].strip()
        if "--" in remainder:
            address, project_summary = [s.strip() for s in remainder.split("--", 1)]
        else:
            address, project_summary = remainder, None

        # Preliminary Recommendation
        pr = re.search(r"Preliminary Recommendation:\s*(.*?)\s*(?:\(|SPEAKER|$)", block, re.DOTALL)
        prelim_recommendation = pr.group(1).strip() if pr else None

        # Continuance
        co = re.search(r"Proposed for Continuance to\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", block)
        continuance_to = co.group(1) if co else None

        # Speakers
        sp = re.search(r"SPEAKER\(S\):(.+?)(?=ACTION:)", block, re.DOTALL)
        speakers_text = sp.group(1) if sp else ""
        speakers_count = len(re.findall(r"[+=-]", speakers_text))

        # Action / Ayes / Nays
        ac = re.search(r"ACTION:\s*([^\n]+)", block)
        action = ac.group(1).strip() if ac else None
        ay = re.search(r"AYES:\s*([^\n]+)", block)
        ayes = ay.group(1).strip() if ay else None
        ny = re.search(r"NAYS:\s*([^\n]+)", block)
        nays = ny.group(1).strip() if ny else None

        # Motion No
        mo = re.search(r"MOTION No:?\s*(\d+)", block)
        motion_no = mo.group(1) if mo else None

        rows.append({
            "item_num": item_num,
            "case_code": case_code,
            "commissioner": commissioner,
            "phone_ext": phone_ext,
            "address": address,
            "project_summary": project_summary,
            "prelim_recommendation": prelim_recommendation,
            "continuance_to": continuance_to,
            "speakers_count": speakers_count,
            "action": action,
            "ayes": ayes,
            "nays": nays,
            "motion_no": motion_no,
        })

    return pd.DataFrame(rows)


# === Main processing ===
for year in range(1998, 2025):
    minutes_year_dir = minutes_raw / str(year)
    for file_path in list(minutes_year_dir.glob("*.pdf")) + list(minutes_year_dir.glob("*.txt")):
        # STEP 1: Read file
        if file_path.suffix.lower() == ".pdf":
            try:
                with fitz.open(str(file_path)) as doc:
                    full_text = "\n".join(page.get_text() for page in doc)
            except Exception:
                try:
                    full_text = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    print(f"⨯ Skipping {file_path.name}")
                    continue
        else:
            try:
                full_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                print(f"⨯ Skipping {file_path.name}")
                continue

        # STEP 2: Split into meetings
        for meeting in split_meetings(full_text):
            meeting_date = meeting["date"]
            date_str = meeting_date.strftime("%Y-%m-%d")
            meeting_text = meeting["text"]

            # STEP 3: Break meeting into sections
            section_headers = list(re.finditer(r"\n([A-F])\.\s+([^\n]+)", meeting_text))
            section_texts = {}
            for i, mh in enumerate(section_headers):
                title = mh.group(2).strip().title()
                start_idx = mh.end()
                end_idx = section_headers[i + 1].start() if i + 1 < len(section_headers) else len(meeting_text)
                section_texts[title] = meeting_text[start_idx:end_idx].strip()

            # STEP 4: Save raw section text
            for title, body in section_texts.items():
                canonical = canonical_names.get(title)
                if not canonical:
                    continue
                out_dir = text_dir / canonical
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{date_str}.txt").write_text(body, encoding="utf-8")

            # STEP 5: Extract projects & save CSV
            project_sections = {
                "Consideration Items Proposed For Continuance": "continuance",
                "Consent Calendar": "consent_calendar",
                "Regular Calendar": "regular_calendar",
            }

            dfs = []
            for display, canon in project_sections.items():
                body = section_texts.get(display, "")
                df = extract_projects(body)
                if not df.empty:
                    df["section"] = canon
                    df["meeting_date"] = meeting_date
                    dfs.append(df)

            if dfs:
                out_df = pd.concat(dfs, ignore_index=True)
                out_df.to_csv(minutes_clean / f"planning_commission_projects_{date_str}.csv", index=False)
                print(f"✓ {file_path.name} → {date_str}")
            else:
                print(f"⚠ No projects in {file_path.name} ({date_str})")
