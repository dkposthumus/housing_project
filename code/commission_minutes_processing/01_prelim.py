import re
import pandas as pd
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime

# === Paths ===
home = Path.home()
work_dir = home / "housing_project"
minutes_raw = work_dir / "data/meeting_minutes/raw"
minutes_clean = work_dir / "data/meeting_minutes/processed"
text_dir = work_dir / "data/meeting_minutes/text"
text_dir.mkdir(parents=True, exist_ok=True)

# === Date splitting ===
date_re = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday),?\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(\d{1,2}),\s*(\d{4})",
    re.IGNORECASE,
)

def split_meetings(full_text):
    """Split full file into meeting‐chunks by date headers."""
    meetings = []
    dates = list(date_re.finditer(full_text))
    for i, m in enumerate(dates):
        start = m.start()
        end = dates[i+1].start() if i+1 < len(dates) else len(full_text)
        month, day, year = m.groups()
        dt = datetime.strptime(f"{month} {day}, {year}", "%B %d, %Y")
        meetings.append({"date": dt, "text": full_text[start:end]})
    return meetings

# === Block‐level extractor for Regular Calendar ===
block_re = re.compile(r"""
    ^\s*(?P<item_num>\d{1,2})\.\s+                 # 1) item number
    (?P<case_code>\d{2,}\.\d+[A-Z]*)\s*            # 2) case code
    \((?P<applicant>[^)]+)\)\s*\n                  # 3) (Applicant)
    (?P<address>.+?)--\s*                          # 4) address up to “--”
    (?P<project_summary>.+?)\n                     # 5) summary up to newline
    (?:.*?\n)*?                                    # skip any intervening lines
    ACTION:\s*(?P<action>[^\n]+)\n                 # 6) ACTION line
    AYES:\s*(?P<ayes>[^\n]+)\n                     # 7) AYES line
    NAYES:\s*(?P<nays>[^\n]+)\n                    # 8) NAYS line
    (?:.*?\n)*?                                    # skip to end of block
""", re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE)

def extract_projects(regular_text: str) -> pd.DataFrame:
    """
    Run block_re.finditer over the Regular Calendar section.
    Returns columns: item_num, case_code, applicant,
    address, project_summary, action, ayes, nays.
    """
    rows = []
    for m in block_re.finditer(regular_text):
        rows.append({
            "item_num": m.group("item_num"),
            "case_code": m.group("case_code"),
            "applicant": m.group("applicant"),
            "address": m.group("address").strip().title(),
            "project_summary": m.group("project_summary").strip(),
            "action": m.group("action").strip(),
            "ayes": m.group("ayes").strip(),
            "nays": m.group("nays").strip(),
        })
    return pd.DataFrame(rows)

# === Main processing ===
for year in range(1998, 2025):
    print(f"Processing year: {year}")
    raw_dir = minutes_raw / str(year)
    if not raw_dir.exists(): continue

    for path in raw_dir.glob("*.*"):
        # 1) read text
        if path.suffix.lower() == ".pdf":
            try:
                with fitz.open(str(path)) as doc:
                    full_text = "\n".join(p.get_text() for p in doc)
            except Exception:
                full_text = path.read_text("utf-8", errors="ignore")
        else:
            full_text = path.read_text("utf-8", errors="ignore")

        # 2) split into meetings
        for meeting in split_meetings(full_text):
            date_str = meeting["date"].strftime("%Y-%m-%d")
            print(f"Processing meeting: {date_str} from {path.name}")
            txt = meeting["text"]

            # 3) carve out the Regular Calendar section
            #    find the 'E.REGULAR CALENDAR' header and next section
            sec_re = re.compile(r"E\.REGULAR CALENDAR(.*?)(?:\n[A-F]\.\s|\Z)", re.DOTALL)
            reg = sec_re.search(txt)
            if not reg:
                continue
            regular_body = reg.group(1)

            # 4) extract project rows
            df = extract_projects(regular_body)
            if df.empty:
                print(f"⚠ no rows in {path.name} @ {date_str}")
                continue

            # 5) save CSV
            out = minutes_clean / f"projects_{date_str}.csv"
            df.to_csv(out, index=False)
            print(f"✓ {path.name} → {date_str} ({len(df)} rows)")
