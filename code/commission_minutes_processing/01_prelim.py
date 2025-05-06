import pandas as pd
from pathlib import Path
import pdfplumber 
import re
# let's create a set of locals referring to our directory and working directory 
home = Path.home()
work_dir = (home / 'housing_project')
data = (work_dir / 'data')
minutes = (data / 'meeting_minutes')
minutes_raw = (minutes / 'pdfs')
minutes_clean = (minutes / 'processed')
code = Path.cwd() 

pdf_path = f"{minutes_raw}/20250327_cpc_min.pdf"

with pdfplumber.open(pdf_path) as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# --- Meeting-level metadata ---
commissioner_present = re.findall(r"COMMISSIONERS PRESENT:\s+(.*)", text)[0].split(", ")
commissioner_absent = re.findall(r"COMMISSIONERS ABSENT:\s+(.*)", text)[0].split(", ")
staff_block = re.search(r"STAFF IN ATTENDANCE:\s+(.*?)\n\n", text, re.DOTALL)
staff_present = [s.strip() for s in staff_block.group(1).split(",")] if staff_block else []

item_blocks = re.split(r"\n(?=\d{1,2}\.\s+\d{4}-\d{6,}[A-Z]*)", text)

records = []
for block in item_blocks[1:]:  # skip header before 1.
    lines = block.strip().split("\n")
    
    header_line = lines[0]
    item_num_match = re.match(r"(\d{1,2})\.\s+(\d{4}-\d{6,}[A-Z]*)\s+\((.*?)\)", header_line)
    if not item_num_match:
        continue

    item_num = item_num_match.group(1)
    code = item_num_match.group(2)
    staff = item_num_match.group(3)

    title_line = lines[1] if len(lines) > 1 else ""
    title = title_line.strip()

    # Address (look for block address patterns)
    address_match = re.search(r"(\d+ [A-Z\s]+(STREET|AVENUE|BOULEVARD|PLACE|ROAD|DRIVE))", block)
    address = address_match.group(1).title() if address_match else None

    # Action
    action_match = re.search(r"ACTION:\s+([^\n]+)", block)
    action = action_match.group(1).strip() if action_match else None

    # Votes
    aye_match = re.search(r"AYES:\s+([^\n]+)", block)
    nay_match = re.search(r"NAYS:\s+([^\n]+)", block)
    aye_votes = aye_match.group(1).strip() if aye_match else None
    nay_votes = nay_match.group(1).strip() if nay_match else None

    # Resolution / Motion
    res_match = re.search(r"(MOTION|RESOLUTION):\s+(\d+)", block)
    motion_or_resolution = res_match.group(2) if res_match else None

    # Speakers
    speakers_match = re.search(r"SPEAKERS:(.*?)(?:ACTION:|AYES:|NAYS:|RESOLUTION:|MOTION:|$)", block, re.DOTALL)
    if speakers_match:
        raw_speakers = speakers_match.group(1).strip()
        speakers_list = re.findall(r"[+=-]\s*.*?(?=\n|$)", raw_speakers)
        speakers_count = len(speakers_list)
        speakers_raw = "; ".join(s.strip() for s in speakers_list)
    else:
        speakers_count = 0
        speakers_raw = ""

    records.append({
        "item_num": item_num,
        "code": code,
        "staff": staff,
        "title": title,
        "address": address,
        "action": action,
        "aye_votes": aye_votes,
        "nay_votes": nay_votes,
        "motion_or_resolution": motion_or_resolution,
        "speakers_count": speakers_count,
        "speakers_raw": speakers_raw
    })

# Save
df = pd.DataFrame(records)
df.to_csv(f"{minutes_clean}/planning_commission_projects.csv", index=False)

# Print meeting-level metadata
print("Commissioners Present:", commissioner_present)
print("Commissioners Absent:", commissioner_absent)
print("Staff Attendance:", staff_present)