"""
inference.py
--------------------------------
Infer JSON on a *tagged* minutes file (<<Project Start>>/End>> markers).
"""

import json, re, torch
from pathlib import Path
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ───────────────────────────────
# 1) Helpers
# ───────────────────────────────
HEADER_DATE_RE = re.compile(
    r'(Monday|Tuesday|Wednesday|Thursday|Friday),?\s*'
    r'(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},\s+\d{4}',
    re.I
)

PROJECT_RE = re.compile(r'<<Project Start>>(.*?)<<Project End>>', re.S)

PROMPT_INSTRUCTION = """You are a strict JSON extractor. Given a block of raw meeting text, extract and return exactly one valid JSON object with these keys (in any order).  
Only output the JSON—no commentary. If a field isn’t present, set it to "".  

Required keys:
- case_number
- project_address
- lot_number
- assessor_block
- project_descr
- type_district
- type_district_descr
- speakers
- action
- modifications
- ayes
- noes
- absent
- vote
- action_name

Raw block:
"""

REQUIRED_KEYS = [
    "case_number","project_address","lot_number","assessor_block","project_descr",
    "type_district","type_district_descr","speakers","action","modifications",
    "ayes","noes","absent","vote","action_name"
]

_scalar_re = {k: re.compile(rf'"{k}"\s*:\s*"([^"]*?)"', re.I) for k in REQUIRED_KEYS}
_list_re   = {k: re.compile(rf'"{k}"\s*:\s*(\[[^\]]*?\])', re.I) for k in REQUIRED_KEYS}

def extract_json(pred: str):
    out = {}
    for k in REQUIRED_KEYS:
        if (m := _list_re[k].search(pred)):
            try: out[k] = json.loads(m.group(1))
            except json.JSONDecodeError: out[k] = []
        elif (m := _scalar_re[k].search(pred)):
            out[k] = m.group(1).strip()
        else:
            out[k] = ""
    return out

# ───────────────────────────────
# 2) Paths (single-file demo)
# ───────────────────────────────
home      = Path.home()
base_dir  = home / "housing_project" / "data" / "meeting_minutes"
tag_dir   = base_dir / "tagged"

target    = tag_dir / "1998" / "December_3_1998.txt"        # <- adjust as needed
model_dir = base_dir / "processed" / "minutes_extractor"

out_path  = base_dir / "processed" / "structured_data.jsonl"
err_log   = base_dir / "processed" / "extraction_errors.log"
out_path.parent.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────
# 3) Load model
# ───────────────────────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model     = T5ForConditionalGeneration.from_pretrained(model_dir).to(device).eval()

# ───────────────────────────────
# 4) Run extraction
# ───────────────────────────────
text = target.read_text(encoding="utf-8", errors="ignore")

# grab meeting date from header line
date_match = HEADER_DATE_RE.search(text)
meeting_date = ""
if date_match:
    raw = date_match.group(0).title()
    raw = re.sub(r',\s*', ', ', raw)
    meeting_date = datetime.strptime(raw, "%B %d, %Y").strftime("%Y-%m-%d")

projects = [m.group(1).strip() for m in PROJECT_RE.finditer(text)]
print(f"{target.name}: found {len(projects)} agenda blocks")

prompts = [PROMPT_INSTRUCTION + blk for blk in projects]
enc     = tokenizer(prompts, padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(device)
outs    = model.generate(**enc, max_new_tokens=512, num_beams=3)
preds   = tokenizer.batch_decode(outs, skip_special_tokens=True)

with open(out_path, "a", encoding="utf-8") as fout, \
     open(err_log, "a", encoding="utf-8") as ferr:
    for blk, pred in zip(projects, preds):
        data  = extract_json(pred)
        entry = {
            "source_file": target.name,
            "meeting_date": meeting_date,
            "block_header": blk.split("\n",1)[0][:120],
            **data
        }
        try:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            ferr.write(f"{target.name} | {entry['block_header']} | {e}\n")

print("✓ Finished – JSON lines saved to", out_path)