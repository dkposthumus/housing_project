import json
import re
from pathlib import Path
from datetime import datetime

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# —————————————————————————————————————————————————————————————————————————————
# 1) Regex & helper for splitting the full text into meetings
# —————————————————————————————————————————————————————————————————————————————
date_pattern = re.compile(
    r'(?:(Monday|Tuesday|Wednesday|Thursday|Friday),?\s*)?'
    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s*'
    r'(\d{1,2}),\s*(\d{4})',
    flags=re.IGNORECASE
)

def split_meetings(text: str):
    """Return list of {'date': datetime, 'text': …}"""
    matches = list(date_pattern.finditer(text))
    meetings = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        month, day, year = m.group(2), m.group(3), m.group(4)
        dt = datetime.strptime(f"{month} {day}, {year}", "%B %d, %Y")
        meetings.append({
            "date": dt,
            "text": text[start:end].strip()
        })
    return meetings

# —————————————————————————————————————————————————————————————————————————————
# 2) Regex for case headers: e.g. "1.   98.226D"  or  "4.  98.290D/DD"
# —————————————————————————————————————————————————————————————————————————————
case_header_re = re.compile(
    r'(?m)^\s*\d{1,2}\.\s*'               # leading index "1.", "23." etc.
    r'\d{2,3}\.\d{3,}(?:[A-Za-z]+)?(?:/[A-Za-z]+)?'  # core case number + optional letter or /DD
)

# —————————————————————————————————————————————————————————————————————————————
# 3) Prompt we feed each block
# —————————————————————————————————————————————————————————————————————————————
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

# the exact keys you promised in your prompt:
REQUIRED_KEYS = [
    "case_number",
    "project_address",
    "lot_number",
    "assessor_block",
    "project_descr",
    "type_district",
    "type_district_descr",
    "speakers",
    "action",
    "modifications",
    "ayes",
    "noes",
    "absent",
    "vote",
    "action_name",
]

# pre‐compile two regexes:
_scalar_re = {
    key: re.compile(rf'"{key}"\s*:\s*"([^"]*?)"', re.IGNORECASE)
    for key in REQUIRED_KEYS
}
_list_re = {
    key: re.compile(rf'"{key}"\s*:\s*(\[[^\]]*?\])', re.IGNORECASE)
    for key in REQUIRED_KEYS
}

def extract_json(pred: str):
    out = {}
    for key in REQUIRED_KEYS:
        # try list first
        m = _list_re[key].search(pred)
        if m:
            try:
                out[key] = json.loads(m.group(1))
            except json.JSONDecodeError:
                out[key] = []
            continue

        # then scalar
        m = _scalar_re[key].search(pred)
        if m:
            out[key] = m.group(1).strip()
        else:
            out[key] = ""  # default if missing

    return out

def main():
    home      = Path.home()
    work_dir  = home / "housing_project" / "data" / "meeting_minutes"
    raw_dir   = work_dir / "raw"
    model_dir = work_dir / "processed" / "minutes_extractor"
    out_file  = work_dir / "processed" / "structured_data.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # load fine-tuned model
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model     = T5ForConditionalGeneration.from_pretrained(model_dir)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # only test on the one file
    target = raw_dir / "1998" / "minutes_1.txt"
    text   = target.read_text(encoding="utf-8", errors="ignore")

    # break into separate meetings by date
    for meeting in split_meetings(text):
        date_str = meeting["date"].strftime("%Y-%m-%d")
        body     = meeting["text"]

        # find every case-header position
        starts = [m.start() for m in case_header_re.finditer(body)]
        if not starts:
            continue
        starts.append(len(body))

        # slice into blocks
        blocks = [
            body[starts[i]:starts[i+1]].strip()
            for i in range(len(starts)-1)
        ]

        # batch the prompts through the model
        prompts = [PROMPT_INSTRUCTION + blk for blk in blocks]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        outs = model.generate(**enc, max_new_tokens=256, num_beams=3, do_sample=False)
        preds = tokenizer.batch_decode(outs, skip_special_tokens=True)

        # write JSONL
        # Inside the main function, replace the JSONL writing block with:
        with open(out_file, "a", encoding="utf-8") as fout, open(work_dir / "processed" / "extraction_errors.log", "a", encoding="utf-8") as errlog:
            for blk, pred in zip(blocks, preds):
                data = extract_json(pred)
                entry = {
                "source_file": target.name,
                "meeting_date": date_str,
                "block_header": blk.split("\n",1)[0].strip(),
                **data
            }
            if "_raw_extraction_error" not in data:
                # Write structured data to JSONL
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            else:
                # Log errors to a separate file
                errlog.write(f"File: {target.name}, Meeting: {date_str}, Block: {entry['block_header']}\nError: {data['_raw_extraction_error']}\n\n")

        print(f"✔ Processed {target.name} for meeting {date_str}")

    print("🎉 Done — see", out_file)

if __name__ == "__main__":
    main()