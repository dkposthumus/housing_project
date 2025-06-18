#!/usr/bin/env python3
import json, re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ── Helpers ────────────────────────────────────────────────────────────────
# Split on lines like "1.  98.226D …"
CASE_HEADER = re.compile(
    r"^\s*\d{1,2}[A-Za-z]?\.\s+"      # item number like “1.” or “12A.”
    r"\d{2,}\.\d{3,}"                 # numeric code “98.226” or “97.470”
    r"(?:[A-Za-z/]+)?",               # optional trailing letters or “/” (e.g. “D”, “ET”, “DD”)
    re.MULTILINE
)

def split_blocks(text: str):
    """Return each project block in the full minutes text."""
    cuts = [m.start() for m in CASE_HEADER.finditer(text)] + [len(text)]
    return [ text[cuts[i]:cuts[i+1]].strip() for i in range(len(cuts)-1) ]

def extract_json(output_str: str):
    """Find the first {...} in the string and parse it."""
    start = output_str.find("{")
    end   = output_str.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(output_str[start:end])

# ── Paths & Device ─────────────────────────────────────────────────────────
home      = Path.home()
minutes   = home/"housing_project"/"data"/"meeting_minutes"
raw_dir   = minutes/"raw"/"1998"
model_dir = minutes/"processed"/"minutes_extractor"
out_file  = minutes/"processed"/"extracted_results.jsonl"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Load model + adapter ────────────────────────────────────────────────────
base  = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map=None)
model = PeftModel.from_pretrained(base, model_dir)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# ── Generation settings ─────────────────────────────────────────────────────
generation_kwargs = {
    "max_new_tokens": 256,
    "temperature": 0.0,     # deterministic
    "num_beams": 1,
}

# ── Process every .txt in 1998 ───────────────────────────────────────────────
with open(out_file, "w", encoding="utf-8") as fout:
    for txt_path in sorted(raw_dir.glob("*.txt")):
        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
        blocks = split_blocks(raw)
        for blk in blocks:
            # Use the block itself as the prompt, no extra markers
            prompt = (
                "You are a JSON extractor. "
                "Given this Planning Commission block, output only a single JSON object "
                "with the keys: case_number, project_address, lot_number, assessor_block, "
                "project_descr, type_district, type_district_descr, speakers, action, "
                "ayes, noes, absent, vote, action_name, modifications, etc.  \n\n"
                f"{blk}\n\n"
                "### JSON:"
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            out_ids = model.generate(**inputs, **generation_kwargs)
            out_str = tokenizer.decode(out_ids[0], skip_special_tokens=True)

            try:
                structured = extract_json(out_str)
            except Exception as e:
                # fallback: store entire raw output
                structured = {"_raw_output": out_str}

            # merge in metadata if you like
            structured["source_file"]   = txt_path.name
            structured["block_header"]  = blk.split("\n",1)[0].strip()

            # write one JSONL line
            fout.write(json.dumps(structured, ensure_ascii=False) + "\n")

        print(f"✔︎ {txt_path.name}: {len(blocks)} blocks processed")

print(f"\nAll done. Results in: {out_file}")
