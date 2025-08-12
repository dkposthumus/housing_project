#!/usr/bin/env python3
"""
training_sample_create.py – build consolidated training.txt from per-year labels & raw minutes

Layout (all in tagged/training/):
  tagged/
    training/
      1996_labeled.json             # JSON array of ~10–15 labeled cases
      1996_sample.txt               # or .rtf — may have multiple files per year:
      1996_sample_part2.rtf         #   1996_sample_*.txt/.rtf are all included
      1997_labeled.json
      1997_sample.rtf
      ...
      training.txt                  # OUTPUT: JSONL for train.py
      logs/
        diagnostics_1996.json
        diagnostics_1997.json
        summary.json
"""

from striprtf.striprtf import rtf_to_text
import json, re
from pathlib import Path

# ───────────────────────── paths ─────────────────────────
home      = Path.home()
base      = home / "housing_project" / "data" / "meeting_minutes"
train_dir = base / "tagged" / "training"
train_dir.mkdir(parents=True, exist_ok=True)

logdir    = train_dir / "logs"
logdir.mkdir(parents=True, exist_ok=True)

outfile   = train_dir / "training.txt"      # consolidated JSONL

# ───────────────────────── constants ─────────────────────
EOJ = "<extra_id_0>"
BLOCK_RE = re.compile(r"<<Project Start>>(.*?)<<Project End>>", re.S)

# Flexible case-number: 2 or 4 digits before dot; ≥3 digits after; optional suffix
CASE_RE  = re.compile(r"\b((?:\d{2}|\d{4})\.\d{3,}(?:[A-Z0-9/]+)?)\b")

YEAR_LABEL_RE = re.compile(r"^(?P<year>\d{4})_labeled\.json$", re.I)

# Keys your model expects; fill missing with "" (lists for ayes/noes/absent)
REQUIRED = [
    "case_number","project_address","lot_number","assessor_block","project_descr",
    "type_district","type_district_descr","speakers","action","modifications",
    "ayes","noes","absent","vote","action_name"
]

def normalise_case(code: str) -> str:
    return code.replace(" ", "").upper() if code else ""

def read_plain_text(p: Path) -> str:
    if p.suffix.lower() == ".rtf":
        return rtf_to_text(p.read_text(encoding="utf-8", errors="ignore"))
    # default: treat as plain text (.txt, etc.)
    return p.read_text(encoding="utf-8", errors="ignore")

def collect_years() -> list[int]:
    years = []
    for fp in train_dir.glob("*_labeled.json"):
        m = YEAR_LABEL_RE.match(fp.name)
        if not m:
            continue
        y = int(m.group("year"))
        # require at least one sample file for the same year (.txt or .rtf)
        has_sample = any(
            train_dir.glob(f"{y}_sample*.{ext}")
            for ext in ("txt", "rtf")
        )
        if has_sample:
            years.append(y)
    return sorted(set(years))

def load_year_labels(year: int) -> list[dict]:
    f = train_dir / f"{year}_labeled.json"
    with f.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"{f.name} must be a JSON array")
        return data

def load_year_blocks(year: int) -> list[str]:
    """
    Read the year's sample file(s) (TXT/RTF) and extract project blocks.
    Accepts multiple files matching {year}_sample*.txt/.rtf.
    """
    texts = []
    for fp in sorted(list(train_dir.glob(f"{year}_sample*.txt")) + list(train_dir.glob(f"{year}_sample*.rtf"))):
        try:
            texts.append(read_plain_text(fp))
        except Exception as e:
            print(f"⚠ Skipping {fp.name}: {e}")
    if not texts:
        return []
    plain = "\n\n".join(texts)
    return [b.strip() for b in BLOCK_RE.findall(plain)]

def make_block_map(blocks: list[str]) -> dict[str, str]:
    """Map case_number → block. If duplicates, keep the longest block."""
    m = {}
    for blk in blocks:
        mm = CASE_RE.search(blk)
        code = normalise_case(mm.group(1) if mm else None)
        if not code:
            continue
        prev = m.get(code)
        if prev is None or len(blk) > len(prev):
            m[code] = blk
    return m

def ensure_required_fields(lbl: dict) -> dict:
    lab = dict(lbl)  # shallow copy
    for k in REQUIRED:
        if k not in lab:
            lab[k] = "" if k not in {"ayes","noes","absent"} else []
    return lab

def build_examples_for_year(year: int) -> tuple[list[dict], dict]:
    labels = load_year_labels(year)
    blocks = load_year_blocks(year)
    block_map = make_block_map(blocks)

    examples = []
    missing = []
    unmatched_blocks = set(block_map.keys())

    for lab in labels:
        code = normalise_case(lab.get("case_number"))
        raw = block_map.get(code)
        if raw is None:
            missing.append(code or "<missing case_number>")
            continue
        unmatched_blocks.discard(code)

        lab_norm = ensure_required_fields(lab)
        comp = json.dumps(lab_norm, ensure_ascii=False) + f" {EOJ}"
        examples.append({"prompt": raw + "\n\n", "completion": comp})

    stats = {
        "year": year,
        "labels": len(labels),
        "blocks_found": len(blocks),
        "paired": len(examples),
        "labels_without_block": len(missing),
        "unmatched_blocks": len(unmatched_blocks)
    }
    # write diagnostics
    diag = {
        "missing_label_case_numbers": missing,
        "unmatched_block_case_numbers": sorted(unmatched_blocks)
    }
    (logdir / f"diagnostics_{year}.json").write_text(
        json.dumps({"stats": stats, "details": diag}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[{year}] labels={stats['labels']} blocks={stats['blocks_found']} "
          f"paired={stats['paired']} missing_labels={stats['labels_without_block']} "
          f"unmatched_blocks={stats['unmatched_blocks']}")
    return examples, stats

def main():
    years = collect_years()
    if not years:
        print("No years found: expected files like '1998_labeled.json' and '1998_sample*.rtf/txt' in tagged/training/.")
        return

    all_examples = []
    all_stats = []
    for y in years:
        ex, st = build_examples_for_year(y)
        all_examples.extend(ex)
        all_stats.append(st)

    # write consolidated training JSONL
    with outfile.open("w", encoding="utf-8") as fout:
        for ex in all_examples:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(all_examples)} examples to {outfile}")
    # summary stats
    (logdir / "summary.json").write_text(
        json.dumps({"years": all_stats, "total_examples": len(all_examples)}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
