#!/usr/bin/env python3
"""
training_sample_create.py – build consolidated training.txt from per-year labels & raw minutes
and produce an inference-safe corpus that excludes the training examples.

Layout (all training inputs live in tagged/training/):
  tagged/
    <year>/<YYYY-MM-DD>.txt                 # <-- produced by your parser (original tagged corpus)
    training/
      1996_labeled.json                     # JSON array of ~10–15 labeled cases
      1996_sample.rtf                       # or .txt (contains <<Project Start>> ... <<Project End>>)
      1997_labeled.json
      1997_sample.txt
      ...
      training.txt                          # OUTPUT: JSONL for train.py
      train_case_ids.txt                    # OUTPUT: list of case_numbers used in training
      logs/
        diagnostics_1996.json
        diagnostics_1997.json
        summary.json
    filtered_for_inference/                 # <-- OUTPUT corpus with training blocks removed
      <year>/<YYYY-MM-DD>.txt

This script:
  1) Builds training.txt from per-year labeled JSON + year_sample.(rtf|txt)
  2) Scans the main tagged corpus (tagged/<year>/*.txt) and writes a filtered copy
     to tagged/filtered_for_inference/ excluding any blocks whose case_number
     appears in the training set (prevents data leakage during evaluation/inference).
"""

from striprtf.striprtf import rtf_to_text
import json, re
from pathlib import Path

# ───────────────────────── paths ─────────────────────────
home      = Path.home()
base      = home / "housing_project" / "data" / "meeting_minutes"
tag_dir   = base / "tagged"                 # originals from parser
train_dir = tag_dir / "training"            # training inputs/outputs live here
train_dir.mkdir(parents=True, exist_ok=True)

logdir    = train_dir / "logs"
logdir.mkdir(parents=True, exist_ok=True)

outfile   = train_dir / "training.txt"      # consolidated JSONL
train_ids_out = train_dir / "train_case_ids.txt"

# Where to write inference-safe corpus
filtered_dir = tag_dir / "filtered_for_inference"
filtered_dir.mkdir(parents=True, exist_ok=True)

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
    return p.read_text(encoding="utf-8", errors="ignore")

def collect_years() -> list[int]:
    years = []
    for fp in train_dir.glob("*_labeled.json"):
        m = YEAR_LABEL_RE.match(fp.name)
        if m:
            y = int(m.group("year"))
            # require a raw sample for the same year
            if any((train_dir / f).exists() for f in (f"{y}_sample.rtf", f"{y}_sample.txt")):
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
    """Read the year's sample file (RTF/TXT) and extract project blocks."""
    texts = []
    for name in (f"{year}_sample.rtf", f"{year}_sample.txt"):
        fp = train_dir / name
        if fp.exists():
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

def build_examples_for_year(year: int) -> tuple[list[dict], dict, set[str]]:
    labels = load_year_labels(year)
    blocks = load_year_blocks(year)
    block_map = make_block_map(blocks)

    examples = []
    missing = []
    unmatched_blocks = set(block_map.keys())
    used_case_ids: set[str] = set()

    for lab in labels:
        code = normalise_case(lab.get("case_number"))
        raw = block_map.get(code)
        if raw is None:
            missing.append(code or "<missing case_number>")
            continue
        unmatched_blocks.discard(code)
        used_case_ids.add(code)

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
    return examples, stats, used_case_ids

# ───────────────────────── leakage guard: filter corpus ──────────────────

def extract_blocks(text: str) -> list[str]:
    """Return list of block texts inside <<Project Start>> ... <<Project End>>."""
    return [b.strip() for b in BLOCK_RE.findall(text)]

def rebuild_with_blocks(blocks: list[str]) -> str:
    """Rebuild a tagged minutes file from a list of blocks."""
    if not blocks:
        return ""
    chunks = []
    for b in blocks:
        chunks.append("<<Project Start>>")
        chunks.append(b.rstrip())
        chunks.append("<<Project End>>")
    return "\n".join(chunks) + "\n"

def filter_tagged_corpus(exclude_case_ids: set[str]) -> dict:
    """
    Read tagged/<year>/*.txt and write filtered copies to tagged/filtered_for_inference/<year>/*.txt,
    excluding any blocks whose case_number is in exclude_case_ids.
    Returns simple stats dict.
    """
    stats = {"files_scanned": 0, "files_written": 0, "blocks_kept": 0, "blocks_removed": 0}
    # walk year directories under tag_dir, skipping 'training' and 'filtered_for_inference'
    for year_dir in sorted(tag_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        name = year_dir.name
        if name in {"training", "filtered_for_inference"}:
            continue
        if not name.isdigit() or len(name) != 4:
            continue

        out_year = filtered_dir / name
        out_year.mkdir(parents=True, exist_ok=True)

        for fp in sorted(year_dir.glob("*.txt")):
            stats["files_scanned"] += 1
            text = fp.read_text(encoding="utf-8", errors="ignore")
            blocks = extract_blocks(text)
            kept = []
            removed_count = 0
            for blk in blocks:
                m = CASE_RE.search(blk)
                code = normalise_case(m.group(1)) if m else ""
                if code and code in exclude_case_ids:
                    removed_count += 1
                    continue
                kept.append(blk)

            if kept:
                out_text = rebuild_with_blocks(kept)
                (out_year / fp.name).write_text(out_text, encoding="utf-8")
                stats["files_written"] += 1
                stats["blocks_kept"] += len(kept)
                stats["blocks_removed"] += removed_count
            else:
                # No blocks left after filtering: skip writing this file.
                stats["blocks_removed"] += removed_count

    # write a small summary file
    (filtered_dir / "FILTERING_SUMMARY.json").write_text(
        json.dumps({
            "excluded_case_ids_count": len(exclude_case_ids),
            "stats": stats
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Leakage guard → filtered corpus at {filtered_dir}")
    print(f"   files_scanned={stats['files_scanned']} files_written={stats['files_written']} "
          f"blocks_kept={stats['blocks_kept']} blocks_removed={stats['blocks_removed']}")
    return stats

# ───────────────────────── main ──────────────────────────────────────────

def main():
    years = collect_years()
    if not years:
        print("No years found: expected files like '1998_labeled.json' and '1998_sample.rtf/txt' in tagged/training/.")
        return

    all_examples = []
    all_stats = []
    all_train_ids: set[str] = set()

    for y in years:
        ex, st, used_ids = build_examples_for_year(y)
        all_examples.extend(ex)
        all_stats.append(st)
        all_train_ids.update(used_ids)

    # write consolidated training JSONL
    with outfile.open("w", encoding="utf-8") as fout:
        for ex in all_examples:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # write train case ids (one per line)
    train_ids_out.write_text("\n".join(sorted(all_train_ids)) + "\n", encoding="utf-8")

    print(f"✓ Wrote {len(all_examples)} examples to {outfile}")
    print(f"✓ Wrote {len(all_train_ids)} training case IDs to {train_ids_out}")

    # summary stats
    (logdir / "summary.json").write_text(
        json.dumps({"years": all_stats, "total_examples": len(all_examples)}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # --- Leakage prevention: build inference-safe mirror excluding training IDs ---
    filter_tagged_corpus(exclude_case_ids=all_train_ids)

if __name__ == "__main__":
    main()
