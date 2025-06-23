import pandas as pd
from pathlib import Path

def jsonl_to_csv(jsonl_path: Path, csv_path: Path):
    # 1️⃣ Read in the JSONL
    df = pd.read_json(jsonl_path, lines=True)

    # 2️⃣ If your inference script emitted an “extracted” column (a nested dict),
    #    unpack it so that each key becomes its own top-level column.
    if "extracted" in df.columns:
        extracted = pd.json_normalize(df["extracted"])
        df = pd.concat([df.drop(columns=["extracted"]), extracted], axis=1)

    # 3️⃣ Turn any list-columns into comma-joined strings
    for col in ("speakers", "ayes", "noes", "absent", "modifications"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    # 4️⃣ (Optional) Pick an order for your columns
    cols = [
        "source_file", "meeting_date", "case_number", "project_address",
        "lot_number", "assessor_block", "project_descr",
        "type_district", "type_district_descr", "action",
        "action_name", "vote", "ayes", "noes", "absent",
        "modifications", "block_header"
    ]
    # Keep only the ones you have, in that order, plus anything else at the end
    present = [c for c in cols if c in df.columns]
    others  = [c for c in df.columns if c not in present]
    df = df[present + others]

    # 5️⃣ Finally write out
    df.to_csv(csv_path, index=False)
    return df

# === Usage ===
home        = Path.home()
work_dir    = home / "housing_project" / "data" / "meeting_minutes"
minutes_clean = work_dir / "processed"
jsonl_path  = minutes_clean / "structured_data.jsonl"
csv_path    = minutes_clean / "extracted_results.csv"

df = jsonl_to_csv(jsonl_path, csv_path)
print(f"✓ Wrote {len(df)} rows to {csv_path}")
