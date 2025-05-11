import pandas as pd
from pathlib import Path

# === Set up paths ===
home = Path.home()
work_dir = home / "housing_project"
data = work_dir / "data"
minutes = data / "meeting_minutes"
minutes_raw = minutes / "pdfs"
minutes_clean = minutes / "processed"
text_dir = minutes / "text"
text_dir.mkdir(parents=True, exist_ok=True)

# now concatenate (vertically) every .csv contained in the processed subfolder 
csv_files = list(minutes_clean.glob("*.csv"))

# Load and concatenate all CSVs
df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Save the master file
df_all.to_csv(f"{minutes}/projects_master.csv", index=False)