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

# now we want to convert the complete .jsonl file into a CSV file 
def jsonl_to_csv(jsonl_path, csv_path):
    # Read the JSONL file into a DataFrame
    df = pd.read_json(jsonl_path, lines=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)
# Convert the JSONL file to CSV
jsonl_path = minutes_clean / "extracted_results.jsonl"
csv_path = minutes_clean / "extracted_results.csv"
jsonl_to_csv(jsonl_path, csv_path)
# Print confirmation
print(f"Converted {jsonl_path} to {csv_path}")