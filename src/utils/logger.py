import csv
from pathlib import Path
from datetime import datetime

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def log_prediction(csv_path: Path, row: dict):
    ensure_parent(csv_path)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def now_iso():
    return datetime.now().isoformat(timespec="seconds")
