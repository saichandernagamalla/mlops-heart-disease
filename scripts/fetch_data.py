"""Fetch and clean the processed Cleveland subset from the official UCI archive."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
from src.data import load_dataset

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

def main():
    raw_dir, processed_dir = Path("data/raw"), Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True); processed_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=30) as response: archive = response.read()
    with ZipFile(BytesIO(archive)) as zf:
        member = next((n for n in zf.namelist() if n.endswith("processed.cleveland.data")), None)
        if not member: raise RuntimeError("processed.cleveland.data not found in UCI archive")
        raw = zf.read(member)
    raw_path = raw_dir / "processed.cleveland.data"; raw_path.write_bytes(raw)
    data = load_dataset(raw_path); data.to_csv(processed_dir / "heart.csv", index=False)
    print(f"Saved {len(data)} rows; {int(data.isna().sum().sum())} missing cells retained for pipeline imputation")

if __name__ == "__main__": main()
