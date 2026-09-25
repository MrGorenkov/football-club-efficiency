"""Download the transfermarkt-datasets tables (dcaribou/transfermarkt-datasets, CC0-1.0) into data/raw/.

The project publishes each table as a gzip CSV on its public bucket (links in the project README).
Writes data/raw/<table>.csv.gz and data/raw/manifest.json (size, SHA-256, HTTP Last-Modified).
"""
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
TABLES = ["games", "club_games", "clubs", "competitions", "players", "player_valuations", "transfers", "appearances"]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for t in TABLES:
        url = f"{BASE}{t}.csv.gz"
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        (RAW / f"{t}.csv.gz").write_bytes(r.content)
        manifest[t] = {"url": url, "bytes": len(r.content), "sha256": hashlib.sha256(r.content).hexdigest(),
                       "last_modified": r.headers.get("Last-Modified")}
        print(t, len(r.content), manifest[t]["last_modified"])
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
