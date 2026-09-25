"""Fetch Crossref metadata for every DOI in lit/dois.txt and write lit/refs.bib (field `verified`) and
lit/crossref_cache.json. Entries whose DOI does not resolve in Crossref are reported and not written."""
import json
import datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
H = {"User-Agent": "papers-work/0.1 (mailto:sgrenkov39@gmail.com)"}


def main():
    cache_f = HERE / "crossref_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    today = datetime.date.today().isoformat()
    bib, bad = [], []
    for line in (HERE / "dois.txt").read_text().split("\n"):
        if not line.strip():
            continue
        key, doi = line.split()
        if doi not in cache:
            r = requests.get(f"https://api.crossref.org/works/{doi}", headers=H, timeout=60)
            if r.status_code != 200:
                bad.append((key, doi, r.status_code))
                continue
            cache[doi] = r.json()["message"]
        m = cache[doi]
        authors = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in m.get("author", [])
                            if a.get("family"))  # Crossref sometimes lists affiliations as nameless authors
        # year of the printed issue when Crossref has it (online-first dates can be a year or two earlier)
        year = next((d["date-parts"][0][0] for d in (m.get("published-print"), m.get("issued")) if d and d.get("date-parts")
                     and d["date-parts"][0][0]), None)
        typ = {"journal-article": "article", "book": "book", "monograph": "book"}.get(m["type"], "misc")
        fields = {"author": authors, "title": (m.get("title") or [""])[0], "year": year,
                  "journal" if typ == "article" else "publisher":
                      (m.get("container-title") or [""])[0] if typ == "article" else m.get("publisher", ""),
                  "volume": m.get("volume"), "number": m.get("issue"), "pages": m.get("page"), "doi": doi, "title_original": (m.get("original-title") or [None])[0],
                  "verified": f"Crossref {today}"}
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
        bib.append(f"@{typ}{{{key},\n{body}\n}}")
    cache_f.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    manual = (HERE / "refs_manual.bib").read_text() if (HERE / "refs_manual.bib").exists() else ""
    (HERE / "refs.bib").write_text("\n\n".join(bib) + "\n\n" + manual)
    print(len(bib), "entries verified;", "not found:", bad)


if __name__ == "__main__":
    main()
