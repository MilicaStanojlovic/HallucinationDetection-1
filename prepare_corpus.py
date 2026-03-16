"""
prepare_corpus.py
─────────────────
Downloads a small public-domain text (War and Peace via Project Gutenberg)
and saves it as data/corpus.txt.
"""

import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
DEST = Path("data/corpus.txt")

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Corpus already at {DEST}  ({DEST.stat().st_size:,} bytes)")
        return
    print(f"Downloading corpus from {URL} …")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {DEST}  ({DEST.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
