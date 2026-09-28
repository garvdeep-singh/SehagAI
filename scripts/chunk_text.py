"""
chunk_text.py

Reads:
    cleaned_text/*.txt

Writes:
    chunks/chunks.jsonl

Each chunk keeps:
    - chunk_id
    - scheme_id
    - source_file
    - title
    - state_or_ministry
    - tags
    - section
    - text

Run:
    python3 scripts/chunk_text.py
"""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "cleaned_text"
OUTPUT_DIR = ROOT / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

# Approximate character-based limits.
# We will tune these later after inspecting the resulting chunks.
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 400


def parse_file(path: Path):
    """Read one cleaned scheme file and extract metadata + sections."""

    text = path.read_text(encoding="utf-8")

    lines = text.splitlines()

    title = ""
    state = ""
    tags = []

    sections = {}
    current_section = None
    current_content = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("TITLE:"):
            title = line[len("TITLE:"):].strip()

        elif line.startswith("STATE_OR_MINISTRY:"):
            state = line[len("STATE_OR_MINISTRY:"):].strip()

        elif line.startswith("TAGS:"):
            raw_tags = line[len("TAGS:"):].strip()
            tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]

        elif line.startswith("## "):
            # Save previous section
            if current_section is not None:
                sections[current_section] = " ".join(current_content).strip()

            current_section = line[3:].strip()
            current_content = []

        else:
            if current_section is not None:
                current_content.append(line)

    # Save final section
    if current_section is not None:
        sections[current_section] = " ".join(current_content).strip()

    return {
        "title": title,
        "state_or_ministry": state,
        "tags": tags,
        "sections": sections,
    }


def split_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split long text into overlapping chunks.

    Tries to break at sentence boundaries instead of cutting
    blindly in the middle of a sentence.
    """

    if len(text) <= size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + size, len(text))

        if end < len(text):
            # Prefer a sentence boundary near the target size.
            sentence_breaks = [
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
            ]

            best_break = max(sentence_breaks)

            if best_break > start + int(size * 0.6):
                end = best_break + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap

        # Prevent getting stuck.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.txt"))

    if not files:
        print("No cleaned text files found.")
        return

    total_chunks = 0
    total_documents = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:

        for path in files:

            data = parse_file(path)

            scheme_id = path.stem

            for section, section_text in data["sections"].items():

                if not section_text:
                    continue

                chunks = split_text(section_text)

                for index, chunk in enumerate(chunks):

                    chunk_id = f"{scheme_id}_{section.lower().replace(' ', '_')}_{index}"

                    record = {
                        "chunk_id": chunk_id,
                        "scheme_id": scheme_id,
                        "source_file": path.name,
                        "title": data["title"],
                        "state_or_ministry": data["state_or_ministry"],
                        "tags": data["tags"],
                        "section": section,
                        "text": chunk,
                    }

                    output.write(
                        json.dumps(record, ensure_ascii=False)
                        + "\n"
                    )

                    total_chunks += 1

            total_documents += 1

    print("=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)
    print(f"Documents processed : {total_documents}")
    print(f"Total chunks        : {total_chunks}")
    print(f"Chunk size          : {CHUNK_SIZE} characters")
    print(f"Chunk overlap       : {CHUNK_OVERLAP} characters")
    print(f"Output              : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()