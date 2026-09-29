import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "chunks" / "chunks.jsonl"

TOP_K = 5


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def main():
    print("Loading chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    corpus = [
        tokenize(
            f"{chunk['title']} "
            f"{chunk['state_or_ministry']} "
            f"{chunk['section']} "
            f"{chunk['tags']} "
            f"{chunk['text']}"
        )
        for chunk in chunks
    ]

    print("Building BM25 index...")

    bm25 = BM25Okapi(corpus)

    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:TOP_K]

    print("\n" + "=" * 70)
    print("TOP BM25 RETRIEVED CHUNKS")
    print("=" * 70)

    for rank, index in enumerate(top_indices, 1):

        chunk = chunks[index]

        print(f"\n[{rank}] Score: {scores[index]:.4f}")
        print(f"Scheme  : {chunk['title']}")
        print(f"State   : {chunk['state_or_ministry']}")
        print(f"Section : {chunk['section']}")
        print(f"Source  : {chunk['source_file']}")

        print("\nText:")
        print(chunk["text"])

        print("-" * 70)


if __name__ == "__main__":
    main()