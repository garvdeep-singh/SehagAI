import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "chunks" / "chunks.jsonl"
OUTPUT_FILE = ROOT / "chunks" / "embeddings.jsonl"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32


def main():
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded.")
    print(f"Reading: {INPUT_FILE}")

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Chunks loaded: {len(records)}")

    texts = []

    for record in records:
        # Include section/title context in the embedding.
        text = (
            f"Scheme: {record['title']}\n"
            f"State/Ministry: {record['state_or_ministry']}\n"
            f"Section: {record['section']}\n"
            f"Content: {record['text']}"
        )

        texts.append(text)

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("Saving embeddings...")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record, embedding in zip(records, embeddings):

            output = {
                **record,
                "embedding": embedding.tolist(),
            }

            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"Chunks embedded : {len(records)}")
    print(f"Embedding model : {MODEL_NAME}")
    print(f"Embedding size  : {len(embeddings[0])}")
    print(f"Output          : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()