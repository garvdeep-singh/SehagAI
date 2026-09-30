import json
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_FILE = ROOT / "chunks" / "embeddings.jsonl"

COLLECTION_NAME = "government_schemes"
VECTOR_SIZE = 1024
BATCH_SIZE = 100


def main():
    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]

    client = QdrantClient(
        url=url,
        api_key=api_key,
    )

    print("Connected to Qdrant Cloud.")

    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")

    points = []

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            points.append(
                PointStruct(
                    id=item["chunk_id"],
                    vector=item["embedding"],
                    payload={
                        "chunk_id": item["chunk_id"],
                        "scheme_id": item["scheme_id"],
                        "source_file": item["source_file"],
                        "title": item["title"],
                        "state_or_ministry": item["state_or_ministry"],
                        "tags": item["tags"],
                        "section": item["section"],
                        "text": item["text"],
                    },
                )
            )

            if len(points) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                print(f"Uploaded {len(points)} vectors")
                points = []

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        print(f"Uploaded {len(points)} vectors")

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )

    print(f"Total vectors in cloud: {count.count}")


if __name__ == "__main__":
    main()