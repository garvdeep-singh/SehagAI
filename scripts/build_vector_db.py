import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "chunks" / "embeddings.jsonl"
DB_PATH = ROOT / "qdrant_db"

COLLECTION_NAME = "government_schemes"

VECTOR_SIZE = 1024
BATCH_SIZE = 100


def main():

    print("Connecting to Qdrant...")

    client = QdrantClient(path=str(DB_PATH))

    # Recreate collection so the script can be safely rerun.
    if client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print("Reading embeddings...")

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Records loaded: {len(records)}")

    points = []

    for idx, record in enumerate(records):

        point = PointStruct(
            id=idx,
            vector=record["embedding"],
            payload={
                "chunk_id": record["chunk_id"],
                "scheme_id": record["scheme_id"],
                "source_file": record["source_file"],
                "title": record["title"],
                "state_or_ministry": record["state_or_ministry"],
                "tags": record["tags"],
                "section": record["section"],
                "text": record["text"],
            },
        )

        points.append(point)

        if len(points) >= BATCH_SIZE:

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )

            points = []

            print(f"Inserted {idx + 1}/{len(records)}")

    # Insert remaining points
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    info = client.get_collection(COLLECTION_NAME)

    print("=" * 60)
    print("QDRANT DATABASE CREATED")
    print("=" * 60)
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Vectors    : {info.points_count}")
    print(f"Dimension  : {VECTOR_SIZE}")
    print(f"Distance   : COSINE")
    print(f"Database   : {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()