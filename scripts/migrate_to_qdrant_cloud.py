# import json
# import os
# from pathlib import Path

# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct

# ROOT = Path(__file__).resolve().parent.parent
# EMBEDDINGS_FILE = ROOT / "chunks" / "embeddings.jsonl"

# COLLECTION_NAME = "government_schemes"
# VECTOR_SIZE = 1024
# BATCH_SIZE = 20
# UPLOAD_TIMEOUT = 120


# def main():
#     url = os.environ["QDRANT_URL"]
#     api_key = os.environ["QDRANT_API_KEY"]

#     client = QdrantClient(
#         url=url,
#         api_key=api_key,
#     )

#     print("Connected to Qdrant Cloud.")

#     existing = [c.name for c in client.get_collections().collections]

#     if COLLECTION_NAME not in existing:
#         client.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config=VectorParams(
#                 size=VECTOR_SIZE,
#                 distance=Distance.COSINE,
#             ),
#         )
#         print(f"Created collection: {COLLECTION_NAME}")
#     else:
#         print(f"Collection already exists: {COLLECTION_NAME}")

#     # points = []
#     # uploaded = 0
#     points = []
#     point_id = 0
#     uploaded = 0

#     with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue

#             item = json.loads(line)

#             points.append(
#                 PointStruct(
#                     # id=item["chunk_id"],
#                     # id=uploaded,
#                     id=point_id,
#                     point_id += 1,
#                     vector=item["embedding"],
#                     payload={
#                         "chunk_id": item["chunk_id"],
#                         "scheme_id": item["scheme_id"],
#                         "source_file": item["source_file"],
#                         "title": item["title"],
#                         "state_or_ministry": item["state_or_ministry"],
#                         "tags": item["tags"],
#                         "section": item["section"],
#                         "text": item["text"],
#                     },
#                 )
#             )

#             if len(points) >= BATCH_SIZE:
#                 client.upsert(
#                     collection_name=COLLECTION_NAME,
#                     points=points,
#                     wait=True,
#                     timeout=UPLOAD_TIMEOUT,
#                 )

#                 uploaded += len(points)
#                 print(f"Uploaded {uploaded}/13671 vectors")
#                 points = []

#     if points:
#         client.upsert(
#             collection_name=COLLECTION_NAME,
#             points=points,
#             wait=True,
#             timeout=UPLOAD_TIMEOUT,
#         )

#         uploaded += len(points)
#         print(f"Uploaded {uploaded}/13671 vectors")

#     count = client.count(
#         collection_name=COLLECTION_NAME,
#         exact=True,
#     )

#     print("=" * 60)
#     print(f"Total vectors in cloud: {count.count}")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()










import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


# ============================================================
# CONFIG
# ============================================================

EMBEDDINGS_FILE = "chunks/embeddings.jsonl"
COLLECTION_NAME = "government_schemes"

VECTOR_SIZE = 1024
BATCH_SIZE = 20
UPLOAD_TIMEOUT = 120


# ============================================================
# CONNECT TO QDRANT CLOUD
# ============================================================

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# ============================================================
# CREATE COLLECTION IF IT DOES NOT EXIST
# ============================================================

collections = client.get_collections().collections

collection_exists = any(
    collection.name == COLLECTION_NAME
    for collection in collections
)

if not collection_exists:
    print(f"Creating collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print("Collection created.")

else:
    print(f"Collection already exists: {COLLECTION_NAME}")


# ============================================================
# UPLOAD EMBEDDINGS
# ============================================================

points = []

# IMPORTANT:
# Qdrant Cloud requires point IDs to be integers or UUIDs.
# We therefore use a unique integer for every embedding.
point_id = 0

uploaded = 0

print("=" * 60)
print("Starting migration to Qdrant Cloud")
print("=" * 60)


with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if not line.strip():
            continue

        item = json.loads(line)

        points.append(
            PointStruct(
                id=point_id,

                vector=item["embedding"],

                payload={
                    # Original chunk identifier
                    "chunk_id": item["chunk_id"],

                    # Metadata
                    "scheme_id": item["scheme_id"],
                    "source_file": item["source_file"],
                    "title": item["title"],
                    "state_or_ministry": item["state_or_ministry"],
                    "tags": item["tags"],
                    "section": item["section"],

                    # Actual chunk text
                    "text": item["text"],
                },
            )
        )

        # Move to the next unique Qdrant point ID
        point_id += 1

        # Upload every BATCH_SIZE points
        if len(points) >= BATCH_SIZE:

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
                timeout=UPLOAD_TIMEOUT,
            )

            uploaded += len(points)

            print(
                f"Uploaded {uploaded} vectors"
            )

            # Clear batch
            points = []


# ============================================================
# UPLOAD REMAINING POINTS
# ============================================================

if points:

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
        timeout=UPLOAD_TIMEOUT,
    )

    uploaded += len(points)

    print(
        f"Uploaded {uploaded} vectors"
    )


# ============================================================
# VERIFY CLOUD COUNT
# ============================================================

count = client.count(
    collection_name=COLLECTION_NAME,
    exact=True,
)

print("=" * 60)
print("Migration complete")
print("=" * 60)

print(f"Vectors uploaded: {uploaded}")
print(f"Vectors in cloud : {count.count}")
print(f"Expected vectors : 13671")

print("=" * 60)

if count.count == 13671:
    print("SUCCESS: All 13,671 vectors are in Qdrant Cloud.")
else:
    print(
        f"WARNING: Expected 13671 vectors, "
        f"but Qdrant Cloud contains {count.count}."
    )