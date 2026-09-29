# import json
# import re
# from pathlib import Path

# from rank_bm25 import BM25Okapi
# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer


# ROOT = Path(__file__).resolve().parent.parent

# CHUNKS_FILE = ROOT / "chunks" / "chunks.jsonl"
# DB_PATH = ROOT / "qdrant_db"

# COLLECTION_NAME = "government_schemes"
# MODEL_NAME = "BAAI/bge-m3"

# TOP_K = 5
# CANDIDATE_K = 20


# def tokenize(text):
#     return re.findall(r"\b\w+\b", text.lower())


# def load_chunks():
#     chunks = []

#     with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
#         for line in f:
#             chunks.append(json.loads(line))

#     return chunks


# def main():

#     print("Loading chunks...")
#     chunks = load_chunks()
#     print(f"Loaded {len(chunks)} chunks.")

#     print("Building BM25 index...")

#     corpus = [
#         tokenize(
#             f"{chunk['title']} "
#             f"{chunk['state_or_ministry']} "
#             f"{chunk['section']} "
#             f"{chunk['tags']} "
#             f"{chunk['text']}"
#         )
#         for chunk in chunks
#     ]

#     bm25 = BM25Okapi(corpus)

#     print("Loading BGE-M3...")
#     model = SentenceTransformer(MODEL_NAME)

#     client = QdrantClient(path=str(DB_PATH))

#     query = input("\nEnter your question: ").strip()

#     if not query:
#         print("Please enter a question.")
#         return

#     # -------------------------
#     # BM25 retrieval
#     # -------------------------

#     query_tokens = tokenize(query)
#     bm25_scores = bm25.get_scores(query_tokens)

#     bm25_indices = bm25_scores.argsort()[::-1][:CANDIDATE_K]

#     # -------------------------
#     # Dense retrieval
#     # -------------------------

#     query_vector = model.encode(
#         query,
#         normalize_embeddings=True,
#     ).tolist()

#     dense_results = client.query_points(
#         collection_name=COLLECTION_NAME,
#         query=query_vector,
#         limit=CANDIDATE_K,
#     ).points

#     # -------------------------
#     # Combine candidates
#     # -------------------------

#     candidates = {}

#     # BM25 candidates
#     for index in bm25_indices:
#         chunk = chunks[index]
#         chunk_id = chunk["chunk_id"]

#         candidates[chunk_id] = {
#             "chunk": chunk,
#             "bm25_score": float(bm25_scores[index]),
#             "dense_score": 0.0,
#         }

#     # Dense candidates
#     for result in dense_results:
#         chunk = result.payload
#         chunk_id = chunk["chunk_id"]

#         if chunk_id not in candidates:
#             candidates[chunk_id] = {
#                 "chunk": chunk,
#                 "bm25_score": 0.0,
#                 "dense_score": 0.0,
#             }

#         candidates[chunk_id]["dense_score"] = float(result.score)

#     # -------------------------
#     # Normalize scores
#     # -------------------------

#     max_bm25 = max(
#         item["bm25_score"] for item in candidates.values()
#     )

#     max_dense = max(
#         item["dense_score"] for item in candidates.values()
#     )

#     for item in candidates.values():

#         bm25_normalized = (
#             item["bm25_score"] / max_bm25
#             if max_bm25 > 0
#             else 0
#         )

#         dense_normalized = (
#             item["dense_score"] / max_dense
#             if max_dense > 0
#             else 0
#         )

#         # Equal weighting for now
#         item["hybrid_score"] = (
#             0.5 * bm25_normalized
#             + 0.5 * dense_normalized
#         )

#     # -------------------------
#     # Final ranking
#     # -------------------------

#     ranked = sorted(
#         candidates.values(),
#         key=lambda x: x["hybrid_score"],
#         reverse=True,
#     )[:TOP_K]

#     # -------------------------
#     # Display results
#     # -------------------------

#     print("\n" + "=" * 70)
#     print("TOP HYBRID RETRIEVED CHUNKS")
#     print("=" * 70)

#     for rank, item in enumerate(ranked, 1):

#         chunk = item["chunk"]

#         print(f"\n[{rank}] Hybrid Score: {item['hybrid_score']:.4f}")
#         print(f"BM25 Score      : {item['bm25_score']:.4f}")
#         print(f"Dense Score     : {item['dense_score']:.4f}")

#         print(f"Scheme  : {chunk['title']}")
#         print(f"State   : {chunk['state_or_ministry']}")
#         print(f"Section : {chunk['section']}")
#         print(f"Source  : {chunk['source_file']}")

#         print("\nText:")
#         print(chunk["text"])

#         print("-" * 70)


# if __name__ == "__main__":
#     main()


import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = ROOT / "chunks" / "chunks.jsonl"
DB_PATH = ROOT / "qdrant_db"

COLLECTION_NAME = "government_schemes"
MODEL_NAME = "BAAI/bge-m3"

TOP_K = 5
CANDIDATE_K = 20
RRF_K = 60


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

    print("Building BM25 index...")

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

    bm25 = BM25Okapi(corpus)

    print("Loading BGE-M3...")
    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(path=str(DB_PATH))

    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    # -------------------------
    # BM25 ranking
    # -------------------------

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = bm25_scores.argsort()[::-1][:CANDIDATE_K]

    # -------------------------
    # Dense ranking
    # -------------------------

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=CANDIDATE_K,
    ).points

    # -------------------------
    # Reciprocal Rank Fusion
    # -------------------------

    candidates = {}

    # BM25 ranks
    for rank, index in enumerate(bm25_indices, 1):

        chunk = chunks[index]
        chunk_id = chunk["chunk_id"]

        candidates.setdefault(
            chunk_id,
            {
                "chunk": chunk,
                "bm25_rank": None,
                "dense_rank": None,
                "rrf_score": 0.0,
            },
        )

        candidates[chunk_id]["bm25_rank"] = rank

        candidates[chunk_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    # Dense ranks
    for rank, result in enumerate(dense_results, 1):

        chunk = result.payload
        chunk_id = chunk["chunk_id"]

        candidates.setdefault(
            chunk_id,
            {
                "chunk": chunk,
                "bm25_rank": None,
                "dense_rank": None,
                "rrf_score": 0.0,
            },
        )

        candidates[chunk_id]["dense_rank"] = rank

        candidates[chunk_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    # -------------------------
    # Final ranking
    # -------------------------

    ranked = sorted(
        candidates.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )[:TOP_K]

    # -------------------------
    # Display
    # -------------------------

    print("\n" + "=" * 70)
    print("TOP RRF HYBRID RETRIEVED CHUNKS")
    print("=" * 70)

    for rank, item in enumerate(ranked, 1):

        chunk = item["chunk"]

        print(f"\n[{rank}] RRF Score: {item['rrf_score']:.6f}")
        print(f"BM25 Rank   : {item['bm25_rank']}")
        print(f"Dense Rank  : {item['dense_rank']}")

        print(f"Scheme  : {chunk['title']}")
        print(f"State   : {chunk['state_or_ministry']}")
        print(f"Section : {chunk['section']}")
        print(f"Source  : {chunk['source_file']}")

        print("\nText:")
        print(chunk["text"])

        print("-" * 70)


if __name__ == "__main__":
    main()