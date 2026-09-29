import json
import re
from pathlib import Path

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = ROOT / "chunks" / "chunks.jsonl"
EVALUATION_FILE = ROOT / "evaluation" / "evaluation_questions.json"
DB_PATH = ROOT / "qdrant_db"

COLLECTION_NAME = "government_schemes"
EMBEDDING_MODEL = "BAAI/bge-m3"

CANDIDATE_K = 20
TOP_K = 5
RRF_K = 60


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def load_questions():
    with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25(chunks):
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

    return BM25Okapi(corpus)


def retrieve(query, model, client, bm25, chunks):
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=CANDIDATE_K
    ).points

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = bm25_scores.argsort()[::-1][:CANDIDATE_K]

    candidates = {}

    # BM25 results
    for rank, index in enumerate(bm25_indices, 1):
        chunk = chunks[index]
        chunk_id = chunk["chunk_id"]

        candidates.setdefault(
            chunk_id,
            {
                "chunk": chunk,
                "rrf_score": 0.0
            }
        )

        candidates[chunk_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    # Dense results
    for rank, result in enumerate(dense_results, 1):
        chunk = result.payload
        chunk_id = chunk["chunk_id"]

        candidates.setdefault(
            chunk_id,
            {
                "chunk": chunk,
                "rrf_score": 0.0
            }
        )

        candidates[chunk_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    ranked = sorted(
        candidates.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return [
        item["chunk"]
        for item in ranked[:TOP_K]
    ]


def evaluate_question(question, results):
    expected_scheme = question["expected_scheme"]
    expected_state = question["expected_state"]

    # Out-of-domain / ambiguous questions don't have
    # a single expected scheme.
    if expected_scheme is None:
        return None

    for rank, chunk in enumerate(results, 1):
        scheme_match = (
            chunk["title"].strip().lower()
            == expected_scheme.strip().lower()
        )

        state_match = True

        if expected_state:
            state_match = (
                chunk["state_or_ministry"]
                .strip()
                .lower()
                == expected_state.strip().lower()
            )

        if scheme_match and state_match:
            return rank

    return None


def main():
    print("Loading chunks...")
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    print("Loading BGE-M3...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to Qdrant...")
    client = QdrantClient(path=str(DB_PATH))

    print("Loading evaluation questions...")
    questions = load_questions()

    print(f"Evaluating {len(questions)} questions...\n")

    results = []

    for question in questions:
        retrieved = retrieve(
            question["question"],
            model,
            client,
            bm25,
            chunks
        )

        rank = evaluate_question(
            question,
            retrieved
        )

        results.append({
            "id": question["id"],
            "question": question["question"],
            "type": question["type"],
            "expected_scheme": question["expected_scheme"],
            "expected_state": question["expected_state"],
            "correct_rank": rank,
            "retrieved": [
                {
                    "title": chunk["title"],
                    "state": chunk["state_or_ministry"],
                    "section": chunk["section"]
                }
                for chunk in retrieved
            ]
        })

        if rank:
            print(
                f"[{question['id']:02d}] "
                f"FOUND at rank {rank} | "
                f"{question['question']}"
            )
        else:
            print(
                f"[{question['id']:02d}] "
                f"NOT FOUND | "
                f"{question['question']}"
            )

    # Only questions with a specific expected scheme
    # are included in recall calculations.
    answerable = [
        result
        for result in results
        if result["expected_scheme"] is not None
    ]

    found = [
        result
        for result in answerable
        if result["correct_rank"] is not None
    ]

    recall_at_5 = (
        len(found) / len(answerable)
        if answerable
        else 0
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Total questions: {len(questions)}")
    print(f"Questions with expected scheme: {len(answerable)}")
    print(f"Scheme found in Top-{TOP_K}: {len(found)}")
    print(f"Recall@{TOP_K}: {recall_at_5:.2%}")

    output_file = ROOT / "evaluation" / "retrieval_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nDetailed results saved to:")
    print(output_file)


if __name__ == "__main__":
    main()