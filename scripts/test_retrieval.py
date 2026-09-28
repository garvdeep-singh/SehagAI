from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

DB_PATH = ROOT / "qdrant_db"

COLLECTION_NAME = "government_schemes"

MODEL_NAME = "BAAI/bge-m3"

TOP_K = 5


def main():

    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(path=str(DB_PATH))

    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
    ).points

    print("\n" + "=" * 70)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 70)

    for i, result in enumerate(results, 1):

        payload = result.payload

        print(f"\n[{i}] Score: {result.score:.4f}")
        print(f"Scheme  : {payload['title']}")
        print(f"State   : {payload['state_or_ministry']}")
        print(f"Section : {payload['section']}")
        print(f"Source  : {payload['source_file']}")

        print("\nText:")
        print(payload["text"])

        print("-" * 70)


if __name__ == "__main__":
    main()