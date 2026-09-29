import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"

sys.path.append(str(SCRIPTS_DIR))

from rag import (
    load_chunks,
    build_bm25,
    retrieve,
    build_context,
    generate_answer,
    EMBEDDING_MODEL,
    DB_PATH,
    COLLECTION_NAME,
)

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai


EVALUATION_FILE = ROOT / "evaluation" / "evaluation_questions.json"
OUTPUT_FILE = ROOT / "evaluation" / "answer_results.json"


def load_questions():
    with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Loading evaluation questions...")
    questions = load_questions()

    # Only evaluate questions with a specific expected scheme.
    # questions = [
    #     q for q in questions
    #     if q["expected_scheme"] is not None
    # ]
    questions = questions
    

    print(f"Questions to evaluate: {len(questions)}")

    print("Loading chunks...")
    chunks = load_chunks()

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    print("Loading BGE-M3...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to Qdrant...")
    client = QdrantClient(path=str(DB_PATH))

    print("Connecting to Gemini...")
    gemini_client = genai.Client()

    results = []

    for number, question in enumerate(questions, 1):
    # for number, question in enumerate(questions[17:], start=18):

        print(
            f"\n[{number}/{len(questions)}] "
            f"{question['question']}"
        )

        retrieved = retrieve(
            question["question"],
            model,
            client,
            bm25,
            chunks
        )

        context = build_context(retrieved)

        answer = generate_answer(
            question["question"],
            context,
            gemini_client
        )

        result = {
            "id": question["id"],
            "question": question["question"],
            "expected_scheme": question["expected_scheme"],
            "expected_state": question["expected_state"],
            "type": question["type"],
            "language": question["language"],
            "answer": answer,
            "sources": [
                {
                    "title": chunk["title"],
                    "state_or_ministry": chunk["state_or_ministry"],
                    "section": chunk["section"],
                    "source_file": chunk["source_file"],
                    "text": chunk["text"]
                }
                for chunk in retrieved
            ]
        }

        results.append(result)

        print("Answer generated.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 60)
    print("ANSWER EVALUATION DATA GENERATED")
    print("=" * 60)

    print(f"Questions evaluated: {len(results)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()