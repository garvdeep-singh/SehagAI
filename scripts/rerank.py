from FlagEmbedding import FlagReranker


MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def main():

    print("Loading reranker...")

    reranker = FlagReranker(
        MODEL_NAME,
        use_fp16=False,
    )

    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    candidates = [
        "The 25% Capital Investment Subsidy Scheme provides financial assistance up to ₹6.25 lakhs.",
        "The scheme is available to eligible entrepreneurs in Lakshadweep.",
        "Applications must be submitted to the concerned authorities with supporting documents.",
    ]

    pairs = [[query, candidate] for candidate in candidates]

    scores = reranker.compute_score(
        pairs,
        normalize=True,
    )

    ranked = sorted(
        zip(scores, candidates),
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    for rank, (score, text) in enumerate(ranked, 1):

        print(f"\n[{rank}] Score: {score:.4f}")
        print(text)


if __name__ == "__main__":
    main()