from pathlib import Path
import sys
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from pydantic import BaseModel


# Allow importing from scripts/
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"

sys.path.append(str(SCRIPTS_DIR))

from rag import (  # noqa: E402
    load_chunks,
    build_bm25,
    retrieve,
    build_context,
    generate_answer,
    SentenceTransformer,
    QdrantClient,
    genai,
    EMBEDDING_MODEL,
    DB_PATH,
    COLLECTION_NAME,
    LLM_MODEL,
)


app = FastAPI(
    title="SehajAI",
    description="Multilingual Government Scheme RAG API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    state_or_ministry: str
    section: str
    source_file: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


# -------------------------
# Load RAG components once
# -------------------------

print("Loading RAG system...")

chunks = load_chunks()
bm25 = build_bm25(chunks)

model = SentenceTransformer(EMBEDDING_MODEL)

qdrant = QdrantClient(
    path=str(DB_PATH)
)

client = genai.Client()

print("RAG system loaded.")


# -------------------------
# Health check
# -------------------------

@app.get("/")
def root():

    return {
        "message": "SehajAI API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# -------------------------
# Ask endpoint
# -------------------------

@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(request: QuestionRequest):

    query = request.question.strip()

    if not query:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    # Retrieve
    results = retrieve(
        query,
        model,
        qdrant,
        bm25,
        chunks,
    )

    # Build context
    context = build_context(results)

    # Generate answer
    answer = generate_answer(
        query,
        context,
        client,
    )

    # Build source list
    sources = []

    for chunk in results:

        sources.append(
            Source(
                title=chunk["title"],
                state_or_ministry=chunk["state_or_ministry"],
                section=chunk["section"],
                source_file=chunk["source_file"],
            )
        )

    return AnswerResponse(
        answer=answer,
        sources=sources,
    )