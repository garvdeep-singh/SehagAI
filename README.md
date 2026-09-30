# SehajAI

SehajAI is a multilingual retrieval-augmented generation (RAG) application for answering questions about Indian government schemes. It combines lexical search with semantic search, then asks Gemini to produce a concise answer grounded in retrieved source chunks.

The project contains:

- A Python/FastAPI backend
- A Next.js frontend
- A preprocessing pipeline for scheme PDFs
- Hybrid BM25 + Qdrant vector retrieval
- Gemini-powered, source-cited answers
- Retrieval and answer evaluation scripts

## Architecture

```text
PDF files
  -> text extraction
  -> text cleaning and deduplication
  -> section-aware chunking
  -> BGE-M3 embeddings
  -> Qdrant collection

User question
  -> BM25 retrieval + Qdrant dense retrieval
  -> reciprocal-rank fusion
  -> top five context chunks
  -> Gemini answer with [Source X] citations
```

The API loads the chunk file, BM25 index, embedding model, Qdrant client, and Gemini client once at startup. The first startup can be slow because `BAAI/bge-m3` may need to be downloaded and loaded.

## Requirements

- Python 3.10 or newer
- Node.js and npm
- A Gemini API key
- A Qdrant Cloud cluster and API key
- Enough disk space and memory for the BGE-M3 model

The checked-in `chunks/chunks.jsonl` file is sufficient to run the API. The original PDF corpus is expected under `text_data/` when rebuilding the dataset; that directory is ignored by Git.

## Configuration

Set these environment variables in the terminal where the backend will run:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export QDRANT_URL="https://your-cluster-url"
export QDRANT_API_KEY="your-qdrant-api-key"
```

Do not commit API keys. A local `.env` file is ignored by the repository, but the current Python code reads variables from the process environment rather than loading `.env` automatically.

Important defaults in `scripts/rag.py`:

- Embedding model: `BAAI/bge-m3`
- Gemini model: `gemini-flash-lite-latest`
- Qdrant collection: `government_schemes`
- Dense/BM25 candidate count: `20`
- Final context size: `5`
- RRF constant: `60`

## Backend Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the FastAPI backend:

```bash
source .venv/bin/activate
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### API Endpoints

`GET /`

Returns a basic running message.

`GET /health`

Returns:

```json
{"status": "healthy"}
```

`POST /ask`

Request:

```json
{"question": "Which schemes support women entrepreneurs?"}
```

Example command:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Which schemes support women entrepreneurs?"}'
```

Response shape:

```json
{
  "answer": "... [Source 1] ...",
  "sources": [
    {
      "title": "Scheme name",
      "state_or_ministry": "State or ministry",
      "section": "Eligibility",
      "source_file": "source.txt"
    }
  ]
}
```

An empty question returns a friendly message and an empty source list. The API currently allows browser requests from `http://localhost:3000` through CORS.

## Frontend Setup

In a second terminal:

```bash
cd sehajai-frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

The frontend currently sends requests to `http://127.0.0.1:8000/ask`, so start the backend on that host and port before asking a question.

Useful frontend commands:

```bash
npm run lint
npm run build
npm run start
```

Run `npm run start` after `npm run build` to serve the production build.

## Rebuild the Dataset

The following pipeline starts with PDFs in `text_data/` and produces the files consumed by retrieval.

### 1. Extract PDF text

```bash
python scripts/extract_text.py
```

Reads `text_data/*.pdf` and writes raw text to `raw_text/`.

### 2. Clean and deduplicate text

```bash
python scripts/clean_text.py
```

Writes cleaned scheme files to `cleaned_text/` and a processing report to `cleaning_manifest.csv`.

### 3. Create chunks

```bash
python scripts/chunk_text.py
```

Writes section-aware JSONL records to `chunks/chunks.jsonl`. Chunks are approximately 3,000 characters with 400 characters of overlap.

### 4. Generate embeddings

```bash
python scripts/embed_chunks.py
```

Writes BGE-M3 vectors to `chunks/embeddings.jsonl`. This downloads and runs the embedding model locally.

### 5. Upload vectors to Qdrant Cloud

```bash
python scripts/migrate_to_qdrant_cloud.py
```

This creates the `government_schemes` collection if necessary and uploads the embeddings using the configured Qdrant environment variables. The collection must use 1,024-dimensional cosine-distance vectors, matching BGE-M3 output and the API configuration.

## Local Qdrant Utilities

`scripts/build_vector_db.py` creates a local Qdrant database in `qdrant_db/`. It is useful for development and inspection, but the current `api.py` and `scripts/rag.py` connect to Qdrant Cloud and require `QDRANT_URL` and `QDRANT_API_KEY`.

To use the local database with the API, the client construction in both runtime paths must be changed from the cloud URL/API-key form to `QdrantClient(path=...)` consistently.

## Evaluation and Tests

Retrieval evaluation questions are stored under `evaluation/`. Available scripts include:

```bash
python scripts/evaluate_retrieval.py
python scripts/evaluate_answers.py
python scripts/test_retrieval.py
python scripts/test_extraction.py
python scripts/validate_cleaned.py
```

Some evaluation scripts still reference the local `qdrant_db/` path, while the production API uses Qdrant Cloud. Confirm the client configuration before running them.

Generated evaluation output is written under `evaluation/` and is ignored where configured by `.gitignore`.

## Repository Layout

```text
.
├── api.py                         FastAPI application
├── requirements.txt               Python dependencies
├── scripts/
│   ├── rag.py                     Retrieval and Gemini generation
│   ├── extract_text.py            PDF to raw text
│   ├── clean_text.py              Cleaning and deduplication
│   ├── chunk_text.py              Chunk generation
│   ├── embed_chunks.py            BGE-M3 embedding generation
│   ├── migrate_to_qdrant_cloud.py Cloud vector upload
│   └── build_vector_db.py         Local Qdrant database builder
├── chunks/
│   ├── chunks.jsonl               Runtime retrieval corpus
│   └── embeddings.jsonl           Generated embedding records
├── cleaned_text/                  Cleaned scheme text
├── raw_text/                      Extracted text files
├── text_data/                     Input PDFs, when rebuilding data
├── evaluation/                    Evaluation questions and outputs
└── sehajai-frontend/              Next.js web application
```

## Troubleshooting

### Backend fails during startup

Check that all three environment variables are set and that the Qdrant collection is named `government_schemes`. Also verify that `chunks/chunks.jsonl` exists and contains valid JSONL records.

### Qdrant returns no results or a collection error

The cloud collection must exist, contain the same chunk payload fields used by the API, and contain 1,024-dimensional vectors. Re-run the embedding and migration steps when the chunk corpus changes.

### Frontend cannot connect

Confirm the backend is running at `http://127.0.0.1:8000` and that the browser origin is `http://localhost:3000`. The frontend URL is currently hard-coded in `sehajai-frontend/src/app/page.tsx`.

### Answers lack information

The model can only use the five chunks returned by hybrid retrieval. Check the chunk quality, BM25 terms, Qdrant collection contents, and source metadata before changing the generation prompt.

## Security Notes

- Keep Gemini and Qdrant credentials outside source control.
- Treat retrieved document text as untrusted data; the generation prompt explicitly prevents retrieved text from acting as instructions.
- Answers are grounded in the indexed corpus and should not be treated as legal or official application advice. Verify important scheme details against the relevant official source.
