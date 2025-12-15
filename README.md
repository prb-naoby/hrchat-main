# HRChat Retrieval API

A FastAPI-based hybrid search service combining **BM25 sparse vectors** and **Gemini dense embeddings** with **Qdrant** vector database.

## Features

- **Hybrid Search**: Combines BM25 sparse vectors + Gemini dense embeddings using RRF fusion
- **BM25 Endpoints**: Standalone sparse embedding generation for ingestion pipelines
- **API Authentication**: Optional API key protection for all endpoints
- **Docker Ready**: Production-ready Dockerfile with GHCR integration

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Main health check |
| `GET` | `/retrieve/health` | Retrieval config status |
| `GET` | `/bm25/health` | BM25 model status |
| `POST` | `/retrieve/hybrid` | Hybrid search query |
| `POST` | `/sparse/bm25` | Generate BM25 sparse vector |
| `POST` | `/sparse/bm25/batch` | Batch BM25 embedding |

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 2. Run with Docker Compose

```bash
docker-compose up -d
```

The API will be available at `http://localhost:8001`

### 3. Test Health Endpoints

```bash
curl http://localhost:8001/healthz
curl http://localhost:8001/docs  # Swagger UI
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `QDRANT_URL` | ✅ | Qdrant server URL |
| `QDRANT_API_KEY` | ✅ | Qdrant API key |
| `QDRANT_COLLECTION` | ✅ | Target collection name |
| `GOOGLE_API_KEY` | ✅ | Google Gemini API key |
| `DENSE_EMBEDDING_MODEL` | ❌ | Gemini model (default: `gemini-embedding-001`) |
| `BM25_MODEL` | ❌ | BM25 model (default: `Qdrant/bm25`) |
| `RETRIEVAL_API_KEY` | ❌ | Protect `/retrieve/*` endpoints |
| `BM25_API_KEY` | ❌ | Protect `/sparse/*` endpoints |

## API Usage

### Hybrid Search

```bash
curl -X POST http://localhost:8001/retrieve/hybrid \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{"query": "What is the leave policy?", "limit": 5}'
```

### Generate BM25 Sparse Vector

```bash
curl -X POST http://localhost:8001/sparse/bm25 \
  -H "Content-Type: application/json" \
  -d '{"text": "Sample document text"}'
```

## Docker Networks

The service connects to external networks:
- `n8n` - For n8n workflow integration
- `qdrant` - For Qdrant vector database

Ensure these networks exist before running:

```bash
docker network create n8n
docker network create qdrant
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python -m uvicorn src.api:app --reload --port 8000
```

## License

MIT
