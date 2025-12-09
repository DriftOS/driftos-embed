# DriftOS Embedding Server

Local embedding server using sentence-transformers. Default model: `paraphrase-MiniLM-L6-v2`.

## Why This Model?

Paraphrase-trained models outperform general-purpose embeddings for drift detection because they're optimized to recognize semantic similarity - exactly what you need for conversation routing.

| Model | Size | Latency | Best For |
|-------|------|---------|----------|
| `paraphrase-MiniLM-L6-v2` | 22M | ~10ms | **Recommended.** Fast, accurate for similarity |
| `all-MiniLM-L6-v2` | 22M | ~10ms | General purpose |
| `all-mpnet-base-v2` | 110M | ~50ms | Higher accuracy, slower |

## Setup

```bash
cd embedding-server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Default: paraphrase-MiniLM-L6-v2
uvicorn server:app --host 0.0.0.0 --port 8100

# Or use alternative model
EMBEDDING_MODEL=all-mpnet-base-v2 uvicorn server:app --host 0.0.0.0 --port 8100
```

First run downloads the model (~90MB for MiniLM, ~420MB for mpnet).

## Endpoints

### POST /embed
```json
{
  "text": "What hotels are near the Eiffel Tower?"
}
```

Response:
```json
{
  "embeddings": [[0.123, -0.456, ...]],
  "dimension": 384,
  "model": "paraphrase-MiniLM-L6-v2"
}
```

### POST /similarity
```json
{
  "text1": "Paris trip planning",
  "text2": "Hotels near Eiffel Tower"
}
```

Response:
```json
{
  "similarity": 0.78
}
```

### GET /health
```json
{
  "status": "healthy",
  "model": "paraphrase-MiniLM-L6-v2",
  "device": "cpu",
  "dimension": 384
}
```

## Performance

Tested on M1 Max Macbook Pro:

| Load | Avg Latency | P95 | Throughput |
|------|-------------|-----|------------|
| Steady | 10ms | 12ms | 600 ops/min |
| Sustained | 37ms | 50ms | 1,800 ops/min |

The server queues requests under heavy load but maintains 100% success rate.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `paraphrase-MiniLM-L6-v2` | sentence-transformers model |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8100` | Server port |

## Integration

The main driftos-embed server connects to this via `EMBEDDING_SERVER_URL`:

```env
EMBEDDING_SERVER_URL=http://localhost:8100
EMBEDDING_MODEL=paraphrase-MiniLM-L6-v2
```
