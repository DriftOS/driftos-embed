# driftos-embed

Lightweight semantic conversation routing engine. Embedding-based drift detection with sub-200ms latency, zero LLM API costs for routing decisions.

## The Problem

AI applications dump entire conversation history into every LLM call:
- Unfocused context → worse responses
- Token waste → higher costs
- No structure → can't query "what did we decide about X?"

## The Solution

driftos-embed uses local embeddings to detect topic shifts and route messages:
- **STAY** - Same topic, continue in current branch
- **BRANCH** - Topic drift detected, create new branch
- **ROUTE** - Return to a previous topic

**Result:** Focused context windows. 20 relevant messages instead of 1000.

## Why Embeddings?

| Approach | Latency | Cost | Accuracy |
|----------|---------|------|----------|
| LLM-based routing | 500-2000ms | $0.001-0.01/call | High |
| **Embedding-based** | **<200ms** | **$0** | Good |

driftos-embed uses `paraphrase-MiniLM-L6-v2` for semantic similarity. Fast enough for real-time, accurate enough for production.

## Quick Start

```bash
# Clone and install
git clone https://github.com/DriftOS/driftos-embed
cd driftos-embed
npm install

# Setup database
cp .env.example .env
npm run db:push

# Start embedding server (Python sidecar)
cd embedding-server && pip install -r requirements.txt
python server.py &

# Run
npm run dev
```

## API

### Route a Message
```bash
POST /api/v1/drift/route
{
  "conversationId": "conv-123",
  "content": "I want to plan a trip to Japan",
  "role": "user"
}
```

Response:
```json
{
  "action": "BRANCH",
  "driftAction": "BRANCH_NEW_CLUSTER",
  "branchId": "branch-456",
  "branchTopic": "I want to plan a trip to Japan",
  "confidence": 1.0,
  "similarity": 0,
  "isNewBranch": true,
  "isNewCluster": true
}
```

### Subsequent Messages
```bash
POST /api/v1/drift/route
{
  "conversationId": "conv-123",
  "content": "What's the best time for cherry blossoms?",
  "role": "user"
}
```

Response:
```json
{
  "action": "STAY",
  "driftAction": "STAY",
  "branchId": "branch-456",
  "similarity": 0.41,
  "isNewBranch": false
}
```

### Topic Shift Detection
```bash
POST /api/v1/drift/route
{
  "conversationId": "conv-123",
  "content": "I need to sort out my tax return",
  "role": "user"
}
```

Response:
```json
{
  "action": "BRANCH",
  "driftAction": "BRANCH_NEW_CLUSTER",
  "branchId": "branch-789",
  "similarity": 0.05,
  "isNewBranch": true,
  "isNewCluster": true
}
```

### Route Back to Previous Topic
```bash
POST /api/v1/drift/route
{
  "conversationId": "conv-123",
  "content": "Back to Japan - should I get a JR rail pass?",
  "role": "user"
}
```

Response:
```json
{
  "action": "ROUTE",
  "driftAction": "STAY",
  "branchId": "branch-456",
  "similarity": 0.49,
  "isNewBranch": false
}
```

### Get Context for LLM
```bash
GET /api/v1/context/{branchId}
```

Response:
```json
{
  "branchId": "branch-456",
  "branchTopic": "I want to plan a trip to Japan",
  "messages": [
    { "role": "user", "content": "I want to plan a trip to Japan" },
    { "role": "user", "content": "What's the best time for cherry blossoms?" },
    { "role": "user", "content": "Back to Japan - should I get a JR rail pass?" }
  ],
  "allFacts": [
    {
      "branchTopic": "I want to plan a trip to Japan",
      "isCurrent": true,
      "facts": [
        { "key": "destination", "value": "Japan", "confidence": 1.0 }
      ]
    }
  ]
}
```

### List Branches
```bash
GET /api/v1/drift/branches/{conversationId}
```

### Extract Facts
```bash
POST /api/v1/facts/{branchId}/extract
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/drift/route` | Route a message to a branch |
| GET | `/api/v1/drift/branches/:conversationId` | List all branches |
| GET | `/api/v1/context/:branchId` | Get optimized LLM context |
| POST | `/api/v1/facts/:branchId/extract` | Extract facts from branch |
| GET | `/api/v1/facts/:branchId` | Get existing facts |

## Configuration

```env
# Required
DATABASE_URL=postgresql://...

# Drift Thresholds (tuned defaults)
DRIFT_STAY_THRESHOLD=0.38        # Above = same topic
DRIFT_NEW_CLUSTER_THRESHOLD=0.15 # Below = new domain
DRIFT_ROUTE_THRESHOLD=0.42       # Above = route to existing

# Embedding Server
EMBEDDING_SERVER_URL=http://localhost:8100
EMBEDDING_MODEL=paraphrase-MiniLM-L6-v2

# Optional: LLM for fact extraction
GROQ_API_KEY=your-key
LLM_MODEL=llama-3.1-8b-instant
```

## How It Works

1. **Embed** - Message is embedded using paraphrase-MiniLM-L6-v2
2. **Compare** - Cosine similarity against current branch centroid
3. **Decide** - Based on thresholds: STAY, BRANCH, or ROUTE
4. **Update** - Branch centroid updated with running average

### Threshold Logic

```
similarity > 0.38  → STAY (same topic)
similarity > 0.42  → ROUTE (if matches another branch)
similarity < 0.15  → BRANCH_NEW_CLUSTER (different domain)
else               → BRANCH_SAME_CLUSTER (related subtopic)
```

## Tuning Guide

Embedding-based routing requires threshold tuning for your use case. The defaults work well for general conversation, but you may need to adjust.

### Choosing an Embedding Model

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `paraphrase-MiniLM-L6-v2` | 22M | ~30ms | **Recommended.** Trained for semantic similarity |
| `all-MiniLM-L6-v2` | 22M | ~30ms | General purpose, slightly less accurate for paraphrase detection |
| `all-mpnet-base-v2` | 110M | ~100ms | Higher accuracy, slower |

**Key insight:** Paraphrase-trained models outperform general-purpose embeddings for drift detection because they're optimized to recognize when two sentences mean the same thing.

### Threshold Tuning

Thresholds control sensitivity. Lower = more branches, higher = fewer branches.

```env
# Conservative (fewer branches, may miss subtle shifts)
DRIFT_STAY_THRESHOLD=0.30
DRIFT_ROUTE_THRESHOLD=0.35
DRIFT_NEW_CLUSTER_THRESHOLD=0.10

# Default (balanced)
DRIFT_STAY_THRESHOLD=0.38
DRIFT_ROUTE_THRESHOLD=0.42
DRIFT_NEW_CLUSTER_THRESHOLD=0.15

# Aggressive (more branches, catches subtle shifts)
DRIFT_STAY_THRESHOLD=0.45
DRIFT_ROUTE_THRESHOLD=0.50
DRIFT_NEW_CLUSTER_THRESHOLD=0.20
```

### Q&A Pair Handling

Questions and answers naturally have lower similarity (different sentence structures). The system applies a 1.3x boost when:
- Previous message contains `?`
- Current message does not contain `?`

This keeps Q&A pairs together in the same branch.

### Debugging Similarity Scores

The response includes `similarity` scores. Use these to tune:

```json
{
  "action": "STAY",
  "similarity": 0.41,  // Just above 0.38 threshold
  "reason": "similar_to_current (0.410 > 0.38)"
}
```

If you're seeing unexpected BRANCHes, check the similarity score and adjust thresholds accordingly.

### Centroid Drift

Branch centroids update with a running average as messages are added. This means:
- Early messages have more influence on the centroid
- Long branches become more "settled" in their topic
- Very long branches may resist ROUTE back from other topics

For high-volume branches, consider periodic centroid recalculation.

## SDK & MCP

Use with the official SDK:

```bash
npm install @driftos/client
```

```typescript
import { createDriftClient } from '@driftos/client';

const client = createDriftClient('http://localhost:3000');

const result = await client.route('conv-123', 'Plan my Japan trip');
const context = await client.getContext(result.branchId);
const prompt = await client.buildPrompt(result.branchId, 'You are a travel assistant');
```

Or use via MCP with Claude Desktop: [driftos-mcp-server](https://github.com/DriftOS/driftos-mcp-server)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     driftos-embed                        │
├─────────────────────────────────────────────────────────┤
│  Routes Layer                                            │
│  └── /drift, /context, /facts, /branches                │
├─────────────────────────────────────────────────────────┤
│  Services Layer                                          │
│  ├── DriftService (routing orchestration)               │
│  ├── ContextService (LLM context assembly)              │
│  └── FactsService (LLM-based extraction)                │
├─────────────────────────────────────────────────────────┤
│  Operations Layer                                        │
│  ├── embedMessage (local embeddings)                    │
│  ├── classifyRouteEmbed (similarity + thresholds)       │
│  └── executeRoute (branch/message creation)             │
├─────────────────────────────────────────────────────────┤
│  Infrastructure                                          │
│  ├── PostgreSQL + Prisma                                │
│  ├── Embedding Server (Python/FastAPI)                  │
│  └── Fastify + TypeScript                               │
└─────────────────────────────────────────────────────────┘
```

## Performance

- **Routing latency:** <200ms
- **Embedding generation:** ~30ms
- **Zero LLM costs** for routing decisions
- **LLM used only** for fact extraction (optional)

## Related Projects

- [driftos-core](https://github.com/DriftOS/driftos-core) - LLM-based routing (higher accuracy, higher latency)
- [drift-sdk](https://github.com/DriftOS/drift-sdk) - TypeScript/JavaScript SDK
- [driftos-mcp-server](https://github.com/DriftOS/driftos-mcp-server) - MCP server for Claude Desktop

## License

MIT

---

**Patent Pending** | [driftos.dev](https://driftos.dev)
