"""
DriftOS Embedding Server
paraphrase-MiniLM-L6-v2 via FastAPI

Best-in-class for drift detection:
- 0.556 gap (related vs unrelated)
- Negative unrelated similarity (-0.08)
- 22M params, ~5ms inference

Run: uvicorn server:app --host 0.0.0.0 --port 8100
"""

import os
import sys
import logging
import json
import warnings
from datetime import datetime, timezone
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from preprocessing import preprocess

# Suppress multiprocessing semaphore leak warning (uvicorn reload artifact)
warnings.filterwarnings("ignore", message="resource_tracker:.*semaphore")


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Concise console formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        time_str = datetime.now().strftime("%H:%M:%S")

        # Compact single-line format
        msg = f"{color}{time_str}{reset} {record.getMessage()}"

        # Add extra fields inline if present
        if hasattr(record, "extra") and record.extra:
            extras = " ".join(f"{k}={v}" for k, v in record.extra.items())
            msg += f" ({extras})"

        return msg


def setup_logging() -> logging.Logger:
    """Configure logging based on environment."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    is_prod = os.getenv("NODE_ENV") == "production"

    logger = logging.getLogger("driftos-embed")
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter() if is_prod else ConsoleFormatter())
    logger.addHandler(handler)

    # Reduce noise from libraries but keep startup/shutdown visible
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Silence per-request logs
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logger


logger = setup_logging()


# Default to paraphrase-MiniLM-L6-v2 - proven best for drift detection
DEFAULT_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Global model reference
model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Loading model on {device}", extra={"model": model_name, "device": device})

    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device=device,
    )

    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model ready (dim={dim})", extra={"dimension": dim})

    yield

    model = None


app = FastAPI(
    title="DriftOS Embedding Server",
    version="0.2.0",
    description="Paraphrase-MiniLM-L6-v2 optimized for drift detection",
    lifespan=lifespan,
)


class EmbedRequest(BaseModel):
    text: str | list[str]
    preprocess: bool = True  # Default ON - preprocessing improves drift detection


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    model: str
    preprocessed_texts: list[str] | None = None


class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    preprocess: bool = True  # Default ON


class SimilarityResponse(BaseModel):
    similarity: float
    adjusted_similarity: float | None = None
    preprocessed_text1: str | None = None
    preprocessed_text2: str | None = None


class DriftRequest(BaseModel):
    """Check if a message has drifted from anchor text."""
    anchor: str
    message: str
    preprocess: bool = True
    stay_threshold: float = 0.38  # Above = STAY
    branch_threshold: float = 0.15  # Below = new cluster


class DriftResponse(BaseModel):
    similarity: float
    action: str  # STAY, BRANCH_SAME_CLUSTER, BRANCH_NEW_CLUSTER
    preprocessed_anchor: str | None = None
    preprocessed_message: str | None = None


class PreprocessRequest(BaseModel):
    text: str | list[str]


class PreprocessResponse(BaseModel):
    original: list[str]
    preprocessed: list[str]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings for text(s)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    texts = [request.text] if isinstance(request.text, str) else request.text
    
    # Optionally preprocess
    preprocessed_texts = None
    if request.preprocess:
        texts = [preprocess(t) for t in texts]
        preprocessed_texts = texts
    
    embeddings = model.encode(texts)
    
    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
        model=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
        preprocessed_texts=preprocessed_texts,
    )


@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(request: SimilarityRequest):
    """Compute cosine similarity between two texts."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    text1, text2 = request.text1, request.text2
    preprocessed_text1, preprocessed_text2 = None, None
    
    # Optionally preprocess
    if request.preprocess:
        text1 = preprocess(text1)
        text2 = preprocess(text2)
        preprocessed_text1 = text1
        preprocessed_text2 = text2
    
    embeddings = model.encode([text1, text2])
    
    # Cosine similarity
    sim = float(torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings[0]).unsqueeze(0),
        torch.tensor(embeddings[1]).unsqueeze(0),
    ))

    has_question1 = '?' in request.text1
    has_question2 = '?' in request.text2

    if has_question1 and not has_question2:
        adjusted_similarity = sim * 1.3
    else:
        adjusted_similarity = sim
    
    
    return SimilarityResponse(
        similarity=sim,
        adjusted_similarity=adjusted_similarity,
        preprocessed_text1=preprocessed_text1,
        preprocessed_text2=preprocessed_text2,
    )


@app.post("/drift", response_model=DriftResponse)
async def check_drift(request: DriftRequest):
    """
    Check if a message has drifted from anchor context.
    
    Thresholds (from gradient benchmark):
    - > 0.47: STAY (same branch)
    - 0.20 - 0.47: BRANCH, same cluster
    - < 0.20: BRANCH, new cluster
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    anchor, message = request.anchor, request.message
    preprocessed_anchor, preprocessed_message = None, None
    
    if request.preprocess:
        anchor = preprocess(anchor)
        message = preprocess(message)
        preprocessed_anchor = anchor
        preprocessed_message = message
    
    embeddings = model.encode([anchor, message])
    
    sim = float(torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings[0]).unsqueeze(0),
        torch.tensor(embeddings[1]).unsqueeze(0),
    ))
    
    # Determine action based on thresholds
    if sim > request.stay_threshold:
        action = "STAY"
    elif sim > request.branch_threshold:
        action = "BRANCH_SAME_CLUSTER"
    else:
        action = "BRANCH_NEW_CLUSTER"
    
    return DriftResponse(
        similarity=sim,
        action=action,
        preprocessed_anchor=preprocessed_anchor,
        preprocessed_message=preprocessed_message,
    )


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_text(request: PreprocessRequest):
    """Preprocess text(s) without embedding."""
    texts = [request.text] if isinstance(request.text, str) else request.text
    preprocessed = [preprocess(t) for t in texts]
    
    return PreprocessResponse(
        original=texts,
        preprocessed=preprocessed,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return HealthResponse(
        status="healthy",
        model=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
        device="mps" if torch.backends.mps.is_available() else "cpu",
        dimension=model.get_sentence_embedding_dimension(),
    )
