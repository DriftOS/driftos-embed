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
import re
import logging
import json
import warnings
from datetime import datetime, timezone
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from preprocessing import preprocess, nlp_full
from nlp_analysis import (
    analyze_message as nlp_analyze_message,
    extract_weighted_entities,
    calculate_entity_overlap,
    should_suppress_anaphoric_floor,
    has_anaphoric_reference,
    is_question as nlp_is_question,
    TOPIC_PIVOT_PATTERNS,
)

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


class EntityOverlapRequest(BaseModel):
    """Check if two texts share significant entities/nouns."""
    text1: str
    text2: str


class EntityOverlapResponse(BaseModel):
    has_overlap: bool
    overlap_score: float  # 0.0 to 1.0
    shared_entities: list[str]
    text1_entities: list[str]
    text2_entities: list[str]


class EntityOverlap(BaseModel):
    has_overlap: bool
    overlap_score: float
    shared_entities: list[str]


class AnalyzeMessageRequest(BaseModel):
    """Analyze context between current and previous message."""
    current: str
    previous: str


class AnalyzeMessageResponse(BaseModel):
    current_is_question: bool
    previous_is_question: bool
    current_has_anaphoric_ref: bool
    has_topic_return_signal: bool
    has_preference: bool = False
    preferred_entity: str | None = None
    rejected_entity: str | None = None
    entity_overlap: EntityOverlap


class AnalyzeDriftRequest(BaseModel):
    """Full drift analysis with similarity calculation and boost application."""
    current: str
    previous: str
    current_embedding: list[float]
    branch_centroid: list[float]


class AnalyzeDriftResponse(BaseModel):
    raw_similarity: float
    boosted_similarity: float
    boost_multiplier: float
    boosts_applied: list[str]
    analysis: AnalyzeMessageResponse


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


@app.post("/entity-overlap", response_model=EntityOverlapResponse)
async def entity_overlap(request: EntityOverlapRequest):
    """
    Check if two texts share significant entities using spaCy NER + noun extraction.
    
    Used to detect when a user reply references something from the previous message,
    e.g., "serpentine belt" -> "maybe the serpentine"
    """
    def extract_entities(text: str) -> set[str]:
        doc = nlp_full(text.lower())
        entities = set()
        
        # Named entities
        for ent in doc.ents:
            entities.add(ent.text.lower())
        
        # Nouns and proper nouns (more than 3 chars to filter noise)
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN') and len(token.text) > 3:
                entities.add(token.lemma_.lower())
                entities.add(token.text.lower())  # Also add raw form
        
        # Noun chunks (compound nouns like "serpentine belt")
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3:
                entities.add(chunk.text.lower())
                # Add ALL words from chunk (not just nouns) - catches adjectives like "serpentine"
                for token in chunk:
                    if len(token.text) > 3 and not token.is_stop:
                        entities.add(token.lemma_.lower())
                        entities.add(token.text.lower())
        
        return entities
    
    entities1 = extract_entities(request.text1)
    entities2 = extract_entities(request.text2)
    
    shared = entities1 & entities2
    
    # Calculate overlap score
    if not entities2:  # Avoid division by zero
        overlap_score = 0.0
    else:
        # Score based on what fraction of the shorter text's entities are shared
        overlap_score = len(shared) / min(len(entities1), len(entities2)) if min(len(entities1), len(entities2)) > 0 else 0.0
    
    return EntityOverlapResponse(
        has_overlap=len(shared) > 0,
        overlap_score=min(overlap_score, 1.0),
        shared_entities=sorted(list(shared)),
        text1_entities=sorted(list(entities1)),
        text2_entities=sorted(list(entities2)),
    )


# Patterns for analyze-message endpoint
# ANAPHORIC_PATTERNS regex kept as fallback
ANAPHORIC_PATTERNS = re.compile(r"\b(that'?s?|this|it'?s?|those|these|the same|them|its)\b", re.IGNORECASE)

# Boost factors (keep in sync with business logic)
QA_BOOST_FACTOR = 1.3
RECENCY_BOOST_FACTOR = 1.6
ANAPHORIC_BOOST_FACTOR = 1.5
TOPIC_RETURN_BOOST_FACTOR = 2.5
ENTITY_OVERLAP_BOOST_FACTOR = 2.0

# Minimum similarity floor when anaphoric reference detected
# If user says "that's cool" or "tell me more about it", they're clearly
# continuing the conversation regardless of raw embedding similarity
ANAPHORIC_SIMILARITY_FLOOR = 0.45

# Short response floor - if previous was question and response is very short,
# it's almost certainly a direct answer ("yes", "no", "okay", "sure")
SHORT_RESPONSE_FLOOR = 0.50
SHORT_RESPONSE_MAX_WORDS = 3

# Response particles - words that only make sense as direct responses
# These indicate continuation regardless of embedding similarity
RESPONSE_PARTICLES = {
    # Affirmative
    'yes', 'yeah', 'yep', 'yup', 'ya', 'aye', 'sure', 'ok', 'okay', 'k',
    'absolutely', 'definitely', 'certainly', 'indeed', 'right', 'correct',
    'agreed', 'exactly', 'true', 'totally', 'yea',
    # Negative  
    'no', 'nope', 'nah', 'never', 'negative',
    # Acknowledgment
    'thanks', 'thank', 'thx', 'ty', 'cheers', 'cool', 'nice', 'great',
    'awesome', 'perfect', 'wonderful', 'excellent', 'good', 'fine',
    # Uncertainty
    'maybe', 'perhaps', 'possibly', 'probably', 'idk', 'dunno',
    # Continuation signals
    'please', 'pls', 'plz', 'go', 'continue', 'more', 'next',
    # Discourse markers
    'well', 'so', 'anyway', 'alright', 'hmm', 'hm', 'oh', 'ah', 'uh',
}

# Floor for response particles
RESPONSE_PARTICLE_FLOOR = 0.55


@app.post("/analyze-message", response_model=AnalyzeMessageResponse)
async def analyze_message(request: AnalyzeMessageRequest):
    """
    Analyze context between current and previous message.
    
    Returns all signals needed for contextual boost calculation.
    Uses advanced spaCy NLP analysis.
    """
    current_doc = nlp_full(request.current)
    previous_doc = nlp_full(request.previous)
    
    # Get full message analysis using new NLP module
    current_analysis = nlp_analyze_message(current_doc, request.current)
    previous_analysis = nlp_analyze_message(previous_doc, request.previous)
    
    # Calculate entity overlap
    overlap_score, shared_entities, _ = calculate_entity_overlap(
        current_analysis.all_entities, previous_analysis.all_entities
    )
    
    return AnalyzeMessageResponse(
        current_is_question=current_analysis.is_question,
        previous_is_question=previous_analysis.is_question,
        current_has_anaphoric_ref=current_analysis.has_anaphoric_ref or bool(ANAPHORIC_PATTERNS.search(request.current)),
        has_topic_return_signal=current_analysis.has_topic_pivot or bool(TOPIC_PIVOT_PATTERNS.search(request.current)),
        has_preference=current_analysis.has_preference,
        preferred_entity=current_analysis.preferred_entity,
        rejected_entity=current_analysis.rejected_entity,
        entity_overlap=EntityOverlap(
            has_overlap=len(shared_entities) > 0,
            overlap_score=min(overlap_score, 1.0),
            shared_entities=sorted(list(shared_entities)),
        ),
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@app.post("/analyze-drift", response_model=AnalyzeDriftResponse)
async def analyze_drift(request: AnalyzeDriftRequest):
    """
    Full drift analysis: NLP analysis + similarity calculation + boost application.
    
    Uses advanced spaCy analysis including:
    - Weighted entity extraction (NER types matter)
    - Preference/comparison detection ("I prefer X to Y")
    - Sentence-level analysis for compound messages
    - Topic pivot detection
    
    Node.js just compares boosted_similarity against thresholds to make routing decisions.
    """
    # Run advanced NLP analysis on both messages
    current_doc = nlp_full(request.current)
    previous_doc = nlp_full(request.previous)
    
    # Get full message analysis
    current_analysis = nlp_analyze_message(current_doc, request.current)
    previous_analysis = nlp_analyze_message(previous_doc, request.previous)
    
    # Extract entities for overlap calculation
    current_entities = current_analysis.all_entities
    previous_entities = previous_analysis.all_entities
    
    # Calculate weighted entity overlap
    overlap_score, shared_entities, new_entity_weight = calculate_entity_overlap(
        current_entities, previous_entities
    )
    has_entity_overlap = len(shared_entities) > 0
    
    # Detect topic pivot/return signals
    has_topic_pivot = current_analysis.has_topic_pivot or bool(TOPIC_PIVOT_PATTERNS.search(request.current))
    
    # Calculate raw similarity
    raw_similarity = cosine_similarity(request.current_embedding, request.branch_centroid)
    
    # If preference detected ("I prefer X to Y"), this is a topic pivot
    # User is explicitly comparing/switching topics
    if current_analysis.has_preference:
        return AnalyzeDriftResponse(
            raw_similarity=raw_similarity,
            boosted_similarity=raw_similarity,  # No boost - let it drift
            boost_multiplier=1.0,
            boosts_applied=['preference_detected'],
            analysis=AnalyzeMessageResponse(
                current_is_question=current_analysis.is_question,
                previous_is_question=previous_analysis.is_question,
                current_has_anaphoric_ref=current_analysis.has_anaphoric_ref,
                has_topic_return_signal=has_topic_pivot,
                has_preference=True,
                preferred_entity=current_analysis.preferred_entity,
                rejected_entity=current_analysis.rejected_entity,
                entity_overlap=EntityOverlap(
                    has_overlap=has_entity_overlap,
                    overlap_score=min(overlap_score, 1.0),
                    shared_entities=sorted(list(shared_entities)),
                ),
            ),
        )
    
    # If topic pivot signal detected, DON'T apply boosts for current branch.
    if has_topic_pivot:
        return AnalyzeDriftResponse(
            raw_similarity=raw_similarity,
            boosted_similarity=raw_similarity,
            boost_multiplier=1.0,
            boosts_applied=[],
            analysis=AnalyzeMessageResponse(
                current_is_question=current_analysis.is_question,
                previous_is_question=previous_analysis.is_question,
                current_has_anaphoric_ref=current_analysis.has_anaphoric_ref,
                has_topic_return_signal=has_topic_pivot,
                entity_overlap=EntityOverlap(
                    has_overlap=has_entity_overlap,
                    overlap_score=min(overlap_score, 1.0),
                    shared_entities=sorted(list(shared_entities)),
                ),
            ),
        )
    
    # Apply boosts
    boosted = raw_similarity
    boosts_applied = []
    
    # Boost 0a: Response particle detection
    # Words like "No", "Yes", "Thanks", "Ok" only make sense as direct responses
    words = [w.lower().strip('.,!?') for w in request.current.split()]
    first_word = words[0] if words else ''
    is_response_particle = first_word in RESPONSE_PARTICLES and len(words) <= 4
    
    if is_response_particle:
        boosted = max(boosted, RESPONSE_PARTICLE_FLOOR)
        boosts_applied.append('response_particle')
    
    # Boost 0b: Ultra-short response (1-2 words) - almost always a direct response
    # "Automatically", "Tomorrow", "Maybe" - single word answers
    elif len(words) <= 2 and not current_analysis.is_question:
        boosted = max(boosted, SHORT_RESPONSE_FLOOR)
        boosts_applied.append('ultra_short_response')
    
    # Boost 1: Q&A pair (previous was question, current is answer)
    # BUT only for short-ish answers, not long new questions
    if previous_analysis.is_question and not current_analysis.is_question and len(words) <= 10:
        boosted *= QA_BOOST_FACTOR
        boosts_applied.append('qa_pair')
    
    # Boost 2: Anaphoric reference with smart floor suppression
    if current_analysis.has_anaphoric_ref:
        # Check if we should suppress the floor (topic pivot, preference, new entities)
        suppress_floor = should_suppress_anaphoric_floor(current_analysis, previous_entities)
        
        if not suppress_floor:
            # Apply full anaphoric floor
            boosted = max(boosted, ANAPHORIC_SIMILARITY_FLOOR)
            boosted *= ANAPHORIC_BOOST_FACTOR
            boosts_applied.append('anaphoric_ref')
        else:
            # Just apply multiplier, no floor
            boosted *= ANAPHORIC_BOOST_FACTOR
            boosts_applied.append('anaphoric_ref_weak')
    
    # Boost 3: Follow-up question
    if current_analysis.is_question:
        boosted *= RECENCY_BOOST_FACTOR
        boosts_applied.append('question')
    
    # Boost 4: Entity overlap (weighted)
    if has_entity_overlap:
        # Scale boost by overlap weight
        overlap_boost = 1.0 + (ENTITY_OVERLAP_BOOST_FACTOR - 1.0) * min(overlap_score, 1.0)
        boosted *= overlap_boost
        boosts_applied.append('entity_overlap')
    
    # Cap at 1.0
    boosted = min(boosted, 1.0)
    
    # Calculate total multiplier
    boost_multiplier = boosted / raw_similarity if raw_similarity > 0 else 1.0
    
    return AnalyzeDriftResponse(
        raw_similarity=raw_similarity,
        boosted_similarity=boosted,
        boost_multiplier=boost_multiplier,
        boosts_applied=boosts_applied,
        analysis=AnalyzeMessageResponse(
            current_is_question=current_analysis.is_question,
            previous_is_question=previous_analysis.is_question,
            current_has_anaphoric_ref=current_analysis.has_anaphoric_ref,
            has_topic_return_signal=has_topic_pivot,
            has_preference=current_analysis.has_preference,
            preferred_entity=current_analysis.preferred_entity,
            rejected_entity=current_analysis.rejected_entity,
            entity_overlap=EntityOverlap(
                has_overlap=has_entity_overlap,
                overlap_score=min(overlap_score, 1.0),
                shared_entities=sorted(list(shared_entities)),
            ),
        ),
    )
