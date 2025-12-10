"""
NLP Analysis Module for DriftOS

Advanced spaCy-based analysis for conversation drift detection:
- Weighted entity extraction (NER types matter)
- Preference/comparison detection ("I prefer X to Y")
- Improved question detection (implicit questions)
- Sentence-level analysis for compound messages
- Topic pivot detection
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from spacy.tokens import Doc, Token


# Entity type weights - higher = more significant for topic detection
ENTITY_WEIGHTS = {
    'PERSON': 3.0,      # People are strong topic indicators
    'ORG': 2.5,         # Organizations
    'GPE': 2.5,         # Countries, cities, states
    'LOC': 2.0,         # Non-GPE locations
    'PRODUCT': 2.0,     # Products, objects
    'EVENT': 2.0,       # Events
    'WORK_OF_ART': 1.5, # Titles of books, songs, etc.
    'NORP': 1.5,        # Nationalities, religious/political groups
    'FAC': 1.5,         # Facilities
    'DATE': 0.5,        # Dates (less topical)
    'TIME': 0.5,        # Times
    'MONEY': 0.5,       # Monetary values
    'QUANTITY': 0.3,    # Quantities
    'CARDINAL': 0.2,    # Numbers
    'ORDINAL': 0.2,     # Ordinal numbers
}

# Default weight for nouns not caught by NER
DEFAULT_NOUN_WEIGHT = 1.0
DEFAULT_PROPN_WEIGHT = 2.0

# Preference/comparison patterns
PREFERENCE_PATTERNS = re.compile(
    r'\b(prefer|rather|instead of|better than|over|compared to|versus|vs\.?)\b',
    re.IGNORECASE
)

# Topic pivot patterns (beyond just "back to")
TOPIC_PIVOT_PATTERNS = re.compile(
    r'\b(back to|returning to|going back to|anyway|speaking of|on another note|'
    r'changing topic|different subject|but about|so about|regarding)\b',
    re.IGNORECASE
)

# Implicit question patterns (functionally questions without ?)
IMPLICIT_QUESTION_PATTERNS = re.compile(
    r'\b(tell me|explain|describe|show me|help me understand|'
    r'i wonder|i\'m curious|wondering if|interested to know|'
    r'want to know|need to know|let me know)\b',
    re.IGNORECASE
)


@dataclass
class WeightedEntity:
    """Entity with weight based on type and context."""
    text: str
    lemma: str
    entity_type: str  # NER label or 'NOUN'/'PROPN'
    weight: float
    
    def __hash__(self):
        return hash(self.lemma)
    
    def __eq__(self, other):
        return self.lemma == other.lemma


@dataclass 
class EntityAnalysis:
    """Results of entity extraction with weights."""
    entities: list[WeightedEntity]
    total_weight: float
    high_value_entities: list[str]  # Entities with weight >= 2.0
    
    def get_entity_set(self) -> set[str]:
        return {e.lemma for e in self.entities}


@dataclass
class SentenceAnalysis:
    """Analysis of a single sentence."""
    text: str
    is_question: bool
    has_anaphoric_ref: bool
    has_preference: bool
    has_topic_pivot: bool
    entities: EntityAnalysis
    preferred_entity: Optional[str] = None  # The thing being preferred
    rejected_entity: Optional[str] = None   # The thing being rejected/compared against


@dataclass
class MessageAnalysis:
    """Full analysis of a message."""
    sentences: list[SentenceAnalysis]
    is_question: bool
    has_anaphoric_ref: bool
    has_preference: bool
    has_topic_pivot: bool
    all_entities: EntityAnalysis
    
    # Compound message detection
    is_compound: bool = False  # Multiple sentences with different intents
    pivot_detected: bool = False  # First part references old, second introduces new
    
    # Preference analysis
    preferred_entity: Optional[str] = None
    rejected_entity: Optional[str] = None


def extract_weighted_entities(doc: Doc) -> EntityAnalysis:
    """
    Extract entities with weights based on NER type.
    
    Higher weight = more significant for topic detection.
    """
    entities = []
    seen_lemmas = set()
    
    # Named entities from NER
    for ent in doc.ents:
        lemma = ent.text.lower()
        if lemma not in seen_lemmas and len(lemma) > 2:
            weight = ENTITY_WEIGHTS.get(ent.label_, 1.0)
            entities.append(WeightedEntity(
                text=ent.text,
                lemma=lemma,
                entity_type=ent.label_,
                weight=weight
            ))
            seen_lemmas.add(lemma)
    
    # Nouns and proper nouns not caught by NER
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in seen_lemmas or len(lemma) <= 3:
            continue
        if token.is_stop:
            continue
            
        if token.pos_ == 'PROPN':
            entities.append(WeightedEntity(
                text=token.text,
                lemma=lemma,
                entity_type='PROPN',
                weight=DEFAULT_PROPN_WEIGHT
            ))
            seen_lemmas.add(lemma)
        elif token.pos_ == 'NOUN':
            entities.append(WeightedEntity(
                text=token.text,
                lemma=lemma,
                entity_type='NOUN',
                weight=DEFAULT_NOUN_WEIGHT
            ))
            seen_lemmas.add(lemma)
    
    # Noun chunks for compound nouns
    for chunk in doc.noun_chunks:
        lemma = chunk.text.lower()
        if lemma not in seen_lemmas and len(lemma) > 4:
            # Weight based on whether it contains a proper noun
            has_propn = any(t.pos_ == 'PROPN' for t in chunk)
            weight = DEFAULT_PROPN_WEIGHT if has_propn else DEFAULT_NOUN_WEIGHT
            entities.append(WeightedEntity(
                text=chunk.text,
                lemma=lemma,
                entity_type='NOUN_CHUNK',
                weight=weight
            ))
            seen_lemmas.add(lemma)
    
    total_weight = sum(e.weight for e in entities)
    high_value = [e.lemma for e in entities if e.weight >= 2.0]
    
    return EntityAnalysis(
        entities=entities,
        total_weight=total_weight,
        high_value_entities=high_value
    )


def has_anaphoric_reference(doc: Doc) -> bool:
    """
    Detect anaphoric references using spaCy POS tagging.
    
    Only returns True if the pronoun likely refers to PREVIOUS context,
    not something mentioned in the same sentence.
    """
    # First, collect all nouns in the message that pronouns could refer to
    local_referents = set()
    for token in doc:
        if token.pos_ in ('NOUN', 'PROPN'):
            local_referents.add(token.lemma_.lower())
    
    for token in doc:
        # Demonstratives as subject/object at START of message
        # "That's cool" vs "I think that's wrong" (different)
        if token.text.lower() in {'this', 'that', 'these', 'those'}:
            # Only count if it's near the start (first 3 tokens) or is the subject
            if token.i <= 2 or token.dep_ in ('nsubj', 'nsubjpass'):
                if token.dep_ in ('nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr'):
                    return True
                if token.pos_ == 'PRON':
                    return True
        
        # Personal pronouns - but only if there's no local referent
        # "my car, it's making noise" - "it" refers to "car" (local)
        # "it's really cool" - "it" likely refers to previous context
        if token.text.lower() in {'it', 'its'}:
            if token.text.lower() == 'it' and token.dep_ == 'expl':
                continue
            # If there's a noun in the message, "it" probably refers to that
            if local_referents:
                continue
            if token.pos_ in ('PRON', 'DET'):
                return True
        
        # "they/them" usually refers to previous context if no plural noun present
        if token.text.lower() in {'they', 'them', 'their'}:
            if token.pos_ in ('PRON', 'DET'):
                # Check if there's a plural noun locally
                has_plural = any(
                    t.tag_ in ('NNS', 'NNPS') for t in doc
                )
                if not has_plural:
                    return True
    
    return False


def is_question(doc: Doc, raw_text: str) -> bool:
    """
    Detect questions - explicit and implicit.
    """
    # Explicit question mark
    if '?' in raw_text:
        return True
    
    # Interrogative words at start
    interrogatives = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose'}
    if doc and len(doc) > 0:
        first_word = doc[0].text.lower()
        if first_word in interrogatives:
            return True
        # Aux verb inversion
        if first_word in {'can', 'could', 'would', 'should', 'do', 'does', 'did', 
                          'is', 'are', 'was', 'were', 'will', 'have', 'has'}:
            return True
    
    # Implicit questions ("tell me about", "I wonder")
    if IMPLICIT_QUESTION_PATTERNS.search(raw_text):
        return True
    
    return False


def detect_preference(doc: Doc, raw_text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Detect preference/comparison statements and extract what's preferred vs rejected.
    
    "I prefer black holes to donald trump"
    -> preferred: "black holes", rejected: "donald trump"
    """
    if not PREFERENCE_PATTERNS.search(raw_text):
        return False, None, None
    
    preferred = None
    rejected = None
    
    # Look for "prefer X to Y" or "X over Y" patterns
    for token in doc:
        if token.text.lower() in {'prefer', 'rather'}:
            # Object of prefer is the preferred thing
            for child in token.children:
                if child.dep_ == 'dobj':
                    # Get the full noun phrase
                    preferred = get_noun_phrase(child)
                elif child.dep_ == 'prep' and child.text.lower() == 'to':
                    # Object of "to" is the rejected thing
                    for pobj in child.children:
                        if pobj.dep_ == 'pobj':
                            rejected = get_noun_phrase(pobj)
        
        # Handle "X over Y" pattern
        if token.text.lower() == 'over' and token.dep_ == 'prep':
            for pobj in token.children:
                if pobj.dep_ == 'pobj':
                    rejected = get_noun_phrase(pobj)
            # The thing before "over" is preferred
            if token.head.pos_ in ('NOUN', 'PROPN'):
                preferred = get_noun_phrase(token.head)
    
    return True, preferred, rejected


def get_noun_phrase(token: Token) -> str:
    """Extract full noun phrase from a token."""
    # Get all tokens in the subtree
    phrase_tokens = list(token.subtree)
    # Sort by position and join
    phrase_tokens.sort(key=lambda t: t.i)
    return ' '.join(t.text for t in phrase_tokens).strip()


def analyze_sentence(doc: Doc, raw_text: str) -> SentenceAnalysis:
    """Analyze a single sentence."""
    entities = extract_weighted_entities(doc)
    has_pref, preferred, rejected = detect_preference(doc, raw_text)
    
    return SentenceAnalysis(
        text=raw_text,
        is_question=is_question(doc, raw_text),
        has_anaphoric_ref=has_anaphoric_reference(doc),
        has_preference=has_pref,
        has_topic_pivot=bool(TOPIC_PIVOT_PATTERNS.search(raw_text)),
        entities=entities,
        preferred_entity=preferred,
        rejected_entity=rejected
    )


def analyze_message(doc: Doc, raw_text: str) -> MessageAnalysis:
    """
    Full analysis of a message with sentence-level breakdown.
    """
    # Split into sentences
    sentences = list(doc.sents)
    
    sentence_analyses = []
    for sent in sentences:
        sent_doc = sent.as_doc()
        sent_analysis = analyze_sentence(sent_doc, sent.text)
        sentence_analyses.append(sent_analysis)
    
    # Aggregate analysis
    all_entities = extract_weighted_entities(doc)
    is_q = any(s.is_question for s in sentence_analyses)
    has_anaph = any(s.has_anaphoric_ref for s in sentence_analyses)
    has_pref = any(s.has_preference for s in sentence_analyses)
    has_pivot = any(s.has_topic_pivot for s in sentence_analyses)
    
    # Detect compound message with pivot
    is_compound = len(sentence_analyses) > 1
    pivot_detected = False
    
    if is_compound and len(sentence_analyses) >= 2:
        first = sentence_analyses[0]
        rest = sentence_analyses[1:]
        
        # First sentence is anaphoric/reactive, rest introduces new entities
        if first.has_anaphoric_ref:
            rest_entities = set()
            for s in rest:
                rest_entities.update(s.entities.get_entity_set())
            
            first_entities = first.entities.get_entity_set()
            new_in_rest = rest_entities - first_entities
            
            if len(new_in_rest) >= 1:
                pivot_detected = True
    
    # Extract preference info
    preferred = None
    rejected = None
    for s in sentence_analyses:
        if s.preferred_entity:
            preferred = s.preferred_entity
        if s.rejected_entity:
            rejected = s.rejected_entity
    
    return MessageAnalysis(
        sentences=sentence_analyses,
        is_question=is_q,
        has_anaphoric_ref=has_anaph,
        has_preference=has_pref,
        has_topic_pivot=has_pivot,
        all_entities=all_entities,
        is_compound=is_compound,
        pivot_detected=pivot_detected,
        preferred_entity=preferred,
        rejected_entity=rejected
    )


def calculate_entity_overlap(
    current_entities: EntityAnalysis, 
    previous_entities: EntityAnalysis
) -> tuple[float, set[str], float]:
    """
    Calculate weighted entity overlap between messages.
    
    Returns: (overlap_score, shared_entities, new_entity_weight)
    """
    current_set = current_entities.get_entity_set()
    previous_set = previous_entities.get_entity_set()
    
    shared = current_set & previous_set
    new_entities = current_set - previous_set
    
    # Calculate weighted overlap
    shared_weight = sum(
        e.weight for e in current_entities.entities 
        if e.lemma in shared
    )
    
    new_weight = sum(
        e.weight for e in current_entities.entities
        if e.lemma in new_entities
    )
    
    # Overlap score based on weights
    if current_entities.total_weight == 0:
        overlap_score = 0.0
    else:
        overlap_score = shared_weight / current_entities.total_weight
    
    return overlap_score, shared, new_weight


def should_suppress_anaphoric_floor(
    current_analysis: MessageAnalysis,
    previous_entities: EntityAnalysis
) -> bool:
    """
    Determine if anaphoric floor should be suppressed.
    
    Suppress when:
    - Preference detected (user comparing/pivoting)
    - Topic pivot phrase detected
    - Compound message with significant new entities in second part
    - High-weight new entities introduced
    """
    if current_analysis.has_preference:
        return True
    
    if current_analysis.has_topic_pivot:
        return True
    
    if current_analysis.pivot_detected:
        return True
    
    # Check for significant new entities
    current_set = current_analysis.all_entities.get_entity_set()
    previous_set = previous_entities.get_entity_set()
    new_entities = current_set - previous_set
    
    # Calculate weight of new entities
    new_weight = sum(
        e.weight for e in current_analysis.all_entities.entities
        if e.lemma in new_entities
    )
    
    # If new entities have high weight, suppress floor
    if new_weight >= 4.0:  # e.g., one PERSON + one GPE
        return True
    
    # If multiple high-value new entities
    high_value_new = [
        e for e in current_analysis.all_entities.entities
        if e.lemma in new_entities and e.weight >= 2.0
    ]
    if len(high_value_new) >= 2:
        return True
    
    return False
