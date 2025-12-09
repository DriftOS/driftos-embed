"""
Text Preprocessing for Drift Detection

Uses spaCy for lemmatization + custom stopword removal.
Produces cleaner embeddings with better semantic separation.
"""

import re
from typing import List
import spacy

# Load spaCy model (small is fast, good enough for lemmatization)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    import subprocess
    import logging
    logging.getLogger("preprocessing").info("Downloading spaCy model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Words to completely remove (don't contribute to topic)
REMOVE_WORDS = {
    # Articles & Determiners
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'some', 'any',
    # Politeness markers
    'please', 'pls', 'plz', 'thanks', 'thank', 'thankyou', 'ty', 'sorry',
    # Fillers
    'just', 'really', 'very', 'quite', 'kind', 'kinda', 'sort', 'sortof',
    'actually', 'basically', 'literally', 'so', 'much', 'um', 'uh', 'well',
    'like', 'ok', 'okay', 'yeah', 'yes', 'no', 'right',
    # Question scaffolding (lemmatized forms)
    'can', 'could', 'would', 'should', 'do', 'be', 'have', 'will',
    'wonder', 'maybe', 'perhaps', 'possible', 'possibly',
    # Common low-signal verbs (lemmatized forms)
    'get', 'go', 'come', 'let', 'make', 'take', 'give', 'need', 'want',
    'know', 'think', 'see', 'look', 'find', 'tell', 'say', 'ask',
    # Pronouns
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
    'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
    'it', 'its', 'they', 'them', 'their', 'theirs',
    '-pron-',  # spaCy's pronoun placeholder
    # Question words
    'here', 'there', 'now', 'then', 'where', 'when', 'what', 'how', 'why', 'which',
    # Prepositions
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    # Conjunctions
    'and', 'or', 'but', 'if', 'because', 'while', 'although',
}


def preprocess(text: str) -> str:
    """
    Preprocess text for drift-optimized embeddings.
    
    Pipeline:
    1. Lowercase
    2. Strip punctuation
    3. Lemmatize (verbs → base, nouns → singular)
    4. Remove stopwords/fillers
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text with only topic-bearing lemmas
    """
    if not text or not text.strip():
        return ""
    
    # Lowercase and clean punctuation
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Process with spaCy
    doc = nlp(text)
    
    # Extract lemmas, filter stopwords
    lemmas = []
    for token in doc:
        lemma = token.lemma_.lower()
        # Skip if: stopword, too short, or in our remove list
        if (
            lemma not in REMOVE_WORDS 
            and len(lemma) > 1 
            and not token.is_punct
            and not token.is_space
        ):
            lemmas.append(lemma)
    
    result = " ".join(lemmas)
    
    # Fallback if too aggressive
    if len(lemmas) < 2:
        basic_filter = {'um', 'uh', 'like', 'just', 'really', 'actually', 'basically'}
        tokens = text.split()
        filtered = [t for t in tokens if t not in basic_filter and len(t) > 1]
        result = " ".join(filtered)
    
    return result


def preprocess_batch(texts: List[str]) -> List[str]:
    """Preprocess multiple texts using spaCy pipe for efficiency."""
    if not texts:
        return []
    
    # Clean texts first
    cleaned = []
    for text in texts:
        if not text or not text.strip():
            cleaned.append("")
        else:
            text = text.lower()
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            cleaned.append(text)
    
    # Batch process with spaCy pipe
    results = []
    for doc in nlp.pipe(cleaned, batch_size=50):
        lemmas = []
        for token in doc:
            lemma = token.lemma_.lower()
            if (
                lemma not in REMOVE_WORDS 
                and len(lemma) > 1 
                and not token.is_punct
                and not token.is_space
            ):
                lemmas.append(lemma)
        
        result = " ".join(lemmas)
        
        # Fallback
        if len(lemmas) < 2:
            basic_filter = {'um', 'uh', 'like', 'just', 'really', 'actually', 'basically'}
            tokens = doc.text.split()
            filtered = [t for t in tokens if t not in basic_filter and len(t) > 1]
            result = " ".join(filtered)
        
        results.append(result)
    
    return results
