/**
 * Embedding Service
 *
 * Calls the Python embedding server (paraphrase-MiniLM-L6-v2)
 * Optimized for drift detection with 0.556 gap between related/unrelated
 */

interface EmbedResponse {
  embeddings: number[][];
  dimension: number;
  model: string;
  preprocessed_texts?: string[];
}

interface SimilarityResponse {
  similarity: number;
  preprocessed_text1?: string;
  preprocessed_text2?: string;
}

interface DriftResponse {
  similarity: number;
  action: 'STAY' | 'BRANCH_SAME_CLUSTER' | 'BRANCH_NEW_CLUSTER';
  preprocessed_anchor?: string;
  preprocessed_message?: string;
}

interface HealthResponse {
  status: string;
  model: string;
  device: string;
  dimension: number;
}

interface EntityOverlapResponse {
  has_overlap: boolean;
  overlap_score: number;
  shared_entities: string[];
  text1_entities: string[];
  text2_entities: string[];
}

interface EntityOverlap {
  has_overlap: boolean;
  overlap_score: number;
  shared_entities: string[];
}

export interface MessageAnalysis {
  current_is_question: boolean;
  previous_is_question: boolean;
  current_has_anaphoric_ref: boolean;
  has_topic_return_signal: boolean;
  entity_overlap: EntityOverlap;
}

export interface DriftAnalysis {
  raw_similarity: number;
  boosted_similarity: number;
  boost_multiplier: number;
  boosts_applied: string[];
  analysis: MessageAnalysis;
}

const EMBEDDING_SERVER_URL = process.env.EMBEDDING_SERVER_URL ?? 'http://localhost:8100';

// Default thresholds from gradient benchmark
const DEFAULT_STAY_THRESHOLD = 0.47;
const DEFAULT_BRANCH_THRESHOLD = 0.20;

/**
 * Generate embedding for a single text
 */
export async function embed(text: string, preprocess = true): Promise<number[]> {
  const response = await fetch(`${EMBEDDING_SERVER_URL}/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, preprocess }),
  });

  if (!response.ok) {
    throw new Error(`Embedding server error: ${response.status} ${response.statusText}`);
  }

  const data: EmbedResponse = await response.json();
  const embedding = data.embeddings[0];
  
  if (!embedding) {
    throw new Error('No embedding returned from server');
  }
  
  return embedding;
}

/**
 * Generate embeddings for multiple texts (batch)
 */
export async function embedBatch(texts: string[], preprocess = true): Promise<number[][]> {
  const response = await fetch(`${EMBEDDING_SERVER_URL}/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: texts, preprocess }),
  });

  if (!response.ok) {
    throw new Error(`Embedding server error: ${response.status} ${response.statusText}`);
  }

  const data: EmbedResponse = await response.json();
  return data.embeddings;
}

/**
 * Compute cosine similarity between two texts
 */
export async function similarity(text1: string, text2: string, preprocess = true): Promise<number> {
  const response = await fetch(`${EMBEDDING_SERVER_URL}/similarity`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text1, text2, preprocess }),
  });

  if (!response.ok) {
    throw new Error(`Embedding server error: ${response.status} ${response.statusText}`);
  }

  const data: SimilarityResponse = await response.json();
  return data.similarity;
}

/**
 * Check drift between anchor context and new message
 * 
 * Returns action based on similarity thresholds:
 * - STAY: > stayThreshold (same topic/subtopic)
 * - BRANCH_SAME_CLUSTER: between thresholds (same domain, different topic)
 * - BRANCH_NEW_CLUSTER: < branchThreshold (different domain)
 */
export async function checkDrift(
  anchor: string,
  message: string,
  options?: {
    preprocess?: boolean;
    stayThreshold?: number;
    branchThreshold?: number;
  }
): Promise<DriftResponse> {
  const response = await fetch(`${EMBEDDING_SERVER_URL}/drift`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      anchor,
      message,
      preprocess: options?.preprocess ?? true,
      stay_threshold: options?.stayThreshold ?? DEFAULT_STAY_THRESHOLD,
      branch_threshold: options?.branchThreshold ?? DEFAULT_BRANCH_THRESHOLD,
    }),
  });

  if (!response.ok) {
    throw new Error(`Embedding server error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

/**
 * Compute cosine similarity from pre-computed embeddings
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Embedding dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    const aVal = a[i] ?? 0;
    const bVal = b[i] ?? 0;
    dotProduct += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Determine drift action from similarity score
 */
export function getDriftAction(
  similarity: number,
  stayThreshold = DEFAULT_STAY_THRESHOLD,
  branchThreshold = DEFAULT_BRANCH_THRESHOLD
): 'STAY' | 'BRANCH_SAME_CLUSTER' | 'BRANCH_NEW_CLUSTER' {
  if (similarity > stayThreshold) return 'STAY';
  if (similarity > branchThreshold) return 'BRANCH_SAME_CLUSTER';
  return 'BRANCH_NEW_CLUSTER';
}

/**
 * Health check for embedding server
 */
export async function healthCheck(): Promise<HealthResponse> {
  const response = await fetch(`${EMBEDDING_SERVER_URL}/health`);

  if (!response.ok) {
    throw new Error(`Embedding server unavailable: ${response.status}`);
  }

  return response.json();
}

/**
 * Check entity overlap between two texts using spaCy NER
 * Used to detect when user references something from previous message
 */
export async function checkEntityOverlap(text1: string, text2: string): Promise<EntityOverlapResponse> {
  const response = await fetch(`${EMBEDDING_SERVER_URL}/entity-overlap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text1, text2 }),
  });

  if (!response.ok) {
    // Non-fatal - return no overlap on error
    return {
      has_overlap: false,
      overlap_score: 0,
      shared_entities: [],
      text1_entities: [],
      text2_entities: [],
    };
  }

  return response.json();
}

/**
 * Analyze context between current and previous message.
 * Returns all signals needed for contextual boost calculation.
 */
export async function analyzeMessage(current: string, previous: string): Promise<MessageAnalysis> {
  const response = await fetch(`${EMBEDDING_SERVER_URL}/analyze-message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ current, previous }),
  });

  if (!response.ok) {
    // Non-fatal - return neutral analysis on error
    return {
      current_is_question: false,
      previous_is_question: false,
      current_has_anaphoric_ref: false,
      has_topic_return_signal: false,
      entity_overlap: {
        has_overlap: false,
        overlap_score: 0,
        shared_entities: [],
      },
    };
  }

  return response.json();
}

/**
 * Full drift analysis: NLP + similarity + boost application.
 * This is the main endpoint for drift detection.
 * 
 * Python handles all the analysis and returns the final boosted similarity.
 * Node just compares against thresholds to make routing decisions.
 */
export async function analyzeDrift(
  current: string,
  previous: string,
  currentEmbedding: number[],
  branchCentroid: number[]
): Promise<DriftAnalysis> {
  const response = await fetch(`${EMBEDDING_SERVER_URL}/analyze-drift`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      current,
      previous,
      current_embedding: currentEmbedding,
      branch_centroid: branchCentroid,
    }),
  });

  if (!response.ok) {
    // Fallback: return raw similarity with no boosts
    // Calculate cosine similarity manually
    const rawSim = cosineSimilarity(currentEmbedding, branchCentroid);
    return {
      raw_similarity: rawSim,
      boosted_similarity: rawSim,
      boost_multiplier: 1.0,
      boosts_applied: [],
      analysis: {
        current_is_question: false,
        previous_is_question: false,
        current_has_anaphoric_ref: false,
        has_topic_return_signal: false,
        entity_overlap: {
          has_overlap: false,
          overlap_score: 0,
          shared_entities: [],
        },
      },
    };
  }

  return response.json();
}

/**
 * Export thresholds for use elsewhere
 */
export const DRIFT_THRESHOLDS = {
  stay: DEFAULT_STAY_THRESHOLD,
  branch: DEFAULT_BRANCH_THRESHOLD,
} as const;
