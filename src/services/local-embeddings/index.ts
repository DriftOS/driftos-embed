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
 * Export thresholds for use elsewhere
 */
export const DRIFT_THRESHOLDS = {
  stay: DEFAULT_STAY_THRESHOLD,
  branch: DEFAULT_BRANCH_THRESHOLD,
} as const;
