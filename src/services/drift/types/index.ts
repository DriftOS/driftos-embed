import type { OperationContext } from '@core/orchestration/index.js';
import type { Branch, Message } from '@prisma/client';

/**
 * Drift Service Types
 * 
 * Based on gradient benchmark results:
 * - STAY threshold: 0.47 (same topic/subtopic)
 * - BRANCH_SAME_CLUSTER threshold: 0.20-0.47 (same domain, different topic)
 * - BRANCH_NEW_CLUSTER threshold: <0.20 (different domain)
 */

export type RouteAction = 'STAY' | 'ROUTE' | 'BRANCH';
export type DriftAction = 'STAY' | 'BRANCH_SAME_CLUSTER' | 'BRANCH_NEW_CLUSTER';

/**
 * Policy configuration for drift routing
 */
export interface DriftPolicy {
  maxBranchesForContext: number;
  /** Above this = STAY in current branch (default: 0.47) */
  stayThreshold: number;
  /** Below this = new cluster (default: 0.20) */
  newClusterThreshold: number;
  /** Above this to another branch = ROUTE (default: 0.55) */
  routeThreshold: number;
}

/**
 * Input for drift routing
 */
export interface DriftInput {
  conversationId: string;
  content: string;
  role?: 'user' | 'assistant';
  currentBranchId?: string;
  policy?: Partial<DriftPolicy>;
  /** Enable automatic fact extraction (default: true) */
  extractFacts?: boolean;
}

/**
 * Branch summary with centroid for embedding comparison
 */
export interface BranchSummary {
  id: string;
  summary: string;
  messageCount: number;
  isCurrentBranch: boolean;
  centroid: number[];
  clusterId?: string;
}

/**
 * Drift detection result from embedding server
 */
export interface DriftDetection {
  similarity: number;
  action: DriftAction;
  preprocessedAnchor?: string;
  preprocessedMessage?: string;
}

/**
 * Classification result with full context
 */
export interface Classification {
  action: RouteAction;
  driftAction: DriftAction;
  targetBranchId?: string;
  targetClusterId?: string;
  newBranchTopic?: string;
  reason: string;
  confidence: number;
  similarity: number;
}

/**
 * Internal context for drift pipeline
 */
export interface DriftContext extends OperationContext {
  conversationId: string;
  content: string;
  role: 'user' | 'assistant';
  currentBranchId?: string;
  policy: DriftPolicy;
  /** Enable automatic fact extraction (default: true) */
  extractFacts: boolean;

  reasonCodes: string[];
  currentBranch?: Branch;
  branches?: BranchSummary[];

  // Last message in current branch (for Q&A pair detection)
  lastMessageContent?: string;

  // Embedding data
  embedding?: number[];

  // Drift detection from embedding server
  driftDetection?: DriftDetection;

  // Final classification
  classification?: Classification;

  // Created entities
  message?: Message;
  branch?: Branch;
}

/**
 * Result from drift routing
 */
export interface DriftResult {
  action: RouteAction;
  driftAction: DriftAction;
  branchId: string;
  messageId: string;
  previousBranchId?: string;
  isNewBranch: boolean;
  isNewCluster: boolean;
  reason: string;
  branchTopic?: string;
  clusterId?: string;
  confidence: number;
  similarity: number;
  reasonCodes: string[];
  metadata?: Record<string, unknown>;
}
