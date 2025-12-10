import { BaseOrchestrator, DefaultPerformanceTracker } from '@core/orchestration';
import type { PipelineStage } from '@core/orchestration';
import type { DriftContext, DriftResult, DriftInput, DriftPolicy } from './types';
import { getConfig } from '@plugins/env';
import * as ops from './operations';

/**
 * Get default policy from environment variables.
 * Falls back to gradient benchmark values if env vars not set.
 *
 * Thresholds:
 * - stayThreshold: 0.47 (above = same topic, STAY)
 * - newClusterThreshold: 0.20 (below = different domain, new cluster)
 * - routeThreshold: 0.55 (above = route to existing branch)
 */
function getDefaultPolicy(): DriftPolicy {
  try {
    const config = getConfig();
    return {
      maxBranchesForContext: config.DRIFT_MAX_BRANCHES_CONTEXT,
      stayThreshold: config.DRIFT_STAY_THRESHOLD,
      newClusterThreshold: config.DRIFT_NEW_CLUSTER_THRESHOLD,
      routeThreshold: config.DRIFT_ROUTE_THRESHOLD,
    };
  } catch {
    // Config not yet initialized (startup), use defaults
    return {
      maxBranchesForContext: 10,
      stayThreshold: 0.38,
      newClusterThreshold: 0.15,
      routeThreshold: 0.42,
    };
  }
}

/**
 * Drift Orchestrator
 *
 * Routes messages to the correct conversation branch using paraphrase-MiniLM-L6-v2
 * embedding similarity with drift-aware thresholds.
 * 
 * Actions:
 * - STAY: Continue in current branch (same topic)
 * - ROUTE: Move to existing branch (returning to previous topic)
 * - BRANCH: Create new branch (topic drift detected)
 *   - BRANCH_SAME_CLUSTER: Related domain, keep in same cluster
 *   - BRANCH_NEW_CLUSTER: Different domain, create new cluster
 */
export class DriftOrchestrator extends BaseOrchestrator<DriftContext, DriftResult, DriftInput> {
  constructor() {
    super({
      name: 'DriftOrchestrator',
      timeout: 10000,
      enableMetrics: true,
      logErrors: true,
    });
  }

  protected async initializeContext(input: DriftInput): Promise<DriftContext> {
    return {
      conversationId: input.conversationId,
      content: input.content,
      role: input.role ?? 'user',
      currentBranchId: input.currentBranchId,
      policy: { ...getDefaultPolicy(), ...input.policy },
      extractFacts: input.extractFacts ?? true,
      requestId: Math.random().toString(36).substr(2, 9),
      startTime: Date.now(),
      perfTracker: new DefaultPerformanceTracker(),
      results: {},
      errors: [],
      metadata: {
        orchestrator: this.getName(),
        model: 'paraphrase-MiniLM-L6-v2',
      },
      reasonCodes: [],
    };
  }

  protected getPipeline(): PipelineStage<DriftContext>[] {
    return [
      { name: 'validate-input', operation: ops.validateInput, critical: true },
      { name: 'load-branches', operation: ops.loadBranches, critical: true },
      { name: 'embed-message', operation: ops.embedMessage, critical: true },
      { name: 'classify-route', operation: ops.classifyRouteEmbed, critical: true },
      { name: 'execute-route', operation: ops.executeRoute, critical: true },
    ];
  }

  protected buildResult(ctx: DriftContext): DriftResult {
    if (!ctx.branch || !ctx.message) {
      throw new Error('Pipeline incomplete: missing branch or message');
    }

    const classification = ctx.classification;
    if (!classification) {
      throw new Error('Pipeline incomplete: missing classification');
    }

    return {
      action: classification.action,
      driftAction: classification.driftAction,
      branchId: ctx.branch.id,
      messageId: ctx.message.id,
      previousBranchId: ctx.currentBranch?.id !== ctx.branch.id ? ctx.currentBranchId : undefined,
      isNewBranch: classification.action === 'BRANCH',
      isNewCluster: classification.driftAction === 'BRANCH_NEW_CLUSTER',
      reason: classification.reason,
      reasonCodes: ctx.reasonCodes,
      metadata: ctx.metadata,
      branchTopic: ctx.branch.summary ?? 'Unknown',
      clusterId: classification.targetClusterId,
      confidence: classification.confidence,
      similarity: classification.similarity,
    };
  }
}
