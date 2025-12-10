import type { DriftContext } from '../types';
import { cosineSimilarity, getDriftAction, analyzeDrift, type DriftAnalysis } from '@services/local-embeddings';

/**
 * Topic return boost factor for routing to other branches.
 * Applied when user says "back to X", "returning to X", etc.
 * This is applied separately since it's used for ROUTE decisions to other branches.
 */
const TOPIC_RETURN_BOOST_FACTOR = 2.5;

/**
 * ClassifyRouteEmbed Operation
 *
 * Uses embedding similarity with drift-aware thresholds:
 * - STAY: similarity > stayThreshold (same topic/subtopic)
 * - BRANCH_SAME_CLUSTER: newClusterThreshold < similarity < stayThreshold (same domain, diff topic)
 * - BRANCH_NEW_CLUSTER: similarity < newClusterThreshold (different domain)
 *
 * Flow:
 * 1. No branches → BRANCH (first message)
 * 2. Call Python /analyze-drift for NLP analysis + boosted similarity
 * 3. If STAY → stay in current branch
 * 4. If drift detected → check all branches for potential ROUTE
 * 5. Route if another branch is above route threshold
 * 6. Otherwise create new branch (same or new cluster based on drift action)
 *
 * Python handles: spaCy NLP, similarity calculation, boost application
 * Node handles: routing decisions based on final numbers
 */
export async function classifyRouteEmbed(ctx: DriftContext): Promise<DriftContext> {
  if (!ctx.embedding) {
    throw new Error('Missing embedding - run embedMessage first');
  }

  // Assistant messages always STAY in current branch - no drift detection
  if (ctx.role === 'assistant') {
    const currentBranch = ctx.branches?.find((b) => b.isCurrentBranch);
    ctx.classification = {
      action: 'STAY',
      driftAction: 'STAY',
      targetBranchId: currentBranch?.id,
      targetClusterId: currentBranch?.clusterId,
      reason: 'assistant_message',
      confidence: 1.0,
      similarity: 1.0,
    };
    ctx.reasonCodes.push('assistant_auto_stay');
    return ctx;
  }

  const { stayThreshold, newClusterThreshold, routeThreshold } = ctx.policy;

  // Case 1: No branches yet - first message creates first branch
  if (!ctx.branches?.length) {
    ctx.classification = {
      action: 'BRANCH',
      driftAction: 'BRANCH_NEW_CLUSTER',
      newBranchTopic: extractTopic(ctx.content),
      reason: 'first_message',
      confidence: 1.0,
      similarity: 0,
    };
    ctx.reasonCodes.push('first_branch');
    return ctx;
  }

  // Find current branch
  const currentBranch = ctx.branches.find((b) => b.isCurrentBranch);

  // Case 2: Current branch has no centroid yet
  if (!currentBranch?.centroid?.length) {
    ctx.classification = {
      action: 'STAY',
      driftAction: 'STAY',
      targetBranchId: currentBranch?.id,
      reason: 'branch_initializing',
      confidence: 1.0,
      similarity: 1.0,
    };
    ctx.reasonCodes.push('branch_no_centroid');
    return ctx;
  }

  // Call Python for full drift analysis (NLP + similarity + boosts)
  let driftAnalysis: DriftAnalysis | null = null;
  let currentSimilarity: number;

  if (ctx.lastMessageContent) {
    // Have previous message - get full analysis with boosts
    driftAnalysis = await analyzeDrift(
      ctx.content,
      ctx.lastMessageContent,
      ctx.embedding,
      currentBranch.centroid
    );
    currentSimilarity = driftAnalysis.boosted_similarity;
  } else {
    // No previous message - just compute raw similarity
    currentSimilarity = cosineSimilarity(ctx.embedding, currentBranch.centroid);
  }

  const driftAction = getDriftAction(currentSimilarity, stayThreshold, newClusterThreshold);

  // Build reason suffix from analysis
  const boostSuffix = driftAnalysis?.boosts_applied.length
    ? `, boosts: ${driftAnalysis.boosts_applied.join('+')}`
    : '';

  // Case 3: STAY - above stay threshold with current branch
  if (driftAction === 'STAY') {
    ctx.classification = {
      action: 'STAY',
      driftAction: 'STAY',
      targetBranchId: currentBranch.id,
      targetClusterId: currentBranch.clusterId,
      reason: `similar_to_current (${currentSimilarity.toFixed(3)} > ${stayThreshold}${boostSuffix})`,
      confidence: currentSimilarity,
      similarity: currentSimilarity,
    };
    ctx.reasonCodes.push('stay_similar');
    driftAnalysis?.boosts_applied.forEach(b => ctx.reasonCodes.push(b));
    return ctx;
  }

  // Drift detected - check all other branches for potential ROUTE
  const otherBranches = ctx.branches.filter(
    (b) => !b.isCurrentBranch && b.centroid?.length
  );

  // Use topic return signal from analysis (already computed by Python)
  const hasTopicReturnSignal = driftAnalysis?.analysis.has_topic_return_signal ?? false;

  if (otherBranches.length > 0) {
    // Apply topic return boost if explicit signal detected
    const branchScores = otherBranches
      .map((branch) => {
        const rawSim = cosineSimilarity(ctx.embedding!, branch.centroid);
        const boostedSim = hasTopicReturnSignal ? rawSim * TOPIC_RETURN_BOOST_FACTOR : rawSim;
        return {
          branch,
          similarity: Math.min(boostedSim, 1.0),
          rawSimilarity: rawSim,
        };
      })
      .sort((a, b) => b.similarity - a.similarity);

    const bestMatch = branchScores[0];

    // Case 4: ROUTE - another branch is above route threshold
    if (bestMatch && bestMatch.similarity > routeThreshold) {
      ctx.classification = {
        action: 'ROUTE',
        driftAction: getDriftAction(bestMatch.similarity, stayThreshold, newClusterThreshold),
        targetBranchId: bestMatch.branch.id,
        targetClusterId: bestMatch.branch.clusterId,
        reason: `routing_to_existing "${bestMatch.branch.summary}" (${bestMatch.similarity.toFixed(3)} > ${routeThreshold}${hasTopicReturnSignal ? ', topic_return_boost' : ''})`,
        confidence: bestMatch.similarity,
        similarity: bestMatch.similarity,
      };
      ctx.reasonCodes.push('route_existing');
      if (hasTopicReturnSignal) ctx.reasonCodes.push('topic_return_signal');
      return ctx;
    }
  }

  // Case 5: BRANCH - create new branch
  ctx.classification = {
    action: 'BRANCH',
    driftAction,
    newBranchTopic: extractTopic(ctx.content),
    targetClusterId: driftAction === 'BRANCH_SAME_CLUSTER' ? currentBranch.clusterId : undefined,
    reason: `topic_drift (${currentSimilarity.toFixed(3)} < ${stayThreshold}, action: ${driftAction})`,
    confidence: 1 - currentSimilarity,
    similarity: currentSimilarity,
  };
  
  ctx.reasonCodes.push(
    driftAction === 'BRANCH_NEW_CLUSTER' ? 'branch_new_cluster' : 'branch_same_cluster'
  );

  return ctx;
}

/**
 * Extract a short topic from message content
 */
function extractTopic(content: string): string {
  const cleaned = content.replace(/\s+/g, ' ').trim();
  return cleaned.length > 100 ? `${cleaned.slice(0, 97)}...` : cleaned;
}
