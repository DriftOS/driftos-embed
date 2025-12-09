import type { DriftContext } from '../types';
import { cosineSimilarity, getDriftAction } from '@services/local-embeddings';

/**
 * Q&A Boost Factor
 *
 * When the previous message was a question and current message is an answer,
 * the semantic similarity is naturally lower. Boost to keep Q&A pairs together.
 */
const QA_BOOST_FACTOR = 1.3;

/**
 * Check if this is a Q&A pair (previous was question, current is answer).
 * Returns boosted similarity if so, otherwise returns original.
 */
function applyQAPairBoost(
  similarity: number,
  currentContent: string,
  lastMessageContent?: string
): number {
  if (!lastMessageContent) return similarity;

  const lastWasQuestion = lastMessageContent.includes('?');
  const currentIsAnswer = !currentContent.includes('?');

  // Boost only when: previous was question AND current is answer
  if (lastWasQuestion && currentIsAnswer) {
    return Math.min(similarity * QA_BOOST_FACTOR, 1.0);
  }
  return similarity;
}

/**
 * ClassifyRouteEmbed Operation
 *
 * Uses embedding similarity with drift-aware thresholds:
 * - STAY: similarity > stayThreshold (same topic/subtopic)
 * - BRANCH_SAME_CLUSTER: newClusterThreshold < similarity < stayThreshold (same domain, diff topic)
 * - BRANCH_NEW_CLUSTER: similarity < newClusterThreshold (different domain)
 *
 * Logic:
 * 1. No branches → BRANCH (first message)
 * 2. Check drift against current branch centroid
 * 3. If STAY → stay in current branch
 * 4. If BRANCH → check all branches for potential ROUTE
 * 5. Route if another branch is above route threshold
 * 6. Otherwise create new branch (same or new cluster based on drift action)
 */
export async function classifyRouteEmbed(ctx: DriftContext): Promise<DriftContext> {
  if (!ctx.embedding) {
    throw new Error('Missing embedding - run embedMessage first');
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

  // Calculate similarity to current branch (with Q&A pair boost if applicable)
  const rawSimilarity = cosineSimilarity(ctx.embedding, currentBranch.centroid);
  const currentSimilarity = applyQAPairBoost(rawSimilarity, ctx.content, ctx.lastMessageContent);
  const driftAction = getDriftAction(currentSimilarity, stayThreshold, newClusterThreshold);

  // Case 3: STAY - above stay threshold with current branch
  if (driftAction === 'STAY') {
    ctx.classification = {
      action: 'STAY',
      driftAction: 'STAY',
      targetBranchId: currentBranch.id,
      targetClusterId: currentBranch.clusterId,
      reason: `similar_to_current (${currentSimilarity.toFixed(3)} > ${stayThreshold})`,
      confidence: currentSimilarity,
      similarity: currentSimilarity,
    };
    ctx.reasonCodes.push('stay_similar');
    return ctx;
  }

  // Drift detected - check all other branches for potential ROUTE
  const otherBranches = ctx.branches.filter(
    (b) => !b.isCurrentBranch && b.centroid?.length
  );

  if (otherBranches.length > 0) {
    // Note: Q&A boost only applies to current branch (where the question was asked)
    // Routing to other branches uses raw similarity
    const branchScores = otherBranches
      .map((branch) => ({
        branch,
        similarity: cosineSimilarity(ctx.embedding!, branch.centroid),
      }))
      .sort((a, b) => b.similarity - a.similarity);

    const bestMatch = branchScores[0];

    // Case 4: ROUTE - another branch is above route threshold
    if (bestMatch && bestMatch.similarity > routeThreshold) {
      ctx.classification = {
        action: 'ROUTE',
        driftAction: getDriftAction(bestMatch.similarity, stayThreshold, newClusterThreshold),
        targetBranchId: bestMatch.branch.id,
        targetClusterId: bestMatch.branch.clusterId,
        reason: `routing_to_existing "${bestMatch.branch.summary}" (${bestMatch.similarity.toFixed(3)} > ${routeThreshold})`,
        confidence: bestMatch.similarity,
        similarity: bestMatch.similarity,
      };
      ctx.reasonCodes.push('route_existing');
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
